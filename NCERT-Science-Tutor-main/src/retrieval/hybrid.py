"""Hybrid retrieval: dense + lexical, combined by Reciprocal Rank Fusion.

## Why RRF and not score blending

The obvious combiner is ``alpha * cosine + (1 - alpha) * bm25``. It does not work
without per-corpus calibration, because the two scores are not comparable
quantities. Cosine similarity over normalised vectors is bounded in [-1, 1] and, on
a topically narrow corpus like one subject's textbooks, occupies a compressed band
near the top of that range — a *bad* match still scores ~0.3. BM25 is unbounded above
and scales with query length and term rarity; a two-word query and a twenty-word
query produce scores an order of magnitude apart. Any fixed ``alpha`` is therefore
implicitly tuned to one query length.

Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009) discards the scores
and keeps only the ordering:

    RRF(d) = Σ_retrievers  weight_r / (k + rank_r(d))

Rank is scale-free, so nothing needs calibrating. The constant ``k`` (60) flattens the
head of the curve: without it, rank 1 would contribute twice what rank 2 does, and a
single confident retriever would dictate the fused order. With k=60, ranks 1 and 2
contribute 1/61 and 1/62 — close enough that *agreement between retrievers*, not
confidence within one, is what lifts a document.

That property is the entire point. A document found at rank 4 by both retrievers
outranks one found at rank 1 by only one of them, which is precisely the signal we
want in front of the reranker.

## Why the two retrievers are queried unconditionally

There is a tempting optimisation: detect "keyword-ish" queries and skip the dense
search. It was rejected. The classifier would need to be right about a distinction
that has no crisp definition, and the two searches run concurrently against local
files in single-digit milliseconds. Spending accuracy to save time we are not short
of is a bad trade — the reranker and the LLM own the latency budget, not retrieval.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Sequence

from langchain_core.documents import Document

from src.config import RetrievalConfig, get_settings
from src.retrieval.lexical import FTS5Index, InMemoryBM25
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Fuses a dense vector search with a lexical search.

    ``lexical`` may be an :class:`FTS5Index`, an :class:`InMemoryBM25`, or ``None``
    (dense-only). The caller decides; this class only cares that it can ask for the
    top-k for a query.
    """

    def __init__(
        self,
        vectorstore,
        lexical: FTS5Index | InMemoryBM25 | None,
        config: RetrievalConfig | None = None,
    ):
        self.config = config or get_settings().retrieval
        self.vectorstore = vectorstore
        self.lexical = lexical
        self._fts_conn: sqlite3.Connection | None = None

        if isinstance(lexical, FTS5Index):
            self._fts_conn = lexical.open()

    # ------------------------------------------------------------------ fusion
    def reciprocal_rank_fusion(
        self,
        ranked_lists: Sequence[tuple[Sequence[Document], float]],
        k: int | None = None,
    ) -> list[Document]:
        """Fuse weighted ranked lists into one ordering.

        Args:
            ranked_lists: ``(documents, weight)`` pairs, each already sorted best-first.
            k: RRF smoothing constant; defaults to the configured value.

        Returns:
            Documents ordered by descending fused score, deduplicated by ``chunk_id``.
        """
        k = self.config.rrf_k if k is None else k
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}
        contributors: dict[str, int] = defaultdict(int)

        for documents, weight in ranked_lists:
            for rank, doc in enumerate(documents):
                doc_id = self._doc_id(doc)

                # First writer wins. Both retrievers return the same passage, but the
                # dense path's Document carries the vector store's metadata while the
                # lexical path's carries FTS row metadata. Keeping the first keeps
                # citations consistent regardless of which retriever found it first.
                doc_map.setdefault(doc_id, doc)
                contributors[doc_id] += 1
                scores[doc_id] += weight / (k + rank + 1)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        fused = []
        for doc_id, score in ordered:
            doc = doc_map[doc_id]
            # Surfaced for the UI diagnostics panel and for eval error analysis:
            # knowing whether a hit came from one retriever or both is the single
            # most useful signal when a retrieval looks wrong.
            doc.metadata["fusion_score"] = round(score, 6)
            doc.metadata["retriever_hits"] = contributors[doc_id]
            fused.append(doc)
        return fused

    @staticmethod
    def _doc_id(doc: Document) -> str:
        """Identity for deduplication.

        ``chunk_id`` is expected, but indexes built before content-hash ids may lack
        it, and a ``KeyError`` deep inside fusion is a poor way to find that out.
        Content hash is the fallback: slower, but it deduplicates correctly, which
        is the only thing fusion actually needs.
        """
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id is not None:
            return str(chunk_id)
        return f"sha:{hash(doc.page_content)}"

    # --------------------------------------------------------------- retrieval
    def _dense_search(self, query: str, class_filter: int | None) -> list[Document]:
        # Chroma pushes metadata filters down into the query rather than filtering
        # after the fact, so a class-filtered search still returns a *full* k
        # results from within that class instead of whatever survives post-hoc.
        where = {"class": class_filter} if class_filter is not None else None
        try:
            return self.vectorstore.similarity_search(query, k=self.config.dense_k, filter=where)
        except Exception as exc:
            logger.error("dense search failed: %s", exc)
            return []

    def _lexical_search(self, query: str, class_filter: int | None) -> list[Document]:
        if self.lexical is None:
            return []
        try:
            if isinstance(self.lexical, FTS5Index):
                assert self._fts_conn is not None
                return self.lexical.search(
                    self._fts_conn, query, k=self.config.lexical_k, class_filter=class_filter
                )
            return self.lexical.search(query, k=self.config.lexical_k, class_filter=class_filter)
        except Exception as exc:
            logger.error("lexical search failed: %s", exc)
            return []

    def retrieve(self, query: str, class_filter: int | None = None) -> list[Document]:
        """Run both retrievers, fuse, and return the reranker's candidate window.

        Args:
            query: The student's question, verbatim.
            class_filter: Restrict to one NCERT class (6-10), or ``None`` for all.

        Returns:
            At most ``candidate_k`` documents, best-first.
        """
        dense = self._dense_search(query, class_filter)
        lexical = self._lexical_search(query, class_filter)

        if not dense and not lexical:
            logger.warning("no candidates for query %r (class_filter=%s)", query, class_filter)
            return []

        fused = self.reciprocal_rank_fusion(
            [
                (dense, self.config.dense_weight),
                (lexical, self.config.lexical_weight),
            ]
        )

        logger.info(
            "retrieved dense=%d lexical=%d -> fused=%d -> candidates=%d",
            len(dense),
            len(lexical),
            len(fused),
            min(len(fused), self.config.candidate_k),
        )
        return fused[: self.config.candidate_k]

    def close(self) -> None:
        if self._fts_conn is not None:
            self._fts_conn.close()
            self._fts_conn = None
