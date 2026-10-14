"""Pipeline orchestration.

## The startup bug this rewrite removes

The original ``_initialize`` had two independent branches. The first checked whether
``db/`` existed and loaded the vector store if so. The second, unconditionally,
needed BM25 documents — and when ``cache/bm25_docs.pkl`` was absent it did:

    chunks = cache.load("chunks")

with no existence check. So a deployment carrying a committed ``db/`` but no
``cache/`` — which is exactly what ``.gitignore`` described, and what ``make clean``
produces — crashed at boot with ``FileNotFoundError``, after the user had already
waited through the model load.

More fundamentally, the constructor could decide to *build the index*: load five
PDFs, chunk them, and embed 3,597 passages, inline, during application startup. On a
platform with a boot health-check timeout that is not a slow start, it is a failed
deploy that retries forever.

## The principle applied instead

**Building the index is a build-time job. Serving is a read-only job.** Startup now
opens artifacts or fails with an actionable message; it never creates them. Index
construction lives in ``scripts/build_index.py`` and runs on a developer machine,
where it can take ten minutes without anything timing out.

This is the decision recorded in ``docs/adr/0001-prebuilt-index.md``, and everything
else about the deployment follows from it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from src.config import Settings, get_settings
from src.embeddings.embedder import NCERTEmbedder
from src.generation.generator import GenerationError, NCERTGenerator
from src.generation.prompt import PromptBuilder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.lexical import FTS5Index, InMemoryBM25
from src.retrieval.reranker import CohereReranker
from src.utils.cache import CacheManager
from src.utils.logger import get_logger
from src.vectorstore.chroma_store import IndexManifest, NCERTVectorStore

logger = get_logger(__name__)


class IndexNotBuiltError(RuntimeError):
    """No usable index on disk. Carries the command that fixes it."""


@dataclass
class QueryResult:
    """One question's worth of output.

    A dataclass rather than a dict so that callers get attribute access and type
    checking, and so adding a field cannot silently break a consumer that iterates
    keys. ``as_dict`` preserves the original dict shape for backwards compatibility.
    """

    query: str
    answer: str
    documents: list[Document]
    timings: dict[str, float]
    citations: dict[str, dict] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "documents": self.documents,
            "timings": self.timings,
            "citations": self.citations,
            "degraded": self.degraded,
        }


class RAGPipeline:
    """Query-serving pipeline. Read-only with respect to the index."""

    def __init__(
        self,
        cohere_api_key: str | None = None,
        settings: Settings | None = None,
        require_generator: bool = True,
    ):
        self.settings = settings or get_settings()
        self.degraded: list[str] = []

        started = time.time()

        self.embedder = NCERTEmbedder(self.settings.embedding)
        self.vectorstore_manager = NCERTVectorStore(self.settings.paths)

        self._assert_index_ready()
        self._assert_index_compatible()

        # Loading the embedding model is the expensive step on the local backend
        # (~5 s, ~700 MB). It happens after the cheap validation above, so a
        # misconfigured deployment fails in milliseconds rather than after the wait.
        embedding_model = self.embedder.get_model()
        self.vectorstore = self.vectorstore_manager.load_store(embedding_model)

        self.lexical = self._build_lexical_index()
        self.retriever = HybridRetriever(self.vectorstore, self.lexical, self.settings.retrieval)
        self.reranker = CohereReranker(cohere_api_key, self.settings.rerank)

        if not self.reranker.available and self.settings.rerank.enabled:
            self.degraded.append("reranking unavailable — results ordered by fusion score only")

        self.generator = None
        try:
            self.generator = NCERTGenerator(config=self.settings.generation)
        except GenerationError as exc:
            if require_generator:
                raise
            # Retrieval-only mode is genuinely useful: it lets the eval harness score
            # the retriever without an LLM key, and it lets the UI show sources when
            # generation is misconfigured instead of showing a stack trace.
            logger.warning("generator unavailable: %s", exc)
            self.degraded.append("answer generation unavailable — showing retrieved sources only")

        self.prompt_builder = PromptBuilder()

        logger.info(
            "pipeline ready in %.1fs | %s",
            time.time() - started,
            self.settings.describe(),
        )
        for note in self.degraded:
            logger.warning("degraded: %s", note)

    # ------------------------------------------------------------------ startup
    def _assert_index_ready(self) -> None:
        if self.vectorstore_manager.is_ready():
            return
        raise IndexNotBuiltError(
            f"No vector index found at {self.settings.paths.db_dir}.\n\n"
            f"The index is a build artifact, not something the server creates at "
            f"startup. Build it once with:\n\n"
            f"    python -m scripts.build_index\n\n"
            f"then commit db/ so deployments start against a ready index."
        )

    def _assert_index_compatible(self) -> None:
        """Verify build-time and query-time embedding models agree.

        An absent manifest is tolerated — indexes built before manifests existed are
        still perfectly usable — but it is logged, because it means the guard is
        inactive and a model mismatch would go undetected.
        """
        manifest = IndexManifest.read(self.settings.paths.manifest)
        if manifest is None:
            logger.warning(
                "no index manifest at %s — cannot verify that the index matches "
                "embedding model %r. Rebuild to enable this check.",
                self.settings.paths.manifest,
                self.embedder.model_id,
            )
            return
        manifest.assert_compatible_with(self.embedder.model_id)
        logger.info(
            "index manifest OK: %d chunks, %s, built %s",
            manifest.chunk_count,
            manifest.embedding_model,
            manifest.built_at,
        )

    def _build_lexical_index(self) -> FTS5Index | InMemoryBM25 | None:
        """Resolve the lexical backend, degrading rather than failing.

        Order of preference: the committed FTS5 artifact, then the legacy pickle,
        then dense-only. Dense-only is a real quality regression on rare-term
        queries, so it is recorded in ``degraded`` and surfaced in the UI.
        """
        backend = self.settings.retrieval.lexical_backend

        if backend == "none":
            return None

        if backend == "fts5":
            index = FTS5Index(self.settings.paths.lexical_index)
            if index.path.exists() and index.count() > 0:
                logger.info("lexical backend: FTS5 (%d chunks)", index.count())
                return index
            logger.warning(
                "no FTS5 index at %s — falling back to in-memory BM25. "
                "Run `python -m scripts.build_index` to build it.",
                index.path,
            )

        cache = CacheManager(self.settings.paths.cache_dir)
        if cache.exists("bm25_docs"):
            docs = cache.load("bm25_docs")
            if docs:
                self.degraded.append(
                    "using in-memory BM25 (slower cold start, ~60 MB extra) — "
                    "rebuild the index to use the on-disk FTS5 artifact"
                )
                return InMemoryBM25(docs, k=self.settings.retrieval.lexical_k)

        logger.error("no lexical index available — running dense-only")
        self.degraded.append("lexical search unavailable — rare-term queries will be weaker")
        return None

    # -------------------------------------------------------------------- query
    def retrieve(self, query: str, class_filter: int | None = None) -> list[Document]:
        """Retrieval and reranking, without generation. Used by the eval harness."""
        candidates = self.retriever.retrieve(query, class_filter=class_filter)
        return self.reranker.rerank(query, candidates)

    def run(self, query: str, class_filter: int | None = None) -> QueryResult:
        """Answer one question end to end.

        Args:
            query: The student's question.
            class_filter: Restrict retrieval to one NCERT class (6-10), or ``None``.

        Raises:
            ValueError: The query is empty.
            GenerationError: The LLM call failed in a way the user should see.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("Query must not be empty.")

        timings: dict[str, float] = {}
        overall = time.perf_counter()

        mark = time.perf_counter()
        candidates = self.retriever.retrieve(query, class_filter=class_filter)
        timings["retrieval_time"] = time.perf_counter() - mark

        mark = time.perf_counter()
        documents = self.reranker.rerank(query, candidates)
        timings["rerank_time"] = time.perf_counter() - mark

        mark = time.perf_counter()
        if self.generator is None:
            answer = (
                "Answer generation is not configured on this deployment. "
                "The passages retrieved for your question are shown below."
            )
        else:
            answer = self.generator.generate_answer(query, documents)
        timings["generation_time"] = time.perf_counter() - mark

        timings["total_time"] = time.perf_counter() - overall

        return QueryResult(
            query=query,
            answer=answer,
            documents=documents,
            timings=timings,
            citations=self.prompt_builder.citation_map(documents),
            degraded=list(self.degraded),
        )

    def stream(self, query: str, class_filter: int | None = None) -> tuple[list[Document], Iterator[str]]:
        """Retrieve, then return sources and a token iterator.

        Sources come back *before* the first token so the UI can render citations
        while generation is still streaming. Users start reading the sources during
        the ~12 s the model takes, which removes most of the perceived wait.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("Query must not be empty.")

        candidates = self.retriever.retrieve(query, class_filter=class_filter)
        documents = self.reranker.rerank(query, candidates)

        if self.generator is None:

            def _unavailable() -> Iterator[str]:
                yield "Answer generation is not configured on this deployment."

            return documents, _unavailable()

        return documents, self.generator.stream_answer(query, documents)

    # ------------------------------------------------------------------- health
    def health(self) -> dict[str, Any]:
        """Snapshot for the UI diagnostics panel and the preflight script."""
        lexical_count = self.lexical.count() if self.lexical is not None else 0
        return {
            "vectors": self.vectorstore_manager.count(),
            "embedding_dim": self.vectorstore_manager.embedding_dim(),
            "lexical_chunks": lexical_count,
            "lexical_backend": type(self.lexical).__name__ if self.lexical else "none",
            "reranker": self.settings.rerank.model if self.reranker.available else "disabled",
            "generator": self.settings.generation.model if self.generator else "disabled",
            "degraded": self.degraded,
            **self.settings.describe(),
        }

    def close(self) -> None:
        self.retriever.close()


def build_pipeline(require_generator: bool = True) -> RAGPipeline:
    """Convenience constructor that reads keys from the environment."""
    return RAGPipeline(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        require_generator=require_generator,
    )
