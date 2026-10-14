"""Cross-encoder reranking.

## What this stage buys

Retrieval is a *bi-encoder* process: the query and each chunk are embedded
independently and compared by cosine distance. That independence is what makes it
fast enough to search thousands of chunks — the document vectors are precomputed —
but it also means the model never sees the query and the document together. It
cannot represent "this passage answers *this* question"; only "this passage is about
a similar topic".

A reranker is a *cross-encoder*: query and document go through the network jointly,
so attention runs across both. It answers the question the bi-encoder structurally
cannot. The price is that nothing can be precomputed, so cost is linear in the
number of candidates — which is exactly why it sits at the narrow end of a funnel,
scoring 15 candidates rather than 3,597.

Concretely, on this corpus, RRF reliably places the right *region* of the textbook in
the top 15 but is indifferent about ordering within it. The reranker is what turns
"the answer is somewhere in these fifteen passages" into "these are the five".

## Two fixes over the original implementation

**The model is pinned.** The previous code called ``client.rerank()`` with no
``model`` argument against an unpinned SDK, relying on a server-side default that is
not part of Cohere's compatibility contract. A silent change there would degrade
answer quality with no error and no diff to point at.

**Failure degrades instead of propagating.** Reranking is a *precision* improvement
over an already-relevant candidate set. If Cohere is down, rate-limits a trial key,
or times out, returning the RRF ordering truncated to the same length is a materially
worse answer — not no answer. An outage in an optional quality stage should not be a
500.
"""

from __future__ import annotations

import os

from langchain_core.documents import Document

from src.config import RerankConfig, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CohereReranker:
    """Reranks candidates with Cohere Rerank, degrading to fusion order on failure."""

    def __init__(self, api_key: str | None = None, config: RerankConfig | None = None):
        self.config = config or get_settings().rerank
        self.available = False
        self._client = None

        if not self.config.enabled:
            logger.info("reranking disabled by configuration; using fusion order")
            return

        key = api_key or os.getenv(self.config.api_key_env)
        if not key:
            # A missing key is a deployment misconfiguration, not a crash. Logged
            # loudly, then the system runs in a documented degraded mode.
            logger.warning(
                "%s is not set — reranking disabled. Retrieval quality will be "
                "noticeably worse; set the key to restore it.",
                self.config.api_key_env,
            )
            return

        try:
            import cohere

            self._client = cohere.ClientV2(api_key=key, timeout=self.config.timeout_s)
            self.available = True
            logger.info("reranker ready: %s", self.config.model)
        except Exception as exc:
            logger.error("could not initialise Cohere client: %s — reranking disabled", exc)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """Reorder ``documents`` by joint query-document relevance.

        Args:
            query: The student's question.
            documents: Candidates from fusion, best-first.
            top_n: How many to keep; defaults to the configured value.

        Returns:
            At most ``top_n`` documents. On any failure, the input order truncated to
            ``top_n`` — never an exception, never an empty list.
        """
        top_n = top_n or get_settings().retrieval.rerank_top_n

        if not documents:
            return []

        if not self.available or self._client is None:
            return documents[:top_n]

        # Cohere rejects top_n greater than the number of documents supplied.
        effective_top_n = min(top_n, len(documents))

        try:
            response = self._client.rerank(
                query=query,
                documents=[doc.page_content for doc in documents],
                model=self.config.model,
                top_n=effective_top_n,
            )
        except Exception as exc:
            logger.warning("rerank failed (%s) — falling back to fusion order", exc)
            return documents[:top_n]

        reranked: list[Document] = []
        for position, result in enumerate(response.results):
            # Guard against an index outside the submitted list. It should not happen,
            # but an IndexError here would take down a request that has a perfectly
            # good fallback available.
            if result.index >= len(documents):
                logger.warning("reranker returned out-of-range index %d; skipped", result.index)
                continue
            doc = documents[result.index]
            doc.metadata["rerank_score"] = round(float(result.relevance_score), 6)
            doc.metadata["rerank_position"] = position + 1
            reranked.append(doc)

        if not reranked:
            logger.warning("reranker returned nothing usable — falling back to fusion order")
            return documents[:top_n]

        return reranked
