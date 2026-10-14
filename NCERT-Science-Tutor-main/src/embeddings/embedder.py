"""Embedding backends.

## The decision this module encodes

Measured on this corpus (3,597 chunks, 768-dim), the serving process with
sentence-transformers loaded peaks at **943 MB RSS**. Render's free and Starter
instances both cap at 512 MB. The model is not the largest contributor — torch
itself is — so shrinking the model does not solve it.

That leaves three ways to ship:

1. Pay for a 2 GB instance and change no code.
2. Move embedding off-process to a hosted API, removing torch from the serving image.
3. Replace torch with an ONNX runtime.

This module implements (1) and (2) behind one interface, selected by
``NCERT_EMBEDDING_BACKEND``. ``docs/adr/0002-embedding-backend.md`` records why (2)
is the deployed default and (3) was rejected.

## The invariant

Two backends means two vector spaces. An index built by one and queried by the other
returns results that are *plausible but wrong* — cosine similarity over mismatched
spaces still yields an ordering, just a meaningless one. There is no exception this
system can throw at query time to notice that. So the model identity is written into
the index manifest at build time and verified at startup; see ``src/vectorstore``.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from src.config import EmbeddingConfig, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """The surface LangChain's vector stores require.

    Deliberately structural rather than a base class: it is exactly LangChain's
    ``Embeddings`` interface, so a raw ``HuggingFaceEmbeddings`` also satisfies it
    and can be passed straight through without a wrapper.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class CohereEmbeddings:
    """Hosted embeddings via Cohere.

    Two details that are easy to get wrong and expensive to debug:

    * **Asymmetric input types.** Cohere's v3 embedding models are trained with
      distinct ``search_document`` and ``search_query`` prefixes. Using the same
      type for both costs several points of retrieval quality — the model has learned
      to place a question and its answer *near* each other, not to place them in the
      same region. This is the most commonly skipped step when adopting these models.
    * **Batch ceiling.** A single embed call accepts at most 96 inputs. Indexing
      3,597 chunks is therefore ~38 calls, comfortably inside a trial key's quota.
    """

    def __init__(self, config: EmbeddingConfig, api_key: str | None = None):
        import cohere  # imported lazily so the local backend never pays for it

        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "NCERT_EMBEDDING_BACKEND=cohere requires COHERE_API_KEY to be set. "
                "Either set it, or switch to NCERT_EMBEDDING_BACKEND=local."
            )

        self.config = config
        self.model = config.cohere_model
        self._client = cohere.ClientV2(api_key=key)

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            response = self._client.embed(
                texts=batch,
                model=self.model,
                input_type=input_type,
                embedding_types=["float"],
            )
            vectors.extend(response.embeddings.float_)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, input_type="search_document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="search_query")[0]


class NCERTEmbedder:
    """Factory that resolves the configured backend.

    The heavy import lives inside the branch that needs it. That is not stylistic:
    importing ``langchain_huggingface`` pulls in torch and costs ~6 s and several
    hundred MB before a single vector is computed. On the Cohere path that cost must
    never be paid, and a module-level import would pay it unconditionally.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or get_settings().embedding
        self._model: EmbeddingProvider | None = None

    def get_model(self) -> EmbeddingProvider:
        if self._model is not None:
            return self._model

        if self.config.backend == "cohere":
            logger.info("embedding backend: cohere (%s)", self.config.cohere_model)
            self._model = CohereEmbeddings(self.config)
        else:
            logger.info("embedding backend: local (%s) — loading torch", self.config.local_model)
            from langchain_huggingface import HuggingFaceEmbeddings

            self._model = HuggingFaceEmbeddings(
                model_name=self.config.local_model,
                # Explicit rather than inferred. Render has no GPU; letting
                # sentence-transformers probe for CUDA wastes startup time and, on
                # some torch builds, emits a warning that looks like a real failure
                # in the deploy logs.
                model_kwargs={"device": "cpu"},
                # all-mpnet-base-v2 is trained with cosine similarity; normalising
                # here lets the vector store use a plain dot product.
                encode_kwargs={"normalize_embeddings": True},
            )

        return self._model

    @property
    def model_id(self) -> str:
        return self.config.model_id
