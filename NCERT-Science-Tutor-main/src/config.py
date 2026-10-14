"""Single source of truth for every tunable in the system.

Design rules this module enforces:

1. **No magic numbers anywhere else.** Retrieval depth, chunk size, model names and
   timeouts all live here. Tuning the pipeline is a diff to one file, not a hunt
   through eight modules.
2. **Paths are anchored to the repository root, never to the process CWD.** The old
   code used relative literals like ``"db"``, which silently pointed at the wrong
   directory whenever the app was launched from anywhere but the repo root — and
   the failure mode was an *empty* index rather than a crash.
3. **Every value is overridable by environment variable.** Render (and any other
   PaaS) configures services through the environment; a config layer that can only
   be edited in source is a config layer you cannot deploy against.

See ``docs/adr/`` for the reasoning behind the defaults chosen here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Anchored at the repo root: src/config.py -> src/ -> <root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EmbeddingBackend = Literal["local", "cohere"]
LexicalBackend = Literal["fts5", "memory", "none"]


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from None


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}") from None


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key)
    if not raw:
        return default
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PathConfig:
    """Filesystem layout. All absolute, all derived from ``PROJECT_ROOT``."""

    root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: _env_path("NCERT_DATA_DIR", PROJECT_ROOT / "data"))
    db_dir: Path = field(default_factory=lambda: _env_path("NCERT_DB_DIR", PROJECT_ROOT / "db"))
    cache_dir: Path = field(default_factory=lambda: _env_path("NCERT_CACHE_DIR", PROJECT_ROOT / "cache"))

    @property
    def chroma_sqlite(self) -> Path:
        """The file Chroma actually writes. Its presence is the real readiness signal."""
        return self.db_dir / "chroma.sqlite3"

    @property
    def lexical_index(self) -> Path:
        """SQLite FTS5 lexical index — a build artifact, committed alongside ``db/``."""
        return self.db_dir / "lexical.sqlite3"

    @property
    def manifest(self) -> Path:
        """Records how the index was built, so serving can refuse an incompatible one."""
        return self.db_dir / "index_manifest.json"


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkConfig:
    """Chunking parameters. Rationale in ``docs/notes/chunking.md``.

    800/150 was chosen because NCERT prose averages ~55 characters per line and a
    self-contained explanatory passage runs 8-14 lines. 800 characters holds one
    complete idea; the 150-character overlap (~19%) keeps a definition from being
    severed from the sentence that qualifies it.
    """

    chunk_size: int = field(default_factory=lambda: _env_int("NCERT_CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("NCERT_CHUNK_OVERLAP", 150))

    # Paragraphs shorter than this are page furniture: headers, figure captions,
    # "Activity 6.3" labels, stray page numbers. They pollute BM25 term statistics
    # and produce embeddings that match everything weakly.
    min_paragraph_chars: int = field(default_factory=lambda: _env_int("NCERT_MIN_PARAGRAPH_CHARS", 80))

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); otherwise splitting never terminates."
            )


# --------------------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingConfig:
    """Which embedding backend serves queries.

    ``local``  — sentence-transformers in-process. Zero marginal cost, no network on
                 the query path, but pulls torch into the image and ~700 MB into RSS.
                 The right choice for development and for hosts with >=2 GB RAM.
    ``cohere`` — hosted embedding API. Removes torch from the serving image entirely,
                 which is what makes a 512 MB instance viable. Costs one ~30 ms API
                 call per query, negligible against a ~12 s generation call.

    The two backends produce *different vector spaces*. An index built with one
    cannot be queried with the other; ``IndexManifest`` enforces this at startup
    rather than letting it degrade into silently poor retrieval.
    """

    backend: EmbeddingBackend = field(
        default_factory=lambda: _env_str("NCERT_EMBEDDING_BACKEND", "local")  # type: ignore[arg-type]
    )
    local_model: str = field(
        default_factory=lambda: _env_str(
            "NCERT_LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
        )
    )
    cohere_model: str = field(
        default_factory=lambda: _env_str("NCERT_COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
    )
    # Cohere caps a single embed call at 96 inputs.
    batch_size: int = field(default_factory=lambda: _env_int("NCERT_EMBEDDING_BATCH_SIZE", 96))

    def __post_init__(self) -> None:
        if self.backend not in ("local", "cohere"):
            raise ValueError(f"NCERT_EMBEDDING_BACKEND must be 'local' or 'cohere', got {self.backend!r}")

    @property
    def model_id(self) -> str:
        """The identifier written into the manifest and checked at startup."""
        return self.local_model if self.backend == "local" else self.cohere_model


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval depth budget. Rationale in ``docs/notes/retrieval.md``.

    The funnel is 10 + 10 -> fuse -> 15 -> rerank -> 5. Each number is a cost/recall
    trade, not a round number:

    * ``dense_k``/``lexical_k`` = 10 — measured recall@10 on the eval set plateaus
      here; going to 20 doubles rerank cost for marginal recall.
    * ``candidate_k`` = 15 — the reranker is the expensive stage (it is a
      cross-encoder: cost is linear in candidates, not sublinear like ANN search).
      15 is the widest window that keeps rerank under ~1 s.
    * ``rerank_top_n`` = 5 — five 800-char chunks is ~1000 tokens of context, which
      keeps the generation prompt inside a comfortable budget for an 8B model whose
      instruction-following degrades noticeably past ~2k tokens of context.
    """

    dense_k: int = field(default_factory=lambda: _env_int("NCERT_DENSE_K", 10))
    lexical_k: int = field(default_factory=lambda: _env_int("NCERT_LEXICAL_K", 10))
    candidate_k: int = field(default_factory=lambda: _env_int("NCERT_CANDIDATE_K", 15))
    rerank_top_n: int = field(default_factory=lambda: _env_int("NCERT_RERANK_TOP_N", 5))

    # Reciprocal Rank Fusion smoothing constant. 60 is the value from Cormack et al.
    # (2009); it flattens the contribution curve so that the difference between rank
    # 1 and rank 2 does not dominate the fused ordering. See docs/notes/retrieval.md.
    rrf_k: int = field(default_factory=lambda: _env_int("NCERT_RRF_K", 60))

    # Relative trust in each retriever. Equal by default: on a textbook corpus,
    # lexical matching is genuinely competitive with dense retrieval because
    # students ask using the exact terminology the book defines.
    dense_weight: float = field(default_factory=lambda: _env_float("NCERT_DENSE_WEIGHT", 1.0))
    lexical_weight: float = field(default_factory=lambda: _env_float("NCERT_LEXICAL_WEIGHT", 1.0))

    lexical_backend: LexicalBackend = field(
        default_factory=lambda: _env_str("NCERT_LEXICAL_BACKEND", "fts5")  # type: ignore[arg-type]
    )

    def __post_init__(self) -> None:
        if self.lexical_backend not in ("fts5", "memory", "none"):
            raise ValueError(
                f"NCERT_LEXICAL_BACKEND must be 'fts5', 'memory' or 'none', got {self.lexical_backend!r}"
            )
        if self.rerank_top_n > self.candidate_k:
            raise ValueError(
                f"rerank_top_n ({self.rerank_top_n}) exceeds candidate_k ({self.candidate_k}); "
                "the reranker cannot return more documents than it is given."
            )


# --------------------------------------------------------------------------------------
# Reranking
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RerankConfig:
    """Cross-encoder reranking stage.

    ``model`` is pinned explicitly. The previous code called ``client.rerank()``
    without a model, relying on a server-side default that is not part of Cohere's
    compatibility contract — a silent-drift hazard on an unpinned SDK.
    """

    enabled: bool = field(default_factory=lambda: _env_bool("NCERT_RERANK_ENABLED", True))
    model: str = field(default_factory=lambda: _env_str("NCERT_RERANK_MODEL", "rerank-english-v3.0"))
    timeout_s: float = field(default_factory=lambda: _env_float("NCERT_RERANK_TIMEOUT", 15.0))
    api_key_env: str = "COHERE_API_KEY"


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationConfig:
    """LLM settings.

    ``temperature=0.0`` is deliberate and not a default left unexamined. The system
    is an *extractive* one: the prompt contract says every statement must be traceable
    to retrieved context. Sampling introduces exactly the failure mode the contract
    forbids, and it also makes the RAGAS faithfulness metric non-reproducible run to
    run. See ``docs/notes/prompting.md``.
    """

    base_url: str = field(
        default_factory=lambda: _env_str("NCERT_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )
    model: str = field(default_factory=lambda: _env_str("NCERT_LLM_MODEL", "meta/llama-3.1-8b-instruct"))
    temperature: float = field(default_factory=lambda: _env_float("NCERT_LLM_TEMPERATURE", 0.0))
    max_tokens: int = field(default_factory=lambda: _env_int("NCERT_LLM_MAX_TOKENS", 700))
    timeout_s: float = field(default_factory=lambda: _env_float("NCERT_LLM_TIMEOUT", 60.0))
    max_retries: int = field(default_factory=lambda: _env_int("NCERT_LLM_MAX_RETRIES", 2))
    api_key_env: str = "NVIDIA_API_KEY"


# --------------------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    paths: PathConfig = field(default_factory=PathConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    log_level: str = field(default_factory=lambda: _env_str("NCERT_LOG_LEVEL", "INFO"))

    # Classes covered by the corpus. Used by the ingestion walk and by the UI filter.
    classes: tuple[int, ...] = (6, 7, 8, 9, 10)

    def describe(self) -> dict[str, object]:
        """Flat summary for logs and the UI diagnostics panel."""
        return {
            "embedding_backend": self.embedding.backend,
            "embedding_model": self.embedding.model_id,
            "lexical_backend": self.retrieval.lexical_backend,
            "rerank_enabled": self.rerank.enabled,
            "rerank_model": self.rerank.model if self.rerank.enabled else None,
            "llm_model": self.generation.model,
            "retrieval_funnel": (
                f"{self.retrieval.dense_k}+{self.retrieval.lexical_k}"
                f" -> {self.retrieval.candidate_k} -> {self.retrieval.rerank_top_n}"
            ),
            "db_dir": str(self.paths.db_dir),
        }


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide settings singleton.

    Built once so that a Streamlit rerun does not re-read the environment and
    produce two configs that disagree. ``refresh=True`` exists for tests that
    manipulate ``os.environ``.
    """
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
