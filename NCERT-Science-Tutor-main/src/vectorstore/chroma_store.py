"""Chroma persistence plus the index manifest.

## The failure this module exists to prevent

An index built with one embedding model and queried with another does not raise. The
vectors have compatible shapes often enough to slip through, cosine similarity still
produces a total ordering, and the system returns five confidently-cited passages
that have nothing to do with the question. There is no runtime signal to check —
the numbers are all valid, they just mean nothing.

The only place this can be caught is at the boundary between build and serve. So the
build writes a manifest recording *how* the index was made, and startup refuses to
serve against a manifest it does not match. A loud failure at boot is worth a great
deal more than quiet nonsense at query time.

## Why the readiness check is not `os.listdir("db")`

The original ``_db_exists`` returned true for any non-empty ``db/`` directory. That
covers three states it should not: a directory holding only an empty ``chroma.sqlite3``,
a partially-copied index, and a build that crashed midway. Each produces a vector
store that opens cleanly and returns zero results forever. Readiness here means
"the collection contains vectors", verified by counting them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import PathConfig, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

MANIFEST_VERSION = 1


class IndexIncompatibleError(RuntimeError):
    """The on-disk index was not built for the current configuration."""


@dataclass
class IndexManifest:
    """What the index is, recorded at build time."""

    manifest_version: int
    embedding_backend: str
    embedding_model: str
    embedding_dim: int
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    built_at: str
    source_files: list[str]

    @classmethod
    def create(
        cls,
        embedding_backend: str,
        embedding_model: str,
        embedding_dim: int,
        chunk_count: int,
        chunk_size: int,
        chunk_overlap: int,
        source_files: Sequence[str],
    ) -> IndexManifest:
        return cls(
            manifest_version=MANIFEST_VERSION,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            chunk_count=chunk_count,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source_files=sorted(source_files),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        logger.info("wrote index manifest to %s", path)

    @classmethod
    def read(cls, path: Path) -> IndexManifest | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**data)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("index manifest at %s is unreadable (%s); ignoring", path, exc)
            return None

    def assert_compatible_with(self, embedding_model: str) -> None:
        """Refuse to serve if the query-time model differs from the build-time one."""
        if self.embedding_model != embedding_model:
            raise IndexIncompatibleError(
                f"Index was built with embedding model {self.embedding_model!r} but the "
                f"process is configured to query with {embedding_model!r}. These produce "
                f"different vector spaces, so retrieval would return confident nonsense.\n"
                f"Fix by either:\n"
                f"  - setting the matching backend/model in the environment, or\n"
                f"  - rebuilding: python -m scripts.build_index --force"
            )


class NCERTVectorStore:
    """Creates and opens the persisted Chroma collection."""

    # Chroma's LangChain integration defaults to this name; pinned explicitly so a
    # library default change cannot orphan the committed index.
    COLLECTION_NAME = "langchain"

    def __init__(self, paths: PathConfig | None = None):
        self.paths = paths or get_settings().paths
        self.persist_directory = str(self.paths.db_dir)

    # ------------------------------------------------------------------- build
    def create_store(self, documents: Sequence[Document], embedding_model) -> Chroma:
        """Build the collection from scratch.

        Documents are added in batches. Chroma's default HTTP/SQLite path degrades
        badly on a single enormous insert, and batching also keeps peak memory
        bounded — which matters when the build runs on a laptop rather than a
        build server.
        """
        self.paths.db_dir.mkdir(parents=True, exist_ok=True)

        store = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=self.persist_directory,
        )

        batch_size = 256
        for start in range(0, len(documents), batch_size):
            batch = list(documents[start : start + batch_size])
            store.add_documents(batch)
            logger.info("indexed %d/%d chunks", min(start + batch_size, len(documents)), len(documents))

        return store

    # -------------------------------------------------------------------- load
    def load_store(self, embedding_model) -> Chroma:
        return Chroma(
            collection_name=self.COLLECTION_NAME,
            persist_directory=self.persist_directory,
            embedding_function=embedding_model,
        )

    # --------------------------------------------------------------- readiness
    def is_ready(self) -> bool:
        """True only if the collection actually holds vectors."""
        if not self.paths.chroma_sqlite.exists():
            logger.info("no index at %s", self.paths.chroma_sqlite)
            return False
        count = self.count()
        if count == 0:
            logger.warning("index at %s exists but is empty", self.paths.db_dir)
            return False
        return True

    def count(self) -> int:
        """Number of vectors in the collection, read without loading an embedder.

        Queried through sqlite3 directly rather than by constructing a Chroma client,
        because on the local backend constructing one would load torch — an
        unacceptable cost for a health check that must answer in milliseconds.
        """
        import sqlite3

        if not self.paths.chroma_sqlite.exists():
            return 0
        try:
            conn = sqlite3.connect(f"file:{self.paths.chroma_sqlite}?mode=ro", uri=True)
            try:
                return int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("could not count vectors: %s", exc)
            return 0

    def embedding_dim(self) -> int | None:
        """Dimensionality recorded by Chroma, used to sanity-check the manifest."""
        import sqlite3

        if not self.paths.chroma_sqlite.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{self.paths.chroma_sqlite}?mode=ro", uri=True)
            try:
                row = conn.execute("SELECT dimension FROM collections LIMIT 1").fetchone()
                return int(row[0]) if row else None
            finally:
                conn.close()
        except sqlite3.Error:
            return None
