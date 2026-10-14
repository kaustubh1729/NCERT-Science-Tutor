"""Lexical (keyword) retrieval.

## Why lexical retrieval is not optional here

Dense retrieval is weakest exactly where a textbook tutor is used most: rare,
precise technical terms. "Stomata", "sonorous", "isobar" and "Dobereiner's triads"
appear a handful of times in the corpus, so an embedding model has seen little
evidence about where they belong and places them imprecisely. BM25 has the opposite
profile — the rarer the term, the *higher* its IDF weight and the sharper the match.
The two methods fail on disjoint query populations, which is the precondition for
fusion to help rather than average out. See ``docs/notes/retrieval.md``.

## Why FTS5 rather than rank_bm25

``rank_bm25`` is an in-memory library. Serving with it requires unpickling all 3,597
chunks, holding their full text as Python objects, and rebuilding the term-frequency
index on every cold start. Measured: ~60 MB resident and ~0.5 s of boot, paid again
after every one of Render's idle spin-downs.

SQLite FTS5 is an on-disk inverted index with a built-in ``bm25()`` ranking function.
It is a committed build artifact; opening it is a file handle, not a computation.
Boot cost drops to ~1 ms and resident cost to the OS page cache — which the kernel
can reclaim under pressure, unlike a Python heap.

There is a real trade being made: FTS5's tokenizer is not identical to
``rank_bm25``'s, and its ``bm25()`` uses fixed k1=1.2/b=0.75 with no exposed knobs.
Since fusion consumes *ranks* and not scores, the scoring difference is largely
absorbed downstream. Tokenizer differences are not, and are documented in
``docs/adr/0003-lexical-index.md``.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger(__name__)

# FTS5 treats these as query syntax. A student typing `What is "matter"?` or
# `NOT flammable` would otherwise produce a syntax error or a silently inverted
# query, so user text is never interpolated raw — every term is quoted.
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def fts5_available() -> bool:
    """Whether this interpreter's SQLite was compiled with FTS5.

    Nearly all mainstream Python builds include it, but Alpine-based and some
    minimal distro builds do not, and the failure appears only at index-build time.
    Callers use this to fall back rather than crash.
    """
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE _probe USING fts5(x)")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


def build_query(text: str) -> str:
    """Turn free-text into a safe FTS5 MATCH expression.

    Each token is double-quoted (making it a literal phrase, disarming operators)
    and the tokens are OR-ed. OR rather than AND is deliberate: a student question
    is a sentence, not a boolean query, and requiring every term to be present would
    return nothing for "What exactly is the function of stomata in a leaf?".
    Ranking, not filtering, is what separates good matches from bad ones here.
    """
    tokens = _TOKEN.findall(text)
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)


class FTS5Index:
    """Read/write wrapper over the on-disk lexical index."""

    def __init__(self, path: Path):
        self.path = Path(path)

    # ------------------------------------------------------------------- build
    def build(self, documents: Sequence[Document]) -> int:
        """Create the index from scratch. Called only by ``scripts/build_index.py``.

        Written to a temporary path and moved into place, so an interrupted build
        cannot leave a half-populated index that serving would happily open and
        return partial results from.
        """
        tmp_path = self.path.with_suffix(".building")
        tmp_path.unlink(missing_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(tmp_path)
        try:
            conn.executescript(
                """
                PRAGMA journal_mode = OFF;
                PRAGMA synchronous = OFF;

                -- Metadata lives in an ordinary table, not in the FTS table.
                -- Columns declared UNINDEXED inside FTS5 still occupy the inverted
                -- index's row storage; keeping them out halves the artifact size.
                CREATE TABLE chunks (
                    rowid      INTEGER PRIMARY KEY,
                    chunk_id   TEXT NOT NULL UNIQUE,
                    class      INTEGER,
                    source     TEXT,
                    page       INTEGER,
                    content    TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    content,
                    content='chunks',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                );
                """
            )

            rows = [
                (
                    i + 1,
                    str(doc.metadata.get("chunk_id", i)),
                    doc.metadata.get("class"),
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.page_content,
                )
                for i, doc in enumerate(documents)
            ]
            conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", rows)

            # 'rebuild' populates the external-content FTS index from `chunks`.
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            # 'optimize' merges the b-tree segments into one. Costs a few seconds at
            # build time and measurably shortens every subsequent query.
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
            conn.execute("CREATE INDEX idx_chunks_class ON chunks(class)")
            conn.commit()
            conn.execute("VACUUM")
            conn.commit()
        finally:
            conn.close()

        self.path.unlink(missing_ok=True)
        tmp_path.replace(self.path)

        size_mb = self.path.stat().st_size / 1e6
        logger.info("built FTS5 index: %d chunks, %.1f MB at %s", len(rows), size_mb, self.path)
        return len(rows)

    # ------------------------------------------------------------------ search
    def open(self) -> sqlite3.Connection:
        """Open read-only.

        ``mode=ro`` is not decoration. The index is a committed artifact and a bug
        that writes to it would corrupt a file tracked in git. Read-only also lets
        SQLite skip taking write locks, which matters because Streamlit serves
        concurrent sessions from one process.
        """
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)

    def search(
        self,
        conn: sqlite3.Connection,
        query: str,
        k: int,
        class_filter: int | None = None,
    ) -> list[Document]:
        match_expr = build_query(query)
        if not match_expr:
            return []

        # bm25() returns *negative* numbers, more negative meaning more relevant,
        # so ascending ORDER BY is the correct direction. Sorting the other way is a
        # silent and thoroughly convincing way to return the worst matches first.
        sql = """
            SELECT c.chunk_id, c.class, c.source, c.page, c.content, bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
        """
        params: list[object] = [match_expr]

        if class_filter is not None:
            sql += " AND c.class = ?"
            params.append(class_filter)

        sql += " ORDER BY score LIMIT ?"
        params.append(k)

        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 query failed (%s); returning no lexical results", exc)
            return []

        return [
            Document(
                page_content=row[4],
                metadata={
                    "chunk_id": row[0],
                    "class": row[1],
                    "source": row[2],
                    "page": row[3],
                    "type": "ncert_science",
                    "lexical_score": row[5],
                },
            )
            for row in rows
        ]

    def count(self) -> int:
        if not self.path.exists():
            return 0
        conn = self.open()
        try:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        finally:
            conn.close()


class InMemoryBM25:
    """``rank_bm25`` fallback.

    Kept for two situations: an index built before the FTS5 artifact existed, and
    interpreters without FTS5 compiled in. Costs ~60 MB and ~0.5 s per cold start,
    which is why it is the fallback and not the default.
    """

    def __init__(self, documents: Iterable[Document], k: int):
        from langchain_community.retrievers import BM25Retriever

        self._documents = list(documents)
        self._retriever = BM25Retriever.from_documents(self._documents)
        self._retriever.k = k
        logger.info("in-memory BM25 built over %d chunks", len(self._documents))

    def search(self, query: str, k: int, class_filter: int | None = None) -> list[Document]:
        self._retriever.k = k if class_filter is None else k * 4  # over-fetch, then filter
        results = self._retriever.invoke(query)
        if class_filter is not None:
            results = [d for d in results if d.metadata.get("class") == class_filter]
        return results[:k]

    def count(self) -> int:
        return len(self._documents)
