#!/usr/bin/env python3
"""Verify a deployment can actually serve, before it tries to.

    python -m scripts.preflight

Replaces the old ``setup_render.py``, which had two problems. It ran as a Render
*background worker* — a service type with its own filesystem, so any index it built
was invisible to the web service — and it would trigger a full index build as a side
effect of a "setup check", inside a container with a boot timeout.

This script only inspects. It exits non-zero when the deployment cannot serve, so it
is usable as a build-command gate: a broken deploy fails during build, with a clear
reason, instead of at 3am with a stack trace in the logs.

Deliberately imports nothing heavy — no torch, no Chroma client. It must be able to
report "the index is missing" in well under a second.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.config import get_settings
from src.vectorstore.chroma_store import IndexManifest

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, check: str, detail: str) -> None:
        self.rows.append((status, check, detail))

    @property
    def failed(self) -> bool:
        return any(status == FAIL for status, _, _ in self.rows)

    def render(self) -> str:
        symbols = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL"}
        width = max(len(check) for _, check, _ in self.rows) if self.rows else 10
        lines = ["", "=" * 78, "  Preflight", "=" * 78]
        for status, check, detail in self.rows:
            lines.append(f"  [{symbols[status]}] {check.ljust(width)}  {detail}")
        lines.append("=" * 78)
        return "\n".join(lines)


def check_index(report: Report) -> None:
    settings = get_settings()
    db = settings.paths.chroma_sqlite

    if not db.exists():
        report.add(
            FAIL,
            "vector index",
            f"missing at {db} - run `python -m scripts.build_index` and commit db/",
        )
        return

    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        count = int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
        dim_row = conn.execute("SELECT dimension FROM collections LIMIT 1").fetchone()
        conn.close()
    except sqlite3.Error as exc:
        report.add(FAIL, "vector index", f"unreadable: {exc}")
        return

    if count == 0:
        report.add(FAIL, "vector index", "present but contains zero vectors - rebuild required")
    else:
        dim = dim_row[0] if dim_row else "?"
        report.add(PASS, "vector index", f"{count} vectors, {dim} dims")


def check_manifest(report: Report) -> None:
    settings = get_settings()
    manifest = IndexManifest.read(settings.paths.manifest)

    if manifest is None:
        report.add(
            WARN,
            "index manifest",
            "absent - the embedding-model compatibility check is inactive",
        )
        return

    configured = settings.embedding.model_id
    if manifest.embedding_model != configured:
        report.add(
            FAIL,
            "index manifest",
            f"built with {manifest.embedding_model!r}, configured for {configured!r} - "
            f"incompatible vector spaces",
        )
    else:
        report.add(PASS, "index manifest", f"{manifest.embedding_model} ({manifest.chunk_count} chunks)")


def check_lexical(report: Report) -> None:
    settings = get_settings()
    backend = settings.retrieval.lexical_backend

    if backend == "none":
        report.add(WARN, "lexical index", "disabled - rare-term queries will be weaker")
        return

    path = settings.paths.lexical_index
    if path.exists():
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            conn.close()
            report.add(PASS, "lexical index", f"FTS5, {count} chunks, {path.stat().st_size / 1e6:.1f} MB")
            return
        except sqlite3.Error as exc:
            report.add(WARN, "lexical index", f"FTS5 file unreadable ({exc}) - will fall back")

    fallback = settings.paths.cache_dir / "bm25_docs.pkl"
    if fallback.exists():
        report.add(
            WARN,
            "lexical index",
            "no FTS5 artifact; falling back to in-memory BM25 (~60 MB, slower cold start)",
        )
    else:
        report.add(FAIL, "lexical index", "no FTS5 artifact and no BM25 cache - run build_index")


def check_env(report: Report) -> None:
    settings = get_settings()

    if os.getenv("NVIDIA_API_KEY"):
        report.add(PASS, "NVIDIA_API_KEY", "set")
    else:
        report.add(FAIL, "NVIDIA_API_KEY", "missing - answer generation will not work")

    if os.getenv("COHERE_API_KEY"):
        report.add(PASS, "COHERE_API_KEY", "set")
    elif settings.rerank.enabled:
        report.add(WARN, "COHERE_API_KEY", "missing - reranking disabled, retrieval quality reduced")
    else:
        report.add(PASS, "COHERE_API_KEY", "not required (reranking disabled)")

    if settings.embedding.backend == "cohere" and not os.getenv("COHERE_API_KEY"):
        report.add(FAIL, "embedding backend", "set to 'cohere' but COHERE_API_KEY is missing")


def check_memory_profile(report: Report) -> None:
    """Flag the configuration that will not fit a small instance.

    This is the check that would have caught the original deployment problem. The
    local embedding backend loads torch, which puts the serving process around
    943 MB RSS on this corpus — over the 512 MB cap on Render's Free and Starter
    plans. It is a warning rather than a failure because it is entirely correct on
    a 2 GB instance.
    """
    settings = get_settings()
    if settings.embedding.backend == "local":
        report.add(
            WARN,
            "memory profile",
            "local embeddings load torch (~943 MB RSS measured) - needs a >=2 GB instance; "
            "set NCERT_EMBEDDING_BACKEND=cohere for 512 MB plans",
        )
    else:
        report.add(PASS, "memory profile", "torch-free serving path (~300 MB RSS expected)")


def main() -> int:
    load_dotenv()
    report = Report()

    check_index(report)
    check_manifest(report)
    check_lexical(report)
    check_env(report)
    check_memory_profile(report)

    print(report.render())

    if report.failed:
        print("\nPreflight failed. The deployment cannot serve queries in this state.\n")
        return 1

    print("\nPreflight passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
