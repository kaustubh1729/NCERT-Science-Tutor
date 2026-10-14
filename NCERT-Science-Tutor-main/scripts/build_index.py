#!/usr/bin/env python3
"""Build the retrieval index. Run on a developer machine, commit the output.

    python -m scripts.build_index               # build if absent
    python -m scripts.build_index --force       # rebuild from scratch
    python -m scripts.build_index --no-cache    # ignore cached chunks, re-read PDFs
    python -m scripts.build_index --dry-run     # chunk and report, write nothing

## Why this is a script and not something the server does

Indexing this corpus means reading ~1,500 PDF pages and computing 3,597 embeddings.
On a laptop that is a few minutes. Inside a web process on a PaaS it is a failed
health check, retried forever, because the platform expects a port to be listening
within its boot timeout and the process is busy running a transformer.

Separating the two also makes the index *reviewable*. It is a build artifact with a
manifest recording the model, dimensionality, chunk count and source files, committed
alongside the code. Whether retrieval changed between two commits is a question you
can answer from the diff.

See ``docs/adr/0001-prebuilt-index.md``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Allow `python scripts/build_index.py` in addition to `python -m scripts.build_index`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.config import get_settings
from src.embeddings.embedder import NCERTEmbedder
from src.ingestion.pdf_loader import NCERTLoader
from src.processing.chunker import NCERTChunker
from src.retrieval.lexical import FTS5Index, fts5_available
from src.utils.cache import CacheManager
from src.utils.logger import configure_logging, get_logger
from src.vectorstore.chroma_store import IndexManifest, NCERTVectorStore

logger = get_logger("build_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the NCERT retrieval index.")
    parser.add_argument("--force", action="store_true", help="delete and rebuild an existing index")
    parser.add_argument("--no-cache", action="store_true", help="re-read PDFs instead of using cached chunks")
    parser.add_argument("--dry-run", action="store_true", help="chunk and report statistics, write nothing")
    return parser.parse_args()


def load_or_build_chunks(use_cache: bool) -> tuple[list, list[str]]:
    """Chunks plus the source filenames they came from."""
    settings = get_settings()
    cache = CacheManager(settings.paths.cache_dir)

    if use_cache:
        cached = cache.load("chunks")
        if cached:
            sources = sorted({c.metadata.get("source", "?") for c in cached})
            logger.info("using %d cached chunks (pass --no-cache to re-read PDFs)", len(cached))
            return cached, sources

    logger.info("reading PDFs from %s", settings.paths.data_dir)
    loader = NCERTLoader(paths=settings.paths)
    pages = loader.load_all()

    if loader.errors:
        logger.warning("%d ingestion problems:", len(loader.errors))
        for error in loader.errors:
            logger.warning("  - %s", error)

    chunks = NCERTChunker(settings.chunking).chunk_documents(pages)
    cache.save("chunks", chunks)

    sources = sorted({c.metadata.get("source", "?") for c in chunks})
    return chunks, sources


def report(chunks: list) -> None:
    """Distribution statistics — the fastest way to notice a chunking regression."""
    lengths = sorted(len(c.page_content) for c in chunks)
    by_class: dict = {}
    for chunk in chunks:
        by_class.setdefault(chunk.metadata.get("class", "?"), 0)
        by_class[chunk.metadata.get("class", "?")] += 1

    def percentile(p: float) -> int:
        if not lengths:
            return 0
        return lengths[min(int(len(lengths) * p), len(lengths) - 1)]

    logger.info("--- chunk statistics ---")
    logger.info("total          : %d", len(chunks))
    logger.info("chars  min/p50 : %d / %d", lengths[0] if lengths else 0, percentile(0.50))
    logger.info("chars  p90/max : %d / %d", percentile(0.90), lengths[-1] if lengths else 0)
    for class_num in sorted(by_class, key=str):
        logger.info("class %-3s      : %d chunks", class_num, by_class[class_num])

    # A large share of chunks at the size ceiling means paragraph splitting is not
    # doing its job and the character splitter is making most of the decisions —
    # exactly the silent failure the old chunker had.
    ceiling = get_settings().chunking.chunk_size
    at_ceiling = sum(1 for length in lengths if length >= ceiling - 20)
    share = at_ceiling / len(lengths) if lengths else 0
    logger.info("at size ceiling: %d (%.0f%%)", at_ceiling, share * 100)
    if share > 0.5:
        logger.warning(
            "over half of chunks are at the size ceiling — paragraph detection may "
            "be failing. Inspect a few chunks before trusting this index."
        )


def main() -> int:
    args = parse_args()
    load_dotenv()
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info("=" * 78)
    logger.info(" NCERT index build")
    logger.info("=" * 78)
    for key, value in settings.describe().items():
        logger.info("  %-18s %s", key, value)

    store = NCERTVectorStore(settings.paths)

    if store.is_ready() and not args.force and not args.dry_run:
        logger.info(
            "index already present at %s with %d vectors. Use --force to rebuild.",
            settings.paths.db_dir,
            store.count(),
        )
        return 0

    started = time.time()
    chunks, sources = load_or_build_chunks(use_cache=not args.no_cache)
    if not chunks:
        logger.error("no chunks produced — nothing to index")
        return 1

    report(chunks)

    if args.dry_run:
        logger.info("dry run: no artifacts written")
        return 0

    # ---------------------------------------------------------------- dense index
    if args.force and settings.paths.db_dir.exists():
        logger.info("removing existing index at %s", settings.paths.db_dir)
        shutil.rmtree(settings.paths.db_dir)

    embedder = NCERTEmbedder(settings.embedding)
    logger.info("embedding %d chunks with %s ...", len(chunks), embedder.model_id)
    embedding_model = embedder.get_model()

    embed_started = time.time()
    store.create_store(chunks, embedding_model)
    logger.info("dense index built in %.0fs", time.time() - embed_started)

    # -------------------------------------------------------------- lexical index
    if fts5_available():
        FTS5Index(settings.paths.lexical_index).build(chunks)
    else:
        logger.warning(
            "this interpreter's SQLite lacks FTS5 — skipping the lexical index. "
            "Serving will fall back to in-memory BM25 from cache/bm25_docs.pkl."
        )
        CacheManager(settings.paths.cache_dir).save("bm25_docs", chunks)

    # -------------------------------------------------------------------- manifest
    probe_dim = len(embedding_model.embed_query("dimensionality probe"))
    IndexManifest.create(
        embedding_backend=settings.embedding.backend,
        embedding_model=embedder.model_id,
        embedding_dim=probe_dim,
        chunk_count=len(chunks),
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        source_files=sources,
    ).write(settings.paths.manifest)

    elapsed = time.time() - started
    db_size = sum(f.stat().st_size for f in settings.paths.db_dir.rglob("*") if f.is_file()) / 1e6

    logger.info("=" * 78)
    logger.info(
        " build complete in %.0fs — %d chunks, %d dims, %.0f MB on disk",
        elapsed,
        len(chunks),
        probe_dim,
        db_size,
    )
    logger.info("=" * 78)
    logger.info("Commit db/ so deployments start against a ready index:")
    logger.info("    git add -f db/ && git commit -m 'rebuild index'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
