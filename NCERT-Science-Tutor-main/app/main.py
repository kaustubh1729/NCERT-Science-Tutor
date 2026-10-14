#!/usr/bin/env python3
"""Command-line interface.

    python -m app.main "What is photosynthesis?"
    python -m app.main --class 9 "State Newton's first law"
    python -m app.main --json "What are stomata?"
    python -m app.main --retrieval-only "isobars"

The original read one question from ``input()`` at import time, which made the module
unimportable — ``import app.main`` blocked on stdin — and unusable in a pipeline. It
now takes arguments, supports JSON output for scripting, and returns a non-zero exit
code on failure so it composes with shell tooling.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import warnings

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv

from src.config import get_settings
from src.generation.generator import GenerationError
from src.pipeline.rag_pipeline import IndexNotBuiltError, RAGPipeline
from src.utils.logger import configure_logging
from src.vectorstore.chroma_store import IndexIncompatibleError

RULE = "-" * 78


@contextlib.contextmanager
def quiet_stdout(enabled: bool):
    """Redirect stdout to stderr for the duration of the block.

    ``--json`` output has to be machine-parseable, which means stdout must carry the
    JSON document and nothing else. Loading the embedding model prints a
    ``Loading weights: 100%|...`` progress bar from deep inside
    transformers/safetensors, and the logger writes there too — either is enough to
    make ``python -m app.main --json ... | jq`` fail with "Extra data".

    Redirecting the whole stream is used rather than chasing individual env vars
    (``HF_HUB_DISABLE_PROGRESS_BARS`` and friends) because it is robust to whichever
    library decides to print next. Diagnostics are not lost — they move to stderr,
    where they belong for a program whose stdout is a data channel.
    """
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(sys.stderr):
        yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.main",
        description="Ask a question of the NCERT Science textbooks.",
    )
    parser.add_argument("question", nargs="*", help="the question; omit to read from stdin")
    parser.add_argument(
        "--class",
        dest="class_filter",
        type=int,
        choices=[6, 7, 8, 9, 10],
        help="restrict retrieval to one class",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of prose")
    parser.add_argument("--retrieval-only", action="store_true", help="skip generation, show passages")
    parser.add_argument("--quiet", action="store_true", help="suppress startup logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    configure_logging(
        "WARNING" if args.quiet else get_settings().log_level,
        # In --json mode stdout is the data channel; diagnostics go to stderr.
        stream=sys.stderr if args.json else None,
    )

    question = " ".join(args.question).strip() or (
        sys.stdin.read().strip() if not sys.stdin.isatty() else input("Question: ").strip()
    )
    if not question:
        print("No question provided.", file=sys.stderr)
        return 2

    try:
        # Everything that might print during startup is confined to stderr in JSON
        # mode, so stdout carries only the document.
        with quiet_stdout(args.json):
            pipeline = RAGPipeline(
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                require_generator=not args.retrieval_only,
            )
    except (IndexNotBuiltError, IndexIncompatibleError, GenerationError) as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    if args.retrieval_only:
        with quiet_stdout(args.json):
            documents = pipeline.retrieve(question, class_filter=args.class_filter)
        if args.json:
            print(
                json.dumps(
                    [{"content": d.page_content, "metadata": d.metadata} for d in documents],
                    indent=2,
                    default=str,
                )
            )
            return 0
        for index, doc in enumerate(documents, start=1):
            meta = doc.metadata
            print(f"\n[P{index}] Class {meta.get('class')} page {meta.get('page')}")
            print(RULE)
            print(doc.page_content)
        return 0

    try:
        with quiet_stdout(args.json):
            result = pipeline.run(question, class_filter=args.class_filter)
    except GenerationError as exc:
        print(f"\nGeneration failed: {exc}\n", file=sys.stderr)
        return 1

    if args.json:
        payload = result.as_dict()
        payload["documents"] = [{"content": d.page_content, "metadata": d.metadata} for d in result.documents]
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(f"\n{RULE}\nANSWER\n{RULE}\n")
    print(result.answer)

    print(f"\n{RULE}\nSOURCES\n{RULE}")
    for index, doc in enumerate(result.documents, start=1):
        meta = doc.metadata
        score = meta.get("rerank_score")
        suffix = f"  relevance={score:.3f}" if score is not None else ""
        print(f"  [P{index}] Class {meta.get('class')} page {meta.get('page')}  {meta.get('source')}{suffix}")

    timings = result.timings
    print(f"\n{RULE}\nTIMINGS\n{RULE}")
    print(
        f"  retrieval {timings['retrieval_time']:5.2f}s  "
        f"rerank {timings['rerank_time']:5.2f}s  "
        f"generation {timings['generation_time']:6.2f}s  "
        f"total {timings['total_time']:6.2f}s"
    )

    for note in result.degraded:
        print(f"\n  ! {note}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
