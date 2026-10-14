#!/usr/bin/env python3
"""Evaluation harness.

    python -m eval.evaluate --retrieval-only     # no LLM, no API key, CI-safe
    python -m eval.evaluate                      # full RAGAS suite
    python -m eval.evaluate --output results.json

## Why retrieval is scored without an LLM

A wrong answer has two possible causes — the passage was never retrieved, or it was
retrieved and misused — and they need different fixes. An end-to-end score cannot
distinguish them, so this harness computes retrieval metrics independently, using
keyword overlap against the ground-truth answer rather than a judge model.

That makes them deterministic, free, and fast enough to run in CI on every commit
touching retrieval. The LLM-judged RAGAS metrics cannot be, on cost grounds.

Reasoning and target values: ``docs/notes/evaluation.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.logger import configure_logging, get_logger

logger = get_logger("eval")

# Words too common in science prose to indicate a genuine content match.
_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "in",
    "on",
    "to",
    "and",
    "or",
    "by",
    "that",
    "this",
    "it",
    "its",
    "as",
    "for",
    "with",
    "from",
    "at",
    "be",
    "been",
    "which",
    "when",
    "process",
    "called",
    "used",
    "using",
    "can",
}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOPWORDS}


def contains_answer(passage: str, expected: str, threshold: float = 0.45) -> bool:
    """Whether a passage plausibly contains the ground-truth answer.

    Keyword-overlap proxy, not semantic matching. It is deliberately crude: the point
    is a metric that is *stable* and needs no API key, so that retrieval regressions
    are detectable in CI. Cross-checking a sample against human judgement before
    trusting the absolute values is advisable — the trend across commits is the
    reliable signal, not the level.
    """
    expected_words = content_words(expected)
    if not expected_words:
        return False
    overlap = expected_words & content_words(passage)
    return len(overlap) / len(expected_words) >= threshold


def retrieval_metrics(pipeline: RAGPipeline, dataset: list[dict]) -> dict[str, Any]:
    """Hit rate and MRR, before and after reranking, broken down by bucket.

    Unanswerable questions are excluded from hit-rate arithmetic — there is no gold
    passage to find, so scoring retrieval against them is meaningless. They are
    scored on refusal behaviour in :func:`ragas_metrics` instead.

    The per-bucket split is the point. An aggregate hit rate cannot tell you whether
    a retrieval change helped rare-term lookups and hurt paraphrased ones, which is
    precisely the trade-off the hybrid design is managing.
    """
    answerable = [item for item in dataset if not item.get("unanswerable")]

    fused_hits = 0
    reranked_hits = 0
    reciprocal_ranks: list[float] = []
    per_question: list[dict] = []
    buckets: dict[str, dict[str, float]] = {}

    for item in answerable:
        query = item["query"]
        expected = item["expected_answer"]
        bucket = item.get("bucket", "unspecified")

        candidates = pipeline.retriever.retrieve(query, class_filter=item.get("class"))
        reranked = pipeline.reranker.rerank(query, candidates)

        fused_hit = any(contains_answer(d.page_content, expected) for d in candidates)
        rank = next(
            (i for i, d in enumerate(reranked, start=1) if contains_answer(d.page_content, expected)),
            None,
        )

        fused_hits += int(fused_hit)
        reranked_hits += int(rank is not None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        stats = buckets.setdefault(bucket, {"n": 0, "hits": 0, "mrr_sum": 0.0})
        stats["n"] += 1
        stats["hits"] += int(rank is not None)
        stats["mrr_sum"] += 1.0 / rank if rank else 0.0

        # How many of the two retrievers found the winning passage — the single most
        # useful diagnostic when a retrieval looks wrong.
        both = sum(1 for d in reranked if d.metadata.get("retriever_hits", 1) > 1)

        per_question.append(
            {
                "id": item.get("id"),
                "bucket": bucket,
                "query": query,
                "hit_in_candidates": fused_hit,
                "rank_after_rerank": rank,
                "candidates": len(candidates),
                "found_by_both_retrievers": both,
            }
        )
        logger.info(
            "%-12s %-42s candidates=%-4s rank=%s",
            bucket,
            query[:42],
            "HIT" if fused_hit else "MISS",
            rank or "MISS",
        )

    n = len(answerable) or 1
    return {
        "n_questions": len(dataset),
        "n_answerable": len(answerable),
        "hit_rate_candidates": round(fused_hits / n, 3),
        "hit_rate_reranked": round(reranked_hits / n, 3),
        "mrr_reranked": round(sum(reciprocal_ranks) / n, 3),
        "by_bucket": {
            name: {
                "n": int(s["n"]),
                "hit_rate": round(s["hits"] / s["n"], 3),
                "mrr": round(s["mrr_sum"] / s["n"], 3),
            }
            for name, s in sorted(buckets.items())
        },
        "per_question": per_question,
    }


def ragas_metrics(pipeline: RAGPipeline, dataset: list[dict]) -> dict[str, Any]:
    """Full RAGAS suite. Requires both API keys and costs one LLM call per metric."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    questions, answers, contexts, ground_truths = [], [], [], []
    unanswerable_total = 0
    unanswerable_refused = 0

    for item in dataset:
        logger.info("running: %s", item["query"])
        result = pipeline.run(item["query"])

        if item.get("unanswerable"):
            # The bucket usually omitted, and the one that matters most: a system
            # that never refuses is a system that hallucinates, and questions which
            # all have answers cannot detect that.
            unanswerable_total += 1
            refused = "Not found in the NCERT textbooks" in result.answer
            unanswerable_refused += int(refused)
            logger.info("  unanswerable -> %s", "REFUSED (correct)" if refused else "ANSWERED (fabrication)")

        questions.append(item["query"])
        answers.append(result.answer)
        contexts.append([d.page_content for d in result.documents])
        ground_truths.append(item["expected_answer"])

    hf_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # RAGAS defaults to OpenAI for the judge. Pointing it at the same NVIDIA endpoint
    # keeps the project to one LLM provider, at the cost of judge and generator
    # sharing a model family — and therefore sharing blind spots. A stronger,
    # different judge would be better practice; noted in docs/notes/evaluation.md.
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("NVIDIA_API_KEY", ""))
    os.environ.setdefault("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")

    scores = evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    return {
        "ragas": {k: round(float(v), 3) for k, v in scores.items() if isinstance(v, (int, float))},
        "refusal_total": sum(1 for a in answers if "Not found in the NCERT textbooks" in a),
        "unanswerable_n": unanswerable_total,
        "unanswerable_refusal_rate": (
            round(unanswerable_refused / unanswerable_total, 3) if unanswerable_total else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the NCERT RAG pipeline.")
    parser.add_argument("--dataset", default="eval/dataset.json")
    parser.add_argument("--retrieval-only", action="store_true", help="skip LLM-judged metrics")
    parser.add_argument("--output", help="write results as JSON to this path")
    args = parser.parse_args()

    load_dotenv()
    configure_logging("INFO")

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    logger.info("loaded %d evaluation questions", len(dataset))

    if len(dataset) < 30:
        logger.warning(
            "%d questions is too few to detect a regression or tune a parameter. "
            "See docs/notes/evaluation.md for the stratification this set needs.",
            len(dataset),
        )

    pipeline = RAGPipeline(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        require_generator=not args.retrieval_only,
    )

    results: dict[str, Any] = {"config": pipeline.settings.describe()}
    results.update(retrieval_metrics(pipeline, dataset))

    if not args.retrieval_only:
        results.update(ragas_metrics(pipeline, dataset))

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    for key, value in results.items():
        if key in ("per_question", "config"):
            continue
        print(f"  {key:<24} {value}")
    print("=" * 70 + "\n")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("wrote %s", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
