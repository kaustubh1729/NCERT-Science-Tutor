"""Reciprocal Rank Fusion tests.

The property these exist to pin down is the one the whole hybrid design rests on:
**agreement between retrievers outranks confidence within one**. If a refactor
quietly changed `rrf_k` to a small value, results would still look plausible while
that property silently disappeared. `test_agreement_beats_single_retriever_confidence`
is the guard.

The old suite's fusion test asserted `len(fused) <= 3` and that the results were still
Mock objects — true for almost any implementation, including a broken one.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.config import RetrievalConfig
from src.retrieval.hybrid import HybridRetriever


def doc(chunk_id: str, content: str = "content") -> Document:
    return Document(page_content=f"{content} {chunk_id}", metadata={"chunk_id": chunk_id})


@pytest.fixture
def fuser() -> HybridRetriever:
    # __new__ bypasses __init__ so fusion can be tested without a vector store,
    # a lexical index, or any I/O. Fusion is pure arithmetic over ranked lists.
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.config = RetrievalConfig()
    return retriever


class TestFusionOrdering:
    def test_agreement_beats_single_retriever_confidence(self, fuser):
        """The central property. See docs/adr/0004-rank-fusion.md.

        `both` is rank 4 in each list; `solo` is rank 1 in one and absent from the
        other. `both` must win:
            both: 1/64 + 1/64 = 0.03125
            solo: 1/61        = 0.01639
        """
        dense = [doc("solo"), doc("f1"), doc("f2"), doc("both")]
        lexical = [doc("g1"), doc("g2"), doc("g3"), doc("both")]

        fused = fuser.reciprocal_rank_fusion([(dense, 1.0), (lexical, 1.0)])

        assert fused[0].metadata["chunk_id"] == "both", (
            "a document found by both retrievers at rank 4 must outrank one found "
            "by a single retriever at rank 1 — this is the entire point of RRF"
        )

    def test_retriever_hit_count_is_recorded(self, fuser):
        dense = [doc("shared"), doc("dense_only")]
        lexical = [doc("shared"), doc("lex_only")]

        fused = {
            d.metadata["chunk_id"]: d for d in fuser.reciprocal_rank_fusion([(dense, 1.0), (lexical, 1.0)])
        }

        assert fused["shared"].metadata["retriever_hits"] == 2
        assert fused["dense_only"].metadata["retriever_hits"] == 1

    def test_deduplicates_by_chunk_id(self, fuser):
        dense = [doc("a"), doc("b")]
        lexical = [doc("a"), doc("b")]
        fused = fuser.reciprocal_rank_fusion([(dense, 1.0), (lexical, 1.0)])
        assert len(fused) == 2

    def test_weights_shift_the_ordering(self, fuser):
        dense = [doc("d_top")]
        lexical = [doc("l_top")]

        lexical_favoured = fuser.reciprocal_rank_fusion([(dense, 0.1), (lexical, 1.0)])
        dense_favoured = fuser.reciprocal_rank_fusion([(dense, 1.0), (lexical, 0.1)])

        assert lexical_favoured[0].metadata["chunk_id"] == "l_top"
        assert dense_favoured[0].metadata["chunk_id"] == "d_top"

    def test_within_a_single_list_original_order_is_preserved(self, fuser):
        ranked = [doc("first"), doc("second"), doc("third")]
        fused = fuser.reciprocal_rank_fusion([(ranked, 1.0)])
        assert [d.metadata["chunk_id"] for d in fused] == ["first", "second", "third"]


class TestFusionRobustness:
    def test_empty_lists_do_not_raise(self, fuser):
        assert fuser.reciprocal_rank_fusion([([], 1.0), ([], 1.0)]) == []

    def test_one_empty_retriever_is_survivable(self, fuser):
        """Degraded mode: the lexical index is missing and returns nothing."""
        fused = fuser.reciprocal_rank_fusion([([doc("a"), doc("b")], 1.0), ([], 1.0)])
        assert [d.metadata["chunk_id"] for d in fused] == ["a", "b"]

    def test_missing_chunk_id_falls_back_instead_of_raising(self, fuser):
        """Regression: `doc.metadata["chunk_id"]` raised KeyError on older indexes."""
        no_id = Document(page_content="a document with no chunk_id", metadata={})
        fused = fuser.reciprocal_rank_fusion([([no_id], 1.0)])
        assert len(fused) == 1

    def test_fusion_score_is_attached(self, fuser):
        # The stored score is rounded to 6 decimals for display in the UI and eval
        # output, so the comparison is to that precision rather than to full float.
        fused = fuser.reciprocal_rank_fusion([([doc("a")], 1.0)])
        assert fused[0].metadata["fusion_score"] == pytest.approx(1 / 61, abs=5e-7)


class TestRetrievalConfigValidation:
    def test_rerank_top_n_cannot_exceed_candidate_k(self):
        with pytest.raises(ValueError, match="rerank_top_n"):
            RetrievalConfig(candidate_k=5, rerank_top_n=10)

    def test_unknown_lexical_backend_is_rejected(self):
        with pytest.raises(ValueError, match="LEXICAL_BACKEND"):
            RetrievalConfig(lexical_backend="elasticsearch")
