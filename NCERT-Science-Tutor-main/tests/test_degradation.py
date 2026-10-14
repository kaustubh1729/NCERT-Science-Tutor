"""Degradation and failure-mode tests.

Each of these corresponds to a row in the table in
``docs/adr/0006-graceful-degradation.md``. The distinction being pinned down is which
failures are allowed to degrade and which must be loud — get that backwards and the
system either shows tracebacks for recoverable problems or silently serves nonsense
for unrecoverable ones.

Nothing here touches the network. The failures are injected.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from src.config import RerankConfig
from src.retrieval.reranker import CohereReranker
from src.vectorstore.chroma_store import IndexIncompatibleError, IndexManifest


def docs(n: int = 5) -> list[Document]:
    return [Document(page_content=f"passage {i}", metadata={"chunk_id": f"c{i}"}) for i in range(n)]


class TestRerankerDegrades:
    """Reranking improves precision over an already-relevant set. Losing it is a
    quality regression, not an outage."""

    def test_missing_api_key_disables_rather_than_raises(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        reranker = CohereReranker(api_key=None)
        assert reranker.available is False

    def test_disabled_reranker_returns_fusion_order_truncated(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        reranker = CohereReranker(api_key=None)
        result = reranker.rerank("any query", docs(10), top_n=5)
        assert [d.metadata["chunk_id"] for d in result] == ["c0", "c1", "c2", "c3", "c4"]

    def test_api_failure_falls_back_to_fusion_order(self):
        """A rate-limited trial key must not produce a 500."""
        reranker = CohereReranker.__new__(CohereReranker)
        reranker.config = RerankConfig()
        reranker.available = True
        reranker._client = Mock()
        reranker._client.rerank.side_effect = RuntimeError("429 rate limit exceeded")

        result = reranker.rerank("query", docs(6), top_n=3)
        assert len(result) == 3
        assert [d.metadata["chunk_id"] for d in result] == ["c0", "c1", "c2"]

    def test_out_of_range_index_from_api_is_skipped(self):
        reranker = CohereReranker.__new__(CohereReranker)
        reranker.config = RerankConfig()
        reranker.available = True
        reranker._client = Mock()
        reranker._client.rerank.return_value = Mock(
            results=[
                Mock(index=1, relevance_score=0.9),
                Mock(index=99, relevance_score=0.8),  # out of range
            ]
        )
        result = reranker.rerank("query", docs(3), top_n=2)
        assert len(result) == 1
        assert result[0].metadata["chunk_id"] == "c1"

    def test_empty_input_returns_empty(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        assert CohereReranker(api_key=None).rerank("query", []) == []

    def test_rerank_scores_are_attached(self):
        reranker = CohereReranker.__new__(CohereReranker)
        reranker.config = RerankConfig()
        reranker.available = True
        reranker._client = Mock()
        reranker._client.rerank.return_value = Mock(results=[Mock(index=2, relevance_score=0.87)])
        result = reranker.rerank("query", docs(3), top_n=1)
        assert result[0].metadata["rerank_score"] == pytest.approx(0.87)
        assert result[0].metadata["rerank_position"] == 1

    def test_model_is_pinned_on_the_api_call(self):
        """Regression: the original called rerank() with no model, relying on a
        server-side default that is not part of Cohere's compatibility contract."""
        reranker = CohereReranker.__new__(CohereReranker)
        reranker.config = RerankConfig()
        reranker.available = True
        reranker._client = Mock()
        reranker._client.rerank.return_value = Mock(results=[])

        reranker.rerank("query", docs(3), top_n=2)
        assert reranker._client.rerank.call_args.kwargs["model"] == "rerank-english-v3.0"


class TestManifestFailsLoudly:
    """The one condition that must NOT degrade: a model/index mismatch produces
    confident nonsense with no runtime signal available to detect it."""

    def _manifest(self, model: str) -> IndexManifest:
        return IndexManifest.create(
            embedding_backend="local",
            embedding_model=model,
            embedding_dim=768,
            chunk_count=3353,
            chunk_size=800,
            chunk_overlap=150,
            source_files=["ncert_class9_science.pdf"],
        )

    def test_mismatched_model_raises(self):
        manifest = self._manifest("sentence-transformers/all-mpnet-base-v2")
        with pytest.raises(IndexIncompatibleError, match="different vector spaces"):
            manifest.assert_compatible_with("embed-english-v3.0")

    def test_matching_model_passes(self):
        manifest = self._manifest("sentence-transformers/all-mpnet-base-v2")
        manifest.assert_compatible_with("sentence-transformers/all-mpnet-base-v2")

    def test_error_names_the_fix(self):
        manifest = self._manifest("model-a")
        with pytest.raises(IndexIncompatibleError) as exc:
            manifest.assert_compatible_with("model-b")
        assert "scripts.build_index" in str(exc.value)

    def test_manifest_round_trips(self, tmp_path):
        path = tmp_path / "index_manifest.json"
        self._manifest("model-x").write(path)
        loaded = IndexManifest.read(path)
        assert loaded is not None
        assert loaded.embedding_model == "model-x"
        assert loaded.chunk_count == 3353

    def test_absent_manifest_is_tolerated(self, tmp_path):
        """Indexes predating manifests remain usable — the check just goes inactive."""
        assert IndexManifest.read(tmp_path / "nope.json") is None

    def test_corrupt_manifest_is_tolerated(self, tmp_path):
        path = tmp_path / "index_manifest.json"
        path.write_text("{ this is not json", encoding="utf-8")
        assert IndexManifest.read(path) is None


class TestCacheMissesAreNotCrashes:
    """Regression: `cache.load("chunks")` with no existence check turned a cold
    cache into a FileNotFoundError during application startup."""

    def test_missing_key_returns_default(self, tmp_path):
        from src.utils.cache import CacheManager

        assert CacheManager(tmp_path).load("nonexistent") is None

    def test_zero_byte_file_is_treated_as_a_miss(self, tmp_path):
        from src.utils.cache import CacheManager

        cache = CacheManager(tmp_path)
        (tmp_path / "truncated.pkl").write_bytes(b"")
        assert cache.exists("truncated") is False
        assert cache.load("truncated") is None

    def test_corrupt_pickle_returns_default(self, tmp_path):
        from src.utils.cache import CacheManager

        cache = CacheManager(tmp_path)
        (tmp_path / "bad.pkl").write_bytes(b"\x80\x04 not a real pickle")
        assert cache.load("bad", default="fallback") == "fallback"

    def test_round_trip(self, tmp_path):
        from src.utils.cache import CacheManager

        cache = CacheManager(tmp_path)
        cache.save("k", {"a": 1})
        assert cache.load("k") == {"a": 1}
