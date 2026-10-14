"""SQLite FTS5 lexical index tests.

The query-escaping tests are the important ones. FTS5 treats bare `AND`, `OR`, `NOT`,
`*` and `"` as query syntax, so interpolating a student's question directly is both a
crash risk and — worse — a silent-wrong-results risk, since `NOT flammable` is valid
FTS5 that inverts the query.
"""

from __future__ import annotations

import sqlite3

import pytest
from langchain_core.documents import Document

from src.retrieval.lexical import FTS5Index, build_query, fts5_available

pytestmark = pytest.mark.skipif(not fts5_available(), reason="SQLite built without FTS5")


CORPUS = [
    ("c1", 9, 99, "Newton's first law states that an object at rest remains at rest."),
    ("c2", 7, 12, "Stomata are tiny pores on the leaf surface regulating gas exchange."),
    ("c3", 6, 40, "Photosynthesis is how green plants prepare food using sunlight."),
    ("c4", 10, 88, "A sonorous material produces a ringing sound when struck."),
    ("c5", 9, 100, "Friction opposes the relative motion between two surfaces."),
]


@pytest.fixture
def index(tmp_path) -> FTS5Index:
    documents = [
        Document(
            page_content=content,
            metadata={"chunk_id": cid, "class": cls, "source": f"class{cls}.pdf", "page": page},
        )
        for cid, cls, page, content in CORPUS
    ]
    idx = FTS5Index(tmp_path / "lexical.sqlite3")
    idx.build(documents)
    return idx


class TestQueryEscaping:
    def test_tokens_are_quoted(self):
        assert build_query("photosynthesis light") == '"photosynthesis" OR "light"'

    def test_fts5_operators_are_neutralised(self):
        """`NOT` must become a search term, not an operator that inverts the query."""
        assert build_query("NOT flammable") == '"NOT" OR "flammable"'

    def test_quotes_and_punctuation_do_not_break_the_expression(self):
        assert build_query('What is "matter"?') == '"What" OR "is" OR "matter"'

    def test_empty_query_yields_empty_expression(self):
        assert build_query("") == ""
        assert build_query("!!!") == ""


class TestSearch:
    def test_rare_term_retrieves_its_passage(self, index):
        """The case dense retrieval handles badly, and lexical handles well."""
        conn = index.open()
        results = index.search(conn, "sonorous", k=5)
        conn.close()
        assert results
        assert results[0].metadata["chunk_id"] == "c4"

    def test_results_are_ranked_best_first(self, index):
        """bm25() returns negative scores; ascending ORDER BY is correct.

        Sorting the other way is a silent and very convincing way to return the
        worst matches first.
        """
        conn = index.open()
        results = index.search(conn, "Newton first law object rest", k=5)
        conn.close()
        assert results[0].metadata["chunk_id"] == "c1"
        scores = [r.metadata["lexical_score"] for r in results]
        assert scores == sorted(scores)

    def test_class_filter_is_applied_in_the_query(self, index):
        conn = index.open()
        results = index.search(conn, "the surface", k=10, class_filter=9)
        conn.close()
        assert all(r.metadata["class"] == 9 for r in results)

    def test_metadata_round_trips(self, index):
        conn = index.open()
        result = index.search(conn, "stomata", k=1)[0]
        conn.close()
        assert result.metadata["class"] == 7
        assert result.metadata["page"] == 12
        assert result.metadata["chunk_id"] == "c2"

    def test_no_match_returns_empty_not_an_error(self, index):
        conn = index.open()
        results = index.search(conn, "quantum chromodynamics", k=5)
        conn.close()
        assert results == []

    def test_empty_query_returns_empty(self, index):
        conn = index.open()
        assert index.search(conn, "", k=5) == []
        conn.close()


class TestIndexIntegrity:
    def test_count_matches_corpus(self, index):
        assert index.count() == len(CORPUS)

    def test_index_is_opened_read_only(self, index):
        """The index is a committed artifact; a stray write would corrupt tracked bytes."""
        conn = index.open()
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM chunks")
        conn.close()

    def test_no_partial_file_is_left_behind(self, index):
        assert not index.path.with_suffix(".building").exists()

    def test_stemming_matches_word_forms(self, index):
        """`porter unicode61` means 'opposes' should be findable via 'opposing'."""
        conn = index.open()
        results = index.search(conn, "opposing motion", k=5)
        conn.close()
        assert any(r.metadata["chunk_id"] == "c5" for r in results)
