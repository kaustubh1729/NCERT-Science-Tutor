"""Chunking tests.

Several of these are **regression tests for bugs that shipped**. The originals were
not caught because the old test suite asserted on the chunker's configuration
(`chunk_size == 800`) rather than on its behaviour — it verified that a constant had
the value it was assigned, which cannot fail and cannot detect anything.
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.config import ChunkConfig
from src.processing.chunker import NCERTChunker


def make_page(text: str, page: int = 1, class_num: int = 9) -> Document:
    return Document(
        page_content=text,
        metadata={"class": class_num, "source": "test.pdf", "page": page, "type": "ncert_science"},
    )


class TestCleanTextPreservesStructure:
    """Regression: `re.sub(r"\\s+", " ")` destroyed every paragraph boundary."""

    def test_blank_lines_survive_cleaning(self):
        chunker = NCERTChunker()
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        cleaned = chunker.clean_text(text)
        assert "\n\n" in cleaned, (
            "clean_text collapsed paragraph breaks. This is the original bug: it made "
            "split_into_paragraphs unable to match anything, so every chunk boundary "
            "fell wherever the character counter landed."
        )

    def test_paragraphs_are_actually_split(self):
        chunker = NCERTChunker(ChunkConfig(chunk_size=800, chunk_overlap=150, min_paragraph_chars=20))
        text = "\n\n".join(
            [
                "Photosynthesis is how green plants prepare their own food from sunlight.",
                "Stomata are tiny pores on the leaf surface that regulate gas exchange.",
                "Chlorophyll is the green pigment that absorbs light energy for the plant.",
            ]
        )
        paragraphs = chunker.split_into_paragraphs(chunker.clean_text(text))
        assert len(paragraphs) == 3

    def test_short_fragments_are_dropped(self):
        chunker = NCERTChunker(ChunkConfig(min_paragraph_chars=80))
        text = "Fig. 6.1\n\n" + ("A long enough paragraph to survive the length floor. " * 3)
        paragraphs = chunker.split_into_paragraphs(chunker.clean_text(text))
        assert all("Fig. 6.1" not in p for p in paragraphs)

    def test_hyphenated_linebreaks_rejoin(self):
        chunker = NCERTChunker()
        cleaned = chunker.clean_text("The process of photo-\nsynthesis requires light.")
        assert "photosynthesis" in cleaned

    def test_standalone_page_numbers_removed_but_inline_numbers_kept(self):
        chunker = NCERTChunker()
        cleaned = chunker.clean_text("Some text here.\n42\nChapter 12 covers 25 grams of magnesium.")
        assert "\n42\n" not in cleaned
        assert "Chapter 12" in cleaned
        assert "25 grams" in cleaned


class TestPacking:
    def test_small_paragraphs_are_merged(self):
        chunker = NCERTChunker(ChunkConfig(chunk_size=800, chunk_overlap=150, min_paragraph_chars=10))
        paragraphs = [f"Short paragraph number {i}." for i in range(10)]
        packed = chunker.pack_paragraphs(paragraphs)
        assert len(packed) < len(paragraphs), "packing did not merge anything"
        assert all(len(p) <= 800 for p in packed)

    def test_packing_never_exceeds_budget(self):
        config = ChunkConfig(chunk_size=200, chunk_overlap=50, min_paragraph_chars=10)
        chunker = NCERTChunker(config)
        packed = chunker.pack_paragraphs(["x" * 60 for _ in range(20)])
        # A single paragraph can exceed the budget on its own; a *pack* must not.
        assert all(len(p) <= config.chunk_size for p in packed)

    def test_oversized_paragraph_passes_through(self):
        config = ChunkConfig(chunk_size=200, chunk_overlap=50, min_paragraph_chars=10)
        chunker = NCERTChunker(config)
        packed = chunker.pack_paragraphs(["y" * 500])
        # Packing does not split; the recursive splitter handles the remainder.
        assert len(packed) == 1


class TestChunkIdentity:
    """Regression: positional IDs renumbered on rebuild and broke fusion dedup."""

    def test_ids_are_content_addressed_not_positional(self):
        chunker = NCERTChunker()
        page = make_page("\n\n".join(["A paragraph with plenty of substance in it. " * 4] * 2))
        chunks = chunker.chunk_documents([page])
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert all(isinstance(i, str) for i in ids)
        assert not all(str(i).isdigit() for i in ids)

    def test_ids_are_stable_across_rebuilds(self):
        text = "\n\n".join(["Stable content that should hash identically every time. " * 4])
        first = NCERTChunker().chunk_documents([make_page(text)])
        second = NCERTChunker().chunk_documents([make_page(text)])
        assert [c.metadata["chunk_id"] for c in first] == [c.metadata["chunk_id"] for c in second]

    def test_same_text_on_different_pages_stays_distinct(self):
        """Identical prose in two books must keep two citations, not collapse to one."""
        text = "Matter is anything that has mass and occupies space. " * 4
        chunks = NCERTChunker().chunk_documents(
            [
                make_page(text, page=10, class_num=6),
                make_page(text, page=55, class_num=9),
            ]
        )
        ids = {c.metadata["chunk_id"] for c in chunks}
        assert len(ids) == len(chunks), "provenance is not part of the chunk identity"

    def test_exact_duplicates_are_removed(self):
        text = "Repeated summary box content that appears twice on one page. " * 3
        page = make_page(text + "\n\n" + text)
        chunks = NCERTChunker().chunk_documents([page])
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))


class TestConfigValidation:
    def test_overlap_must_be_smaller_than_size(self):
        import pytest

        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkConfig(chunk_size=100, chunk_overlap=100)
