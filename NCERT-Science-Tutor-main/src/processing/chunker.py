"""Turn raw PDF page text into retrieval units.

## The bug this module used to have

The original ``clean_text`` ended with ``re.sub(r"\\s+", " ", text)``, which collapses
*every* run of whitespace — including the blank lines between paragraphs — into a
single space. ``split_into_paragraphs`` then split on ``\\n{2,}``, a pattern that can
no longer match anything, so it always returned exactly one "paragraph": the entire
page as a single line.

The consequence was subtle rather than loud. Nothing crashed; the paragraph-aware
layer was simply inert, and every chunk boundary was decided by
``RecursiveCharacterTextSplitter`` counting characters. Chunks routinely began
mid-sentence and ended mid-definition, which is the single most damaging thing you
can do to a RAG system: the retriever finds the right *region* and hands the
generator a fragment that does not contain the answer.

The fix is ordering. Structure is extracted from the text **while the structure is
still present**, and whitespace normalisation happens per-paragraph afterwards.

## Why paragraph-first at all

NCERT prose is written in self-contained explanatory paragraphs — that is the unit
the textbook authors used to express one idea. Splitting on that boundary first, and
falling back to character-count splitting only for paragraphs that overflow, means
most chunks are semantically whole by construction rather than by luck.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ChunkConfig, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Running headers/footers stamped on every page of the 2024+ NCERT science series.
# These repeat thousands of times across the corpus; left in, they become the highest
# document-frequency terms in the collection and actively distort BM25 scoring.
_BOILERPLATE_PATTERNS = [
    re.compile(r"Curiosity\s*\|\s*Textbook of Science\s*\|\s*Grade\s*\d+", re.IGNORECASE),
    re.compile(r"Exploration\s*\|\s*Grade\s*\d+", re.IGNORECASE),
    re.compile(r"^\s*Reprint\s+\d{4}-\d{2}\s*$", re.MULTILINE),
]

# A line that is nothing but a number is a page number. Anchored to whole lines so
# that "12" inside "Chapter 12" or "12 g of magnesium" survives.
_STANDALONE_PAGE_NUMBER = re.compile(r"^[ \t]*\d{1,4}[ \t]*$", re.MULTILINE)

# PyMuPDF emits a hard line break at every rendered line, so a sentence wrapped
# across two lines arrives as "photo-\nsynthesis". Rejoin hyphenated splits before
# anything else looks at the text.
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")

# Typographic ligatures. These books are typeset with them, and PyMuPDF extracts the
# single ligature glyph plus — because of how the glyph is positioned — a spurious
# space before the rest of the word:
#
#     "reflection"  ->  "reﬂ ection"
#     "difference"  ->  "diﬀ erence"
#     "specific"    ->  "speciﬁ c"
#
# Measured on this corpus: 7% of chunks contain at least one word split this way.
#
# The cost is concentrated on lexical retrieval, where it is severe. FTS5 tokenises
# "reﬂ ection" as something other than "reflection", so a student searching for
# "reflection" cannot match the passage that defines it — and reflection is a core
# Class 7 and Class 10 topic. Dense retrieval is more forgiving (subword tokenisers
# degrade rather than fail) but still loses signal.
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
# Ligature + whitespace + lowercase letter is a broken word, not a word boundary:
# the space is an extraction artifact and must be removed along with the expansion.
_LIGATURE_SPLIT = re.compile(f"([{''.join(_LIGATURES)}])[ \t]+(?=[a-z])")
_LIGATURE_ANY = re.compile(f"[{''.join(_LIGATURES)}]")

# Three or more newlines carry no more meaning than two.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# Horizontal whitespace only — deliberately excludes \n, which is load-bearing here.
# The non-breaking space is written as an escape rather than as the literal
# character: PDF extraction emits U+00A0 routinely, and a literal one in source is
# invisible to a reader and trivially deleted by accident.
_HORIZONTAL_SPACE = re.compile("[ \t\xa0]+")

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class NCERTChunker:
    """Paragraph-aware chunker with character-count fallback."""

    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or get_settings().chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            # Ordered most-preferred first. The splitter tries each in turn and only
            # falls through to the next when a piece is still too large, so a chunk
            # is cut at a sentence boundary rather than a word boundary whenever the
            # text allows it.
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            length_function=len,
        )

    # ---------------------------------------------------------------- cleaning
    def clean_text(self, text: str) -> str:
        """Strip page furniture while **preserving paragraph structure**.

        Newlines survive this function by design. Collapsing them is the caller's
        job, and only after structure has been extracted.
        """
        # Ligatures first: every later pattern and every downstream tokenizer sees
        # normalised words. The split form is handled before the bare form so the
        # spurious space is consumed rather than left behind.
        text = _LIGATURE_SPLIT.sub(lambda m: _LIGATURES[m.group(1)], text)
        text = _LIGATURE_ANY.sub(lambda m: _LIGATURES[m.group(0)], text)

        text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

        for pattern in _BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)

        text = _STANDALONE_PAGE_NUMBER.sub("", text)

        # Normalise horizontal runs but leave vertical structure intact.
        text = _HORIZONTAL_SPACE.sub(" ", text)
        text = _EXCESS_BLANK_LINES.sub("\n\n", text)

        return text.strip()

    def split_into_paragraphs(self, text: str) -> list[str]:
        """Split on blank lines and drop fragments too short to be prose.

        The length floor removes figure captions, activity numbers and stray
        headings. These are not merely useless: a 20-character chunk embeds to a
        vector that sits near the centroid of the space and therefore returns a
        mediocre similarity score for *every* query, crowding out real content.
        """
        paragraphs = []
        for raw in _PARAGRAPH_BREAK.split(text):
            # Now — and only now — flatten the intra-paragraph line wrapping that
            # PyMuPDF introduced.
            paragraph = _HORIZONTAL_SPACE.sub(" ", raw.replace("\n", " ")).strip()
            if len(paragraph) >= self.config.min_paragraph_chars:
                paragraphs.append(paragraph)
        return paragraphs

    # ----------------------------------------------------------------- packing
    def pack_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Merge consecutive paragraphs up to the size budget.

        Splitting on real paragraph boundaries produces chunks with a median of ~190
        characters on this corpus, because NCERT prose is written in short paragraphs.
        That is excellent for retrieval precision and poor for generation: five
        retrieved chunks would supply ~300 tokens of context, too thin for the model
        to write a grounded explanation from.

        Packing resolves the tension rather than trading one against the other.
        Consecutive paragraphs from the same page are concatenated until the next one
        would overflow ``chunk_size``. Every boundary in the result is still a real
        paragraph boundary — no sentence is ever cut — but chunks now carry enough
        surrounding context to answer from.

        The trailing paragraph of each chunk is carried into the next when it fits the
        overlap budget. This is overlap in the units the text is actually made of: a
        whole paragraph repeated, rather than the last 150 characters of one, which is
        what a character-based overlap window would give and which typically begins
        mid-sentence.
        """
        packed: list[str] = []
        current: list[str] = []
        current_len = 0

        for paragraph in paragraphs:
            addition = len(paragraph) + (2 if current else 0)

            if current and current_len + addition > self.config.chunk_size:
                packed.append("\n\n".join(current))

                # Carry the tail paragraph forward as overlap, but only when it is
                # small enough to be context rather than content. A large trailing
                # paragraph repeated in full would duplicate real material across two
                # chunks, which inflates the index and lets one passage occupy two
                # slots in the reranker's window.
                tail = current[-1]
                if len(tail) <= self.config.chunk_overlap:
                    current = [tail]
                    current_len = len(tail)
                else:
                    current = []
                    current_len = 0
                addition = len(paragraph) + (2 if current else 0)

            current.append(paragraph)
            current_len += addition

        if current:
            packed.append("\n\n".join(current))

        return packed

    # ---------------------------------------------------------------- chunking
    def chunk_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Page documents in, chunk documents out.

        Every chunk carries a ``chunk_id`` that is a hash of its own content plus
        provenance — **not** an enumeration index. This matters more than it looks:

        * Rebuilding the index after editing one PDF used to renumber every
          subsequent chunk, invalidating any cached evaluation results keyed by id.
        * Reciprocal Rank Fusion deduplicates by ``chunk_id``. With positional ids,
          a dense hit and a lexical hit on the same passage from two different index
          builds would be treated as two distinct documents and both survive fusion,
          wasting a slot in the reranker's window.

        A content hash is stable across rebuilds and collides only for genuinely
        identical passages, which *should* deduplicate.
        """
        paragraph_docs: list[Document] = []
        pages_in = 0

        for doc in documents:
            pages_in += 1
            cleaned = self.clean_text(doc.page_content)
            paragraphs = self.split_into_paragraphs(cleaned)
            # Packing is per page: a chunk never spans two pages, so the page number
            # in its citation is unambiguous. A student following a citation must
            # land on the page the text is actually printed on.
            for packed in self.pack_paragraphs(paragraphs):
                paragraph_docs.append(Document(page_content=packed, metadata=dict(doc.metadata)))

        # Only chunks still over budget after packing reach the character splitter —
        # a single paragraph longer than chunk_size. On this corpus that is ~2% of
        # chunks, so the character splitter is now the exception rather than, as
        # before, the mechanism deciding every boundary.
        chunks = self.splitter.split_documents(paragraph_docs)

        seen: set[str] = set()
        kept: list[Document] = []
        dropped_short = 0
        dropped_duplicate = 0

        for chunk in chunks:
            # The recursive splitter emits a remainder chunk whenever a paragraph
            # does not divide evenly, and that remainder is often a 20-40 character
            # sentence fragment. The length floor is applied to paragraphs *before*
            # splitting, so these tails slip past it and have to be caught here.
            # They are worse than useless: too short to answer anything, but still
            # occupying a slot in the reranker's fixed-size candidate window.
            if len(chunk.page_content.strip()) < self.config.min_paragraph_chars:
                dropped_short += 1
                continue

            chunk_id = self._chunk_id(chunk)
            if chunk_id in seen:
                dropped_duplicate += 1
                continue

            seen.add(chunk_id)
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["char_count"] = len(chunk.page_content)
            kept.append(chunk)

        logger.info(
            "chunked %d pages -> %d paragraphs -> %d chunks (dropped %d too-short, %d duplicate)",
            pages_in,
            len(paragraph_docs),
            len(kept),
            dropped_short,
            dropped_duplicate,
        )
        return kept

    @staticmethod
    def _chunk_id(chunk: Document) -> str:
        """Stable content-addressed identifier.

        Provenance is folded in so that the same sentence appearing in the Class 8
        and Class 10 books stays two retrievable chunks with two different citations,
        rather than collapsing into one and losing a citation.
        """
        payload = "|".join(
            [
                str(chunk.metadata.get("source", "")),
                str(chunk.metadata.get("page", "")),
                chunk.page_content,
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
