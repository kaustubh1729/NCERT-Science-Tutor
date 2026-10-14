"""PDF ingestion.

## Why PyMuPDF and not pypdf

The NCERT science books are typeset in a two-column layout with figures, coloured
"Activity" boxes and marginal notes. ``pypdf`` extracts text in content-stream order,
which for this layout interleaves the two columns — you get the first line of column
one, then the first line of column two, and the prose is destroyed before chunking
ever sees it. PyMuPDF reconstructs reading order from the text blocks' geometry, so
paragraphs come out intact.

This is a case where the extraction library choice bounds the quality of everything
downstream. No amount of clever chunking recovers from interleaved columns.

## Why failures are per-file and not fatal

A corrupt or missing PDF for one class should still leave a system that answers
questions about the other four. Each failure is logged with the file that caused it
and collected into a report the caller can act on.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

from src.config import PathConfig, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NCERTLoader:
    """Loads the NCERT science PDFs into one Document per page."""

    def __init__(self, data_path: str | Path | None = None, paths: PathConfig | None = None):
        settings_paths = paths or get_settings().paths
        self.data_path = Path(data_path) if data_path is not None else settings_paths.data_dir
        self.classes = get_settings().classes
        self.errors: list[str] = []

    def load_class_pdfs(self, class_num: int) -> list[Document]:
        """Load every PDF for one class.

        Raises:
            FileNotFoundError: The class directory does not exist.
        """
        class_path = self.data_path / f"class_{class_num}"
        if not class_path.exists():
            raise FileNotFoundError(f"No directory for class {class_num} at {class_path}")

        documents: list[Document] = []
        for pdf_path in sorted(class_path.glob("*.pdf")):
            documents.extend(self._load_pdf(pdf_path, class_num))
        return documents

    @staticmethod
    def _page_text(page) -> str:
        """Extract a page as blocks joined by blank lines.

        This is the single highest-leverage line in the ingestion path, and it is not
        obvious why.

        ``page.get_text("text")`` returns the page as a flat string with a newline at
        every *rendered* line break and — measured on this corpus — **no blank lines
        at all**: splitting a page on ``\\n\\s*\\n`` yields exactly one group. Any
        downstream "paragraph-aware" chunking is therefore inert by construction, and
        chunk boundaries fall wherever a character counter happens to land.

        ``page.get_text("blocks")`` returns PyMuPDF's layout analysis instead: a list
        of geometric text blocks that correspond to actual typeset paragraphs, 12 per
        page on average here. Joining them with a blank line gives the chunker real
        structure to split on.

        Blocks are taken in document order rather than sorted by position. These books
        use a two-column layout, and sorting blocks top-to-bottom would interleave the
        columns — producing text that reads as alternating fragments of two different
        paragraphs. The PDF content stream emits each column contiguously, so document
        order is the one that preserves reading order.
        """
        blocks = page.get_text("blocks")
        # Tuple layout: (x0, y0, x1, y1, text, block_no, block_type).
        # block_type 1 is an image; it carries no text and would contribute an empty
        # paragraph that survives into the corpus as noise.
        paragraphs = [block[4].strip() for block in blocks if block[6] == 0 and block[4].strip()]
        return "\n\n".join(paragraphs)

    def _load_pdf(self, pdf_path: Path, class_num: int) -> Iterator[Document]:
        # A Git LFS pointer file is ~130 bytes of text. Checking out this repo
        # without `git lfs pull` leaves exactly that in place of every PDF, and
        # PyMuPDF's error for it ("cannot recognize file format") does not hint at
        # the real cause. Detecting it here turns a confusing failure into a fix.
        size = pdf_path.stat().st_size
        if size < 1024:
            message = (
                f"{pdf_path.name} is only {size} bytes — this is almost certainly an "
                f"un-pulled Git LFS pointer. Run: git lfs pull"
            )
            logger.error(message)
            self.errors.append(message)
            return

        try:
            document = fitz.open(pdf_path)
        except Exception as exc:
            message = f"could not open {pdf_path.name}: {exc}"
            logger.error(message)
            self.errors.append(message)
            return

        # Captured before the finally-block closes the document: this is a generator,
        # so the close runs when iteration ends and any attribute read after that
        # raises "document closed".
        page_count = document.page_count
        empty_pages = 0
        try:
            for page_number in range(page_count):
                text = self._page_text(document[page_number])
                if not text.strip():
                    # Full-page figures are common in these books and legitimately
                    # have no extractable text. Counted, not warned about per page.
                    empty_pages += 1
                    continue

                yield Document(
                    page_content=text,
                    metadata={
                        "class": class_num,
                        "source": pdf_path.name,
                        # 1-based: this is shown to students, who will look it up in
                        # a physical book that starts at page 1.
                        "page": page_number + 1,
                        "type": "ncert_science",
                    },
                )
        finally:
            document.close()

        logger.info(
            "loaded %s: %d pages (%d with no extractable text)",
            pdf_path.name,
            page_count,
            empty_pages,
        )

    def load_all(self) -> list[Document]:
        """Load every class. Missing classes are reported, not fatal."""
        all_documents: list[Document] = []
        for class_num in self.classes:
            try:
                documents = self.load_class_pdfs(class_num)
                logger.info("class %d: %d pages with text", class_num, len(documents))
                all_documents.extend(documents)
            except FileNotFoundError as exc:
                message = f"class {class_num} skipped: {exc}"
                logger.warning(message)
                self.errors.append(message)

        if not all_documents:
            raise RuntimeError(
                f"No PDF pages could be read from {self.data_path}.\n"
                + ("\n".join(f"  - {e}" for e in self.errors) if self.errors else "")
            )

        logger.info("loaded %d pages across %d classes", len(all_documents), len(self.classes))
        return all_documents
