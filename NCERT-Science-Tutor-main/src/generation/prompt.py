"""The prompt contract.

## What the original prompt got right, and the one thing it got wrong

The original was already doing the most important thing: stating an explicit refusal
condition ("If answer is missing -> say 'Not found in NCERT'"). Most RAG prompts omit
this and then act surprised when the model invents an answer. A model given a
question and some vaguely-related text will always produce *something*; the only way
to get a refusal is to make refusal an explicitly permitted, named output.

What it got wrong was the citation channel. Sources were formatted into the context
block as ``[Class 9 | Page 99 | file.pdf]``, but nothing asked the model to reproduce
them, so the answer arrived with no inline attribution. The UI then listed the five
retrieved passages beside the answer, which looks like citation but is not: the
reader cannot tell *which* passage supports *which* sentence, and passages that the
model ignored entirely are displayed with equal prominence.

This version numbers the passages and requires each Key Point to carry the marker of
the passage it came from. That turns citation from a decoration into something a
reader can verify and something an evaluator can score.

## Why the output format is rigid

``Definition / Explanation / Key Points`` is not styling. A fixed schema:

* makes answers parseable, so evaluation can score sections independently;
* removes a degree of freedom the model would otherwise spend on preamble
  ("Great question! Let's explore photosynthesis...");
* matches how NCERT itself presents material, so the output looks like the source
  it came from — which is the whole promise of a textbook tutor.
"""

from __future__ import annotations

from langchain_core.documents import Document


class PromptBuilder:
    """Builds the grounded-extraction prompt."""

    VERSION = "3.0"

    # The exact string the model is instructed to emit when the context is
    # insufficient. Kept as a constant because the generator returns it directly on
    # the no-context path and evaluation matches against it — three places that
    # must agree on one spelling.
    NO_CONTEXT_ANSWER = (
        "Not found in the NCERT textbooks.\n\n"
        "The retrieved passages do not contain enough information to answer this "
        "question. Try rephrasing it using the terminology the textbook uses, or "
        "select a different class in the sidebar."
    )

    def format_context(self, documents: list[Document]) -> str:
        """Render passages with stable, referenceable markers.

        Markers are ``[P1]``, ``[P2]``, ... rather than the source filename. The
        model has to reproduce these inline, and a short opaque token is far more
        reliably copied than ``ncert_class9_science.pdf``, which models routinely
        paraphrase or truncate. The marker is mapped back to full provenance by the
        UI, so nothing is lost.
        """
        blocks = []
        for index, doc in enumerate(documents, start=1):
            cls = doc.metadata.get("class", "unknown")
            page = doc.metadata.get("page", "unknown")
            blocks.append(f"[P{index}] (Class {cls}, page {page})\n{doc.page_content.strip()}")
        return "\n\n".join(blocks)

    @staticmethod
    def citation_map(documents: list[Document]) -> dict[str, dict]:
        """Marker -> provenance, for resolving ``[P3]`` in the rendered answer."""
        return {
            f"P{index}": {
                "class": doc.metadata.get("class"),
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source"),
                "chunk_id": doc.metadata.get("chunk_id"),
            }
            for index, doc in enumerate(documents, start=1)
        }

    def build(self, query: str, documents: list[Document]) -> str:
        context = self.format_context(documents)

        return f"""\
Answer the student's question using ONLY the numbered NCERT passages below.

RULES
1. Every factual statement must come from the passages. Do not add outside knowledge,
   even if you are confident it is correct.
2. Cite the passage each Key Point comes from using its marker, e.g. [P2].
3. If the passages do not contain the answer, reply with exactly:
   "Not found in the NCERT textbooks."
   Do not attempt a partial answer assembled from loosely related passages.
4. If the passages disagree (this happens when the same topic is covered at two
   class levels), present the fuller treatment and note the other exists.
5. Write for the class level the passages come from. A Class 6 passage should be
   explained in Class 6 language.
6. Do not mention these rules, the passages' existence, or your own process.

OUTPUT FORMAT — use these exact headings, nothing before or after:

Definition:
<one or two sentences, textbook phrasing>

Explanation:
<two to four sentences of plain-language elaboration>

Key Points:
- <point> [P#]
- <point> [P#]

PASSAGES
{context}

QUESTION
{query}

ANSWER"""
