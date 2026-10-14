"""Streamlit interface.

## Three problems in the original UI, and what replaced them

**The class filter did nothing.** A ``selectbox`` offered "Class 6" through
"Class 10", the value was assigned to ``selected_class``, and it was never read
again. Every query searched all five books regardless. It is now plumbed through
retrieval as a metadata filter on both the dense and lexical paths.

**Model output was interpolated into raw HTML.** The answer was rendered with
``st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)``.
Two consequences: markdown in the answer never rendered (it was inside an HTML
block), so bullet points arrived as literal asterisks; and any HTML the model
emitted was injected into the page. The model is not an attacker, but it is
repeating text from PDFs, and "render untrusted text as HTML" is not a property
worth having. Answers now go through Streamlit's own markdown renderer.

**A 15-second blocking spinner.** Generation is ~12 s of that. The answer now
streams token by token, and sources render *before* generation starts, so there is
something to read during the wait.
"""

from __future__ import annotations

import os
import warnings

# Must precede the transformers import chain.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
from dotenv import load_dotenv

from src.config import get_settings
from src.generation.generator import GenerationError
from src.pipeline.rag_pipeline import IndexNotBuiltError, RAGPipeline
from src.utils.logger import configure_logging
from src.vectorstore.chroma_store import IndexIncompatibleError

load_dotenv()
configure_logging(get_settings().log_level)

st.set_page_config(
    page_title="NCERT Science Tutor",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling is confined to layout and never wraps model output. Colours are expressed
# against Streamlit's theme variables so the page follows the user's light/dark
# preference instead of hardcoding a light palette that becomes unreadable in dark
# mode — which the previous fixed #ffffff answer box did.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 62rem; }
      .ncert-title { font-size: 2.1rem; font-weight: 700; margin-bottom: .1rem; }
      .ncert-sub { color: var(--text-color-light, #6b7280); margin-bottom: 1.4rem; }
      .passage {
        border-left: 3px solid rgba(128,128,128,.35);
        padding: .1rem 0 .1rem .9rem;
        font-size: .92rem;
        line-height: 1.55;
      }
      .pill {
        display: inline-block; padding: .1rem .55rem; border-radius: 999px;
        background: rgba(128,128,128,.16); font-size: .74rem; margin-right: .3rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the index…")
def load_pipeline() -> RAGPipeline:
    """Build the pipeline once per process.

    ``cache_resource`` (not ``cache_data``) because the return value is a live object
    holding a model and open file handles, and it must be shared across sessions
    rather than copied per user — copying it would multiply a ~900 MB process by the
    number of concurrent visitors.

    ``require_generator=False`` so a missing LLM key degrades to retrieval-only
    instead of showing every visitor a stack trace.
    """
    return RAGPipeline(cohere_api_key=os.getenv("COHERE_API_KEY"), require_generator=False)


def fatal(title: str, detail: str, fix: str | None = None) -> None:
    st.error(f"**{title}**\n\n{detail}")
    if fix:
        st.code(fix, language="bash")
    st.stop()


try:
    pipeline = load_pipeline()
except IndexNotBuiltError as exc:
    fatal("No search index found", str(exc), "python -m scripts.build_index")
except IndexIncompatibleError as exc:
    fatal("Index does not match the configured embedding model", str(exc))
except Exception as exc:
    fatal("The application could not start", f"`{type(exc).__name__}: {exc}`")

settings = get_settings()
health = pipeline.health()

# ----------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Scope")
    class_choice = st.selectbox(
        "Restrict to one class",
        ["All classes (6-10)"] + [f"Class {n}" for n in settings.classes],
        index=0,
        help=(
            "Narrowing the search removes cross-class competition. Useful when a "
            "topic is taught at several levels and you want the treatment written "
            "for yours — 'reflection of light' appears in Class 7 and Class 10 with "
            "very different depth."
        ),
    )
    class_filter = None if class_choice.startswith("All") else int(class_choice.split()[-1])

    st.markdown("### Index")
    left, right = st.columns(2)
    left.metric("Chunks", f"{health['vectors']:,}")
    right.metric("Dims", health["embedding_dim"] or "—")

    st.markdown("### Pipeline")
    st.markdown(
        f"""
        <span class="pill">retrieval {health["retrieval_funnel"]}</span>
        <span class="pill">{health["embedding_backend"]} embeddings</span>
        <span class="pill">{health["lexical_backend"]}</span>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"**Embeddings** {health['embedding_model']}  \n"
        f"**Reranker** {health['reranker']}  \n"
        f"**Generator** {health['generator']}"
    )

    for note in health["degraded"]:
        st.warning(note, icon="⚠️")

    with st.expander("How this works"):
        st.markdown(
            """
            1. **Two searches run in parallel.** A dense vector search finds passages
               that mean something similar; BM25 finds passages containing the exact
               terms. They fail on different questions, which is why both are used.
            2. **Reciprocal Rank Fusion merges them** by rank rather than score — the
               two scores are not on comparable scales, so blending them directly
               requires calibration that does not survive a change in query length.
            3. **A cross-encoder reranks the top 15.** Unlike the search stage it
               reads the question and passage *together*, which is what distinguishes
               "about this topic" from "answers this question".
            4. **The model may only use those passages** and must cite each point.
               If they do not contain the answer, it is instructed to say so rather
               than assemble something plausible.
            """
        )

# -------------------------------------------------------------------------- main
st.markdown('<div class="ncert-title">📘 NCERT Science Tutor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ncert-sub">Answers grounded in the NCERT Science textbooks for classes 6-10, '
    "with the passage behind every claim.</div>",
    unsafe_allow_html=True,
)

EXAMPLES = [
    "What is photosynthesis?",
    "Why do we see a rainbow?",
    "How does sound travel through a medium?",
    "What are the properties of acids and bases?",
]

if "query" not in st.session_state:
    st.session_state["query"] = ""

cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES, strict=True):
    if col.button(example, use_container_width=True):
        st.session_state["query"] = example

query = st.text_input(
    "Your question",
    key="query",
    placeholder="Ask anything from the NCERT Science syllabus…",
)
ask = st.button("Answer", type="primary")

if (ask or query) and query.strip():
    try:
        with st.spinner("Searching the textbooks…"):
            documents, token_stream = pipeline.stream(query, class_filter=class_filter)
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Retrieval failed: `{type(exc).__name__}: {exc}`")
        st.stop()

    if not documents:
        st.warning(
            "Nothing in the retrieved passages matches this question. "
            + (
                f"You have narrowed the search to Class {class_filter}; try 'All classes'."
                if class_filter
                else "Try rephrasing it using the textbook's terminology."
            )
        )
        st.stop()

    # Sources first: they render immediately and give the reader something to do
    # during the ~12 s the model spends generating.
    st.markdown("#### Sources")
    st.caption("Each Key Point in the answer cites one of these by its marker.")
    for index, doc in enumerate(documents, start=1):
        meta = doc.metadata
        label = f"**[P{index}]** Class {meta.get('class', '?')} · page {meta.get('page', '?')}"
        score = meta.get("rerank_score")
        if score is not None:
            label += f" · relevance {score:.2f}"
        with st.expander(label):
            # st.text, never st.markdown: this is verbatim textbook content and must
            # not be interpreted as markup.
            st.text(doc.page_content)
            st.caption(
                f"{meta.get('source', 'unknown')} · found by {meta.get('retriever_hits', 1)} of 2 retrievers"
            )

    st.markdown("#### Answer")
    try:
        # write_stream renders incrementally through Streamlit's markdown pipeline —
        # model output is never treated as HTML.
        answer = st.write_stream(token_stream)
    except GenerationError as exc:
        st.error(str(exc))
        st.info("The passages above were retrieved successfully and may still answer your question.")
        st.stop()

    if answer and "Not found in the NCERT textbooks" in answer:
        st.info(
            "The model declined to answer rather than guess — the retrieved passages "
            "did not contain the answer. That refusal is the system working as "
            "designed, not a failure."
        )

st.divider()
st.caption(
    "NCERT textbooks are the property of NCERT and are used here for educational purposes. "
    "Answers are extracted from the textbook text and should be checked against the book."
)
