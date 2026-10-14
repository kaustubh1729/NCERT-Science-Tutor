# NCERT Science Tutor

A retrieval-augmented question-answering system over the NCERT Science textbooks for
classes 6–10. Ask a question in your own words; get an answer assembled only from the
textbook, with the passage behind every claim.

```
$ python -m app.main "what is a sonorous material"

Definition:
Sonority is the property of metals that enables them to produce a ringing sound
when struck.

Explanation:
When a metal spoon or coin is struck, it produces a clear ringing sound, while
materials such as coal and wood produce dull sounds. Metals that show this
property are described as sonorous. [...]

Key Points:
- Metals produce a ringing sound when struck [P1]
- This property is called sonority [P1]
- Non-metals such as sulfur are not sonorous [P2]

SOURCES
  [P1] Class 7  page 46   ncert_class7_science.pdf   relevance=0.94
  [P2] Class 7  page 53   ncert_class7_science.pdf   relevance=0.71
  [P3] Class 9  page 184  ncert_class9_science.pdf   relevance=0.68

TIMINGS
  retrieval 1.28s  rerank 0.94s  generation 12.60s  total 14.82s
```

---

## Design decisions

The interesting part of a RAG system is everything that decides what the model gets
to read - and most of those decisions fail silently when wrong. A retriever returning
a passage cut mid-definition produces a confident, well-cited, wrong answer with no
exception raised anywhere.

[`DESIGN.md`](DESIGN.md) is a write-up of the major decisions and the reasoning
behind them: the chunking bug that made paragraph splitting inert, the memory
measurement that ruled out the free tier with local embeddings, why score blending
doesn't work for combining dense and lexical rankings, and how the system decides
what to fail loudly vs. degrade quietly.

---

## Architecture

```
                         "why does a ball stop rolling?"
                                      │
                ┌─────────────────────┴─────────────────────┐
                ▼                                           ▼
      dense retrieval  k=10                      lexical retrieval  k=10
      Chroma · 768-dim vectors                   SQLite FTS5 · BM25
      finds paraphrase:                          finds rare terms:
      "ball stops" ≈ "friction opposes           "sonorous", "isobar",
      relative motion"                           "Dobereiner's triads"
                │                                           │
                └─────────────────────┬─────────────────────┘
                                      ▼
                      Reciprocal Rank Fusion  (k=60)
                      agreement between retrievers outweighs
                      confidence within either one
                                      │
                                 15 candidates
                                      ▼
                      Cohere cross-encoder rerank
                      reads query + passage TOGETHER -
                      the thing a bi-encoder structurally cannot
                                      │
                                  5 passages
                                      ▼
                      Llama 3.1 8B · temperature 0
                      extraction only · must cite [P#]
                      may answer "Not found in the NCERT textbooks"
                                      │
                                      ▼
                        answer + inline citations + timings
```

The funnel widths (10+10 → 15 → 5) are a cost model, not round numbers - the reranker
is the only stage whose cost is linear in candidate count.
[Derivation](docs/notes/retrieval.md#3-the-funnel-budget-10--10--15--5).

---

## Quick start

```bash
git clone https://github.com/KRISHNA-12082006/Science_NCERT.git
cd Science_NCERT
git lfs pull                     # the PDFs; only needed to rebuild the index

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-local.txt

cp .env.example .env             # add COHERE_API_KEY and NVIDIA_API_KEY
python -m scripts.preflight      # verifies the deployment can actually serve
```

The index ships with the repository, so there is nothing to build:

```bash
python -m streamlit run app/ui.py        # web UI
python -m app.main "what is photosynthesis?"
python -m app.main --class 9 --retrieval-only "newton's first law"
```

Rebuild the index only after changing chunking, the embedding model, or the PDFs:

```bash
python -m scripts.build_index --force
```

---

## Deploying to Render

**Read [`DEPLOYMENT.md`](DEPLOYMENT.md) before deploying - there is one decision you
have to make, and getting it wrong means the service starts, passes its health check,
and is then OOM-killed on the first real query.**

The short version, measured rather than assumed:

| Configuration | RSS | Cold start | Cheapest working plan |
|---|---|---|---|
| `NCERT_EMBEDDING_BACKEND=local` | **943 MB** | ~15 s | Standard, $25/mo (2 GB) |
| `NCERT_EMBEDDING_BACKEND=cohere` | **~300 MB** | ~3 s | Free (512 MB) |

The difference is torch. Local embeddings load it into the serving process; hosted
embeddings do not. Using a *smaller* local model does not help - torch's runtime is
the floor, not the model weights. Full reasoning in
[ADR 0002](docs/adr/0002-embedding-backend.md).

---

## Project layout

```
src/
  config.py            all tunables, env-overridable, one place
  ingestion/           PDF → pages (block extraction, LFS-pointer detection)
  processing/          pages → chunks (paragraph packing, ligature repair)
  embeddings/          local | cohere behind one interface
  vectorstore/         Chroma + the index manifest that prevents silent mismatch
  retrieval/           FTS5 lexical, hybrid fusion, cross-encoder rerank
  generation/          prompt contract + LLM client
  pipeline/            orchestration; read-only w.r.t. the index
scripts/
  build_index.py       offline index build (the only writer)
  preflight.py         can this deployment serve? exits non-zero if not
eval/                  retrieval metrics (no LLM) + RAGAS suite
docs/adr/              architecture decision records
docs/notes/            research notes and measurements
db/                    committed index artifacts + manifest
```

---

## Measured results

Numbers from running it, not from estimating it. Full detail and caveats in
[`VERIFICATION.md`](VERIFICATION.md).

**Chunk quality** - the metric that exposed the original defect:

| | Before | After |
|---|---|---|
| Chunks | 3,597 | 3,353 |
| Median chunk length | - | 615 chars |
| **Boundaries at a real paragraph end** | **~0%** | **93%** |
| Chunks with ligature-broken words | 7% | **0%** |

**Retrieval**, 25 stratified questions, hit rate @ 15 candidates = **0.857**:

| Bucket | n | MRR | |
|---|---|---|---|
| `rare_term` | 5 | **0.80** | best - the case the lexical index exists for |
| `multi_hop` | 3 | 0.67 | |
| `definition` | 5 | 0.64 | |
| `cross_class` | 3 | 0.42 | competing treatments at two class levels |
| `paraphrase` | 5 | 0.27 | weakest, and partly a metric artifact |

`rare_term` scoring highest is the design claim in
[ADR 0003](docs/adr/0003-lexical-index.md) holding up. These were measured with
reranking degraded (no network in the test environment), so they understate the live
pipeline - see the caveat in [`VERIFICATION.md` §5](VERIFICATION.md).

---

## Testing

```bash
pytest -q                                    # 55 tests, no API keys, no network
python -m eval.evaluate --retrieval-only     # retrieval metrics, no LLM
python -m eval.evaluate                      # full RAGAS suite
```

---

## Known limitations

- **Evaluation set is 25 questions** - enough to compare retrieval across query types, not enough to tune `rrf_k` or retriever weights. Those are at literature defaults.
- **All performance figures are single-user.** Streamlit's per-session overhead under concurrent load is the main unknown in the 512 MB claim.
- **English only.** The Porter stemmer in the FTS5 index doesn't work for NCERT's Hindi editions.
- **Block-order PDF extraction is a heuristic** that works for these five books. There's no general answer to PDF reading order.

---

## Licence

MIT - see [LICENSE](LICENSE). NCERT textbooks are the property of NCERT and are
included here for educational use.
