# Design Notes

Everything here is something I got wrong, had to measure, or had to think carefully about before arriving at what's in the code. Not a changelog - more like the reasoning that doesn't fit in a comment.

---

## The chunking problem

This was the thing that mattered most, and it was hard to spot because the code *looked* correct.

The original chunker was written to be paragraph-aware - `split_into_paragraphs()` used `re.split(r'\n{2,}', text)`. The problem was that `clean_text()` ran `re.sub(r"\s+", " ", text)` first, which collapses every newline into a space. By the time `split_into_paragraphs` saw the text, there were no newlines left to split on. So it returned the entire page as one paragraph, every time, and `RecursiveCharacterTextSplitter` made every single chunking decision by counting characters.

It looked like paragraph splitting was working. The bug was silent.

Fixing the ordering didn't fix the problem either. When I measured the actual PDF text using `page.get_text("text")`, I got zero blank-line paragraph separations on every page. PyMuPDF's text mode emits a newline at each rendered line break and never emits a blank line between paragraphs. So even with the ordering corrected, `\n{2,}` would still have found nothing.

The fix was switching to `page.get_text("blocks")` - PyMuPDF's geometric layout analysis, which returns the actual typeset blocks (12 per page on average) corresponding to real paragraphs. One thing to be careful about here: I had to keep blocks in document order rather than sorting by position. These are two-column books, and sorting blocks top-to-bottom interleaves the columns and produces alternating fragments of two different paragraphs. The PDF content stream emits each column contiguously, so document order already gives reading order.

Splitting on true block boundaries gave real paragraph boundaries, but then the median chunk was 192 characters - good for retrieval precision but way too short to give the generator useful context. Five 192-character chunks is maybe 200 tokens of reading material; the model starts hedging instead of answering. So I added a packing step: merge consecutive same-page paragraphs until the next would push the chunk past 800 characters. Every boundary stays at a real paragraph end, chunks just carry enough prose to answer from.

The numbers before and after:

| | Before | After |
|---|---|---|
| Chunks | 3,597 | 3,353 |
| Median chunk length | ~arbitrary | 615 chars |
| Boundaries at a paragraph end | ~0% | 93% |

While I was measuring all of this, I found another issue: about 7% of chunks had ligature-broken words - `reﬂ ection`, `diﬀ erence`, `ﬁ lter`. These are Unicode ligature characters (ﬁ = U+FB01, ﬀ = U+FB00) followed by spaces that shouldn't be there - a common artifact of PDF text extraction. FTS5 can't match "reflection" to "reﬂ ection". I added ligature normalization in `clean_text()` before any other transformation, and after the rebuild the FTS5 match count for "reflection" went from 0 to 92.

---

## Why the index is built offline

The original pipeline built the index in `RAGPipeline.__init__()`. If the index didn't exist, the constructor would read all five PDFs, chunk them, and embed 3,353 chunks before the process was ready to serve a single request. That path takes 3–6 minutes on CPU.

On Render, there's a boot health check. The build passes it - the process starts immediately - but then the first query triggers the index build, and the request times out. Any concurrent request triggers another build. Race condition, plus a terrible first-user experience.

The deeper problem was Render's ephemeral filesystem. Services spin down after 15 minutes idle on the free tier. When they spin back up, the filesystem is fresh - the index built at the previous startup is gone. So the build would be re-run on essentially every visit.

The fix is strict separation: `scripts/build_index.py` runs on my machine, writes `db/chroma.sqlite3`, `db/lexical.sqlite3`, and `db/index_manifest.json`, and those get committed to the repo. The pipeline is read-only. Deployment is a checkout, not a build.

One thing I found when doing this: the original `_db_exists()` check returned True for any non-empty `db/` directory, including a `chroma.sqlite3` from a build that crashed halfway. That file loads cleanly and returns zero results for every query. I replaced it with an actual row count: `SELECT COUNT(*) FROM embeddings`.

---

## The memory problem

I measured the serving process RSS stage by stage:

```
baseline Python 3.12                      ~30 MB
+ import langchain_huggingface (torch)   ~280 MB   ← torch runtime, not weights
+ load all-mpnet-base-v2 weights         ~700 MB
+ open Chroma collection                 ~750 MB
+ load BM25 index into RAM               ~810 MB
+ import streamlit                       ~900 MB
+ first query                            943 MB    ← peak
```

Render's free and starter plans are capped at 512 MB. The service would start, pass its health check, and be OOM-killed on the first real query - with exit code 137 (128 + SIGKILL), which doesn't look much like an out-of-memory error unless you know to look for it.

The non-obvious part: switching to a smaller sentence-transformers model doesn't fix this. MiniLM-L6 has 88 MB of weights compared to mpnet's 419 MB. But torch's runtime - the memory allocator, MKL, CUDA-less kernel code - is roughly 250 MB regardless of which model loads. The weights are not the whole problem.

The fix is removing torch from the serving image entirely. I added a Cohere embedding backend that calls their hosted API on the query path. With that:

```
baseline + langchain + chromadb + streamlit + cohere SDK ≈ 300 MB
```

Comfortably within 512 MB, and cold start drops from ~15s to ~3s (no model to load).

The tradeoff is a network call per query. But context helps: the query already spends ~12 seconds in generation. A 30ms embedding call is 0.25% of end-to-end time, and I'm not adding a new vendor dependency - the reranker already calls Cohere.

The constraint this creates: the two backends produce different vector spaces. You can't rebuild with mpnet and then query with Cohere. I wrote the model identity and embedding dimensions into `db/index_manifest.json` and made startup raise `IndexIncompatibleError` on a mismatch rather than serving confident wrong results.

---

## Why both retrievers

Dense (bi-encoder) retrieval and BM25 fail on different query types, and a textbook tutor gets both types constantly.

Dense retrieval learns semantic similarity from context. Its strength is paraphrase: "why does a ball stop rolling?" retrieves the friction passage without sharing a single content word with it. Its weakness is rare terms - if the embedding model saw "sonorous" or "Dobereiner's triads" infrequently during training, those vectors land in roughly the right neighbourhood but not precisely. On a corpus where everything is science prose, "roughly the right neighbourhood" still leaves hundreds of candidates with similar scores.

BM25 inverts this exactly. IDF (inverse document frequency) means rarer terms get higher weight. "Sonorous" appears in maybe 3 of 3,353 chunks, so a query containing it hits those 3 specifically and nothing else. But BM25 scores "how do plants make food" against "photosynthesis" as zero - no shared tokens.

Students ask both kinds of questions: "what is a sonometer?" (rare term they just met in the book) and "why does ice feel cold?" (phenomenon described in their own words). Neither retriever alone handles both populations. Both failure modes are common, and their failure sets are genuinely disjoint - which is the precondition for fusion to add information rather than average two mediocre rankings.

---

## Combining two rankings with Reciprocal Rank Fusion

The intuitive approach is a weighted score blend: `alpha * cosine_similarity + (1 - alpha) * bm25_score`. This doesn't work, and the reason is worth understanding.

Cosine similarity on a narrow corpus compresses into roughly [0.2, 0.7]. A genuinely bad match still scores 0.3. BM25 is unbounded - its magnitude scales with query length and with how rare the matched terms are. A two-word rare-term query and a twenty-word common-word query can produce scores an order of magnitude apart for matches of equal actual relevance. Any fixed `alpha` is silently tuned to one query length and one term-rarity profile, and breaks when the query distribution shifts.

Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009) sidesteps this by discarding scores and using only rankings:

```
RRF(d) = Σ_r  w_r / (k + rank_r(d))
```

Rank 3 means the same thing to both retrievers - there's nothing left to calibrate.

The constant `k = 60` is what makes it work well. With k=0, the formula degenerates: rank 1 is worth twice rank 2, so a single confident retriever dominates. With k=60, rank 1 contributes 1/61 and rank 2 contributes 1/62 - a 1.6% difference. The whole top-10 sits within 15% of itself. What discriminates is agreement between retrievers, not confidence within one:

```
found by both at rank 4:   1/64 + 1/64 = 0.0313
found by one  at rank 1:   1/61        = 0.0164   ← loses by ~2×
```

A passage that's both semantically close and contains the exact query terminology is more likely to answer the question than one satisfying only a single criterion. RRF encodes that prior directly.

k=60 is from the original TREC experiments, not tuned on this corpus. I'd need a labelled relevance set of ~100+ questions to move it defensibly.

---

## SQLite FTS5 over in-memory BM25

The original implementation used `rank_bm25` (via LangChain's `BM25Retriever`), which unpickles all 3,353 documents and rebuilds the index in RAM on every cold start. Measured cost: ~60 MB and ~0.5 seconds. On Render's free tier where services spin down after 15 minutes idle, this is paid by essentially every visitor, not once at startup.

SQLite FTS5 is a better fit. The index is built once offline and committed as `db/lexical.sqlite3`. Serving opens it read-only in ~1ms. The memory it uses sits in the OS page cache, which the kernel can reclaim under memory pressure; the BM25 heap allocation is live Python objects the GC can never touch.

There's a secondary benefit: class filtering. Filtering rank_bm25 results by class requires over-fetching and discarding in Python, so a filtered search might return far fewer than k results when one class has sparse coverage. FTS5 pushes the predicate into the query itself - `WHERE chunks.class = ? ORDER BY bm25(chunks_fts) LIMIT k` - and always returns k results from within the selected class.

One thing to note for anyone reading the code: FTS5's `bm25()` function returns negative scores, with more negative meaning more relevant. So `ORDER BY bm25(chunks_fts) LIMIT k` is ascending, which is correct but counterintuitive. There's a comment in `lexical.py` explaining this.

The Porter stemmer (`tokenize='porter unicode61'`) means "reflected" matches "reflection", which is what a student looking up "reflection of light" wants. The tokenizer differs from rank_bm25's whitespace-only splitting, so results aren't identical - but stemming is a genuine improvement here, not just a change.

---

## What fails loudly vs. what degrades quietly

This runs on free-tier trial API keys. Rate limits happen. I needed to decide which failures should take the whole request down and which should just degrade the output.

The rule I arrived at: fail loudly when the alternative is being silently wrong; degrade when the alternative is being usefully partial.

**Fail at startup:**
- Vector index absent - system cannot serve anything; raise `IndexNotBuiltError` with the build command
- Index/model mismatch - this one I care most about, because the alternative is serving five confidently-cited passages with no relationship to the question. Cosine similarity still produces a total ordering even when the query and index are in different vector spaces. There's no runtime signal; the only place to catch it is at startup. So `IndexIncompatibleError` is raised before the first query.

**Degrade and record:**
- Lexical index unavailable - fall back to in-memory BM25, then dense-only if that fails too. Rare-term queries get worse but not broken.
- Reranker unavailable - return fusion-ordered candidates. Fusion already narrowed 3,353 chunks to 15 good ones; losing the reranker means slightly worse precision in the final 5, not wrong results.
- Generator unavailable - serve retrieved passages without prose. Still useful for a student who wants to see the relevant textbook section.

Every degradation is recorded in `result.degraded` and shown in the UI sidebar. A degraded system says so, rather than quietly returning worse output.

---

## The prompting approach

By the time the generator sees a prompt, we've narrowed 3,353 chunks to the 5 most relevant passages. The model's job is extraction and citation, not reasoning from internal knowledge.

Temperature is 0.0. The task is identifying which sentences answer the question and which book they came from - variance is noise, not creativity.

Passages are numbered [P1]–[P5] rather than cited by filename. This makes inline citations (`friction opposes relative motion [P1]`) easy for the model to write and easy for the reader to verify without opening a PDF.

The prompt includes an explicit refusal: the model is instructed to respond "Not found in the NCERT textbooks" when the retrieved passages don't support an answer. Without naming the refusal as a valid output, the model confabulates - it's been trained to be helpful, and "the passages don't say" doesn't feel like a helpful response to it. Naming it changes this; the model produces clean refusals on out-of-scope questions.

The system prompt also tells the model not to mention its own process ("I found in passage P2...") and to handle cross-class disagreement explicitly when Class 8 and Class 10 treat the same topic differently.

---

## What I'd do differently

- **Tune `rrf_k` on this corpus.** k=60 is a literature default. It might not be optimal for five textbooks, single subject, short questions. Tuning needs ~100 labelled question/passage pairs.
- **Query expansion.** Student questions are short and often use non-textbook wording. Generating 2–3 paraphrases and fusing all their result lists (RRF handles this natively - just more input lists) would probably be the highest-value improvement.
- **Soft class preference over hard filtering.** The current class filter is a binary predicate. A soft prior - prefer the student's class, still allow passages from other classes - would handle topics taught at multiple levels more gracefully.
- **ONNX embeddings.** If the Cohere dependency becomes a problem, `fastembert` with bge-small gives ~300 MB without a network call. It would need a full re-index since the model family differs, but the `EmbeddingProvider` interface already exists for exactly this swap.
