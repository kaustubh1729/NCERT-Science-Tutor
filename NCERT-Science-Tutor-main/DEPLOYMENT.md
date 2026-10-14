# Deploying to Render

## Read this first

There is exactly one decision that determines whether this deploys successfully, and
the failure mode when you get it wrong is deceptive: the service builds, starts,
passes its health check, reports healthy in the dashboard - and is OOM-killed on the
first real query.

**Measured serving memory:**

| `NCERT_EMBEDDING_BACKEND` | RSS | Cold start | Works on 512 MB? |
|---|---|---|---|
| `local` | **943 MB** | ~15 s | **No** |
| `cohere` | **~300 MB** | ~3 s | Yes |

**Render's plans:**

| Plan | RAM | Verdict |
|---|---|---|
| Free | 512 MB | `cohere` only |
| Starter ($7/mo) | 512 MB | `cohere` only |
| Standard ($25/mo) | 2 GB | either |

The difference is torch. `local` loads sentence-transformers in-process; `cohere`
calls a hosted embedding API and never imports torch. Choosing a *smaller* local
model does not help - torch's runtime is the floor, not the model weights. Reasoning
and the full measurement breakdown: [ADR 0002](docs/adr/0002-embedding-backend.md).

Pick a path:

- **[Path A - Free/Starter tier](#path-a--freestarter-512-mb)** · needs a one-time
  index rebuild with Cohere embeddings
- **[Path B - Standard tier](#path-b--standard-2-gb)** · deploy the repository as-is,
  $25/mo

---

## What ships in the repository

The retrieval index is a **committed build artifact**, not something the server
creates ([ADR 0001](docs/adr/0001-prebuilt-index.md)). A deploy is a checkout.

```
db/chroma.sqlite3        ~19 MB   dense vectors
db/lexical.sqlite3       ~3 MB    FTS5 keyword index
db/index_manifest.json            embedding model, dims, chunk count
data/*.pdf              375 MB    Git LFS - NOT needed at runtime
```

The PDFs are only needed to *rebuild* the index. Render does not fetch Git LFS content
by default, so they arrive as ~130-byte pointer files. This is harmless: nothing on
the serving path opens them. The build command attempts `git lfs pull` and is
explicitly allowed to fail.

---

## Path A - Free/Starter (512 MB)

### 1. Rebuild the index with Cohere embeddings

Required once. The `local` index that ships in the repository is 768-dimensional
mpnet; Cohere's `embed-english-v3.0` is a different vector space, and startup will
refuse to serve on a mismatch rather than return confident nonsense.

```bash
export COHERE_API_KEY=...                        # Windows: set COHERE_API_KEY=...
export NCERT_EMBEDDING_BACKEND=cohere

python -m scripts.build_index --force
```

Cost: 3,353 chunks ÷ 96 per batch = **35 API calls**, well inside a trial key's
1,000/month. Takes about a minute.

```bash
git add -f db/
git commit -m "rebuild index with cohere embeddings"
git push
```

### 2. Create the service

In the Render dashboard: **New → Blueprint**, select the repository. `render.yaml` is
detected automatically. Change `plan: starter` to `plan: free` if you want the free
tier.

### 3. Set the secrets

Render will prompt for the two variables marked `sync: false`:

| Variable | Where to get it |
|---|---|
| `COHERE_API_KEY` | https://dashboard.cohere.com/api-keys |
| `NVIDIA_API_KEY` | https://build.nvidia.com |

Everything else is already in `render.yaml`.

### 4. Deploy

The build runs `python -m scripts.preflight`, which exits non-zero if the deployment
cannot serve. A broken deploy fails during build with a diagnosis instead of at
runtime with a stack trace.

Expected build log tail:

```
[PASS] vector index    3353 vectors, 1024 dims
[PASS] index manifest  embed-english-v3.0 (3353 chunks)
[PASS] lexical index   FTS5, 3353 chunks, 3.2 MB
[PASS] NVIDIA_API_KEY  set
[PASS] COHERE_API_KEY  set
[PASS] memory profile  torch-free serving path (~300 MB RSS expected)
Preflight passed.
```

---

## Path B - Standard (2 GB)

Deploy the repository unchanged. In `render.yaml`:

```yaml
plan: standard

buildCommand: |
  set -e
  pip install --upgrade pip
  pip install -r requirements.txt -r requirements-local.txt
  git lfs pull || echo "skipped"
  python -m scripts.preflight

envVars:
  - key: NCERT_EMBEDDING_BACKEND
    value: local
```

No re-index, no per-query embedding call, no Cohere dependency on the retrieval path.
Preflight will emit a `WARN` about the memory profile; on a 2 GB instance that is
informational.

---

## Environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `NVIDIA_API_KEY` | yes | - | Generation. Without it the app serves retrieved passages only. |
| `COHERE_API_KEY` | yes¹ | - | Reranking; also embeddings on Path A. |
| `NCERT_EMBEDDING_BACKEND` | - | `local` | `local` \| `cohere`. **The decision above.** |
| `NCERT_LEXICAL_BACKEND` | - | `fts5` | `fts5` \| `memory` \| `none` |
| `NCERT_RERANK_ENABLED` | - | `true` | Set `false` to run without Cohere entirely |
| `NCERT_LLM_MODEL` | - | `meta/llama-3.1-8b-instruct` | |
| `NCERT_LLM_BASE_URL` | - | NVIDIA NIM | Any OpenAI-compatible endpoint |
| `NCERT_LOG_LEVEL` | - | `INFO` | |
| `PYTHON_VERSION` | - | `3.12.7` | Pinned; Render's default moves |

¹ Strictly optional - without it the system degrades to fusion-ordered results and
says so in the UI ([ADR 0006](docs/adr/0006-graceful-degradation.md)). Mandatory on
Path A, where it also serves embeddings.

---

## Free-tier behaviour worth knowing

**Spin-down.** Free services sleep after 15 minutes idle. The next visitor waits for a
cold start - ~3 s on Path A, ~15 s on Path B - *plus* Render's container start, so
30–60 s in practice. This is why cold start was worth optimising: it is paid by most
visitors, not once per deploy.

**Build minutes.** 500/month on free. A build here is 2–4 minutes on Path A
(no torch) and 6–10 on Path B.

**No disk.** Deliberately not requested. A Render disk would pin the service to one
instance and store a derived artifact that is already in git.

---

## Troubleshooting

### Service is killed shortly after a query - exit status 137

Out of memory. `137 = 128 + 9` (SIGKILL). You are on Path B configuration with a
Path A plan. Either set `NCERT_EMBEDDING_BACKEND=cohere` (and rebuild the index) or
move to `plan: standard`.

### `IndexIncompatibleError` at startup

```
Index was built with embedding model 'sentence-transformers/all-mpnet-base-v2'
but the process is configured to query with 'embed-english-v3.0'.
```

Working as intended. The two produce different vector spaces; querying across them
returns plausible, confidently-cited nonsense with no runtime signal. Rebuild with
the backend you intend to serve with, and commit `db/`.

### `IndexNotBuiltError` at startup

`db/` did not reach the server. Confirm it is committed - it is intentionally *not*
gitignored:

```bash
git ls-files db/ | head
```

### Build fails on `git lfs pull`

It should not - the command ends in `|| echo`. If it is fatal, remove that line
entirely; serving never opens the PDFs.

### Answers are correct but slow, or arrive all at once

Streaming needs the websocket to reach the container. Confirm
`--server.enableCORS false` is in the start command: Render terminates TLS at its
proxy, and with CORS protection on, Streamlit rejects the proxied websocket handshake.
The page loads and never connects.

### `429` in the logs, answers still appear

A Cohere trial-key rate limit. Reranking degraded to fusion order for that request;
the sidebar says so. Not an outage.

---

## Pre-deploy checklist

```bash
python -m scripts.preflight       # must exit 0
pytest -q                         # 55 tests, no network
git status --short                # db/ committed, .env NOT committed
git ls-files | grep -c "^\.env$"  # must print 0
```

---

## Other platforms

The application is a plain Streamlit process reading committed SQLite files, so it
carries no Render-specific code.

- **Hugging Face Spaces** - good fit: 16 GB RAM on the free CPU tier, no spin-down.
  Path B works unchanged. Requires a public repo, and LFS quota applies to the PDFs.
- **Fly.io** - needs a Dockerfile; 256 MB default is too small, use 512 MB with
  Path A.
- **Railway** - same shape as Render; $5/month credit.

Not suitable: Vercel and other serverless platforms. Cold start would reload the
index on every invocation, and the 250 MB bundle limit excludes the index.
