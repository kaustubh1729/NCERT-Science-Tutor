# Contributing

Setup

```bash
git clone https://github.com/KRISHNA-12082006/Science_NCERT.git
cd Science_NCERT
git lfs pull                          # only needed to rebuild the index

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                  # add your API keys
python -m scripts.preflight           # confirms the checkout can serve
```

## Before opening a PR

```bash
ruff check src/ app/ scripts/ eval/ tests/
pytest -q
```

CI runs both on Python 3.10 and 3.12.

## Two things worth knowing

**Serving code never writes to `db/`.** The index is built offline by `scripts/build_index.py` and committed. If you need to change chunking, the embedding model, or add PDFs, run `python -m scripts.build_index --force` and commit the rebuilt `db/`. Nothing in `src/` or `app/` should touch it.

**Tests don't touch the network.** No test should need `COHERE_API_KEY`, `NVIDIA_API_KEY`, or the actual PDF files. Build any test fixtures in `tmp_path`; mock external calls.

## Changing retrieval

If you change chunking, embedding, or retrieval logic, rebuild the index and check the quality metrics before committing:

```bash
python -m scripts.build_index --force
python -m eval.evaluate --retrieval-only
git add -f db/
```

## Code style

Ruff for linting and formatting (config in `pyproject.toml`). Type hints on public functions. All tunables in `src/config.py`. See `DESIGN.md` for the reasoning behind major architectural decisions.
