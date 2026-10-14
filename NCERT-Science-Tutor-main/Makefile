.PHONY: help install install-serve test lint format check run cli demo eval eval-full index index-force preflight clean

help:
	@echo ""
	@echo "  NCERT Science Tutor"
	@echo ""
	@echo "  Setup"
	@echo "    install         full dev environment (torch, tests, eval)"
	@echo "    install-serve   serving deps only, no torch - mirrors the Render image"
	@echo ""
	@echo "  Run"
	@echo "    run             Streamlit UI"
	@echo "    cli Q=\"...\"     one question from the command line"
	@echo "    preflight       can this checkout serve? exits non-zero if not"
	@echo ""
	@echo "  Index  (see docs/adr/0001-prebuilt-index.md)"
	@echo "    index           build if absent"
	@echo "    index-force     rebuild from scratch, re-reading the PDFs"
	@echo ""
	@echo "  Quality"
	@echo "    test            pytest - no network, no API keys"
	@echo "    lint            ruff check + format check"
	@echo "    format          apply ruff formatting"
	@echo "    check           lint + test"
	@echo "    eval            retrieval metrics only - no LLM"
	@echo "    eval-full       full RAGAS suite - needs both API keys"
	@echo ""

# --------------------------------------------------------------------------- setup
install:
	pip install --upgrade pip
	pip install -r requirements.txt -r requirements-local.txt -r requirements-dev.txt

install-serve:
	pip install --upgrade pip
	pip install -r requirements.txt

# ----------------------------------------------------------------------------- run
run:
	python -m streamlit run app/ui.py

cli:
	@python -m app.main "$(Q)"

preflight:
	python -m scripts.preflight

# --------------------------------------------------------------------------- index
index:
	python -m scripts.build_index

index-force:
	python -m scripts.build_index --force --no-cache

# ------------------------------------------------------------------------- quality
test:
	pytest -q --cov=src --cov-report=term-missing

lint:
	ruff check src/ app/ scripts/ eval/ tests/
	ruff format --check src/ app/ scripts/ eval/ tests/

format:
	ruff format src/ app/ scripts/ eval/ tests/
	ruff check --fix src/ app/ scripts/ eval/ tests/

check: lint test

eval:
	python -m eval.evaluate --retrieval-only

eval-full:
	python -m eval.evaluate --output eval/results/latest.json

# --------------------------------------------------------------------------- clean
# Note: does NOT touch db/. That is a committed artifact, not a cache
# (docs/adr/0001-prebuilt-index.md). Use `make index-force` to rebuild it.
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path "./ncert/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned caches; db/ left intact (it is a build artifact, not a cache)"
