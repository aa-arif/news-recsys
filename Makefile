# news-recsys pipeline. Every stage is idempotent and safe to re-run.
#
#   make m1                 # data: download -> parquet -> folds -> stats
#   make DATASET=large m1   # same pipeline on MIND-large (config change only)
#   make smoke              # whole pipeline on a tiny synthetic dataset (what CI runs)

UV      ?= uv
PY      := $(UV) run python
DATASET ?= small

.DEFAULT_GOAL := help

.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- environment ------------------------------------------------------------
.PHONY: setup lint fmt types test
setup: ## install the project and dev dependencies
	$(UV) sync --extra dev

lint: ## ruff check
	$(UV) run ruff check .

fmt: ## ruff format
	$(UV) run ruff format .

types: ## pyright
	$(UV) run pyright

test: ## pytest (unit tests only; slow tests need the real dataset)
	$(UV) run pytest -m "not slow and not integration"

# --- M1: data ---------------------------------------------------------------
.PHONY: m1 data parquet splits stats
m1: data parquet splits stats ## milestone 1: raw MIND -> parquet -> folds -> stats

data: ## download the MIND archives (needs NEWSREC_HF_TOKEN)
	$(PY) scripts/download_data.py --dataset $(DATASET)

parquet: ## parse behaviors.tsv / news.tsv into parquet
	$(PY) scripts/build_parquet.py --dataset $(DATASET)

splits: ## assign chronological train/val/test folds
	$(PY) scripts/make_splits.py --dataset $(DATASET)

stats: ## dataset sizes + cold-start rate -> results/metrics/
	$(PY) scripts/data_stats.py --dataset $(DATASET)

# --- synthetic end-to-end ---------------------------------------------------
.PHONY: synthetic smoke
synthetic: ## write a tiny synthetic MIND-format dataset
	$(PY) scripts/make_synthetic.py

smoke: synthetic ## run the pipeline end-to-end on synthetic data (CI)
	$(PY) scripts/build_parquet.py --dataset synthetic
	$(PY) scripts/make_splits.py --dataset synthetic
	$(PY) scripts/data_stats.py --dataset synthetic

# --- housekeeping -----------------------------------------------------------
.PHONY: clean clean-data
clean: ## remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__

clean-data: ## remove derived data (keeps the downloaded archives)
	rm -rf data/processed artifacts
