# news-recsys pipeline. Every stage is idempotent and safe to re-run.
#
#   make m1 m2 m3 m4        # offline: data -> features -> retrieval -> ranking
#   make m5                 # serving: ONNX export, redis, seed
#   make m6                 # load test (needs `make serve` running in another shell)
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
.PHONY: setup lint fmt types test check
setup: ## install the project and dev dependencies
	$(UV) sync --extra dev

lint: ## ruff check
	$(UV) run ruff check .

fmt: ## ruff format
	$(UV) run ruff format .

types: ## pyright
	$(UV) run pyright

test: ## pytest (unit tests; slow/integration tests need data or a server)
	$(UV) run pytest -m "not slow and not integration"

check: lint types test ## everything CI runs except the smoke test

# --- M1: data ---------------------------------------------------------------
.PHONY: m1 data parquet splits stats sysinfo
m1: data parquet splits stats sysinfo ## milestone 1: raw MIND -> parquet -> folds -> stats

data: ## download the MIND archives (needs NEWSREC_HF_TOKEN)
	$(PY) scripts/download_data.py --dataset $(DATASET)

parquet: ## parse behaviors.tsv / news.tsv into parquet
	$(PY) scripts/build_parquet.py --dataset $(DATASET)

splits: ## assign chronological train/val/test folds
	$(PY) scripts/make_splits.py --dataset $(DATASET)

stats: ## dataset sizes + cold-start rate -> results/metrics/
	$(PY) scripts/data_stats.py --dataset $(DATASET)

sysinfo: ## record the hardware every latency number is measured on
	$(PY) scripts/system_info.py --dataset $(DATASET)

# --- M2: features and baselines ---------------------------------------------
.PHONY: m2 embeddings features baselines
m2: embeddings features baselines ## milestone 2: embeddings -> shared features -> baselines

embeddings: ## vocabulary + sentence-transformer article embeddings
	$(PY) scripts/embed_news.py --dataset $(DATASET)

features: ## ordered replay -> leak-free feature matrices + serving snapshot
	$(PY) scripts/build_features.py --dataset $(DATASET)

baselines: ## time-aware popularity + LightGBM LambdaRank
	$(PY) scripts/train_baselines.py --dataset $(DATASET)

# --- M3: retrieval ----------------------------------------------------------
.PHONY: m3 two-tower index retrieval
m3: two-tower index retrieval ## milestone 3: two-tower -> FAISS HNSW -> recall/latency

two-tower: ## train the two-tower model (in-batch softmax + logQ correction)
	$(PY) scripts/train_two_tower.py --dataset $(DATASET)

index: ## build the FAISS HNSW index over item vectors
	$(PY) scripts/build_index.py --dataset $(DATASET)

retrieval: ## Recall@K over the full catalogue + efSearch sweep
	$(PY) scripts/eval_retrieval.py --dataset $(DATASET)

# --- M4: ranking ------------------------------------------------------------
.PHONY: m4 ranker
m4: ranker ## milestone 4: DIN + DCN-v2 ranker, cold-start slices, calibration

ranker: ## train and evaluate the ranker on the impression logs
	$(PY) scripts/train_ranker.py --dataset $(DATASET)

# --- M5: serving ------------------------------------------------------------
.PHONY: m5 onnx redis seed serve skew compose-up compose-down
m5: onnx redis seed ## milestone 5: ONNX export + redis + seeded online feature store

onnx: ## export the user tower and ranker to ONNX (verified against torch)
	$(PY) scripts/export_onnx.py --dataset $(DATASET)

redis: ## start the redis container
	docker compose up -d redis

seed: ## load the feature-store snapshot and user histories into redis
	$(PY) scripts/seed_redis.py --dataset $(DATASET) --flush

serve: ## run the API locally (uvicorn, one worker)
	$(UV) run uvicorn news_recsys.serving.app:app --host 127.0.0.1 --port 8000 --workers 1 --log-level warning

skew: ## training/serving skew check against the running server
	$(PY) scripts/check_skew.py --dataset $(DATASET)

compose-up: ## API + redis in docker
	docker compose up -d --build

compose-down: ## stop the docker stack
	docker compose down

# --- M6: load test ----------------------------------------------------------
.PHONY: m6 loadtest loadtest-tuned
m6: loadtest ## milestone 6: latency vs QPS ladder (server must be running)

loadtest: ## locust ladder against the running server
	$(PY) scripts/load_test.py --dataset $(DATASET) --label baseline

loadtest-tuned: ## same ladder with the tuned configuration
	$(PY) scripts/load_test.py --dataset $(DATASET) --label tuned

# --- M7: stretch ------------------------------------------------------------
.PHONY: m7 rerank
m7: rerank ## milestone 7: MMR diversity trade-off

rerank: ## nDCG vs intra-list diversity as the MMR lambda varies
	$(PY) scripts/eval_rerank.py --dataset $(DATASET)

# --- reporting --------------------------------------------------------------
.PHONY: readme
readme: ## regenerate the README tables from results/metrics/*.json
	$(PY) scripts/build_readme.py --dataset $(DATASET)

# --- synthetic end-to-end ---------------------------------------------------
.PHONY: synthetic smoke
synthetic: ## write a tiny synthetic MIND-format dataset
	$(PY) scripts/make_synthetic.py

smoke: synthetic ## run the pipeline end-to-end on synthetic data (CI)
	$(PY) scripts/build_parquet.py --dataset synthetic
	$(PY) scripts/make_splits.py --dataset synthetic
	$(PY) scripts/data_stats.py --dataset synthetic
	$(PY) scripts/embed_news.py --dataset synthetic
	$(PY) scripts/build_features.py --dataset synthetic
	$(PY) scripts/train_baselines.py --dataset synthetic --num-boost-round 40
	$(PY) scripts/train_two_tower.py --dataset synthetic --epochs 1 --batch-size 64 --val-sample 100
	$(PY) scripts/build_index.py --dataset synthetic
	$(PY) scripts/eval_retrieval.py --dataset synthetic --latency-queries 100
	$(PY) scripts/train_ranker.py --dataset synthetic --epochs 1 --val-impressions 50
	$(PY) scripts/export_onnx.py --dataset synthetic
	$(PY) scripts/eval_rerank.py --dataset synthetic --max-impressions 100

# --- housekeeping -----------------------------------------------------------
.PHONY: clean clean-data
clean: ## remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__

clean-data: ## remove derived data (keeps the downloaded archives)
	rm -rf data/processed artifacts
