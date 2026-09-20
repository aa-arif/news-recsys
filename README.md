# news-recsys

Two-stage news recommender on the [MIND](https://msnews.github.io/) dataset: two-tower
retrieval over FAISS plus a DIN + DCN-v2 ranker, served behind FastAPI with ONNX Runtime
and Redis.

Results, architecture diagram and reproduction steps land here as milestones complete.
Every number in this file is produced by a script in this repo and stored under `results/`.

**Status:** in progress — see the Makefile for the pipeline stages.
