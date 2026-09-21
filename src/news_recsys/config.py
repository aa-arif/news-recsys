"""Central configuration.

Everything that differs between MIND-small and MIND-large lives here, so that
milestone 7 ("re-run on MIND-large") is a config change and nothing else.

Values are read from the environment with the ``NEWSREC_`` prefix, falling back to
the defaults below. A local ``.env`` is honoured but never committed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

DatasetVariant = Literal["small", "large", "synthetic"]

#: Official download locations, linked from https://msnews.github.io/
HF_DATASET_REPO = "yjw1029/MIND"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main"

#: Expected zip sizes in bytes, verified by HTTP HEAD on 2026-09-20. Used as a cheap
#: integrity check so a truncated download fails loudly instead of silently.
EXPECTED_ZIP_BYTES: dict[str, int] = {
    "MINDsmall_train.zip": 52_953_372,
    "MINDsmall_dev.zip": 30_946_172,
    "MINDlarge_train.zip": 530_197_363,
    "MINDlarge_dev.zip": 103_456_953,
    "MINDlarge_test.zip": 604_624_665,
}


class Settings(BaseSettings):
    """Runtime configuration for the whole pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="NEWSREC_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- dataset -----------------------------------------------------------
    dataset: DatasetVariant = "small"
    hf_token: str | None = None

    # ---- paths -------------------------------------------------------------
    root_dir: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    results_dir: Path = REPO_ROOT / "results"

    # ---- reproducibility ---------------------------------------------------
    seed: int = 42

    # ---- text embeddings ---------------------------------------------------
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_dim: int = 384
    text_max_tokens: int = 96
    embed_batch_size: int = 256

    # ---- history / sequence ------------------------------------------------
    max_history: int = 50

    # ---- time-aware features ----------------------------------------------
    #: Beta-prior strength for smoothed CTR: ctr = (clicks + a) / (impressions + a + b).
    ctr_prior_clicks: float = 1.0
    ctr_prior_impressions: float = 50.0
    #: Half-lives (hours) for the exponentially decayed popularity counters.
    popularity_half_lives_hours: tuple[float, ...] = (1.0, 6.0, 24.0)

    # ---- retrieval ---------------------------------------------------------
    two_tower_dim: int = 128
    two_tower_epochs: int = 3
    two_tower_batch_size: int = 512
    two_tower_lr: float = 1e-3
    faiss_hnsw_m: int = 32
    faiss_ef_construction: int = 200
    faiss_ef_search: int = 64
    retrieval_candidates: int = 200
    #: Fraction of ``retrieval_candidates`` drawn from the "most popular right now" source
    #: rather than the two-tower ANN. Measured in M3: on MIND-small a user-independent
    #: recency/popularity retriever has far higher recall than the learned tower, so the
    #: candidate set blends both (0.0 disables the popularity source).
    #:
    #: A *share* rather than a count, so that changing the candidate budget - which is a
    #: latency knob - does not silently change the source mix, which is a quality decision.
    popularity_share: float = 0.5
    #: Size of the trending list maintained in Redis for that source.
    popularity_list_size: int = 500

    # ---- ranking -----------------------------------------------------------
    ranker_epochs: int = 3
    ranker_impressions_per_batch: int = 64
    ranker_lr: float = 1e-3
    #: Shown-but-not-clicked rows kept per impression during training. Positives are
    #: always kept; the resulting calibration bias is corrected in closed form at
    #: inference (see models/calibration.py).
    ranker_negative_sample_rate: float = 0.25
    #: The ranker attends over a shorter history than retrieval: target attention costs
    #: one MLP evaluation per (candidate, history item) pair, which dominates CPU time.
    ranker_max_history: int = 30
    ranker_item_dim: int = 64
    ranker_attention_dim: int = 32
    ranker_cross_layers: int = 3
    ranker_mlp_dims: tuple[int, ...] = (256, 128, 64)
    ranker_dropout: float = 0.1

    # ---- serving -----------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    serve_host: str = "127.0.0.1"
    serve_port: int = 8000
    serve_default_k: int = 10
    #: FastAPI runs sync endpoints in a thread pool. Its default (40) is far more than
    #: this box has cores, and oversubscribing turns GIL contention into p99 latency:
    #: queueing in the accept backlog is cheaper than queueing on the interpreter lock.
    serve_threadpool_size: int = 8
    ort_intra_op_threads: int = 2
    ort_inter_op_threads: int = 1
    user_embedding_cache_size: int = 0  # 0 disables the cache (M6 toggles this)

    # ---- evaluation --------------------------------------------------------
    ndcg_cutoffs: tuple[int, ...] = (5, 10)
    recall_cutoffs: tuple[int, ...] = (10, 50, 100, 200, 500)

    # ---- derived paths -----------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw" / self.dataset

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed" / self.dataset

    @computed_field  # type: ignore[prop-decorator]
    @property
    def artifact_dir(self) -> Path:
        return self.artifacts_dir / self.dataset

    @computed_field  # type: ignore[prop-decorator]
    @property
    def metrics_dir(self) -> Path:
        return self.results_dir / "metrics"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def zip_names(self) -> dict[str, str]:
        """Split name -> zip file name for the configured dataset variant."""
        if self.dataset == "synthetic":
            return {}
        tag = "MINDsmall" if self.dataset == "small" else "MINDlarge"
        return {"train": f"{tag}_train.zip", "dev": f"{tag}_dev.zip"}

    def ensure_dirs(self) -> None:
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.artifact_dir,
            self.metrics_dir,
            self.figures_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=8)
def _cached_settings(overrides: tuple[tuple[str, object], ...]) -> Settings:
    return Settings(**dict(overrides))  # type: ignore[arg-type]


def get_settings(**overrides: object) -> Settings:
    """Return settings, optionally overriding fields (used by tests and scripts)."""
    return _cached_settings(tuple(sorted(overrides.items())))


def seed_everything(seed: int | None = None) -> int:
    """Seed python, numpy and torch. Returns the seed actually used."""
    import random

    import numpy as np

    resolved = seed if seed is not None else get_settings().seed
    random.seed(resolved)
    np.random.seed(resolved)
    os.environ["PYTHONHASHSEED"] = str(resolved)
    try:
        import torch

        torch.manual_seed(resolved)
        torch.use_deterministic_algorithms(False)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass
    return resolved
