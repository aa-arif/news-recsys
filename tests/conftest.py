"""Shared fixtures.

Every test runs against the synthetic MIND-format dataset, so the suite never needs the
licensed data and still exercises the real parsing / splitting / feature code paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from news_recsys.config import Settings
from news_recsys.data.parse import build_parquet
from news_recsys.data.splits import assign_folds
from news_recsys.data.synthetic import write_synthetic_mind


@pytest.fixture(scope="session")
def synthetic_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """A fully built tiny dataset: raw TSVs, parquet tables and folds."""
    root = tmp_path_factory.mktemp("newsrec")
    settings = Settings(
        dataset="synthetic",
        data_dir=root / "data",
        artifacts_dir=root / "artifacts",
        results_dir=root / "results",
        hf_token=None,
    )
    settings.ensure_dirs()
    write_synthetic_mind(settings.raw_dir, n_users=60, n_news=120, impressions_per_day=80, seed=7)
    build_parquet(settings.raw_dir, settings.processed_dir)
    assign_folds(settings)
    return settings


@pytest.fixture(scope="session")
def raw_dir(synthetic_settings: Settings) -> Path:
    return synthetic_settings.raw_dir
