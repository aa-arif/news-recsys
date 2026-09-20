"""Shared fixtures.

Every test runs against the synthetic MIND-format dataset, so the suite never needs the
licensed data and still exercises the real parsing, feature, retrieval and serving code.

Unit tests use *random* article embeddings rather than running the sentence-transformer:
nothing here is testing MiniLM, and a 90 MB model download would add minutes to every CI
run. The real embedding path is exercised end to end by ``make smoke``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from news_recsys.config import Settings
from news_recsys.data.parse import build_parquet
from news_recsys.data.splits import assign_folds
from news_recsys.data.synthetic import write_synthetic_mind
from news_recsys.features.build import build_features
from news_recsys.features.text import save_embeddings
from news_recsys.features.vocab import build_vocabulary

EMBEDDING_DIM = 32


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
        text_dim=EMBEDDING_DIM,
    )
    settings.ensure_dirs()
    write_synthetic_mind(settings.raw_dir, n_users=60, n_news=120, impressions_per_day=80, seed=7)
    build_parquet(settings.raw_dir, settings.processed_dir)
    assign_folds(settings)
    return settings


@pytest.fixture(scope="session")
def synthetic_artifacts(synthetic_settings: Settings) -> Settings:
    """Vocabulary, (random) embeddings, feature matrices and the serving snapshot."""
    vocabulary = build_vocabulary(synthetic_settings)
    vocabulary.save(synthetic_settings.artifact_dir)

    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(vocabulary.n_news, EMBEDDING_DIM)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    save_embeddings(embeddings, synthetic_settings.artifact_dir)

    folds = build_features(synthetic_settings, vocabulary=vocabulary)
    for fold in folds.values():
        fold.save(synthetic_settings.artifact_dir / "features")
    return synthetic_settings


@pytest.fixture(scope="session")
def raw_dir(synthetic_settings: Settings) -> Path:
    return synthetic_settings.raw_dir
