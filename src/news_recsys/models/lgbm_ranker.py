"""LightGBM LambdaRank over the engineered features.

This is the "strong tabular baseline" a ranking team reaches for before anything neural:
gradient-boosted trees on the same time-aware features, trained with a listwise objective
that optimises nDCG within each impression - the metric MIND actually reports.

Groups are impressions. Early stopping watches validation nDCG@10, so the number of trees
is chosen on validation and the test fold stays sealed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from news_recsys.config import Settings
from news_recsys.eval.metrics import group_boundaries
from news_recsys.features.build import FoldFeatures
from news_recsys.features.time_aware import feature_names
from news_recsys.features.vocab import Vocabulary
from news_recsys.logging_utils import get_logger

logger = get_logger("models.lgbm")

MODEL_FILENAME = "lgbm_lambdarank.txt"


def group_sizes(impression_keys: NDArray[np.int64]) -> NDArray[np.int32]:
    bounds = group_boundaries(impression_keys)
    return np.diff(bounds).astype(np.int32)


@dataclass
class LambdaRankModel:
    """LightGBM LambdaRank plus the feature assembly both training and serving use."""

    settings: Settings
    vocabulary: Vocabulary
    booster: lgb.Booster | None = None
    params: dict[str, Any] = field(default_factory=dict)
    best_iteration: int = 0

    @property
    def feature_names(self) -> list[str]:
        return [*feature_names(self.settings), "category_id", "subcategory_id"]

    @property
    def categorical_features(self) -> list[str]:
        return ["category_id", "subcategory_id"]

    def design_matrix(self, fold: FoldFeatures) -> NDArray[np.float32]:
        """Dense features plus the two categorical ids LightGBM splits on directly."""
        return design_matrix(fold.features, fold.news_index, self.vocabulary)

    def fit(
        self, train: FoldFeatures, validation: FoldFeatures, **overrides: Any
    ) -> LambdaRankModel:
        params: dict[str, Any] = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": list(self.settings.ndcg_cutoffs),
            "lambdarank_truncation_level": 30,
            "label_gain": [0, 1],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "max_bin": 127,
            "num_threads": 0,
            "seed": self.settings.seed,
            "verbosity": -1,
            "force_row_wise": True,
        }
        params.update(overrides)
        self.params = params

        train_set = lgb.Dataset(
            self.design_matrix(train),
            label=train.labels.astype(np.int32),
            group=group_sizes(train.impression_key),
            feature_name=self.feature_names,
            categorical_feature=self.categorical_features,
            free_raw_data=True,
        )
        valid_set = lgb.Dataset(
            self.design_matrix(validation),
            label=validation.labels.astype(np.int32),
            group=group_sizes(validation.impression_key),
            reference=train_set,
            feature_name=self.feature_names,
            categorical_feature=self.categorical_features,
            free_raw_data=True,
        )

        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=overrides.pop("num_boost_round", 600),
            valid_sets=[valid_set],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )
        self.best_iteration = int(self.booster.best_iteration or self.booster.num_trees())
        logger.info("LambdaRank stopped at iteration %d", self.best_iteration)
        return self

    def predict(self, fold: FoldFeatures) -> NDArray[np.float64]:
        if self.booster is None:
            raise RuntimeError("model is not trained")
        predictions = self.booster.predict(
            self.design_matrix(fold), num_iteration=self.best_iteration
        )
        return np.asarray(predictions, dtype=np.float64)

    def importance(self, top: int = 20) -> list[dict[str, Any]]:
        if self.booster is None:
            raise RuntimeError("model is not trained")
        gains = self.booster.feature_importance(importance_type="gain")
        names = self.booster.feature_name()
        order = np.argsort(-gains)[:top]
        total = float(gains.sum()) or 1.0
        return [
            {
                "feature": names[index],
                "gain": float(gains[index]),
                "gain_share": float(gains[index] / total),
            }
            for index in order
        ]

    def save(self, directory: Path) -> Path:
        if self.booster is None:
            raise RuntimeError("model is not trained")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MODEL_FILENAME
        self.booster.save_model(str(path), num_iteration=self.best_iteration)
        return path

    @classmethod
    def load(cls, directory: Path, settings: Settings, vocabulary: Vocabulary) -> LambdaRankModel:
        booster = lgb.Booster(model_file=str(directory / MODEL_FILENAME))
        model = cls(settings=settings, vocabulary=vocabulary, booster=booster)
        model.best_iteration = booster.num_trees()
        return model


def design_matrix(
    features: NDArray[np.float32], news_index: NDArray[np.int32], vocabulary: Vocabulary
) -> NDArray[np.float32]:
    """Attach category / subcategory ids to the dense features (``-1`` for unknown)."""
    known = news_index >= 0
    safe = np.where(known, news_index, 0)
    category = np.where(known, vocabulary.news_category[safe], -1).astype(np.float32)
    subcategory = np.where(known, vocabulary.news_subcategory[safe], -1).astype(np.float32)
    return np.column_stack([features, category, subcategory]).astype(np.float32)
