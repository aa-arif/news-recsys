"""Time-aware popularity: the baseline every news recommender must beat.

Ranking by "what is popular right now" is a genuinely strong baseline on news, and it is
the honest floor for the rest of the project: a two-tower model that cannot beat a decayed
CTR counter is not earning its serving cost.

The score is a single column of the shared feature matrix - a smoothed, exponentially
decayed click-through rate that only ever saw events strictly before the impression. Which
column (which half-life) is used is *selected on validation*, never on test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from news_recsys.config import Settings
from news_recsys.eval.metrics import evaluate_ranking
from news_recsys.features.build import FoldFeatures
from news_recsys.features.time_aware import feature_names
from news_recsys.logging_utils import get_logger

logger = get_logger("models.popularity")


def candidate_columns(settings: Settings) -> list[str]:
    """Popularity-ish columns that are legitimate stand-alone scores."""
    names = feature_names(settings)
    candidates = ["art_ctr_smooth"]
    candidates += [name for name in names if name.startswith("art_ctr_decay_")]
    candidates += [name for name in names if name.startswith("art_impr_decay_")]
    return [name for name in candidates if name in names]


@dataclass
class PopularityBaseline:
    """Ranks by one time-aware popularity column, chosen on validation."""

    settings: Settings
    column_name: str = "art_ctr_smooth"
    selection: dict[str, float] = field(default_factory=dict)

    @property
    def column_index(self) -> int:
        return feature_names(self.settings).index(self.column_name)

    def fit(self, validation: FoldFeatures) -> PopularityBaseline:
        """Pick the column with the best validation nDCG@10."""
        best_name, best_score = None, -np.inf
        for name in candidate_columns(self.settings):
            index = feature_names(self.settings).index(name)
            report = evaluate_ranking(
                validation.labels.astype(np.float64),
                validation.features[:, index].astype(np.float64),
                validation.impression_key,
                cutoffs=self.settings.ndcg_cutoffs,
            )
            score = report.means()["ndcg@10"]
            self.selection[name] = score
            logger.info("popularity candidate %-28s val nDCG@10 = %.4f", name, score)
            if score > best_score:
                best_name, best_score = name, score

        assert best_name is not None
        self.column_name = best_name
        logger.info("selected %s (val nDCG@10 = %.4f)", best_name, best_score)
        return self

    def predict(self, fold: FoldFeatures) -> NDArray[np.float64]:
        return fold.features[:, self.column_index].astype(np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "time_aware_popularity",
            "column": self.column_name,
            "selection_val_ndcg@10": self.selection,
        }
