"""One evaluation entry point, so every model in this repo is scored identically.

Given a fold's features and a model's scores it produces:

* the MIND ranking metrics with bootstrap confidence intervals,
* the same metrics restricted to cold-start impressions, under **two** definitions,
* log loss, Brier score and a calibration curve for the probability estimates.

Two cold-start definitions, because they are different failure modes and differ by three
orders of magnitude on MIND-small:

``unseen_in_train``
    The clicked article never appeared in a *training* impression, so nothing about it
    was learned during training - only its content and its live counters can help. This
    covers most of the MIND-small test set and answers "does the model generalise to
    articles published after training?".

``no_prior_impressions``
    The article had not been shown to *anyone* at the moment of the request, so even its
    live counters are empty. This is the genuine first-serve case, and it is rare,
    because an article that goes live at 00:05 is already warm by 09:00.

Models that emit unbounded scores (LambdaRank, a listwise-trained neural ranker) are
passed through a calibrator fitted on the *validation* fold before the probability
metrics are computed - never on test.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from news_recsys.config import Settings
from news_recsys.eval.metrics import (
    RankingReport,
    evaluate_ranking,
    group_boundaries,
    probability_report,
)
from news_recsys.features.build import FoldFeatures
from news_recsys.features.time_aware import feature_names

COLD_DEFINITIONS = ("unseen_in_train", "no_prior_impressions")

_DEFINITION_TEXT = {
    "unseen_in_train": "clicked article absent from every training impression",
    "no_prior_impressions": "clicked article had never been shown to anyone before this request",
}


def row_cold_masks(
    fold: FoldFeatures, settings: Settings, *, train_news_index: NDArray[np.int64] | None
) -> dict[str, NDArray[np.bool_]]:
    """Row-level cold flags under both definitions."""
    cold_column = feature_names(settings).index("art_is_cold")
    masks: dict[str, NDArray[np.bool_]] = {
        "no_prior_impressions": fold.features[:, cold_column] > 0.5
    }
    if train_news_index is not None:
        size = int(max(train_news_index.max(initial=-1), fold.news_index.max(initial=-1))) + 2
        seen = np.zeros(size, dtype=bool)
        seen[train_news_index[train_news_index >= 0]] = True
        safe = np.where(fold.news_index >= 0, fold.news_index, size - 1)
        masks["unseen_in_train"] = ~seen[safe]
    return masks


def impression_cold_masks(
    fold: FoldFeatures, row_masks: dict[str, NDArray[np.bool_]]
) -> tuple[NDArray[np.int64], dict[str, NDArray[np.bool_]]]:
    """An impression is cold when the article the user clicked is cold."""
    bounds = group_boundaries(fold.impression_key)
    impression_ids: list[int] = []
    flags: dict[str, list[bool]] = {name: [] for name in row_masks}

    for start, end in pairwise(bounds):
        labels = fold.labels[start:end]
        impression_ids.append(int(fold.impression_key[start]))
        clicked = labels == 1
        for name, mask in row_masks.items():
            flags[name].append(bool(clicked.any() and mask[start:end][clicked].any()))

    return (
        np.asarray(impression_ids, dtype=np.int64),
        {name: np.asarray(values, dtype=bool) for name, values in flags.items()},
    )


def _slice_report(
    report: RankingReport,
    impression_ids: NDArray[np.int64],
    flags: NDArray[np.bool_],
    *,
    cold: bool,
) -> dict[str, Any]:
    lookup = dict(zip(impression_ids.tolist(), flags.tolist(), strict=True))
    selector = np.asarray([lookup.get(int(key), False) == cold for key in report.impression_ids])
    if not selector.any():
        return {"n_impressions": 0}
    return report.subset(selector).to_dict()


def evaluate_predictions(
    fold: FoldFeatures,
    scores: NDArray[np.float64],
    settings: Settings,
    *,
    probabilities: NDArray[np.float64] | None = None,
    train_news_index: NDArray[np.int64] | None = None,
    with_slices: bool = True,
) -> dict[str, Any]:
    """Full report for one model on one fold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = fold.labels.astype(np.float64)
    report = evaluate_ranking(labels, scores, fold.impression_key, cutoffs=settings.ndcg_cutoffs)

    payload: dict[str, Any] = {
        "fold": fold.fold,
        "overall": report.to_dict(),
        "rows": int(labels.size),
    }

    if with_slices:
        row_masks = row_cold_masks(fold, settings, train_news_index=train_news_index)
        impression_ids, impression_flags = impression_cold_masks(fold, row_masks)
        payload["cold_start"] = {
            name: {
                "definition": _DEFINITION_TEXT[name],
                "cold_row_share": float(row_masks[name].mean()),
                "cold_impression_share": float(impression_flags[name].mean()),
                "cold": _slice_report(report, impression_ids, impression_flags[name], cold=True),
                "warm": _slice_report(report, impression_ids, impression_flags[name], cold=False),
            }
            for name in row_masks
        }

    if probabilities is not None:
        payload["probability"] = probability_report(
            labels, np.asarray(probabilities, dtype=np.float64)
        )

    return payload
