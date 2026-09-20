"""The metric implementation is checked against independent references.

AUC is checked against scikit-learn, and MRR / nDCG against a transcription of the
reference implementation published with MIND, so a subtle ranking bug cannot quietly
inflate every number in the results table.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from news_recsys.eval.metrics import (
    calibration_curve,
    evaluate_ranking,
    group_boundaries,
    log_loss,
)


def reference_mrr(labels: np.ndarray, scores: np.ndarray) -> float:
    """MIND's reference MRR."""
    order = np.argsort(scores)[::-1]
    ordered = np.take(labels, order)
    reciprocal = ordered / (np.arange(len(ordered)) + 1)
    return float(np.sum(reciprocal) / np.sum(labels))


def reference_ndcg(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """MIND's reference nDCG@k."""

    def dcg(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        gains = 2**y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return float(np.sum(gains / discounts))

    best = dcg(labels, labels, k)
    return dcg(labels, scores, k) / best


def test_group_boundaries_finds_runs() -> None:
    groups = np.array([7, 7, 7, 9, 9, 11])
    assert group_boundaries(groups).tolist() == [0, 3, 5, 6]
    assert group_boundaries(np.array([], dtype=np.int64)).tolist() == [0]


def test_matches_reference_implementations_on_random_slates() -> None:
    rng = np.random.default_rng(0)
    labels_all, scores_all, groups_all = [], [], []
    expected_auc, expected_mrr, expected_ndcg5 = [], [], []

    for impression in range(200):
        size = int(rng.integers(5, 40))
        labels = (rng.random(size) < 0.15).astype(np.float64)
        if labels.sum() in (0, size):  # keep only slates the protocol scores
            labels[0], labels[-1] = 1.0, 0.0
        scores = rng.normal(size=size) + labels * 0.7  # a model that is better than chance
        labels_all.append(labels)
        scores_all.append(scores)
        groups_all.append(np.full(size, impression))
        expected_auc.append(roc_auc_score(labels, scores))
        expected_mrr.append(reference_mrr(labels, scores))
        expected_ndcg5.append(reference_ndcg(labels, scores, 5))

    report = evaluate_ranking(
        np.concatenate(labels_all), np.concatenate(scores_all), np.concatenate(groups_all)
    )

    np.testing.assert_allclose(report.auc, expected_auc, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(report.mrr, expected_mrr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(report.ndcg[5], expected_ndcg5, rtol=1e-12, atol=1e-12)
    assert report.n_impressions == 200
    assert report.n_skipped == 0


def test_degenerate_slates_are_skipped_not_scored() -> None:
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    groups = np.array([1, 1, 2, 2])  # one all-negative slate, one all-positive slate
    report = evaluate_ranking(labels, scores, groups)
    assert report.n_impressions == 0
    assert report.n_skipped == 2


def test_ties_share_rank_so_slate_order_cannot_leak() -> None:
    # A constant scorer must land exactly on 0.5 AUC regardless of where positives sit.
    labels = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    scores = np.zeros(5)
    groups = np.zeros(5, dtype=np.int64)
    report = evaluate_ranking(labels, scores, groups)
    assert report.auc[0] == pytest.approx(0.5)


def test_perfect_and_inverted_rankings() -> None:
    labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    groups = np.zeros(5, dtype=np.int64)
    perfect = evaluate_ranking(labels, np.array([1.0, 0.9, 0.2, 0.1, 0.0]), groups)
    assert perfect.auc[0] == pytest.approx(1.0)
    assert perfect.mrr[0] == pytest.approx(0.75)  # positives at ranks 1 and 2
    assert perfect.ndcg[5][0] == pytest.approx(1.0)

    inverted = evaluate_ranking(labels, np.array([0.0, 0.1, 0.2, 0.9, 1.0]), groups)
    assert inverted.auc[0] == pytest.approx(0.0)


def test_bootstrap_ci_brackets_the_mean() -> None:
    rng = np.random.default_rng(3)
    labels, scores, groups = [], [], []
    for impression in range(300):
        size = 20
        slate = np.zeros(size)
        slate[rng.integers(0, size)] = 1.0
        labels.append(slate)
        scores.append(rng.normal(size=size) + slate * 0.5)
        groups.append(np.full(size, impression))
    report = evaluate_ranking(
        np.concatenate(labels), np.concatenate(scores), np.concatenate(groups)
    )
    ci = report.bootstrap_ci(n_resamples=200)
    low, high = ci["auc"]
    assert low < report.means()["auc"] < high


def test_log_loss_matches_manual_computation() -> None:
    labels = np.array([1.0, 0.0])
    probabilities = np.array([0.8, 0.3])
    expected = -(np.log(0.8) + np.log(0.7)) / 2
    assert log_loss(labels, probabilities) == pytest.approx(expected)


def test_calibration_curve_detects_a_biased_model() -> None:
    rng = np.random.default_rng(1)
    labels = (rng.random(20_000) < 0.04).astype(np.float64)
    honest = np.full(20_000, 0.04)
    inflated = np.full(20_000, 0.20)
    assert calibration_curve(labels, honest)["ece"] < 0.01
    assert calibration_curve(labels, inflated)["ece"] > 0.15
