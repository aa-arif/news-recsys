"""MIND's evaluation protocol, implemented once and used by every model here.

MIND scores a model *per impression*, not over a pooled set of rows: for each impression
the candidates shown to that user are ranked, a metric is computed on that slate, and the
metrics are averaged over impressions. That matters because impressions differ wildly in
slate size and positive count; pooling rows would let a few huge slates dominate.

Metrics follow the definitions used by the official MIND scorer:

* **AUC**     - ROC AUC within the impression (equivalently the Mann-Whitney statistic).
* **MRR**     - mean of ``1 / rank`` over the positives of the impression.
* **nDCG@k**  - binary-gain nDCG truncated at k, normalised by the ideal ranking.

Impressions that are all-positive or all-negative have no defined AUC/MRR/nDCG and are
excluded; the count of excluded impressions is reported alongside, never silently hidden.

Everything returns per-impression arrays as well as the mean, which is what makes the
bootstrap confidence intervals and the cold-start slices possible downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def group_boundaries(group_ids: NDArray[Any]) -> IntArray:
    """Start offsets of each contiguous run in ``group_ids`` (plus the final length).

    The event tables are sorted by ``(time, impression_key, position)``, so rows of one
    impression are contiguous and a single pass finds the slate boundaries.
    """
    if group_ids.size == 0:
        return np.zeros(1, dtype=np.int64)
    changes = np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1
    return np.concatenate(([0], changes, [group_ids.size])).astype(np.int64)


def _average_ranks(scores: FloatArray) -> FloatArray:
    """Ranks of ``scores`` in ascending order, ties sharing their average rank."""
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    sorted_scores = scores[order]
    # Average the ranks inside each run of equal scores so tied candidates cannot be
    # ordered by their (arbitrary) position in the slate.
    start = 0
    for index in range(1, sorted_scores.size + 1):
        if index == sorted_scores.size or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def _auc(labels: FloatArray, scores: FloatArray) -> float:
    """ROC AUC via the rank-sum identity; tie-safe and allocation-light."""
    positives = labels.sum()
    negatives = labels.size - positives
    ranks = _average_ranks(scores)
    rank_sum = ranks[labels > 0].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _mrr(labels: FloatArray, scores: FloatArray) -> float:
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    reciprocal = ordered / np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(reciprocal.sum() / ordered.sum())


def _dcg(ordered_labels: FloatArray, k: int) -> float:
    top = ordered_labels[:k]
    discounts = np.log2(np.arange(2, top.size + 2, dtype=np.float64))
    return float(((2**top - 1) / discounts).sum())


def _ndcg(labels: FloatArray, scores: FloatArray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")
    actual = _dcg(labels[order], k)
    ideal = _dcg(np.sort(labels)[::-1], k)
    return actual / ideal if ideal > 0 else 0.0


@dataclass
class RankingReport:
    """Per-impression metric arrays plus their means."""

    impression_ids: IntArray
    auc: FloatArray
    mrr: FloatArray
    ndcg: dict[int, FloatArray]
    n_impressions: int
    n_skipped: int
    cutoffs: tuple[int, ...] = (5, 10)
    extra: dict[str, Any] = field(default_factory=dict)

    def means(self) -> dict[str, float]:
        out = {"auc": float(self.auc.mean()), "mrr": float(self.mrr.mean())}
        for cutoff in self.cutoffs:
            out[f"ndcg@{cutoff}"] = float(self.ndcg[cutoff].mean())
        return out

    def bootstrap_ci(
        self, *, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 42
    ) -> dict[str, list[float]]:
        """Percentile bootstrap over impressions for every metric.

        Impressions are the independent unit here, so resampling impressions (not rows)
        is the right thing; the interval widths say whether a model difference is real.
        """
        rng = np.random.default_rng(seed)
        n = self.auc.size
        if n == 0:
            return {}
        draws = rng.integers(0, n, size=(n_resamples, n))
        out: dict[str, list[float]] = {}
        for name, values in [("auc", self.auc), ("mrr", self.mrr)] + [
            (f"ndcg@{cutoff}", self.ndcg[cutoff]) for cutoff in self.cutoffs
        ]:
            samples = values[draws].mean(axis=1)
            low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
            out[name] = [float(low), float(high)]
        return out

    def to_dict(self, *, with_ci: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **self.means(),
            "n_impressions": self.n_impressions,
            "n_skipped_impressions": self.n_skipped,
        }
        if with_ci:
            payload["ci95"] = self.bootstrap_ci()
        payload.update(self.extra)
        return payload

    def subset(self, mask: NDArray[np.bool_]) -> RankingReport:
        """Restrict to a subset of impressions (used for the cold-start slices)."""
        return RankingReport(
            impression_ids=self.impression_ids[mask],
            auc=self.auc[mask],
            mrr=self.mrr[mask],
            ndcg={cutoff: values[mask] for cutoff, values in self.ndcg.items()},
            n_impressions=int(mask.sum()),
            n_skipped=self.n_skipped,
            cutoffs=self.cutoffs,
        )


def evaluate_ranking(
    labels: NDArray[Any],
    scores: NDArray[Any],
    group_ids: NDArray[Any],
    *,
    cutoffs: tuple[int, ...] = (5, 10),
) -> RankingReport:
    """Score a model the way MIND does: one metric per impression, then average.

    ``labels``, ``scores`` and ``group_ids`` are row-aligned and must be grouped by
    impression (the loaders in :mod:`news_recsys.data.splits` guarantee that ordering).
    """
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    bounds = group_boundaries(np.asarray(group_ids))

    kept_ids: list[int] = []
    auc_values: list[float] = []
    mrr_values: list[float] = []
    ndcg_values: dict[int, list[float]] = {cutoff: [] for cutoff in cutoffs}
    skipped = 0

    for start, end in pairwise(bounds):
        slate_labels = labels[start:end]
        positives = slate_labels.sum()
        if positives == 0 or positives == slate_labels.size:
            skipped += 1
            continue
        slate_scores = scores[start:end]
        kept_ids.append(int(group_ids[start]))
        auc_values.append(_auc(slate_labels, slate_scores))
        mrr_values.append(_mrr(slate_labels, slate_scores))
        for cutoff in cutoffs:
            ndcg_values[cutoff].append(_ndcg(slate_labels, slate_scores, cutoff))

    return RankingReport(
        impression_ids=np.asarray(kept_ids, dtype=np.int64),
        auc=np.asarray(auc_values, dtype=np.float64),
        mrr=np.asarray(mrr_values, dtype=np.float64),
        ndcg={
            cutoff: np.asarray(values, dtype=np.float64) for cutoff, values in ndcg_values.items()
        },
        n_impressions=len(kept_ids),
        n_skipped=skipped,
        cutoffs=cutoffs,
    )


def log_loss(labels: NDArray[Any], probabilities: NDArray[Any], *, eps: float = 1e-7) -> float:
    """Binary cross entropy over rows (not impressions): a calibration-sensitive metric."""
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1 - eps)
    return float(
        -(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)).mean()
    )


def brier_score(labels: NDArray[Any], probabilities: NDArray[Any]) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return float(((probabilities - labels) ** 2).mean())


def calibration_curve(
    labels: NDArray[Any],
    probabilities: NDArray[Any],
    *,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> dict[str, Any]:
    """Reliability curve plus expected calibration error.

    Quantile bins by default: with a ~4% positive rate, uniform bins put almost every row
    in the first bucket and the curve says nothing.
    """
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if strategy == "quantile":
        edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    if edges.size < 2:
        # A constant scorer has no quantile structure; score it as a single bin rather
        # than silently returning an empty (and therefore perfect-looking) curve.
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    bin_index = np.digitize(probabilities, edges[1:-1], right=False)
    rows: list[dict[str, float]] = []
    ece = 0.0
    total = labels.size
    for index in range(len(edges) - 1):
        mask = bin_index == index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted = float(probabilities[mask].mean())
        observed = float(labels[mask].mean())
        rows.append(
            {
                "bin": index,
                "count": count,
                "mean_predicted": mean_predicted,
                "observed_rate": observed,
            }
        )
        ece += count / total * abs(mean_predicted - observed)

    return {
        "bins": rows,
        "ece": float(ece),
        "mean_predicted": float(probabilities.mean()),
        "observed_rate": float(labels.mean()),
    }


def probability_report(labels: NDArray[Any], probabilities: NDArray[Any]) -> dict[str, Any]:
    """Everything that judges the *value* of a score rather than its ordering."""
    return {
        "log_loss": log_loss(labels, probabilities),
        "brier": brier_score(labels, probabilities),
        "calibration": calibration_curve(labels, probabilities),
    }
