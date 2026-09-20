"""Turning ranking scores into probabilities, fitted on validation only.

Ranking losses (LambdaRank, listwise softmax) produce scores whose ordering is
meaningful but whose scale is arbitrary, so log loss and a reliability curve are
meaningless until the scores are mapped to probabilities. Platt scaling - a
one-dimensional logistic regression on the score - is the standard, minimal way to do
that. It is fitted on the **validation** fold; the test fold only ever sees the fitted
transform.

:class:`PriorCorrection` handles the other calibration problem in this project: when
negatives are downsampled during ranker training, the model learns the sampled positive
rate rather than the true one, and the fix is a closed-form shift of the logit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression


def _logit(probabilities: NDArray[np.float64], eps: float = 1e-7) -> NDArray[np.float64]:
    clipped = np.clip(probabilities, eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


@dataclass
class PlattCalibrator:
    """Logistic calibration of a single score column."""

    slope: float = 1.0
    intercept: float = 0.0
    fitted_on: str = "val"

    def fit(self, scores: NDArray[np.float64], labels: NDArray[np.float64]) -> PlattCalibrator:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(labels, dtype=np.float64)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(scores, labels)
        self.slope = float(np.ravel(model.coef_)[0])
        self.intercept = float(np.ravel(model.intercept_)[0])
        return self

    def transform(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        z = self.slope * np.asarray(scores, dtype=np.float64) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict[str, float | str]:
        return {"slope": self.slope, "intercept": self.intercept, "fitted_on": self.fitted_on}

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> PlattCalibrator:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            slope=float(payload["slope"]),
            intercept=float(payload["intercept"]),
            fitted_on=str(payload.get("fitted_on", "val")),
        )


@dataclass
class PriorCorrection:
    """Undo the bias introduced by downsampling negatives.

    If negatives are kept with probability ``w``, a model trained on the sample estimates
    ``p_s`` where ``odds_s = odds_true / w``. The corrected probability is therefore
    obtained by subtracting ``log(1/w)`` from the logit - a shift that leaves the ranking
    within an impression untouched but makes the numbers mean something.
    """

    negative_keep_rate: float = 1.0

    @property
    def logit_shift(self) -> float:
        return float(np.log(self.negative_keep_rate))

    def apply(self, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.negative_keep_rate >= 1.0:
            return np.asarray(probabilities, dtype=np.float64)
        corrected = _logit(np.asarray(probabilities, dtype=np.float64)) + self.logit_shift
        return 1.0 / (1.0 + np.exp(-corrected))
