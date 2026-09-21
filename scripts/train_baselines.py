"""M2: train and evaluate the baselines (time-aware popularity, LightGBM LambdaRank).

Selection happens on validation - the popularity column and LightGBM's tree count are
both chosen there. The test fold is scored once, at the end.
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any

import numpy as np

from news_recsys.config import get_settings, seed_everything
from news_recsys.eval.runner import evaluate_predictions
from news_recsys.features.build import load_fold_features, subsample_negatives
from news_recsys.features.vocab import load_vocabulary
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger, timed
from news_recsys.models.calibration import PlattCalibrator
from news_recsys.models.lgbm_ranker import LambdaRankModel
from news_recsys.models.popularity import PopularityBaseline
from news_recsys.plots import plot_calibration

logger = get_logger("scripts.train_baselines")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--num-boost-round", type=int, default=600)
    parser.add_argument("--variant", default="", help="feature variant to use")
    parser.add_argument("--label", default="", help="suffix for the results file")
    parser.add_argument(
        "--train-negative-rate",
        type=float,
        default=1.0,
        help="fraction of shown-not-clicked TRAINING rows to keep (evaluation is unaffected)",
    )
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    seed_everything(settings.seed)

    vocabulary = load_vocabulary(settings)
    folds = {
        fold: load_fold_features(fold, settings, variant=args.variant)
        for fold in ("train", "val", "test")
    }
    # Articles the training fold ever showed: the reference set for the cold-start slice.
    train_news_index = folds["train"].news_index.astype(np.int64)
    logger.info(
        "train %s | val %s | test %s",
        folds["train"].features.shape,
        folds["val"].features.shape,
        folds["test"].features.shape,
    )

    results: dict[str, Any] = {
        "dataset": settings.dataset,
        "variant": args.variant,
        "protocol": (
            "val = last day of MIND train (model selection); test = MIND-small dev, scored once. "
            "Probabilities come from Platt scaling fitted on val."
        ),
        "models": {},
    }

    # -- baseline 1: time-aware popularity ---------------------------------
    with timed(logger, "popularity baseline") as popularity_timing:
        popularity = PopularityBaseline(settings).fit(folds["val"])
    scores = {fold: popularity.predict(features) for fold, features in folds.items()}
    calibrator = PlattCalibrator().fit(scores["val"], folds["val"].labels.astype(np.float64))
    results["models"]["popularity"] = {
        **popularity.to_dict(),
        "train_seconds": popularity_timing["seconds"],
        "calibrator": calibrator.to_dict(),
        "val": evaluate_predictions(
            folds["val"],
            scores["val"],
            settings,
            probabilities=calibrator.transform(scores["val"]),
            train_news_index=train_news_index,
        ),
        "test": evaluate_predictions(
            folds["test"],
            scores["test"],
            settings,
            probabilities=calibrator.transform(scores["test"]),
            train_news_index=train_news_index,
            save_per_impression=(
                settings.artifact_dir
                / "per_impression"
                / f"popularity_{args.label or 'default'}.npz"
            ),
        ),
    }
    popularity_curve = results["models"]["popularity"]["test"]["probability"]["calibration"]

    # -- baseline 2: LightGBM LambdaRank -----------------------------------
    model = LambdaRankModel(settings=settings, vocabulary=vocabulary)
    training_fold = subsample_negatives(
        folds["train"], args.train_negative_rate, seed=settings.seed
    )
    if args.train_negative_rate < 1.0:
        logger.info(
            "training LambdaRank on %d of %d rows (negatives kept at %.2f)",
            training_fold.labels.size,
            folds["train"].labels.size,
            args.train_negative_rate,
        )
    training_rows = training_fold.labels.size
    with timed(logger, "LightGBM LambdaRank") as lgbm_timing:
        model.fit(training_fold, folds["val"], num_boost_round=args.num_boost_round)
    if not args.label:
        model.save(settings.artifact_dir)

    # The training design matrix is the biggest object in the process; drop it before
    # scoring so the largest dataset variant does not need both at once.
    del training_fold
    gc.collect()

    lgbm_scores = {}
    inference_seconds = {}
    for fold, features in folds.items():
        start = time.perf_counter()
        lgbm_scores[fold] = model.predict(features)
        inference_seconds[fold] = time.perf_counter() - start

    lgbm_calibrator = PlattCalibrator().fit(
        lgbm_scores["val"], folds["val"].labels.astype(np.float64)
    )
    if not args.label:
        lgbm_calibrator.save(settings.artifact_dir / "lgbm_calibrator.json")

    results["models"]["lgbm_lambdarank"] = {
        "model": "lightgbm_lambdarank",
        "params": model.params,
        "best_iteration": model.best_iteration,
        "train_negative_rate": args.train_negative_rate,
        "train_rows_used": int(training_rows),
        "train_seconds": lgbm_timing["seconds"],
        "inference_seconds": inference_seconds,
        "rows_per_second_test": float(
            folds["test"].labels.size / max(inference_seconds["test"], 1e-9)
        ),
        "feature_importance_top20": model.importance(),
        "calibrator": lgbm_calibrator.to_dict(),
        "val": evaluate_predictions(
            folds["val"],
            lgbm_scores["val"],
            settings,
            probabilities=lgbm_calibrator.transform(lgbm_scores["val"]),
            train_news_index=train_news_index,
        ),
        "test": evaluate_predictions(
            folds["test"],
            lgbm_scores["test"],
            settings,
            probabilities=lgbm_calibrator.transform(lgbm_scores["test"]),
            train_news_index=train_news_index,
            save_per_impression=(
                settings.artifact_dir / "per_impression" / f"lgbm_{args.label or 'default'}.npz"
            ),
        ),
    }
    lgbm_curve = results["models"]["lgbm_lambdarank"]["test"]["probability"]["calibration"]

    label = f"_{args.label}" if args.label else ""
    figure_path = settings.figures_dir / f"calibration_baselines{label}_{settings.dataset}.png"
    plot_calibration(
        {"time-aware popularity": popularity_curve, "LightGBM LambdaRank": lgbm_curve},
        figure_path,
        title=f"Calibration on test (MIND-{settings.dataset})",
    )
    results["figures"] = {"calibration": str(figure_path.relative_to(settings.root_dir))}

    path = write_json(settings.metrics_dir / f"baselines{label}_{settings.dataset}.json", results)
    for name, payload in results["models"].items():
        means = payload["test"]["overall"]
        logger.info(
            "%-24s test AUC=%.4f MRR=%.4f nDCG@5=%.4f nDCG@10=%.4f logloss=%.4f",
            name,
            means["auc"],
            means["mrr"],
            means["ndcg@5"],
            means["ndcg@10"],
            payload["test"]["probability"]["log_loss"],
        )
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
