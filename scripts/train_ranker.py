"""M4: train and evaluate the DIN + DCN-v2 ranker on the impression logs.

Negatives are downsampled during training, so the probabilities the model emits are
biased upwards. Three probability variants are reported so the fix is visible rather than
asserted: the raw sigmoid, the closed-form prior correction, and the prior correction
followed by Platt scaling fitted on validation.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import torch

from news_recsys.config import get_settings, seed_everything
from news_recsys.data.sequences import build_impression_histories
from news_recsys.eval.metrics import probability_report
from news_recsys.eval.runner import evaluate_predictions
from news_recsys.features.build import load_fold_features
from news_recsys.features.text import load_embeddings
from news_recsys.features.vocab import load_vocabulary
from news_recsys.io_utils import read_json, write_json
from news_recsys.logging_utils import get_logger, timed
from news_recsys.models.calibration import PlattCalibrator, PriorCorrection
from news_recsys.models.ranker_train import (
    RankerBatcher,
    build_ranker_dataset,
    predict,
    train_ranker,
)
from news_recsys.plots import plot_calibration

logger = get_logger("scripts.train_ranker")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--negative-rate", type=float, default=None)
    parser.add_argument("--val-impressions", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.epochs:
        overrides["ranker_epochs"] = args.epochs
    if args.negative_rate is not None:
        overrides["ranker_negative_sample_rate"] = args.negative_rate
    settings = get_settings(**overrides)
    settings.ensure_dirs()
    seed_everything(settings.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    vocabulary = load_vocabulary(settings)
    embeddings = np.asarray(load_embeddings(settings, mmap=False), dtype=np.float32)

    with timed(logger, "assemble ranker datasets"):
        folds = {fold: load_fold_features(fold, settings) for fold in ("train", "val", "test")}
        histories = {
            fold: build_impression_histories(fold, settings, vocabulary=vocabulary)
            for fold in ("train", "val", "test")
        }
        datasets = {
            fold: build_ranker_dataset(folds[fold], histories[fold], settings)
            for fold in ("train", "val", "test")
        }
    train_news_index = folds["train"].news_index.astype(np.int64)
    logger.info(
        "train %d rows / %d impressions | val %d rows | test %d rows",
        datasets["train"].n_rows,
        datasets["train"].n_groups,
        datasets["val"].n_rows,
        datasets["test"].n_rows,
    )

    with timed(logger, "train DIN+DCNv2 ranker") as timing:
        model, selection = train_ranker(
            datasets["train"],
            datasets["val"],
            embeddings,
            vocabulary,
            settings,
            val_impressions=args.val_impressions,
        )
    model.save(settings.artifact_dir, extra={"selection": selection})

    # -- score val and test in full ----------------------------------------
    scores: dict[str, np.ndarray] = {}
    row_order: dict[str, np.ndarray] = {}
    inference_seconds: dict[str, float] = {}
    for fold in ("val", "test"):
        batcher = RankerBatcher(datasets[fold], embeddings, vocabulary)
        start = time.perf_counter()
        logits, rows = predict(
            model, batcher, impressions_per_batch=settings.ranker_impressions_per_batch
        )
        inference_seconds[fold] = time.perf_counter() - start
        order = np.argsort(rows)  # restore the fold's row order
        scores[fold] = logits[order]
        row_order[fold] = rows[order]
        assert np.array_equal(row_order[fold], np.arange(datasets[fold].n_rows))
        logger.info(
            "%s scored %d rows in %.1fs (%.0f rows/s)",
            fold,
            logits.size,
            inference_seconds[fold],
            logits.size / max(inference_seconds[fold], 1e-9),
        )

    # -- calibration: raw vs prior-corrected vs prior + Platt --------------
    correction = PriorCorrection(negative_keep_rate=settings.ranker_negative_sample_rate)
    raw = {fold: 1.0 / (1.0 + np.exp(-value)) for fold, value in scores.items()}
    corrected = {fold: correction.apply(value) for fold, value in raw.items()}
    platt = PlattCalibrator().fit(scores["val"], folds["val"].labels.astype(np.float64))
    platt.save(settings.artifact_dir / "ranker_calibrator.json")
    final = {fold: platt.transform(value) for fold, value in scores.items()}

    calibration_variants = {
        fold: {
            "raw_sigmoid": probability_report(folds[fold].labels.astype(np.float64), raw[fold]),
            "prior_corrected": probability_report(
                folds[fold].labels.astype(np.float64), corrected[fold]
            ),
            "prior_corrected_plus_platt": probability_report(
                folds[fold].labels.astype(np.float64), final[fold]
            ),
        }
        for fold in ("val", "test")
    }

    results: dict[str, Any] = {
        "dataset": settings.dataset,
        "model": "din_dcnv2",
        "train_seconds": timing["seconds"],
        "torch_threads": torch.get_num_threads(),
        "inference_seconds": inference_seconds,
        "rows_per_second_test": float(
            datasets["test"].n_rows / max(inference_seconds["test"], 1e-9)
        ),
        "selection": selection,
        "negative_sample_rate": settings.ranker_negative_sample_rate,
        "prior_correction_logit_shift": correction.logit_shift,
        "calibrator": platt.to_dict(),
        "calibration_variants": calibration_variants,
        "val": evaluate_predictions(
            folds["val"],
            scores["val"],
            settings,
            probabilities=final["val"],
            train_news_index=train_news_index,
        ),
        "test": evaluate_predictions(
            folds["test"],
            scores["test"],
            settings,
            probabilities=final["test"],
            train_news_index=train_news_index,
        ),
    }

    figure = plot_calibration(
        {
            "raw sigmoid (downsampled negatives)": calibration_variants["test"]["raw_sigmoid"][
                "calibration"
            ],
            "prior-corrected": calibration_variants["test"]["prior_corrected"]["calibration"],
            "prior-corrected + Platt": calibration_variants["test"]["prior_corrected_plus_platt"][
                "calibration"
            ],
        },
        settings.figures_dir / f"calibration_ranker_{settings.dataset}.png",
        title=f"Ranker calibration on test (MIND-{settings.dataset})",
    )
    results["figures"] = {"calibration": str(figure.relative_to(settings.root_dir))}

    # -- side-by-side with the baselines -----------------------------------
    baselines_path = settings.metrics_dir / f"baselines_{settings.dataset}.json"
    if baselines_path.exists():
        baselines = read_json(baselines_path)
        comparison = {
            name: payload["test"]["overall"] for name, payload in baselines["models"].items()
        }
        comparison["din_dcnv2"] = results["test"]["overall"]
        results["comparison_test"] = comparison

    write_json(settings.metrics_dir / f"ranker_{settings.dataset}.json", results)
    means = results["test"]["overall"]
    logger.info(
        "ranker test AUC=%.4f MRR=%.4f nDCG@5=%.4f nDCG@10=%.4f",
        means["auc"],
        means["mrr"],
        means["ndcg@5"],
        means["ndcg@10"],
    )
    for definition, slices in results["test"]["cold_start"].items():
        cold, warm = slices["cold"], slices["warm"]
        logger.info(
            "%-20s cold AUC=%.4f (n=%d) | warm AUC=%.4f (n=%d)",
            definition,
            cold.get("auc", float("nan")),
            cold.get("n_impressions", 0),
            warm.get("auc", float("nan")),
            warm.get("n_impressions", 0),
        )


if __name__ == "__main__":
    main()
