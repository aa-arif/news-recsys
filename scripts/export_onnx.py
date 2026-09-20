"""M5: export the user tower and the ranker to ONNX for serving."""

from __future__ import annotations

import argparse

import numpy as np

from news_recsys.config import get_settings
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger
from news_recsys.models.onnx_export import export_ranker, export_user_tower
from news_recsys.models.ranker import DinDcnRanker
from news_recsys.models.two_tower import TwoTowerModel

logger = get_logger("scripts.export_onnx")


def real_ranker_batch(settings: object) -> dict[str, np.ndarray] | None:
    """One real validation impression, for a meaningful export check (see export_ranker)."""
    try:
        from news_recsys.data.sequences import build_impression_histories
        from news_recsys.features.build import load_fold_features
        from news_recsys.features.text import load_embeddings
        from news_recsys.features.vocab import load_vocabulary
        from news_recsys.models.ranker_train import RankerBatcher, build_ranker_dataset

        vocabulary = load_vocabulary(settings)
        embeddings = np.asarray(load_embeddings(settings, mmap=False), dtype=np.float32)
        fold = load_fold_features("val", settings)
        histories = build_impression_histories("val", settings, vocabulary=vocabulary)
        dataset = build_ranker_dataset(fold, histories, settings)
        batch = RankerBatcher(dataset, embeddings, vocabulary).batch(np.array([0]))
        return {
            name: batch[name].numpy()
            for name in (
                "dense",
                "candidate_text",
                "candidate_category",
                "candidate_subcategory",
                "history_text",
                "history_category",
                "history_subcategory",
                "history_mask",
            )
        }
    except FileNotFoundError:
        logger.warning("no validation features on disk - verifying the export on random tensors")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()

    two_tower, _ = TwoTowerModel.load(settings.artifact_dir)
    user_report = export_user_tower(
        two_tower, settings.artifact_dir, settings, history_length=settings.max_history
    )

    ranker, _ = DinDcnRanker.load(settings.artifact_dir)
    ranker_report = export_ranker(
        ranker,
        settings.artifact_dir,
        settings,
        history_length=settings.ranker_max_history,
        verification_inputs=real_ranker_batch(settings),
    )

    write_json(
        settings.metrics_dir / f"onnx_{settings.dataset}.json",
        {
            "dataset": settings.dataset,
            "opset": 17,
            "user_tower": {
                "path": str(user_report.path.relative_to(settings.root_dir)),
                "megabytes": round(user_report.path.stat().st_size / 1024 / 1024, 2),
                "max_abs_diff_vs_torch": user_report.max_absolute_difference,
                "inputs": user_report.inputs,
                "history_length": settings.max_history,
            },
            "ranker": {
                "path": str(ranker_report.path.relative_to(settings.root_dir)),
                "megabytes": round(ranker_report.path.stat().st_size / 1024 / 1024, 2),
                "max_abs_diff_vs_torch": ranker_report.max_absolute_difference,
                "ordering_identical_vs_torch": ranker_report.ordering_identical,
                "verified_on": ranker_report.verified_on,
                "inputs": ranker_report.inputs,
                "history_length": settings.ranker_max_history,
            },
        },
    )
    logger.info(
        "exported ONNX (user tower diff %.2e, ranker diff %.2e)",
        user_report.max_absolute_difference,
        ranker_report.max_absolute_difference,
    )


if __name__ == "__main__":
    main()
