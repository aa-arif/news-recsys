"""M2: replay the event log in time order and materialise leak-free features."""

from __future__ import annotations

import argparse

from news_recsys.config import get_settings, seed_everything
from news_recsys.features.build import build_features
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger, timed

logger = get_logger("scripts.build_features")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    seed_everything(settings.seed)

    with timed(logger, "build features") as timing:
        folds = build_features(settings)

    directory = settings.artifact_dir / "features"
    summary = {"dataset": settings.dataset, "seconds": timing["seconds"], "folds": {}}
    for fold, features in folds.items():
        path = features.save(directory)
        summary["folds"][fold] = {
            "rows": int(features.features.shape[0]),
            "features": int(features.features.shape[1]),
            "positives": int(features.labels.sum()),
            "path": str(path),
            "megabytes": round(path.stat().st_size / 1024 / 1024, 1),
        }
        logger.info("%-5s %s -> %s", fold, features.features.shape, path.name)
    summary["feature_names"] = list(folds["train"].names)
    write_json(settings.metrics_dir / f"features_{settings.dataset}.json", summary)


if __name__ == "__main__":
    main()
