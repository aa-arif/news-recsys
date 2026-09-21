"""M2: replay the event log in time order and materialise leak-free features."""

from __future__ import annotations

import argparse

from news_recsys.config import get_settings, seed_everything
from news_recsys.features.build import build_features, features_dir
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger, timed

logger = get_logger("scripts.build_features")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="withhold an event from the counters until it is this old",
    )
    parser.add_argument(
        "--daily-batch",
        action="store_true",
        help="counters refresh only at midnight (what this repo's server actually gets)",
    )
    parser.add_argument(
        "--variant", default="", help="name this feature set so regimes can coexist"
    )
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    seed_everything(settings.seed)

    with timed(logger, "build features") as timing:
        folds = build_features(
            settings,
            delay_seconds=args.delay_seconds,
            daily_batch=args.daily_batch,
            variant=args.variant,
        )

    directory = features_dir(settings, args.variant)
    summary = {
        "dataset": settings.dataset,
        "seconds": timing["seconds"],
        "variant": args.variant,
        "delay_seconds": args.delay_seconds,
        "daily_batch": args.daily_batch,
        "folds": {},
    }
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
    label = f"_{args.variant}" if args.variant else ""
    write_json(settings.metrics_dir / f"features{label}_{settings.dataset}.json", summary)


if __name__ == "__main__":
    main()
