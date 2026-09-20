"""M1: dataset sizes, split boundaries and cold-start rate -> results/metrics/data_stats_*.json."""

from __future__ import annotations

import argparse

from news_recsys.config import get_settings
from news_recsys.data.stats import compute_stats
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger, timed

logger = get_logger("scripts.data_stats")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    args = parser.parse_args()
    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()

    with timed(logger, "dataset statistics"):
        stats = compute_stats(settings)
    path = write_json(settings.metrics_dir / f"data_stats_{settings.dataset}.json", stats)

    for fold, info in stats["folds"].items():
        logger.info(
            "%-5s impressions=%7d events=%8d pos=%6.3f%% users=%6d articles=%6d slate=%.1f hist=%.1f",
            fold,
            info["impressions"],
            info["events"],
            100 * info["positive_rate"],
            info["users"],
            info["articles_in_slates"],
            info["mean_slate_size"],
            info["mean_history_len"],
        )
    cold = stats["cold_start"]["test_vs_train_slates"]
    logger.info(
        "cold-start (test vs train slates): %.1f%% of articles, %.1f%% of rows, %.1f%% of clicks",
        100 * cold["cold_article_share"],
        100 * cold["cold_event_share"],
        100 * cold["cold_click_share"],
    )
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
