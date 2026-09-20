"""Write a tiny synthetic MIND-format dataset (used by the CI smoke test)."""

from __future__ import annotations

import argparse

from news_recsys.config import get_settings
from news_recsys.data.synthetic import write_synthetic_mind
from news_recsys.logging_utils import get_logger

logger = get_logger("scripts.make_synthetic")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=120)
    parser.add_argument("--news", type=int, default=300)
    parser.add_argument("--impressions-per-day", type=int, default=200)
    args = parser.parse_args()

    settings = get_settings(dataset="synthetic")
    settings.ensure_dirs()
    raw_dir = write_synthetic_mind(
        settings.raw_dir,
        n_users=args.users,
        n_news=args.news,
        impressions_per_day=args.impressions_per_day,
        seed=settings.seed,
    )
    logger.info("synthetic dataset written to %s", raw_dir)


if __name__ == "__main__":
    main()
