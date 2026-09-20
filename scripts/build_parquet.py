"""M1: convert the raw MIND TSVs into typed Parquet tables."""

from __future__ import annotations

import argparse

from news_recsys.config import get_settings
from news_recsys.data.parse import build_parquet
from news_recsys.logging_utils import get_logger, timed

logger = get_logger("scripts.build_parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    with timed(logger, f"parse {settings.dataset} -> parquet"):
        written = build_parquet(settings.raw_dir, settings.processed_dir)
    for name, path in written.items():
        logger.info("%-20s %8.1f MiB  %s", name, path.stat().st_size / 1024 / 1024, path)


if __name__ == "__main__":
    main()
