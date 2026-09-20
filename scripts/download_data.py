"""M1: download the MIND archives for the configured dataset variant."""

from __future__ import annotations

import argparse

from news_recsys.config import get_settings
from news_recsys.data.download import download_all
from news_recsys.logging_utils import get_logger

logger = get_logger("scripts.download")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    results = download_all(settings, force=args.force)
    for result in results:
        logger.info(
            "%-6s %s (%.1f MiB) cached=%s",
            result.split,
            result.extract_dir,
            result.bytes_downloaded / 1024 / 1024,
            result.cached,
        )


if __name__ == "__main__":
    main()
