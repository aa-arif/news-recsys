"""M1: assign chronological train/val/test folds."""

from __future__ import annotations

import argparse

from news_recsys.config import get_settings
from news_recsys.data.splits import assign_folds
from news_recsys.logging_utils import get_logger, timed

logger = get_logger("scripts.make_splits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    args = parser.parse_args()
    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    with timed(logger, "assign folds"):
        assign_folds(settings)


if __name__ == "__main__":
    main()
