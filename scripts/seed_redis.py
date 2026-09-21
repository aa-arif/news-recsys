"""M5: load the offline feature-store snapshot and user histories into Redis.

The snapshot is the counter state at the moment the test day begins - exactly what the
offline pipeline used to compute the first test-fold features. Seeding Redis from it is
what makes the training/serving skew test meaningful: any difference that shows up is a
difference in *code*, not in data.

In a real deployment this is the backfill step; the steady state would be a stream
consumer applying impressions and clicks to the same keys.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import redis

from news_recsys.config import get_settings
from news_recsys.features.build import load_snapshot
from news_recsys.features.vocab import load_vocabulary
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger, timed
from news_recsys.serving.seed import seed_counters, seed_histories, seed_popularity

logger = get_logger("scripts.seed_redis")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--redis-url", default=None)
    parser.add_argument("--flush", action="store_true", help="drop existing keys first")
    parser.add_argument(
        "--history-fold",
        default="test",
        choices=["train", "val", "test"],
        help="fold whose impression histories become each user's current history",
    )
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    client = redis.from_url(args.redis_url or settings.redis_url)
    client.ping()
    if args.flush:
        client.flushdb()

    vocabulary = load_vocabulary(settings)
    store, as_of = load_snapshot(settings, vocabulary=vocabulary)

    counts: dict[str, Any] = {"as_of": as_of, "dataset": settings.dataset}
    with timed(logger, "seed counters") as counter_timing:
        counts.update(seed_counters(client, store, vocabulary, settings))
    counts["counter_seconds"] = counter_timing["seconds"]

    with timed(logger, "seed trending list"):
        counts["popular_articles"] = seed_popularity(client, store, settings, as_of)

    with timed(logger, f"seed {args.history_fold} histories") as history_timing:
        counts["histories"] = seed_histories(client, settings, fold=args.history_fold)
    counts["history_seconds"] = history_timing["seconds"]
    counts["history_fold"] = args.history_fold

    info = client.info("memory")
    counts["redis_used_memory_mb"] = round(info.get("used_memory", 0) / 1024 / 1024, 2)
    counts["redis_keys"] = client.dbsize()
    counts["seeded_at"] = time.time()

    write_json(settings.metrics_dir / f"redis_seed_{settings.dataset}.json", counts)
    logger.info(
        "seeded %d articles, %d users, %d histories (%.1f MB in redis, %d keys)",
        counts["articles"],
        counts["users"],
        counts["histories"],
        counts["redis_used_memory_mb"],
        counts["redis_keys"],
    )


if __name__ == "__main__":
    main()
