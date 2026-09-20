"""M5: assert the running server computes the same features as the offline pipeline.

This is the HTTP version of ``tests/test_train_serve_skew.py``: instead of calling the
feature code in-process, it asks the *server* for the feature rows of real test-fold
impressions and compares them with the offline replay, element by element. It therefore
covers the parts a unit test cannot - the Redis deployment, the packing format actually
on the wire, the request handling, the artifact copies the server loaded at boot.

Exits non-zero on any mismatch, so it can gate a deploy.
"""

from __future__ import annotations

import argparse
from typing import Any

import httpx
import numpy as np

from news_recsys.config import get_settings, seed_everything
from news_recsys.data.sequences import build_impression_histories
from news_recsys.data.splits import load_events, load_impressions
from news_recsys.features.build import load_snapshot
from news_recsys.features.time_aware import feature_names
from news_recsys.features.vocab import load_vocabulary
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger, timed

logger = get_logger("scripts.check_skew")


def sample_requests(settings: Any, vocabulary: Any, limit: int) -> list[dict[str, Any]]:
    """First test-fold impression per user - the state Redis was seeded with."""
    impressions = load_impressions("test", settings)
    events = load_events("test", settings, columns=["impression_key", "news_id"])
    histories = build_impression_histories("test", settings, vocabulary=vocabulary)
    history_by_key = {
        int(key): histories.history[row][histories.mask[row] > 0]
        for row, key in enumerate(histories.impression_key)
    }

    slates: dict[int, list[str]] = {}
    for key, news_id in zip(
        events["impression_key"].to_list(), events["news_id"].to_list(), strict=True
    ):
        slates.setdefault(int(key), []).append(news_id)

    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, user_id in zip(
        impressions["impression_key"].to_list(), impressions["user_id"].to_list(), strict=True
    ):
        if user_id in seen:
            continue
        seen.add(user_id)
        candidates = slates.get(int(key))
        if not candidates:
            continue
        requests.append(
            {
                "user_id": user_id,
                "news_ids": candidates,
                "history": history_by_key.get(int(key), np.empty(0, dtype=np.int64)),
            }
        )
        if len(requests) >= limit:
            break
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=60)
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    seed_everything(settings.seed)

    vocabulary = load_vocabulary(settings)
    store, as_of = load_snapshot(settings, vocabulary=vocabulary)
    names = feature_names(settings)

    with timed(logger, "sample requests"):
        requests = sample_requests(settings, vocabulary, args.requests)
    logger.info("comparing %d requests at as_of=%.0f", len(requests), as_of)

    mismatches: list[dict[str, Any]] = []
    rows_compared = 0
    max_difference = 0.0
    worst_feature: str | None = None

    with httpx.Client(base_url=args.host, timeout=60.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        server_as_of = float(health.json()["snapshot_as_of"])
        if abs(server_as_of - as_of) > 1e-6:
            raise SystemExit(
                f"server snapshot ({server_as_of}) differs from the local one ({as_of}); "
                "re-run scripts/seed_redis.py against this server"
            )

        for request in requests:
            news_ids = request["news_ids"][: args.max_candidates]
            response = client.get(
                "/features",
                params={
                    "user_id": request["user_id"],
                    "news_ids": ",".join(news_ids),
                    "as_of": as_of,
                },
            )
            response.raise_for_status()
            served = np.asarray(response.json()["features"], dtype=np.float32)

            offline = store.features_for_impression(
                vocabulary.news_indices(news_ids),
                vocabulary.user_index(request["user_id"]),
                request["history"],
                as_of,
            )
            rows_compared += offline.shape[0]

            if not np.array_equal(offline, served):
                difference = np.abs(offline.astype(np.float64) - served.astype(np.float64))
                column = int(np.unravel_index(int(np.argmax(difference)), difference.shape)[1])
                if float(difference.max()) > max_difference:
                    max_difference = float(difference.max())
                    worst_feature = names[column]
                mismatches.append(
                    {
                        "user_id": request["user_id"],
                        "max_abs_diff": float(difference.max()),
                        "feature": names[column],
                        "n_differing": int((difference > 0).sum()),
                    }
                )

    payload = {
        "dataset": settings.dataset,
        "host": args.host,
        "as_of": as_of,
        "requests_compared": len(requests),
        "rows_compared": rows_compared,
        "features_per_row": len(names),
        "values_compared": rows_compared * len(names),
        "mismatched_requests": len(mismatches),
        "max_abs_difference": max_difference,
        "worst_feature": worst_feature,
        "exact_match": not mismatches,
        "examples": mismatches[:5],
    }
    write_json(settings.metrics_dir / f"skew_{settings.dataset}.json", payload)

    if mismatches:
        logger.error(
            "SKEW DETECTED in %d/%d requests (max |diff| %.3e on %s)",
            len(mismatches),
            len(requests),
            max_difference,
            worst_feature,
        )
        raise SystemExit(1)

    logger.info(
        "no skew: %d requests, %d rows, %d feature values identical",
        len(requests),
        rows_compared,
        payload["values_compared"],
    )


if __name__ == "__main__":
    main()
