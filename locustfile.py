"""Locust load profile for ``GET /recommend``.

Open-loop-ish by construction: each simulated user issues a fixed number of requests per
second (``constant_throughput``), so offered load is ``users x rate`` and does not
collapse when the server slows down - which is the failure mode of a naive closed-loop
test, where a slow server simply receives fewer requests and looks fine.

The per-stage timings the API returns with every response are accumulated here and
written to JSON when the run stops, so the load test reports where the time went under
load and not just the total.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

from locust import HttpUser, constant_throughput, events, task

USERS_FILE = Path(os.environ.get("NEWSREC_USERS_FILE", "artifacts/small/serving_users.txt"))
STAGE_OUTPUT = Path(os.environ.get("NEWSREC_STAGE_OUTPUT", "results/logs/stage_timings.json"))
K = int(os.environ.get("NEWSREC_LOAD_K", "10"))
RPS_PER_USER = float(os.environ.get("NEWSREC_RPS_PER_USER", "1"))
AS_OF = os.environ.get("NEWSREC_AS_OF", "")

_STAGES: defaultdict[str, list[float]] = defaultdict(list)
_TOTALS: list[float] = []
_CACHE_HITS = [0, 0]


def load_user_ids() -> list[str]:
    if USERS_FILE.exists():
        ids = [
            line.strip()
            for line in USERS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if ids:
            return ids
    # Falling back to made-up ids would silently measure the cold-user path only.
    raise RuntimeError(f"{USERS_FILE} is missing - run scripts/seed_redis.py first")


USER_IDS = load_user_ids()


class RecommendUser(HttpUser):
    wait_time = constant_throughput(RPS_PER_USER)

    @task
    def recommend(self) -> None:
        user_id = random.choice(USER_IDS)
        query = f"/recommend?user_id={user_id}&k={K}"
        if AS_OF:
            query += f"&as_of={AS_OF}"
        with self.client.get(query, name="/recommend", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}")
                return
            try:
                payload = response.json()
            except ValueError:
                response.failure("invalid json")
                return
            for stage, value in payload.get("timings_ms", {}).items():
                _STAGES[stage].append(float(value))
            _TOTALS.append(float(payload.get("total_ms", 0.0)))
            _CACHE_HITS[0 if payload.get("cache_hit") else 1] += 1


@events.test_stop.add_listener
def write_stage_timings(**_kwargs: object) -> None:
    """Dump per-stage percentiles measured inside the server."""
    import numpy as np

    def percentiles(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean_ms": float(array.mean()) if array.size else 0.0,
            "p50_ms": float(np.percentile(array, 50)) if array.size else 0.0,
            "p95_ms": float(np.percentile(array, 95)) if array.size else 0.0,
            "p99_ms": float(np.percentile(array, 99)) if array.size else 0.0,
        }

    payload = {
        "stages": {stage: percentiles(values) for stage, values in _STAGES.items()},
        "server_total": percentiles(_TOTALS),
        "cache_hits": _CACHE_HITS[0],
        "cache_misses": _CACHE_HITS[1],
        "rps_per_user": RPS_PER_USER,
        "k": K,
    }
    STAGE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    STAGE_OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _STAGES.clear()
    _TOTALS.clear()
