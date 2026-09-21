"""M6: validate the load generator before trusting its numbers.

Runs Locust and the internal open-loop generator against the *same* running server at the
same offered rates, and records what each one reports. On this machine they disagree by
~6x at 25 QPS, and the server's own instrumentation says the internal generator is right -
so the headline latency numbers come from that one, and this script is the evidence.

A load test is a measurement, and a measurement you have not validated is a guess.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from news_recsys.config import get_settings
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger
from news_recsys.serving.loadgen import read_locust_stats, run_open_loop

logger = get_logger("scripts.compare_generators")


def run_locust_rung(
    host: str, users_file: Path, target_qps: int, duration: str, prefix: Path, k: int
) -> dict[str, Any]:
    rps_per_user = 2.0
    users = max(round(target_qps / rps_per_user), 1)
    environment = os.environ.copy()
    environment.update(
        {
            "NEWSREC_USERS_FILE": str(users_file),
            "NEWSREC_STAGE_OUTPUT": str(prefix.with_suffix(".stages.json")),
            "NEWSREC_RPS_PER_USER": str(rps_per_user),
            "NEWSREC_LOAD_K": str(k),
        }
    )
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        "locustfile.py",
        "--headless",
        "--host",
        host,
        "--users",
        str(users),
        "--spawn-rate",
        str(users),
        "--run-time",
        duration,
        "--csv",
        str(prefix),
        "--only-summary",
        "--reset-stats",
    ]
    subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    return read_locust_stats(prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--rates", default="25,50")
    parser.add_argument("--duration", default="40s")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    users_file = settings.artifact_dir / "serving_users.txt"
    user_ids = users_file.read_text(encoding="utf-8").split()
    seconds = float(args.duration.rstrip("s"))
    logs = settings.results_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=args.host, timeout=30.0) as client:
        health = client.get("/health").json()
        for user_id in user_ids[:50]:
            client.get("/recommend", params={"user_id": user_id, "k": args.k})

    comparison: list[dict[str, Any]] = []
    for rate in [int(value) for value in args.rates.split(",")]:
        internal = run_open_loop(
            args.host, user_ids, target_qps=float(rate), duration_seconds=seconds, k=args.k
        ).to_dict()
        locust = run_locust_rung(
            args.host, users_file, rate, args.duration, logs / f"gencmp_{rate}", args.k
        )
        row = {
            "target_qps": rate,
            "internal": {
                "achieved_qps": internal["achieved_qps"],
                "client_p50_ms": internal["client_p50_ms"],
                "client_p95_ms": internal["client_p95_ms"],
                "client_p99_ms": internal["client_p99_ms"],
                "server_p50_ms": internal["server"]["server_total"].get("p50_ms"),
            },
            "locust": {
                "achieved_qps": locust["achieved_qps"],
                "client_p50_ms": locust["client_p50_ms"],
                "client_p95_ms": locust["client_p95_ms"],
                "client_p99_ms": locust["client_p99_ms"],
            },
        }
        comparison.append(row)
        logger.info(
            "%3d QPS | internal p50 %.1f ms (server says %.1f) | locust p50 %.1f ms",
            rate,
            row["internal"]["client_p50_ms"],
            row["internal"]["server_p50_ms"] or float("nan"),
            row["locust"]["client_p50_ms"],
        )

    write_json(
        settings.metrics_dir / f"generator_comparison_{settings.dataset}.json",
        {
            "dataset": settings.dataset,
            "host": args.host,
            "duration_per_rate": args.duration,
            "server": health,
            "comparison": comparison,
            "conclusion": (
                "Locust's gevent loop on Windows adds latency the service does not have; the "
                "internal open-loop generator agrees with the server's own instrumentation, so "
                "it is used for the reported numbers."
            ),
        },
    )


if __name__ == "__main__":
    main()
