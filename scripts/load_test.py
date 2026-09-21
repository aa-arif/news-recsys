"""M6: drive the API at increasing QPS with Locust and report the latency ladder.

For each target QPS the script runs a headless Locust process, reads the client-side
percentiles from Locust's CSV and the server-side per-stage percentiles the locustfile
collects, and records both. The reported "max QPS under the SLO" is the highest rung of
the ladder whose **p99** stayed under the target.

Client-side and server-side numbers are both kept on purpose: the gap between them is
queueing plus HTTP overhead, and a run where the server says 8 ms while the client sees
80 ms is a capacity problem, not a model problem.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from news_recsys.config import get_settings
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger
from news_recsys.plots import plot_latency_vs_qps, plot_stage_latency
from news_recsys.serving.loadgen import read_locust_stats, run_open_loop, sequential_anchor

logger = get_logger("scripts.load_test")

DEFAULT_LADDER = (10, 25, 50, 100, 150, 200, 300)


def run_locust(
    *,
    host: str,
    users: int,
    rps_per_user: float,
    duration: str,
    prefix: Path,
    stage_output: Path,
    users_file: Path,
    as_of: float | None,
    k: int,
) -> dict[str, Any]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "NEWSREC_USERS_FILE": str(users_file),
            "NEWSREC_STAGE_OUTPUT": str(stage_output),
            "NEWSREC_RPS_PER_USER": str(rps_per_user),
            "NEWSREC_LOAD_K": str(k),
        }
    )
    if as_of is not None:
        environment["NEWSREC_AS_OF"] = repr(as_of)

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
        str(max(users // 2, 1)),
        "--run-time",
        duration,
        "--csv",
        str(prefix),
        "--only-summary",
        "--reset-stats",
    ]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.warning("locust exited with %d: %s", result.returncode, result.stderr[-2000:])

    stats = read_locust_stats(prefix)
    if stage_output.exists():
        stats["server"] = json.loads(stage_output.read_text(encoding="utf-8"))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--duration", default="45s")
    parser.add_argument("--ladder", default=",".join(str(value) for value in DEFAULT_LADDER))
    parser.add_argument("--rps-per-user", type=float, default=2.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--slo-ms", type=float, default=50.0)
    parser.add_argument("--label", default="baseline", help="name for this configuration")
    parser.add_argument("--warmup-requests", type=int, default=50)
    parser.add_argument(
        "--generator",
        default="internal",
        choices=["internal", "locust"],
        help="internal = the validated open-loop generator; locust = the gevent one",
    )
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    users_file = settings.artifact_dir / "serving_users.txt"
    if not users_file.exists():
        raise SystemExit(f"{users_file} missing - run scripts/seed_redis.py first")

    with httpx.Client(base_url=args.host, timeout=30.0) as client:
        health = client.get("/health").json()
        logger.info("server: %s", health)
        sample_users = users_file.read_text(encoding="utf-8").splitlines()[: args.warmup_requests]
        for user_id in sample_users:  # warm caches, page in the index, JIT the ORT graph
            client.get("/recommend", params={"user_id": user_id, "k": args.k})

    all_user_ids = users_file.read_text(encoding="utf-8").split()
    anchor_before = sequential_anchor(args.host, all_user_ids, k=args.k)
    logger.info(
        "anchor before: sequential p50 %.1f ms (server %.1f ms)",
        anchor_before["client_p50_ms"],
        anchor_before["server_p50_ms"],
    )

    ladder = [int(value) for value in args.ladder.split(",") if value.strip()]
    points: list[dict[str, Any]] = []
    logs = settings.results_dir / "logs"

    all_users = users_file.read_text(encoding="utf-8").split()
    seconds = float(args.duration.rstrip("s"))

    for target_qps in ladder:
        users = max(round(target_qps / args.rps_per_user), 1)
        prefix = logs / f"locust_{args.label}_{target_qps}"
        stage_output = logs / f"stages_{args.label}_{target_qps}.json"
        logger.info("target %d QPS (generator: %s)", target_qps, args.generator)
        if args.generator == "locust":
            stats = run_locust(
                host=args.host,
                users=users,
                rps_per_user=args.rps_per_user,
                duration=args.duration,
                prefix=prefix,
                stage_output=stage_output,
                users_file=users_file,
                as_of=None,
                k=args.k,
            )
            stats.update({"target_qps": target_qps, "users": users, "qps": stats["achieved_qps"]})
        else:
            stats = run_open_loop(
                args.host,
                all_users,
                target_qps=float(target_qps),
                duration_seconds=seconds,
                k=args.k,
            ).to_dict()
            stats["users"] = users
        points.append(stats)
        logger.info(
            "  achieved %.1f QPS | client p50 %.1f p95 %.1f p99 %.1f ms | failures %d",
            stats["achieved_qps"],
            stats["client_p50_ms"],
            stats["client_p95_ms"],
            stats["client_p99_ms"],
            stats["failures"],
        )
        time.sleep(2)  # let the server drain between rungs

    anchor_after = sequential_anchor(args.host, all_user_ids, k=args.k)
    logger.info(
        "anchor after: sequential p50 %.1f ms (server %.1f ms)",
        anchor_after["client_p50_ms"],
        anchor_after["server_p50_ms"],
    )

    within_slo = [
        point
        for point in points
        if point["failures"] == 0 and point["client_p99_ms"] <= args.slo_ms
    ]
    best = max(within_slo, key=lambda point: point["achieved_qps"]) if within_slo else None

    with httpx.Client(base_url=args.host, timeout=30.0) as client:
        server_stats = client.get("/stats").json()

    payload: dict[str, Any] = {
        "dataset": settings.dataset,
        "label": args.label,
        "host": args.host,
        "duration_per_rung": args.duration,
        "generator": args.generator,
        "slo_ms": args.slo_ms,
        "k": args.k,
        "ladder": points,
        "anchor_before": anchor_before,
        "anchor_after": anchor_after,
        "max_qps_within_slo": best["achieved_qps"] if best else None,
        "max_qps_rung": best["target_qps"] if best else None,
        "server_stats": server_stats,
        "hardware": {
            "note": "fill from scripts/system_info.py",
        },
    }

    system_info = settings.metrics_dir / "system_info.json"
    if system_info.exists():
        payload["hardware"] = json.loads(system_info.read_text(encoding="utf-8"))

    path = write_json(
        settings.metrics_dir / f"load_test_{args.label}_{settings.dataset}.json", payload
    )

    plot_latency_vs_qps(
        {args.label: points},
        settings.figures_dir / f"latency_qps_{args.label}_{settings.dataset}.png",
        title=f"End-to-end p99 vs offered load ({args.label})",
        slo_ms=args.slo_ms,
    )
    heaviest = max(points, key=lambda point: point["achieved_qps"])
    if "server" in heaviest:
        plot_stage_latency(
            heaviest["server"]["stages"],
            settings.figures_dir / f"stage_latency_{args.label}_{settings.dataset}.png",
            title=f"Per-stage latency at {heaviest['achieved_qps']:.0f} QPS ({args.label})",
        )

    logger.info("max QPS holding p99 <= %.0f ms: %s", args.slo_ms, payload["max_qps_within_slo"])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
