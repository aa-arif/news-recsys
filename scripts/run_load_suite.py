"""M6: run the whole load-test suite - baseline and tuned - in one command.

Starts the API as a subprocess with a given configuration, waits for it to be healthy,
runs the QPS ladder against it, shuts it down, and repeats for the next configuration.
Doing it this way means the before/after comparison cannot accidentally compare two
different servers, two different warm-up states, or two different machines.
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
from news_recsys.plots import plot_latency_vs_qps

logger = get_logger("scripts.load_suite")

# Which knobs are worth turning was decided by the idle per-stage breakdown, not by
# guesswork: ranking 200 candidates costs ~15 ms of a ~28 ms request, the two Redis round
# trips ~8 ms, and the ANN search only ~0.7 ms - so efSearch is not where the time is.
CONFIGURATIONS: dict[str, dict[str, str]] = {
    # The straight path: no caching, 200 candidates, ORT with 2 intra-op threads.
    "baseline": {
        "NEWSREC_USER_EMBEDDING_CACHE_SIZE": "0",
        "NEWSREC_ORT_INTRA_OP_THREADS": "2",
        "NEWSREC_RETRIEVAL_CANDIDATES": "200",
    },
    # One change at a time, so the tuned result can be attributed.
    "cache_only": {
        "NEWSREC_USER_EMBEDDING_CACHE_SIZE": "50000",
        "NEWSREC_ORT_INTRA_OP_THREADS": "2",
        "NEWSREC_RETRIEVAL_CANDIDATES": "200",
    },
    "candidates100_only": {
        "NEWSREC_USER_EMBEDDING_CACHE_SIZE": "0",
        "NEWSREC_ORT_INTRA_OP_THREADS": "2",
        "NEWSREC_RETRIEVAL_CANDIDATES": "100",
    },
    "threads1_only": {
        "NEWSREC_USER_EMBEDDING_CACHE_SIZE": "0",
        "NEWSREC_ORT_INTRA_OP_THREADS": "1",
        "NEWSREC_RETRIEVAL_CANDIDATES": "200",
    },
    # Everything together.
    "tuned": {
        "NEWSREC_USER_EMBEDDING_CACHE_SIZE": "50000",
        "NEWSREC_ORT_INTRA_OP_THREADS": "1",
        "NEWSREC_RETRIEVAL_CANDIDATES": "100",
    },
}


def start_server(environment: dict[str, str], port: int) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env.update(environment)
    env["NEWSREC_SERVE_PORT"] = str(port)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "news_recsys.serving.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
        "--log-level",
        "warning",
    ]
    return subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def wait_for_health(
    host: str, process: subprocess.Popen[bytes], timeout: float = 180.0
) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
            raise RuntimeError(f"server exited early:\n{stderr[-3000:]}")
        try:
            with httpx.Client(base_url=host, timeout=5.0) as client:
                response = client.get("/health")
                if response.status_code == 200:
                    return response.json()
        except Exception as error:  # retry until the deadline
            last_error = error
        time.sleep(2.0)
    raise TimeoutError(f"server did not become healthy within {timeout}s: {last_error}")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover
        process.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--configs", default="baseline,tuned")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--duration", default="40s")
    parser.add_argument("--ladder", default="25,50,75,100,150,200")
    parser.add_argument("--rps-per-user", type=float, default=2.0)
    parser.add_argument("--slo-ms", type=float, default=50.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--generator", default="internal", choices=["internal", "locust"])
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    host = f"http://127.0.0.1:{args.port}"
    summary: dict[str, Any] = {
        "dataset": settings.dataset,
        "generator": args.generator,
        "configurations": {},
    }
    series: dict[str, list[dict[str, Any]]] = {}

    for name in [item.strip() for item in args.configs.split(",") if item.strip()]:
        environment = CONFIGURATIONS[name]
        logger.info("=== %s: %s ===", name, environment)
        process = start_server(environment, args.port)
        try:
            health = wait_for_health(host, process)
            logger.info(
                "server healthy: cache=%s threads=%s ef=%s",
                health.get("user_embedding_cache_size"),
                health.get("ort_intra_op_threads"),
                health.get("ef_search"),
            )
            command = [
                sys.executable,
                "scripts/load_test.py",
                "--dataset",
                settings.dataset,
                "--host",
                host,
                "--duration",
                args.duration,
                "--ladder",
                args.ladder,
                "--rps-per-user",
                str(args.rps_per_user),
                "--slo-ms",
                str(args.slo_ms),
                "--k",
                str(args.k),
                "--label",
                name,
                "--generator",
                args.generator,
            ]
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                logger.warning("load_test.py exited with %d for %s", result.returncode, name)
        finally:
            stop_server(process)
            time.sleep(3)

        path = settings.metrics_dir / f"load_test_{name}_{settings.dataset}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary["configurations"][name] = {
                "environment": environment,
                "max_qps_within_slo": payload.get("max_qps_within_slo"),
                "ladder": payload.get("ladder"),
                "server_stats": payload.get("server_stats"),
            }
            series[name] = payload.get("ladder", [])

    if series:
        figure = plot_latency_vs_qps(
            series,
            settings.figures_dir / f"latency_qps_comparison_{settings.dataset}.png",
            title="End-to-end p99 vs offered load, before and after tuning",
            slo_ms=args.slo_ms,
        )
        summary["figures"] = {"comparison": str(Path(figure).relative_to(settings.root_dir))}

    write_json(settings.metrics_dir / f"load_suite_{settings.dataset}.json", summary)
    for name, payload in summary["configurations"].items():
        logger.info("%-14s max QPS within SLO: %s", name, payload["max_qps_within_slo"])


if __name__ == "__main__":
    main()
