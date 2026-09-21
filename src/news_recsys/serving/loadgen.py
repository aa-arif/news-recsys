"""A small open-loop load generator, used to cross-check Locust.

Why this exists: on this Windows machine, Locust reports p50 = 92 ms at 25 QPS against a
server whose own instrumentation says 12 ms and which answers a sequential client in 15 ms.
An independent generator at the same offered rate measures 15 ms. The difference is
gevent's event loop on Windows, not the service - so reporting Locust's number as the
system's latency would be reporting a property of the load generator.

Both generators are kept and both are reported (``scripts/load_test.py --generator``).
Validating the instrument before trusting it is the whole point.

Open loop: request arrivals are scheduled on a fixed timetable and handed to a worker
pool. If the server slows down, arrivals keep coming and the queue grows - which is what
real traffic does. A closed-loop generator would quietly reduce its own offered load and
make an overloaded service look healthy.
"""

from __future__ import annotations

import csv
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np


@dataclass
class LoadResult:
    target_qps: float
    achieved_qps: float
    requests: int
    failures: int
    latencies_ms: np.ndarray
    server_totals_ms: np.ndarray
    stage_samples: dict[str, list[float]] = field(default_factory=dict)

    def percentiles(self) -> dict[str, float]:
        if self.latencies_ms.size == 0:
            return {}
        p50, p95, p99 = np.percentile(self.latencies_ms, [50, 95, 99])
        return {
            "client_mean_ms": float(self.latencies_ms.mean()),
            "client_p50_ms": float(p50),
            "client_p95_ms": float(p95),
            "client_p99_ms": float(p99),
            "client_max_ms": float(self.latencies_ms.max()),
        }

    def server_stages(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for stage, values in self.stage_samples.items():
            array = np.asarray(values, dtype=np.float64)
            if array.size == 0:
                continue
            out[stage] = {
                "count": int(array.size),
                "mean_ms": float(array.mean()),
                "p50_ms": float(np.percentile(array, 50)),
                "p95_ms": float(np.percentile(array, 95)),
                "p99_ms": float(np.percentile(array, 99)),
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        totals = self.server_totals_ms
        server_total = {}
        if totals.size:
            server_total = {
                "count": int(totals.size),
                "mean_ms": float(totals.mean()),
                "p50_ms": float(np.percentile(totals, 50)),
                "p95_ms": float(np.percentile(totals, 95)),
                "p99_ms": float(np.percentile(totals, 99)),
            }
        return {
            "target_qps": self.target_qps,
            "achieved_qps": self.achieved_qps,
            "qps": self.achieved_qps,
            "requests": self.requests,
            "failures": self.failures,
            **self.percentiles(),
            "server": {"stages": self.server_stages(), "server_total": server_total},
        }


def run_open_loop(
    host: str,
    user_ids: list[str],
    *,
    target_qps: float,
    duration_seconds: float,
    k: int = 10,
    workers: int = 24,
    timeout: float = 30.0,
) -> LoadResult:
    """Fire requests on a fixed schedule and record client and server-side latency."""
    pending: queue.Queue[str] = queue.Queue()
    latencies: list[float] = []
    server_totals: list[float] = []
    stages: dict[str, list[float]] = {}
    failures = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal failures
        with httpx.Client(base_url=host, timeout=timeout) as client:
            while True:
                try:
                    user_id = pending.get(timeout=1.0)
                except queue.Empty:
                    return
                started = time.perf_counter()
                try:
                    response = client.get("/recommend", params={"user_id": user_id, "k": k})
                    elapsed = (time.perf_counter() - started) * 1000.0
                    ok = response.status_code == 200
                    payload = response.json() if ok else {}
                except Exception:  # a failed request is data, not a crash
                    elapsed = (time.perf_counter() - started) * 1000.0
                    ok, payload = False, {}
                with lock:
                    latencies.append(elapsed)
                    if not ok:
                        failures += 1
                        continue
                    server_totals.append(float(payload.get("total_ms", 0.0)))
                    for stage, value in payload.get("timings_ms", {}).items():
                        stages.setdefault(stage, []).append(float(value))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for thread in threads:
        thread.start()

    started = time.perf_counter()
    deadline = started + duration_seconds
    issued = 0
    while True:
        due = started + issued / target_qps
        if due >= deadline:
            break
        now = time.perf_counter()
        if now < due:
            time.sleep(due - now)
        pending.put(user_ids[issued % len(user_ids)])
        issued += 1

    for thread in threads:
        thread.join(timeout=timeout + 5)

    elapsed_seconds = time.perf_counter() - started
    return LoadResult(
        target_qps=target_qps,
        achieved_qps=len(latencies) / max(elapsed_seconds, 1e-9),
        requests=len(latencies),
        failures=failures,
        latencies_ms=np.asarray(latencies, dtype=np.float64),
        server_totals_ms=np.asarray(server_totals, dtype=np.float64),
        stage_samples=stages,
    )


def read_locust_stats(prefix: Path) -> dict[str, Any]:
    """Parse the aggregated row of Locust's ``*_stats.csv`` into our point schema."""
    path = prefix.with_name(prefix.name + "_stats.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    aggregated = next((row for row in rows if row.get("Name") == "Aggregated"), rows[-1])

    def number(*names: str) -> float:
        for name in names:
            value = aggregated.get(name)
            if value not in (None, "", "N/A"):
                try:
                    return float(value)
                except ValueError:
                    continue
        return float("nan")

    return {
        "requests": int(number("Request Count")),
        "failures": int(number("Failure Count")),
        "achieved_qps": number("Requests/s"),
        "client_p50_ms": number("50%", "50%ile"),
        "client_p95_ms": number("95%", "95%ile"),
        "client_p99_ms": number("99%", "99%ile"),
        "client_mean_ms": number("Average Response Time"),
        "client_max_ms": number("Max Response Time"),
    }


def sequential_anchor(
    host: str, user_ids: list[str], *, requests: int = 100, k: int = 10, warmup: int = 30
) -> dict[str, float]:
    """Single-request latency with no concurrency: a calibration anchor for the run.

    This laptop is ~3x slower after hours of sustained load than it is cold, so an
    absolute QPS number without a same-run anchor is not reproducible. Recording the
    anchor before and after each ladder makes the machine's state part of the result
    instead of an unstated assumption.
    """
    latencies: list[float] = []
    server: list[float] = []
    with httpx.Client(base_url=host, timeout=30.0) as client:
        for index in range(warmup):
            client.get("/recommend", params={"user_id": user_ids[index % len(user_ids)], "k": k})
        for index in range(requests):
            user_id = user_ids[(warmup + index) % len(user_ids)]
            started = time.perf_counter()
            response = client.get("/recommend", params={"user_id": user_id, "k": k})
            latencies.append((time.perf_counter() - started) * 1000.0)
            if response.status_code == 200:
                server.append(float(response.json().get("total_ms", 0.0)))

    array = np.asarray(latencies, dtype=np.float64)
    server_array = np.asarray(server, dtype=np.float64)
    return {
        "requests": float(array.size),
        "client_p50_ms": float(np.percentile(array, 50)),
        "client_p99_ms": float(np.percentile(array, 99)),
        "server_p50_ms": float(np.percentile(server_array, 50))
        if server_array.size
        else float("nan"),
    }
