"""Figures for the README. Every figure is generated from a results JSON file.

Matplotlib only, no seaborn, no styling that depends on the machine it runs on, so the
figures regenerate identically from a clean checkout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2f6f9f", "#d1495b", "#66a182", "#edae49", "#8d6a9f", "#3d3d3d"]


def _prepare(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_calibration(curves: dict[str, dict[str, Any]], path: Path, *, title: str) -> Path:
    """Reliability diagram: predicted probability vs observed click rate."""
    _prepare(path)
    figure, axis = plt.subplots(figsize=(6.0, 5.0), dpi=150)

    limit = 0.0
    for index, (name, curve) in enumerate(curves.items()):
        predicted = [row["mean_predicted"] for row in curve["bins"]]
        observed = [row["observed_rate"] for row in curve["bins"]]
        limit = max(limit, max(predicted + observed, default=0.0))
        axis.plot(
            predicted,
            observed,
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=PALETTE[index % len(PALETTE)],
            label=f"{name} (ECE={curve['ece']:.4f})",
        )

    limit = min(1.0, limit * 1.1) or 1.0
    axis.plot(
        [0, limit], [0, limit], color="#999999", linestyle="--", linewidth=1.0, label="perfect"
    )
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("mean predicted probability")
    axis.set_ylabel("observed click rate")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_recall_latency(points: list[dict[str, Any]], path: Path, *, title: str, k: int) -> Path:
    """Recall@k against index search latency as efSearch varies."""
    _prepare(path)
    figure, axis = plt.subplots(figsize=(6.4, 4.6), dpi=150)

    latencies = [point["p95_ms"] for point in points]
    recalls = [point[f"recall@{k}"] for point in points]
    axis.plot(latencies, recalls, marker="o", color=PALETTE[0], linewidth=1.8)
    for point in points:
        axis.annotate(
            f"ef={point['ef_search']}",
            (point["p95_ms"], point[f"recall@{k}"]),
            textcoords="offset points",
            xytext=(6, -8),
            fontsize=7,
            color="#444444",
        )
    if points and "exact_recall" in points[0]:
        axis.axhline(
            points[0]["exact_recall"],
            color=PALETTE[1],
            linestyle="--",
            linewidth=1.0,
            label="exact search",
        )
        axis.legend(loc="lower right", fontsize=8)

    axis.set_xlabel("HNSW search latency, p95 (ms)")
    axis.set_ylabel(f"Recall@{k}")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_latency_vs_qps(
    series: dict[str, list[dict[str, Any]]], path: Path, *, title: str, slo_ms: float
) -> Path:
    """p50/p95/p99 end-to-end latency as offered load increases."""
    _prepare(path)
    figure, axis = plt.subplots(figsize=(6.6, 4.6), dpi=150)

    for index, (name, points) in enumerate(series.items()):
        axis.plot(
            [point["qps"] for point in points],
            [point.get("client_p99_ms", point.get("p99_ms", float("nan"))) for point in points],
            marker="o",
            linewidth=1.8,
            color=PALETTE[index % len(PALETTE)],
            label=name,
        )
    axis.axhline(slo_ms, color="#999999", linestyle="--", linewidth=1.0, label=f"SLO {slo_ms:g} ms")
    axis.set_xlabel("offered load (requests/s)")
    axis.set_ylabel("end-to-end p99 latency (ms)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_stage_latency(stages: dict[str, dict[str, float]], path: Path, *, title: str) -> Path:
    """Stacked per-stage latency (p50 / p95 / p99) for the serving path."""
    _prepare(path)
    names = list(stages)
    figure, axis = plt.subplots(figsize=(6.8, 4.2), dpi=150)

    width = 0.26
    for index, percentile in enumerate(("p50_ms", "p95_ms", "p99_ms")):
        positions = [position + (index - 1) * width for position in range(len(names))]
        axis.bar(
            positions,
            [stages[name][percentile] for name in names],
            width=width,
            color=PALETTE[index],
            label=percentile.replace("_ms", ""),
        )
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axis.set_ylabel("latency (ms)")
    axis.set_title(title)
    axis.grid(alpha=0.25, axis="y")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_diversity_tradeoff(points: list[dict[str, Any]], path: Path, *, title: str) -> Path:
    """nDCG@10 against intra-list category diversity as the MMR lambda varies."""
    _prepare(path)
    figure, axis = plt.subplots(figsize=(6.2, 4.6), dpi=150)
    axis.plot(
        [point["diversity"] for point in points],
        [point["ndcg@10"] for point in points],
        marker="o",
        color=PALETTE[2],
        linewidth=1.8,
    )
    for point in points:
        axis.annotate(
            f"$\\lambda$={point['lambda']:g}",
            (point["diversity"], point["ndcg@10"]),
            textcoords="offset points",
            xytext=(6, -9),
            fontsize=7,
            color="#444444",
        )
    axis.set_xlabel("intra-list category diversity (distinct categories / k)")
    axis.set_ylabel("nDCG@10")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
