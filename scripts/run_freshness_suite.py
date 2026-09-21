"""Quantify what counter freshness is worth, end to end.

The offline replay applies a click to the counters the instant it happens. The server in
this repo boots from a start-of-day snapshot and a static trending list, so the features it
reads are up to a day stale. Both sides call the same feature code - the skew test proves
that - which is exactly why the skew test cannot see this gap: it compares the two paths at
the *same* timestamp against the *same* store. The gap lives in the data pipeline, not the
code.

This suite rebuilds the features under several counter-latency regimes, retrains the ranker
and the LightGBM baseline on each, scores the trending retriever under the same latency,
and additionally scores the zero-delay ranker on daily-batch features *without retraining* -
which is what the deployed server actually experiences.

Every step is skipped when its output already exists, so the suite is resumable; nothing is
selected on test, and each configuration is scored on the test fold once.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from news_recsys.config import Settings, get_settings
from news_recsys.io_utils import read_json, write_json
from news_recsys.logging_utils import get_logger

logger = get_logger("scripts.freshness")


@dataclass(frozen=True)
class Regime:
    """One counter-latency policy."""

    key: str
    label: str
    delay_seconds: float = 0.0
    daily_batch: bool = False
    seeds: tuple[int, ...] = (42,)

    @property
    def variant(self) -> str:
        # The zero-delay features are the repo's default set, so they keep the empty name.
        return "" if self.key == "d0" else self.key


REGIMES = (
    Regime("d0", "live counters (0 s)", seeds=(42, 43, 44)),
    Regime("d300", "5 minutes", delay_seconds=300.0),
    Regime("d3600", "1 hour", delay_seconds=3600.0),
    Regime("d21600", "6 hours", delay_seconds=21600.0),
    Regime("daily", "daily batch (midnight)", daily_batch=True, seeds=(42, 43, 44)),
)


def run(command: list[str], *, description: str) -> None:
    logger.info("-> %s", description)
    result = subprocess.run([sys.executable, *command], check=False)
    if result.returncode != 0:
        raise SystemExit(f"step failed ({result.returncode}): {description}")


def metrics_path(settings: Settings, stem: str) -> Path:
    return settings.metrics_dir / f"{stem}_{settings.dataset}.json"


def build_variants(settings: Settings, regimes: tuple[Regime, ...], force: bool) -> None:
    for regime in regimes:
        if not regime.variant:
            continue  # the default feature set already exists
        marker = metrics_path(settings, f"features_{regime.variant}")
        if marker.exists() and not force:
            logger.info("features for %s already built", regime.key)
            continue
        command = [
            "scripts/build_features.py",
            "--dataset",
            settings.dataset,
            "--variant",
            regime.variant,
        ]
        if regime.daily_batch:
            command.append("--daily-batch")
        else:
            command += ["--delay-seconds", str(regime.delay_seconds)]
        run(command, description=f"build features [{regime.label}]")


def train_baselines(settings: Settings, regimes: tuple[Regime, ...], force: bool) -> None:
    for regime in regimes:
        stem = f"baselines_{regime.key}"
        if metrics_path(settings, stem).exists() and not force:
            logger.info("baselines for %s already trained", regime.key)
            continue
        run(
            [
                "scripts/train_baselines.py",
                "--dataset",
                settings.dataset,
                "--variant",
                regime.variant,
                "--label",
                regime.key,
            ],
            description=f"LightGBM + popularity [{regime.label}]",
        )


def train_rankers(settings: Settings, regimes: tuple[Regime, ...], force: bool) -> None:
    for regime in regimes:
        for seed in regime.seeds:
            label = f"{regime.key}_s{seed}"
            if metrics_path(settings, f"ranker_{label}").exists() and not force:
                logger.info("ranker %s already trained", label)
                continue
            command = [
                "scripts/train_ranker.py",
                "--dataset",
                settings.dataset,
                "--variant",
                regime.variant,
                "--label",
                label,
                "--seed",
                str(seed),
            ]
            # The deployed situation: trained on live counters, served stale ones.
            if regime.key == "d0" and seed == regime.seeds[0]:
                command += ["--also-score-variant", "daily"]
            run(command, description=f"ranker [{regime.label}] seed {seed}")


def train_ablation(settings: Settings, force: bool) -> None:
    if metrics_path(settings, "ranker_no_counters").exists() and not force:
        logger.info("counter ablation already trained")
        return
    run(
        [
            "scripts/train_ranker.py",
            "--dataset",
            settings.dataset,
            "--label",
            "no_counters",
            "--drop-counters",
        ],
        description="ranker without any counter feature (ablation)",
    )


def score_trending(settings: Settings, regimes: tuple[Regime, ...], force: bool) -> None:
    for regime in regimes:
        stem = f"retrieval_fresh_{regime.key}"
        if metrics_path(settings, stem).exists() and not force:
            logger.info("trending retriever for %s already scored", regime.key)
            continue
        command = [
            "scripts/eval_retrieval_fresh.py",
            "--dataset",
            settings.dataset,
            "--label",
            regime.key,
        ]
        if regime.daily_batch:
            command.append("--daily-batch")
        else:
            command += ["--delay-seconds", str(regime.delay_seconds)]
        run(command, description=f"trending retriever [{regime.label}]")


# --- aggregation -----------------------------------------------------------------


def load_per_impression(settings: Settings, label: str) -> dict[str, np.ndarray] | None:
    path = settings.artifact_dir / "per_impression" / f"{label}.npz"
    if not path.exists():
        return None
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}


def align(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict both models to the impressions they both scored, in the same order."""
    shared = np.intersect1d(left["impression_ids"], right["impression_ids"])
    left_order = np.searchsorted(
        left["impression_ids"], shared, sorter=np.argsort(left["impression_ids"])
    )
    right_order = np.searchsorted(
        right["impression_ids"], shared, sorter=np.argsort(right["impression_ids"])
    )
    left_sorted = np.argsort(left["impression_ids"])
    right_sorted = np.argsort(right["impression_ids"])
    return (
        left[metric][left_sorted][left_order],
        right[metric][right_sorted][right_order],
    )


def paired_bootstrap(
    left: np.ndarray, right: np.ndarray, *, n_resamples: int = 2000, seed: int = 42
) -> dict[str, float]:
    """Bootstrap the mean *difference* over the impressions both models scored.

    Pairing matters: the two models see the same impressions, and impression difficulty
    varies enormously, so comparing two independent intervals wastes most of the power.
    """
    difference = left - right
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, difference.size, size=(n_resamples, difference.size))
    samples = difference[draws].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_difference": float(difference.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "impressions": int(difference.size),
        "excludes_zero": bool(low > 0 or high < 0),
    }


def aggregate(settings: Settings) -> dict[str, Any]:
    regimes: dict[str, Any] = {}
    for regime in REGIMES:
        baselines_path = metrics_path(settings, f"baselines_{regime.key}")
        trending_path = metrics_path(settings, f"retrieval_fresh_{regime.key}")
        entry: dict[str, Any] = {"label": regime.label, "delay_seconds": regime.delay_seconds}
        if regime.daily_batch:
            entry["delay_seconds"] = None
            entry["policy"] = "daily batch"

        seeds: dict[str, Any] = {}
        for seed in regime.seeds:
            path = metrics_path(settings, f"ranker_{regime.key}_s{seed}")
            if path.exists():
                payload = read_json(path)
                seeds[str(seed)] = payload["test"]["overall"]
                if "cross_variant" in payload:
                    entry["cross_variant"] = {
                        "variant": payload["cross_variant"]["variant"],
                        "test": payload["cross_variant"]["test"]["overall"],
                    }
        if seeds:
            entry["ranker_seeds"] = seeds
            entry["ranker_mean"] = {
                metric: float(np.mean([seed[metric] for seed in seeds.values()]))
                for metric in ("auc", "mrr", "ndcg@5", "ndcg@10")
            }
            entry["ranker_spread_auc"] = float(
                np.max([seed["auc"] for seed in seeds.values()])
                - np.min([seed["auc"] for seed in seeds.values()])
            )
        if baselines_path.exists():
            baselines = read_json(baselines_path)
            entry["lgbm"] = baselines["models"]["lgbm_lambdarank"]["test"]["overall"]
            entry["popularity"] = baselines["models"]["popularity"]["test"]["overall"]
        if trending_path.exists():
            fresh = read_json(trending_path)
            pool = fresh["pools"].get("fresh_24h", {})
            entry["trending_retriever"] = {
                "recall@50": pool.get("retrievers", {}).get("trending", {}).get("recall@50"),
                "recall@200": pool.get("retrievers", {}).get("trending", {}).get("recall@200"),
                "two_tower_recall@200": pool.get("retrievers", {})
                .get("two_tower", {})
                .get("recall@200"),
            }
        regimes[regime.key] = entry

    # Paired bootstrap: ranker (averaged over seeds) minus LightGBM, same impressions.
    comparisons: dict[str, Any] = {}
    for regime in REGIMES:
        if len(regime.seeds) < 2:
            continue
        lgbm = load_per_impression(settings, f"lgbm_{regime.key}")
        per_seed = [load_per_impression(settings, f"{regime.key}_s{seed}") for seed in regime.seeds]
        per_seed = [item for item in per_seed if item is not None]
        if lgbm is None or not per_seed:
            continue

        aligned = []
        for seed_arrays in per_seed:
            ranker_auc, lgbm_auc = align(seed_arrays, lgbm, "auc")
            aligned.append(ranker_auc)
        ranker_mean = np.mean(np.vstack(aligned), axis=0)
        _, lgbm_auc = align(per_seed[0], lgbm, "auc")
        comparisons[regime.key] = {
            "metric": "per-impression AUC",
            "seeds": list(regime.seeds),
            "ranker_minus_lgbm": paired_bootstrap(ranker_mean, lgbm_auc),
            "per_seed_mean_auc": [float(values.mean()) for values in aligned],
            "lgbm_mean_auc": float(lgbm_auc.mean()),
        }

    # The deployed gap: same weights, live-counter training, daily-batch serving.
    deployed = None
    cross = regimes.get("d0", {}).get("cross_variant")
    if cross and "ranker_seeds" in regimes.get("d0", {}):
        own = regimes["d0"]["ranker_seeds"][str(REGIMES[0].seeds[0])]
        deployed = {
            "trained_on": "live counters",
            "served_features": cross["variant"],
            "auc_on_own_features": own["auc"],
            "auc_on_served_features": cross["test"]["auc"],
            "auc_drop": own["auc"] - cross["test"]["auc"],
            "ndcg@10_on_own_features": own["ndcg@10"],
            "ndcg@10_on_served_features": cross["test"]["ndcg@10"],
        }
        served = load_per_impression(settings, f"{REGIMES[0].key}_s{REGIMES[0].seeds[0]}_on_daily")
        own_arrays = load_per_impression(settings, f"{REGIMES[0].key}_s{REGIMES[0].seeds[0]}")
        if served is not None and own_arrays is not None:
            own_auc, served_auc = align(own_arrays, served, "auc")
            deployed["paired_bootstrap_auc_drop"] = paired_bootstrap(own_auc, served_auc)

    ablation_path = metrics_path(settings, "ranker_no_counters")
    ablation = read_json(ablation_path)["test"]["overall"] if ablation_path.exists() else None

    return {
        "dataset": settings.dataset,
        "protocol": (
            "Counters are withheld from the replay until the policy makes them visible; "
            "features, ranker and LightGBM are rebuilt per regime. Selection on validation "
            "only, test scored once per configuration."
        ),
        "regimes": regimes,
        "paired_comparisons": comparisons,
        "deployed_gap": deployed,
        "counter_ablation": ablation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-run steps even if outputs exist")
    parser.add_argument("--skip-trending", action="store_true")
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()

    if not args.aggregate_only:
        build_variants(settings, REGIMES, args.force)
        train_baselines(settings, REGIMES, args.force)
        train_rankers(settings, REGIMES, args.force)
        train_ablation(settings, args.force)
        if not args.skip_trending:
            score_trending(settings, REGIMES, args.force)

    summary = aggregate(settings)
    path = write_json(metrics_path(settings, "freshness"), summary)

    for key, entry in summary["regimes"].items():
        ranker = entry.get("ranker_mean", {})
        lgbm = entry.get("lgbm", {})
        logger.info(
            "%-7s %-24s ranker AUC %s | lgbm AUC %s | trending R@200 %s",
            key,
            entry["label"],
            f"{ranker.get('auc', float('nan')):.4f}",
            f"{lgbm.get('auc', float('nan')):.4f}",
            f"{(entry.get('trending_retriever') or {}).get('recall@200', float('nan')):.4f}",
        )
    if summary["deployed_gap"]:
        gap = summary["deployed_gap"]
        logger.info(
            "deployed gap: AUC %.4f -> %.4f (%.4f) when live-counter weights read daily-batch features",
            gap["auc_on_own_features"],
            gap["auc_on_served_features"],
            -gap["auc_drop"],
        )
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
