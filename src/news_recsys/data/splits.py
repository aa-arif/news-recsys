"""Time-based folds and the loaders every downstream stage uses.

The split is chronological, never random:

* **test**  — the whole MIND ``dev`` split (a later day than anything in ``train``).
* **val**   — the last calendar day present in MIND ``train``.
* **train** — everything in MIND ``train`` before that day.

That ordering (train < val < test in time) is what makes the offline numbers mean
anything: the model is always asked about the future, exactly as it would be online.
Folds are stored as a tiny ``impression_key -> fold`` table so the event tables are
never duplicated and can never drift apart.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from news_recsys.config import Settings, get_settings
from news_recsys.logging_utils import get_logger

logger = get_logger("data.splits")

FOLDS = ("train", "val", "test")
FOLD_ORDER = {fold: index for index, fold in enumerate(FOLDS)}


def _processed(settings: Settings) -> Path:
    return settings.processed_dir


def assign_folds(settings: Settings | None = None) -> pl.DataFrame:
    """Compute the chronological folds and persist ``folds.parquet``."""
    settings = settings or get_settings()
    processed = _processed(settings)

    train_impressions = pl.read_parquet(processed / "impressions_train.parquet").select(
        "impression_key", "time"
    )
    dev_impressions = pl.read_parquet(processed / "impressions_dev.parquet").select(
        "impression_key", "time"
    )

    last_train_day: date = train_impressions.select(pl.col("time").dt.date().max()).item()
    logger.info("validation day (last day of MIND train): %s", last_train_day)

    train_folds = train_impressions.with_columns(
        pl.when(pl.col("time").dt.date() == pl.lit(last_train_day))
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
        .alias("fold")
    )
    dev_folds = dev_impressions.with_columns(pl.lit("test").alias("fold"))

    folds = pl.concat([train_folds, dev_folds], how="vertical").sort("time")
    folds.write_parquet(processed / "folds.parquet", compression="zstd")

    manifest = {
        "validation_day": str(last_train_day),
        "folds": {
            fold: {
                "impressions": int(part.height),
                "time_min": str(part.select(pl.col("time").min()).item()),
                "time_max": str(part.select(pl.col("time").max()).item()),
            }
            for fold, part in ((fold, folds.filter(pl.col("fold") == fold)) for fold in FOLDS)
        },
        "policy": (
            "test = MIND-small dev split (sealed); val = last calendar day of MIND-small "
            "train; train = everything earlier. No random splits."
        ),
    }
    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    (settings.metrics_dir / "splits.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for fold in FOLDS:
        info = manifest["folds"][fold]
        logger.info(
            "%-5s %7d impressions  %s .. %s",
            fold,
            info["impressions"],
            info["time_min"],
            info["time_max"],
        )
    return folds


def load_folds(settings: Settings | None = None) -> pl.DataFrame:
    settings = settings or get_settings()
    path = _processed(settings) / "folds.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/make_splits.py first")
    return pl.read_parquet(path)


def _raw_split_for(fold: str) -> str:
    return "dev" if fold == "test" else "train"


def load_events(
    fold: str, settings: Settings | None = None, *, columns: list[str] | None = None
) -> pl.DataFrame:
    """Load the exploded (impression, news, label) rows for one fold, time-ordered."""
    settings = settings or get_settings()
    if fold not in FOLDS:
        raise ValueError(f"unknown fold {fold!r}; expected one of {FOLDS}")
    keys = load_folds(settings).filter(pl.col("fold") == fold).select("impression_key")
    events = pl.read_parquet(_processed(settings) / f"events_{_raw_split_for(fold)}.parquet")
    events = events.join(keys, on="impression_key", how="semi").sort(
        ["time", "impression_key", "position"]
    )
    return events.select(columns) if columns else events


def load_impressions(fold: str, settings: Settings | None = None) -> pl.DataFrame:
    """Load impression-level rows (including click history) for one fold, time-ordered."""
    settings = settings or get_settings()
    if fold not in FOLDS:
        raise ValueError(f"unknown fold {fold!r}; expected one of {FOLDS}")
    keys = load_folds(settings).filter(pl.col("fold") == fold).select("impression_key")
    impressions = pl.read_parquet(
        _processed(settings) / f"impressions_{_raw_split_for(fold)}.parquet"
    )
    return impressions.join(keys, on="impression_key", how="semi").sort(["time", "impression_key"])


def load_all_events(settings: Settings | None = None) -> pl.DataFrame:
    """Every labelled event across folds, time-ordered, with its fold attached.

    This is the stream the time-aware feature builder replays: features for an event in
    ``val`` may use ``train`` events that happened earlier, which is exactly what a model
    serving on that day would have known.
    """
    settings = settings or get_settings()
    folds = load_folds(settings)
    frames = [
        pl.read_parquet(_processed(settings) / f"events_{split}.parquet")
        for split in ("train", "dev")
    ]
    events = pl.concat(frames, how="vertical")
    return events.join(
        folds.select("impression_key", "fold"), on="impression_key", how="inner"
    ).sort(["time", "impression_key", "position"])


def load_news(settings: Settings | None = None) -> pl.DataFrame:
    settings = settings or get_settings()
    return pl.read_parquet(_processed(settings) / "news.parquet")
