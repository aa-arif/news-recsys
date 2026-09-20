"""M1 tests: parsing fidelity and the chronological split contract."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from news_recsys.config import Settings
from news_recsys.data.parse import parse_behaviors, parse_news
from news_recsys.data.splits import FOLDS, load_events, load_folds, load_impressions, load_news


def test_parse_behaviors_matches_raw_lines(raw_dir: Path) -> None:
    impressions, events = parse_behaviors(raw_dir / "train" / "behaviors.tsv", "train")
    raw_lines = (
        (raw_dir / "train" / "behaviors.tsv").read_text(encoding="utf-8").strip().splitlines()
    )

    assert impressions.height == len(raw_lines)
    # One event row per slate entry across the whole file.
    expected_events = sum(len(line.split("\t")[4].split(" ")) for line in raw_lines)
    assert events.height == expected_events
    assert set(events["label"].unique().to_list()) <= {0, 1}
    assert events["position"].min() == 1


def test_parse_behaviors_roundtrips_one_impression(raw_dir: Path) -> None:
    path = raw_dir / "train" / "behaviors.tsv"
    impressions, events = parse_behaviors(path, "train")
    line = path.read_text(encoding="utf-8").strip().splitlines()[0]
    impression_id, user_id, stamp, history, slate = line.split("\t")

    row = impressions.filter(pl.col("impression_id") == int(impression_id)).row(0, named=True)
    assert row["user_id"] == user_id
    assert row["history"] == ([] if history == "" else history.split(" "))
    assert row["time"] == datetime.strptime(stamp, "%m/%d/%Y %I:%M:%S %p")

    parsed_slate = events.filter(pl.col("impression_key") == row["impression_key"]).sort("position")
    assert parsed_slate["news_id"].to_list() == [entry.split("-")[0] for entry in slate.split(" ")]
    assert parsed_slate["label"].to_list() == [
        int(entry.split("-")[1]) for entry in slate.split(" ")
    ]


def test_parse_news_has_no_nulls(raw_dir: Path) -> None:
    news = parse_news(raw_dir / "train" / "news.tsv", "train")
    assert news.height > 0
    assert news.null_count().sum_horizontal().item() == 0
    assert news["news_id"].n_unique() == news.height


def test_folds_are_chronological_and_disjoint(synthetic_settings: Settings) -> None:
    folds = load_folds(synthetic_settings)
    assert folds["impression_key"].n_unique() == folds.height

    def span(fold: str) -> tuple[datetime, datetime]:
        times = folds.filter(pl.col("fold") == fold)["time"]
        return cast(datetime, times.min()), cast(datetime, times.max())

    bounds = {fold: span(fold) for fold in FOLDS}
    # train strictly before val strictly before test: the whole point of the split.
    assert bounds["train"][1] < bounds["val"][0]
    assert bounds["val"][1] < bounds["test"][0]


def test_val_is_exactly_the_last_training_day(synthetic_settings: Settings) -> None:
    folds = load_folds(synthetic_settings)
    val_days = folds.filter(pl.col("fold") == "val").select(pl.col("time").dt.date().unique())
    assert val_days.height == 1


@pytest.mark.parametrize("fold", FOLDS)
def test_loaders_are_time_ordered(fold: str, synthetic_settings: Settings) -> None:
    events = load_events(fold, synthetic_settings)
    assert events["time"].is_sorted()
    impressions = load_impressions(fold, synthetic_settings)
    assert impressions["time"].is_sorted()
    assert events.height > 0


def test_every_event_article_is_in_the_news_catalogue(synthetic_settings: Settings) -> None:
    news_ids = set(load_news(synthetic_settings)["news_id"].to_list())
    for fold in FOLDS:
        events = load_events(fold, synthetic_settings, columns=["news_id"])
        assert set(events["news_id"].unique().to_list()) <= news_ids


def test_unknown_fold_raises(synthetic_settings: Settings) -> None:
    with pytest.raises(ValueError, match="unknown fold"):
        load_events("nope", synthetic_settings)
