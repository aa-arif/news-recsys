"""Parse MIND's TSV files into typed Parquet.

Two tables come out of ``behaviors.tsv``:

* **impressions** — one row per impression (the slate shown to a user at a timestamp),
  carrying the user's click history as a list of news ids.
* **events** — one row per (impression, news) pair with its binary click label and its
  position in the slate. This is the table models train on.

``news.tsv`` becomes a single deduplicated catalogue across splits. The entity JSON
columns are dropped (we do not use the knowledge graph) but their counts are kept, since
"number of linked entities" is a cheap, leak-free article feature.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from news_recsys.logging_utils import get_logger

logger = get_logger("data.parse")

#: MIND timestamps look like ``11/11/2019 9:05:58 AM``.
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"

#: Stable integer code per split, used to make impression keys unique across splits.
SPLIT_CODES: dict[str, int] = {"train": 0, "dev": 1, "test": 2}

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]


def impression_key(split: str, impression_id: pl.Expr) -> pl.Expr:
    """Globally unique impression key: ``split_code * 1e9 + impression_id``."""
    return (pl.lit(SPLIT_CODES[split] * 1_000_000_000, dtype=pl.Int64) + impression_id.cast(pl.Int64)).alias(
        "impression_key"
    )


def parse_news(path: Path, split: str) -> pl.DataFrame:
    """Read ``news.tsv`` into a typed frame."""
    frame = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=NEWS_COLUMNS,
        quote_char=None,
        schema_overrides={name: pl.Utf8 for name in NEWS_COLUMNS},
    )
    return frame.select(
        pl.col("news_id"),
        pl.col("category").fill_null(""),
        pl.col("subcategory").fill_null(""),
        pl.col("title").fill_null(""),
        pl.col("abstract").fill_null(""),
        pl.col("url").fill_null(""),
        # ``[]`` is an empty entity list; count objects by counting '"Label"' keys.
        pl.col("title_entities").fill_null("[]").str.count_matches(r'"Label"').alias("n_title_entities"),
        pl.col("abstract_entities").fill_null("[]").str.count_matches(r'"Label"').alias("n_abstract_entities"),
        pl.lit(split).alias("source_split"),
    )


def parse_behaviors(path: Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read ``behaviors.tsv`` into (impressions, events)."""
    raw = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=BEHAVIOR_COLUMNS,
        quote_char=None,
        schema_overrides={
            "impression_id": pl.Int64,
            "user_id": pl.Utf8,
            "time": pl.Utf8,
            "history": pl.Utf8,
            "impressions": pl.Utf8,
        },
    )

    base = raw.select(
        impression_key(split, pl.col("impression_id")),
        pl.col("impression_id").cast(pl.Int32),
        pl.col("user_id"),
        pl.col("time").str.strptime(pl.Datetime("ms"), TIME_FORMAT).alias("time"),
        pl.col("history").fill_null("").str.strip_chars().alias("history_raw"),
        pl.col("impressions").fill_null("").str.strip_chars().alias("impressions_raw"),
        pl.lit(split).alias("split"),
    )

    impressions = base.select(
        "impression_key",
        "impression_id",
        "user_id",
        "time",
        "split",
        pl.when(pl.col("history_raw") == "")
        .then(pl.lit([], dtype=pl.List(pl.Utf8)))
        .otherwise(pl.col("history_raw").str.split(" "))
        .alias("history"),
    ).with_columns(pl.col("history").list.len().cast(pl.Int32).alias("n_history"))

    events = (
        base.select(
            "impression_key",
            "user_id",
            "time",
            "split",
            pl.col("impressions_raw").str.split(" ").alias("slate"),
        )
        .with_columns(pl.col("slate").list.len().cast(pl.Int16).alias("slate_size"))
        .explode("slate")
        .with_columns(
            pl.col("slate").str.split("-").list.get(0).alias("news_id"),
            pl.col("slate").str.split("-").list.get(1).cast(pl.Int8).alias("label"),
        )
        .drop("slate")
        .with_columns(pl.col("impression_key").cum_count().over("impression_key").cast(pl.Int16).alias("position"))
    )

    n_missing_label = int(events.select(pl.col("label").is_null().sum()).item())
    if n_missing_label:
        logger.warning("%s: %d slate entries had no label and were dropped", split, n_missing_label)
        events = events.drop_nulls("label")

    return impressions, events.select(
        "impression_key", "user_id", "time", "news_id", "label", "position", "slate_size", "split"
    )


def build_parquet(raw_dir: Path, processed_dir: Path, splits: tuple[str, ...] = ("train", "dev")) -> dict[str, Path]:
    """Convert every split's TSVs to Parquet and write a deduplicated news catalogue."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    news_frames: list[pl.DataFrame] = []

    for split in splits:
        split_dir = raw_dir / split
        impressions, events = parse_behaviors(split_dir / "behaviors.tsv", split)
        impressions_path = processed_dir / f"impressions_{split}.parquet"
        events_path = processed_dir / f"events_{split}.parquet"
        impressions.write_parquet(impressions_path, compression="zstd")
        events.write_parquet(events_path, compression="zstd")
        written[f"impressions_{split}"] = impressions_path
        written[f"events_{split}"] = events_path
        logger.info(
            "%s: %d impressions, %d events (%.2f%% positive)",
            split,
            impressions.height,
            events.height,
            100.0 * float(events.select(pl.col("label").mean()).item()),
        )
        news_frames.append(parse_news(split_dir / "news.tsv", split))

    news = pl.concat(news_frames, how="vertical").unique(subset=["news_id"], keep="first", maintain_order=True)
    news_path = processed_dir / "news.parquet"
    news.write_parquet(news_path, compression="zstd")
    written["news"] = news_path
    logger.info("news catalogue: %d unique articles", news.height)
    return written
