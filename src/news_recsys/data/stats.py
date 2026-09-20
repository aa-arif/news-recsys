"""Dataset statistics, including the cold-start rate the later analyses lean on."""

from __future__ import annotations

from typing import Any

import polars as pl

from news_recsys.config import Settings, get_settings
from news_recsys.data.splits import FOLDS, load_events, load_impressions, load_news
from news_recsys.logging_utils import get_logger

logger = get_logger("data.stats")


def _fold_stats(fold: str, settings: Settings) -> dict[str, Any]:
    events = load_events(fold, settings)
    impressions = load_impressions(fold, settings)

    positives = int(events.select(pl.col("label").sum()).item())
    history_lengths = impressions.select("n_history")
    return {
        "impressions": int(impressions.height),
        "events": int(events.height),
        "positives": positives,
        "positive_rate": positives / max(events.height, 1),
        "users": int(impressions.select(pl.col("user_id").n_unique()).item()),
        "articles_in_slates": int(events.select(pl.col("news_id").n_unique()).item()),
        "mean_slate_size": float(events.height / max(impressions.height, 1)),
        "mean_positives_per_impression": positives / max(impressions.height, 1),
        "mean_history_len": float(history_lengths.select(pl.col("n_history").mean()).item()),
        "median_history_len": float(history_lengths.select(pl.col("n_history").median()).item()),
        "empty_history_share": float(
            history_lengths.select((pl.col("n_history") == 0).mean()).item()
        ),
        "time_min": str(impressions.select(pl.col("time").min()).item()),
        "time_max": str(impressions.select(pl.col("time").max()).item()),
    }


def _seen_article_sets(settings: Settings) -> tuple[set[str], set[str]]:
    """Articles observable from the training fold.

    Two definitions, because they answer different questions:

    * ``slate`` - articles that appeared in a training impression. This is what a model
      trained on impression logs has actually seen labelled examples for.
    * ``slate_or_history`` - the above plus every article in any training click history.
      A content model can have an embedding for these even without labels.
    """
    train_events = load_events("train", settings, columns=["news_id"])
    slate_seen = set(train_events["news_id"].unique().to_list())

    train_impressions = load_impressions("train", settings)
    history_seen = set(
        train_impressions.select(pl.col("history").explode().drop_nulls().unique())[
            "history"
        ].to_list()
    )
    return slate_seen, slate_seen | history_seen


def _cold_start_stats(settings: Settings) -> dict[str, Any]:
    slate_seen, seen_or_history = _seen_article_sets(settings)
    out: dict[str, Any] = {
        "train_articles_in_slates": len(slate_seen),
        "train_articles_in_slates_or_history": len(seen_or_history),
    }

    for fold in ("val", "test"):
        events = load_events(fold, settings, columns=["news_id", "label"])
        articles = events["news_id"].unique().to_list()
        n_articles = len(articles)

        for label, seen in (
            ("vs_train_slates", slate_seen),
            ("vs_train_slates_or_history", seen_or_history),
        ):
            unseen = [article for article in articles if article not in seen]
            unseen_series = pl.Series("unseen", unseen, dtype=pl.Utf8)
            flagged = events.select(
                pl.col("news_id").is_in(unseen_series).alias("cold"), pl.col("label")
            )
            out[f"{fold}_{label}"] = {
                "articles": n_articles,
                "cold_articles": len(unseen),
                "cold_article_share": len(unseen) / max(n_articles, 1),
                "cold_event_share": float(flagged.select(pl.col("cold").mean()).item()),
                "cold_click_share": float(
                    flagged.filter(pl.col("label") == 1).select(pl.col("cold").mean()).item()
                ),
            }
    return out


def _user_overlap(settings: Settings) -> dict[str, Any]:
    users = {
        fold: set(load_impressions(fold, settings)["user_id"].unique().to_list()) for fold in FOLDS
    }
    test_only = users["test"] - users["train"]
    val_only = users["val"] - users["train"]
    return {
        "train_users": len(users["train"]),
        "test_users": len(users["test"]),
        "test_users_unseen_in_train": len(test_only),
        "test_users_unseen_in_train_share": len(test_only) / max(len(users["test"]), 1),
        "val_users_unseen_in_train_share": len(val_only) / max(len(users["val"]), 1),
    }


def _news_stats(settings: Settings) -> dict[str, Any]:
    news = load_news(settings)
    return {
        "articles": int(news.height),
        "categories": int(news.select(pl.col("category").n_unique()).item()),
        "subcategories": int(news.select(pl.col("subcategory").n_unique()).item()),
        "empty_abstract_share": float(
            news.select((pl.col("abstract").str.len_chars() == 0).mean()).item()
        ),
        "mean_title_chars": float(news.select(pl.col("title").str.len_chars().mean()).item()),
        "mean_abstract_chars": float(news.select(pl.col("abstract").str.len_chars().mean()).item()),
    }


def compute_stats(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    logger.info("computing dataset statistics for MIND-%s", settings.dataset)
    return {
        "dataset": settings.dataset,
        "news": _news_stats(settings),
        "folds": {fold: _fold_stats(fold, settings) for fold in FOLDS},
        "cold_start": _cold_start_stats(settings),
        "users": _user_overlap(settings),
    }
