"""Replay the event log once, in time order, and materialise training features.

One pass over every impression across all three folds:

1. read features for the impression from the counter state (which only contains events
   strictly earlier than this impression),
2. write those rows to the fold's feature matrix,
3. apply the impression's outcomes to the counter state.

Because the pass is ordered and read-before-write, "no leakage" is a property of the loop
rather than a claim about individual features. The state at the moment the test fold
starts is snapshotted to disk: that snapshot is what the serving path loads into Redis,
so online features start from exactly the state offline evaluation used.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from news_recsys.config import Settings, get_settings
from news_recsys.data.splits import FOLDS, load_all_events, load_impressions, load_news
from news_recsys.features.text import load_embeddings
from news_recsys.features.time_aware import TimeAwareFeatureStore, feature_names
from news_recsys.features.vocab import Vocabulary, load_vocabulary
from news_recsys.logging_utils import get_logger, timed

logger = get_logger("features.build")

SNAPSHOT_FILENAME = "feature_store_snapshot.npz"


@dataclass
class FoldFeatures:
    """Feature matrix plus everything needed to evaluate or train on it."""

    fold: str
    features: NDArray[np.float32]
    labels: NDArray[np.int8]
    impression_key: NDArray[np.int64]
    news_index: NDArray[np.int32]
    user_index: NDArray[np.int32]
    timestamp: NDArray[np.float64]
    names: tuple[str, ...]

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"features_{self.fold}.npz"
        np.savez(
            path,
            features=self.features,
            labels=self.labels,
            impression_key=self.impression_key,
            news_index=self.news_index,
            user_index=self.user_index,
            timestamp=self.timestamp,
            names=np.asarray(self.names),
        )
        return path

    @classmethod
    def load(cls, directory: Path, fold: str) -> FoldFeatures:
        path = directory / f"features_{fold}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run scripts/build_features.py first")
        payload = np.load(path, allow_pickle=False)
        return cls(
            fold=fold,
            features=payload["features"],
            labels=payload["labels"],
            impression_key=payload["impression_key"],
            news_index=payload["news_index"],
            user_index=payload["user_index"],
            timestamp=payload["timestamp"],
            names=tuple(str(name) for name in payload["names"]),
        )


def _article_static(news: pl.DataFrame) -> dict[str, NDArray[np.float64]]:
    return {
        "title_chars": news["title"].str.len_chars().to_numpy().astype(np.float64),
        "abstract_chars": news["abstract"].str.len_chars().to_numpy().astype(np.float64),
        "title_entities": news["n_title_entities"].to_numpy().astype(np.float64),
        "abstract_entities": news["n_abstract_entities"].to_numpy().astype(np.float64),
    }


def _index_column(
    frame: pl.DataFrame, column: str, mapping: pl.DataFrame, target: str
) -> pl.DataFrame:
    """Vectorised string -> dense id join (much faster than per-row dict lookups)."""
    return frame.join(
        mapping, left_on=column, right_on=mapping.columns[0], how="left"
    ).with_columns(pl.col(target).fill_null(-1).cast(pl.Int64))


def _mapping_frame(values: list[str], key: str, target: str) -> pl.DataFrame:
    return pl.DataFrame({key: values, target: np.arange(len(values), dtype=np.int64)})


def build_features(
    settings: Settings | None = None, *, vocabulary: Vocabulary | None = None
) -> dict[str, FoldFeatures]:
    """Run the ordered replay and return one :class:`FoldFeatures` per fold."""
    settings = settings or get_settings()
    vocabulary = vocabulary or load_vocabulary(settings)
    embeddings = np.asarray(load_embeddings(settings, mmap=False), dtype=np.float32)
    news = load_news(settings)

    news_map = _mapping_frame(vocabulary.news_ids, "news_id", "news_index")
    user_map = _mapping_frame(vocabulary.user_ids, "user_id", "user_index")

    with timed(logger, "load and index events"):
        events = load_all_events(settings)
        events = _index_column(events, "news_id", news_map, "news_index")
        events = events.with_columns(
            (pl.col("time").dt.epoch("ms").cast(pl.Float64) / 1000.0).alias("ts")
        )

        impressions = pl.concat(
            [load_impressions(fold, settings) for fold in FOLDS], how="vertical"
        )
        impressions = _index_column(impressions, "user_id", user_map, "user_index")
        # History ids -> dense indices, keeping order, as a list column.
        history = (
            impressions.select("impression_key", "history")
            .explode("history")
            .join(news_map, left_on="history", right_on="news_id", how="left")
            .with_columns(pl.col("news_index").fill_null(-1).cast(pl.Int64))
            .group_by("impression_key", maintain_order=True)
            .agg(pl.col("news_index").alias("history_index"))
        )
        impressions = impressions.join(history, on="impression_key", how="left")

    # Row-aligned numpy views of the event stream, ordered by (time, impression, position).
    event_impression = events["impression_key"].to_numpy()
    event_news_index = events["news_index"].to_numpy()
    event_labels = events["label"].to_numpy().astype(np.int8)
    event_ts = events["ts"].to_numpy()
    event_fold = events["fold"].to_numpy()

    boundaries = _group_boundaries(event_impression)
    impression_lookup = {
        int(key): (int(user_index), history_index)
        for key, user_index, history_index in zip(
            impressions["impression_key"].to_list(),
            impressions["user_index"].to_list(),
            impressions["history_index"].to_list(),
            strict=True,
        )
    }

    names = feature_names(settings)
    store = TimeAwareFeatureStore(
        vocabulary, embeddings, settings, article_static=_article_static(news)
    )

    outputs: dict[str, NDArray[np.float32]] = {
        fold: np.empty((int((event_fold == fold).sum()), len(names)), dtype=np.float32)
        for fold in FOLDS
    }
    cursors: dict[str, int] = dict.fromkeys(FOLDS, 0)
    snapshot_written = False

    with timed(logger, f"replay {len(boundaries) - 1} impressions"):
        for start, end in pairwise(boundaries):
            fold = str(event_fold[start])
            if fold == "test" and not snapshot_written:
                # Freeze the state the online system would have at the start of the test
                # day; the serving path loads exactly this into Redis.
                save_snapshot(store, settings, as_of=float(event_ts[start]))
                snapshot_written = True

            key = int(event_impression[start])
            user_index, history_index = impression_lookup[key]
            history = np.asarray(history_index if history_index is not None else [], dtype=np.int64)
            article_indices = event_news_index[start:end]
            now = float(event_ts[start])

            matrix = store.features_for_impression(article_indices, user_index, history, now)
            cursor = cursors[fold]
            outputs[fold][cursor : cursor + matrix.shape[0]] = matrix
            cursors[fold] = cursor + matrix.shape[0]

            store.update(article_indices, event_labels[start:end], user_index, now)

    if not snapshot_written:  # datasets without a test fold (not expected, but be explicit)
        save_snapshot(store, settings, as_of=float(event_ts[-1]))

    results: dict[str, FoldFeatures] = {}
    for fold in FOLDS:
        mask = event_fold == fold
        results[fold] = FoldFeatures(
            fold=fold,
            features=outputs[fold],
            labels=event_labels[mask],
            impression_key=event_impression[mask],
            news_index=event_news_index[mask].astype(np.int32),
            user_index=_user_index_per_row(event_impression[mask], impression_lookup),
            timestamp=event_ts[mask],
            names=names,
        )
        logger.info("%-5s features %s", fold, results[fold].features.shape)
    return results


def _user_index_per_row(
    impression_keys: NDArray[np.int64], lookup: dict[int, tuple[int, Any]]
) -> NDArray[np.int32]:
    unique, inverse = np.unique(impression_keys, return_inverse=True)
    per_unique = np.asarray([lookup[int(key)][0] for key in unique], dtype=np.int32)
    return per_unique[inverse]


def _group_boundaries(values: NDArray[Any]) -> NDArray[np.int64]:
    if values.size == 0:
        return np.zeros(1, dtype=np.int64)
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    return np.concatenate(([0], changes, [values.size])).astype(np.int64)


def save_snapshot(store: TimeAwareFeatureStore, settings: Settings, *, as_of: float) -> Path:
    """Persist the counter state (used to seed Redis for serving)."""
    directory = settings.artifact_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SNAPSHOT_FILENAME
    np.savez_compressed(
        path,
        as_of=np.asarray([as_of], dtype=np.float64),
        article_impressions=store.article_impressions,
        article_clicks=store.article_clicks,
        article_decay_impressions=store.article_decay_impressions,
        article_decay_clicks=store.article_decay_clicks,
        article_decay_ts=store.article_decay_ts,
        article_first_seen=store.article_first_seen,
        article_last_seen=store.article_last_seen,
        category_impressions=store.category_impressions,
        category_clicks=store.category_clicks,
        subcategory_impressions=store.subcategory_impressions,
        subcategory_clicks=store.subcategory_clicks,
        user_impressions=store.user_impressions,
        user_clicks=store.user_clicks,
        user_last_seen=store.user_last_seen,
        user_category_clicks=store.user_category_clicks,
        title_chars=store.title_chars,
        abstract_chars=store.abstract_chars,
        title_entities=store.title_entities,
        abstract_entities=store.abstract_entities,
    )
    logger.info("feature-store snapshot at ts=%.0f -> %s", as_of, path)
    return path


def load_snapshot(
    settings: Settings | None = None, *, vocabulary: Vocabulary | None = None
) -> tuple[TimeAwareFeatureStore, float]:
    """Rebuild a feature store from the snapshot. Returns (store, as_of_timestamp)."""
    settings = settings or get_settings()
    vocabulary = vocabulary or load_vocabulary(settings)
    embeddings = np.asarray(load_embeddings(settings, mmap=False), dtype=np.float32)
    payload = np.load(settings.artifact_dir / SNAPSHOT_FILENAME)

    store = TimeAwareFeatureStore(vocabulary, embeddings, settings)
    for name in (
        "article_impressions",
        "article_clicks",
        "article_decay_impressions",
        "article_decay_clicks",
        "article_decay_ts",
        "article_first_seen",
        "article_last_seen",
        "category_impressions",
        "category_clicks",
        "subcategory_impressions",
        "subcategory_clicks",
        "user_impressions",
        "user_clicks",
        "user_last_seen",
        "user_category_clicks",
        "title_chars",
        "abstract_chars",
        "title_entities",
        "abstract_entities",
    ):
        setattr(store, name, payload[name])
    return store, float(payload["as_of"][0])


def load_fold_features(fold: str, settings: Settings | None = None) -> FoldFeatures:
    settings = settings or get_settings()
    return FoldFeatures.load(settings.artifact_dir / "features", fold)
