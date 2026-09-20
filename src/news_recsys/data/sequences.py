"""Click sequences: the (history, clicked article) pairs the towers train on.

History is taken from the impression log itself, so it is exactly what the serving path
will have at request time - the user's clicks *before* this impression, nothing else. The
history is right-aligned in a fixed-width matrix (the most recent click is the last
column), which is what the attention pooling and the ONNX export expect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from news_recsys.config import Settings, get_settings
from news_recsys.data.splits import load_events, load_impressions
from news_recsys.features.vocab import Vocabulary
from news_recsys.logging_utils import get_logger

logger = get_logger("data.sequences")

PAD = -1


@dataclass
class ClickSequences:
    """One row per click: the user's history, and the article they clicked."""

    history: NDArray[np.int64]  # (n, max_history), right-aligned, PAD elsewhere
    mask: NDArray[np.float32]  # (n, max_history)
    positive: NDArray[np.int64]  # (n,)
    user_index: NDArray[np.int64]  # (n,)
    impression_key: NDArray[np.int64]  # (n,)
    timestamp: NDArray[np.float64]  # (n,)

    def __len__(self) -> int:
        return int(self.positive.shape[0])


@dataclass
class ImpressionHistories:
    """One row per impression (used to score retrieval recall per impression)."""

    history: NDArray[np.int64]
    mask: NDArray[np.float32]
    impression_key: NDArray[np.int64]
    user_index: NDArray[np.int64]
    timestamp: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.impression_key.shape[0])


def _history_matrix(
    histories: list[list[int] | None], max_history: int
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    matrix = np.full((len(histories), max_history), PAD, dtype=np.int64)
    mask = np.zeros((len(histories), max_history), dtype=np.float32)
    for row, history in enumerate(histories):
        if not history:
            continue
        recent = [index for index in history if index is not None and index >= 0][-max_history:]
        if not recent:
            continue
        matrix[row, max_history - len(recent) :] = recent
        mask[row, max_history - len(recent) :] = 1.0
    return matrix, mask


def _indexed_impressions(fold: str, settings: Settings, vocabulary: Vocabulary) -> pl.DataFrame:
    impressions = load_impressions(fold, settings)
    news_map = pl.DataFrame(
        {"news_id": vocabulary.news_ids, "news_index": np.arange(vocabulary.n_news, dtype=np.int64)}
    )
    user_map = pl.DataFrame(
        {
            "user_id": vocabulary.user_ids,
            "user_index": np.arange(vocabulary.n_users, dtype=np.int64),
        }
    )
    history = (
        impressions.select("impression_key", "history")
        .explode("history")
        .join(news_map, left_on="history", right_on="news_id", how="left")
        .with_columns(pl.col("news_index").fill_null(PAD).cast(pl.Int64))
        .group_by("impression_key", maintain_order=True)
        .agg(pl.col("news_index").alias("history_index"))
    )
    return (
        impressions.join(user_map, on="user_id", how="left")
        .with_columns(pl.col("user_index").fill_null(PAD).cast(pl.Int64))
        .join(history, on="impression_key", how="left")
        .with_columns((pl.col("time").dt.epoch("ms").cast(pl.Float64) / 1000.0).alias("ts"))
    )


def build_impression_histories(
    fold: str, settings: Settings | None = None, *, vocabulary: Vocabulary
) -> ImpressionHistories:
    settings = settings or get_settings()
    frame = _indexed_impressions(fold, settings, vocabulary)
    history, mask = _history_matrix(frame["history_index"].to_list(), settings.max_history)
    return ImpressionHistories(
        history=history,
        mask=mask,
        impression_key=frame["impression_key"].to_numpy(),
        user_index=frame["user_index"].to_numpy(),
        timestamp=frame["ts"].to_numpy(),
    )


def build_click_sequences(
    fold: str, settings: Settings | None = None, *, vocabulary: Vocabulary
) -> ClickSequences:
    """Every click in the fold, paired with the history available at that moment."""
    settings = settings or get_settings()
    impressions = build_impression_histories(fold, settings, vocabulary=vocabulary)
    lookup = {int(key): row for row, key in enumerate(impressions.impression_key)}

    events = load_events(fold, settings, columns=["impression_key", "news_id", "label"])
    clicks = events.filter(pl.col("label") == 1)
    news_map = pl.DataFrame(
        {"news_id": vocabulary.news_ids, "news_index": np.arange(vocabulary.n_news, dtype=np.int64)}
    )
    clicks = clicks.join(news_map, on="news_id", how="left").with_columns(
        pl.col("news_index").fill_null(PAD).cast(pl.Int64)
    )
    clicks = clicks.filter(pl.col("news_index") >= 0)

    keys = clicks["impression_key"].to_numpy()
    rows = np.asarray([lookup[int(key)] for key in keys], dtype=np.int64)
    logger.info("%s: %d clicks over %d impressions", fold, keys.size, len(impressions))

    return ClickSequences(
        history=impressions.history[rows],
        mask=impressions.mask[rows],
        positive=clicks["news_index"].to_numpy(),
        user_index=impressions.user_index[rows],
        impression_key=keys,
        timestamp=impressions.timestamp[rows],
    )
