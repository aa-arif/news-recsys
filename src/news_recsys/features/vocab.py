"""Dense integer ids for articles, users, categories and subcategories.

Every array in the feature store is indexed by these ids, so the mapping has to be built
once and then frozen: if the article at index 17 changed between training and serving,
every learned embedding and every counter would silently point at the wrong article. The
vocabulary is therefore written to disk next to the model artifacts and loaded by both
the training pipeline and the server.

Unknown ids map to ``-1``. That is a real case in production (an article created after
the last index build) and the feature code treats it as a cold article rather than
crashing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from news_recsys.config import Settings, get_settings
from news_recsys.data.splits import FOLDS, load_impressions, load_news
from news_recsys.logging_utils import get_logger

logger = get_logger("features.vocab")

UNKNOWN = -1
VOCAB_FILENAME = "vocab.json"


@dataclass
class Vocabulary:
    """Frozen id maps plus the per-article static attributes the models need."""

    news_ids: list[str]
    user_ids: list[str]
    categories: list[str]
    subcategories: list[str]
    news_category: NDArray[np.int32]
    news_subcategory: NDArray[np.int32]

    def __post_init__(self) -> None:
        self._news_index = {news_id: index for index, news_id in enumerate(self.news_ids)}
        self._user_index = {user_id: index for index, user_id in enumerate(self.user_ids)}

    # -- lookups ------------------------------------------------------------
    @property
    def n_news(self) -> int:
        return len(self.news_ids)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_categories(self) -> int:
        return len(self.categories)

    @property
    def n_subcategories(self) -> int:
        return len(self.subcategories)

    def news_index(self, news_id: str) -> int:
        return self._news_index.get(news_id, UNKNOWN)

    def user_index(self, user_id: str) -> int:
        return self._user_index.get(user_id, UNKNOWN)

    def news_indices(self, news_ids: list[str]) -> NDArray[np.int64]:
        return np.fromiter(
            (self._news_index.get(news_id, UNKNOWN) for news_id in news_ids),
            dtype=np.int64,
            count=len(news_ids),
        )

    def user_indices(self, user_ids: list[str]) -> NDArray[np.int64]:
        return np.fromiter(
            (self._user_index.get(user_id, UNKNOWN) for user_id in user_ids),
            dtype=np.int64,
            count=len(user_ids),
        )

    def map_column(self, frame: pl.DataFrame, column: str, kind: str) -> NDArray[np.int64]:
        """Map a string column of a frame to dense ids (``kind`` is ``news`` or ``user``)."""
        values = frame[column].to_list()
        return self.news_indices(values) if kind == "news" else self.user_indices(values)

    # -- persistence --------------------------------------------------------
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / VOCAB_FILENAME
        path.write_text(
            json.dumps(
                {
                    "news_ids": self.news_ids,
                    "user_ids": self.user_ids,
                    "categories": self.categories,
                    "subcategories": self.subcategories,
                    "news_category": self.news_category.tolist(),
                    "news_subcategory": self.news_subcategory.tolist(),
                }
            ),
            encoding="utf-8",
        )
        logger.info(
            "vocabulary: %d articles, %d users, %d categories, %d subcategories -> %s",
            self.n_news,
            self.n_users,
            self.n_categories,
            self.n_subcategories,
            path,
        )
        return path

    @classmethod
    def load(cls, directory: Path) -> Vocabulary:
        payload = json.loads((directory / VOCAB_FILENAME).read_text(encoding="utf-8"))
        return cls(
            news_ids=payload["news_ids"],
            user_ids=payload["user_ids"],
            categories=payload["categories"],
            subcategories=payload["subcategories"],
            news_category=np.asarray(payload["news_category"], dtype=np.int32),
            news_subcategory=np.asarray(payload["news_subcategory"], dtype=np.int32),
        )


def build_vocabulary(settings: Settings | None = None) -> Vocabulary:
    """Build the vocabulary from the parsed tables (article order = news.parquet order)."""
    settings = settings or get_settings()
    news = load_news(settings)

    categories = sorted(news["category"].unique().to_list())
    subcategories = sorted(news["subcategory"].unique().to_list())
    category_index = {name: index for index, name in enumerate(categories)}
    subcategory_index = {name: index for index, name in enumerate(subcategories)}

    users: list[str] = []
    seen: set[str] = set()
    for fold in FOLDS:
        for user_id in load_impressions(fold, settings)["user_id"].to_list():
            if user_id not in seen:
                seen.add(user_id)
                users.append(user_id)

    return Vocabulary(
        news_ids=news["news_id"].to_list(),
        user_ids=users,
        categories=categories,
        subcategories=subcategories,
        news_category=np.asarray(
            [category_index[name] for name in news["category"]], dtype=np.int32
        ),
        news_subcategory=np.asarray(
            [subcategory_index[name] for name in news["subcategory"]], dtype=np.int32
        ),
    )


def load_vocabulary(settings: Settings | None = None) -> Vocabulary:
    settings = settings or get_settings()
    return Vocabulary.load(settings.artifact_dir)
