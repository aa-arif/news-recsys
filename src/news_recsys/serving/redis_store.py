"""Redis as the online feature store.

Layout (one key per entity, values packed as little-endian float64 so a round trip is
bit-exact - `repr()` round trips too, but packing is half the bytes and avoids any
locale/precision questions):

===================  =========================================================
``art:{index}``      impressions, clicks, decay_impr[H], decay_clicks[H],
                     decay_ts, first_seen, last_seen
``cat:{index}``      impressions, clicks
``sub:{index}``      impressions, clicks
``usr:{index}``      impressions, clicks, last_seen, category_clicks[C]
``hist:{user_id}``   list of article ids, oldest first (RPUSH appends the newest)
===================  =========================================================

One request touches at most four key groups, and every group is fetched with a single
``MGET``/``LRANGE`` inside one pipeline, so the whole feature fetch is one round trip.
That is the difference between ~1 ms and ~50 ms of Redis time per request when the
candidate set is 200 articles.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from news_recsys.config import Settings
from news_recsys.features.time_aware import CandidateBlock, UserBlock, decay_factors
from news_recsys.features.vocab import UNKNOWN, Vocabulary

ARTICLE_PREFIX = "art:"
CATEGORY_PREFIX = "cat:"
SUBCATEGORY_PREFIX = "sub:"
USER_PREFIX = "usr:"
HISTORY_PREFIX = "hist:"


def article_format(n_half_lives: int) -> str:
    return f"<{2 + 2 * n_half_lives + 3}d"


def user_format(n_categories: int) -> str:
    return f"<{3 + n_categories}d"


PAIR_FORMAT = "<2d"


@dataclass
class ArticleStatic:
    """Article attributes that never change; kept in process memory, not Redis."""

    category: NDArray[np.int64]
    subcategory: NDArray[np.int64]
    title_chars: NDArray[np.float64]
    abstract_chars: NDArray[np.float64]
    title_entities: NDArray[np.float64]
    abstract_entities: NDArray[np.float64]


class RedisFeatureStore:
    """Reads the online counters and assembles the blocks ``compute_features`` needs."""

    def __init__(
        self,
        client: Any,
        vocabulary: Vocabulary,
        embeddings: NDArray[np.float32],
        static: ArticleStatic,
        settings: Settings,
    ) -> None:
        self.client = client
        self.vocabulary = vocabulary
        self.embeddings = embeddings
        self.static = static
        self.settings = settings
        self.half_lives = np.asarray(settings.popularity_half_lives_hours, dtype=np.float64)
        self.n_half_lives = self.half_lives.shape[0]
        self.article_struct = struct.Struct(article_format(self.n_half_lives))
        self.user_struct = struct.Struct(user_format(vocabulary.n_categories))
        self.pair_struct = struct.Struct(PAIR_FORMAT)

    # -- writes (seeding) ---------------------------------------------------
    def pack_article(self, values: NDArray[np.float64]) -> bytes:
        return self.article_struct.pack(*values.tolist())

    def pack_user(self, values: NDArray[np.float64]) -> bytes:
        return self.user_struct.pack(*values.tolist())

    def pack_pair(self, impressions: float, clicks: float) -> bytes:
        return self.pair_struct.pack(impressions, clicks)

    # -- reads --------------------------------------------------------------
    def fetch_history(self, user_id: str) -> NDArray[np.int64]:
        """The user's click history, oldest first, as dense article indices.

        Fetched on its own because the user embedding needs it *before* the candidate
        set exists - the counter fetch that follows is one round trip for everything else.
        """
        raw = self.client.lrange(f"{HISTORY_PREFIX}{user_id}", 0, -1) or []
        ids = [item.decode() if isinstance(item, bytes) else str(item) for item in raw]
        return self.vocabulary.news_indices(ids)

    def fetch_counters(
        self,
        user_id: str,
        candidate_indices: NDArray[np.int64],
        now: float,
        history_indices: NDArray[np.int64],
    ) -> tuple[CandidateBlock, UserBlock]:
        """One pipelined round trip: article, category, subcategory and user counters."""
        known = candidate_indices >= 0
        safe = np.where(known, candidate_indices, 0)
        category = np.where(known, self.static.category[safe], UNKNOWN)
        subcategory = np.where(known, self.static.subcategory[safe], UNKNOWN)

        unique_categories = np.unique(category[category >= 0])
        unique_subcategories = np.unique(subcategory[subcategory >= 0])
        user_index = self.vocabulary.user_index(user_id)

        pipeline = self.client.pipeline(transaction=False)
        pipeline.mget([f"{ARTICLE_PREFIX}{int(index)}" for index in candidate_indices])
        if unique_categories.size:
            pipeline.mget([f"{CATEGORY_PREFIX}{int(index)}" for index in unique_categories])
        if unique_subcategories.size:
            pipeline.mget([f"{SUBCATEGORY_PREFIX}{int(index)}" for index in unique_subcategories])
        pipeline.get(f"{USER_PREFIX}{user_index}" if user_index >= 0 else f"{USER_PREFIX}-1")
        responses = pipeline.execute()

        article_raw = responses[0]
        cursor = 1
        category_raw = responses[cursor] if unique_categories.size else []
        cursor += 1 if unique_categories.size else 0
        subcategory_raw = responses[cursor] if unique_subcategories.size else []
        cursor += 1 if unique_subcategories.size else 0
        user_raw = responses[cursor]

        candidates = self._candidate_block(
            candidate_indices,
            known,
            safe,
            category,
            subcategory,
            article_raw,
            dict(zip(unique_categories.tolist(), category_raw, strict=True)),
            dict(zip(unique_subcategories.tolist(), subcategory_raw, strict=True)),
            now,
        )
        user = self._user_block(user_index, user_raw, history_indices)
        return candidates, user

    def _candidate_block(
        self,
        candidate_indices: NDArray[np.int64],
        known: NDArray[np.bool_],
        safe: NDArray[np.int64],
        category: NDArray[np.int64],
        subcategory: NDArray[np.int64],
        article_raw: list[bytes | None],
        category_values: dict[int, bytes | None],
        subcategory_values: dict[int, bytes | None],
        now: float,
    ) -> CandidateBlock:
        n = candidate_indices.shape[0]
        width = self.article_struct.size // 8
        counters = np.zeros((n, width), dtype=np.float64)
        for row, payload in enumerate(article_raw):
            if payload:
                counters[row] = self.article_struct.unpack(payload)

        h = self.n_half_lives
        impressions = counters[:, 0]
        clicks = counters[:, 1]
        decay_impressions = counters[:, 2 : 2 + h]
        decay_clicks = counters[:, 2 + h : 2 + 2 * h]
        decay_ts = counters[:, 2 + 2 * h]
        first_seen = counters[:, 3 + 2 * h]
        last_seen = counters[:, 4 + 2 * h]

        factors = decay_factors(now, decay_ts, self.half_lives)

        def pair(
            values: dict[int, bytes | None], ids: NDArray[np.int64]
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            impressions_out = np.zeros(n, dtype=np.float64)
            clicks_out = np.zeros(n, dtype=np.float64)
            for row, identifier in enumerate(ids.tolist()):
                payload = values.get(int(identifier)) if identifier >= 0 else None
                if payload:
                    impressions_out[row], clicks_out[row] = self.pair_struct.unpack(payload)
            return impressions_out, clicks_out

        category_impressions, category_clicks = pair(category_values, category)
        subcategory_impressions, subcategory_clicks = pair(subcategory_values, subcategory)

        zero = np.zeros(n, dtype=np.float64)
        return CandidateBlock(
            impressions=impressions,
            clicks=clicks,
            decay_impressions=decay_impressions * factors,
            decay_clicks=decay_clicks * factors,
            first_seen=first_seen,
            last_seen=last_seen,
            is_known=known,
            category=category,
            subcategory=subcategory,
            category_impressions=category_impressions,
            category_clicks=category_clicks,
            subcategory_impressions=subcategory_impressions,
            subcategory_clicks=subcategory_clicks,
            title_chars=np.where(known, self.static.title_chars[safe], zero),
            abstract_chars=np.where(known, self.static.abstract_chars[safe], zero),
            title_entities=np.where(known, self.static.title_entities[safe], zero),
            abstract_entities=np.where(known, self.static.abstract_entities[safe], zero),
            embeddings=np.where(known[:, None], self.embeddings[safe], 0.0).astype(np.float32),
        )

    def _user_block(
        self, user_index: int, user_raw: bytes | None, history_indices: NDArray[np.int64]
    ) -> UserBlock:
        n_categories = self.vocabulary.n_categories
        if user_raw:
            values = np.asarray(self.user_struct.unpack(user_raw), dtype=np.float64)
            impressions, clicks, last_seen = values[0], values[1], values[2]
            category_clicks = values[3:]
        else:
            impressions = clicks = last_seen = 0.0
            category_clicks = np.zeros(n_categories, dtype=np.float64)

        history = history_indices[history_indices >= 0][-self.settings.max_history :]
        history_embeddings = (
            self.embeddings[history]
            if history.size
            else np.zeros((0, self.embeddings.shape[1]), np.float32)
        )
        return UserBlock(
            impressions=float(impressions),
            clicks=float(clicks),
            last_seen=float(last_seen),
            is_known=user_index >= 0,
            category_clicks=category_clicks,
            history_length=int(history.size),
            history_embeddings=history_embeddings,
        )
