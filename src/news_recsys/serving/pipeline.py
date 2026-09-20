"""The online recommendation path, one stage at a time.

    history (Redis) -> user vector (ONNX) -> ANN top-N (FAISS)
        -> counters (Redis) -> features (shared module) -> ranker (ONNX) -> top-k

Every stage is timed individually and the timings travel back with the response, because
"the endpoint takes 40 ms" is not actionable and "the ranker takes 31 ms of it" is.

The optional user-embedding cache is the first optimisation M6 measures: the user tower
is pure - history in, vector out - so as long as the history has not changed the vector
cannot have. The cache key is therefore the history itself, not just the user id, which
keeps it correct when a click lands between two requests.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from news_recsys.features.time_aware import compute_features
from news_recsys.features.vocab import UNKNOWN
from news_recsys.models.onnx_export import run_ranker, run_user_tower
from news_recsys.serving.redis_store import RedisFeatureStore
from news_recsys.serving.state import ServingArtifacts

STAGES = (
    "history_fetch",
    "user_embedding",
    "retrieval",
    "counter_fetch",
    "feature_build",
    "ranking",
    "postprocess",
)


@dataclass
class Recommendation:
    news_id: str
    title: str
    category: str
    score: float
    probability: float
    retrieval_score: float
    is_cold: bool


@dataclass
class RecommendResult:
    user_id: str
    k: int
    as_of: float
    items: list[Recommendation]
    timings_ms: dict[str, float]
    candidates: int
    history_length: int
    cache_hit: bool = False
    total_ms: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


class UserEmbeddingCache:
    """Tiny LRU keyed by the history itself (see the module docstring)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._store: OrderedDict[tuple[int, ...], NDArray[np.float32]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.capacity > 0

    def get(self, key: tuple[int, ...]) -> NDArray[np.float32] | None:
        if not self.enabled:
            return None
        value = self._store.get(key)
        if value is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: tuple[int, ...], value: NDArray[np.float32]) -> None:
        if not self.enabled:
            return
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "capacity": self.capacity,
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }


class RecommendationPipeline:
    """Stateless per request; holds the artifacts, the Redis store and the cache."""

    def __init__(self, artifacts: ServingArtifacts, redis_client: Any) -> None:
        self.artifacts = artifacts
        self.settings = artifacts.settings
        self.store = RedisFeatureStore(
            redis_client,
            artifacts.vocabulary,
            np.asarray(artifacts.embeddings),
            artifacts.static,
            artifacts.settings,
        )
        self.cache = UserEmbeddingCache(artifacts.settings.user_embedding_cache_size)

    # -- individual stages --------------------------------------------------
    def history_tensors(
        self, history_indices: NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
        """Right-aligned history tensors for the towers (batch dimension of 1)."""
        length = self.settings.max_history
        history = history_indices[history_indices >= 0][-length:]
        padded = np.full(length, UNKNOWN, dtype=np.int64)
        mask = np.zeros(length, dtype=np.float32)
        if history.size:
            padded[length - history.size :] = history
            mask[length - history.size :] = 1.0

        known = padded >= 0
        safe = np.where(known, padded, 0)
        text = (np.asarray(self.artifacts.embeddings[safe], dtype=np.float32) * mask[:, None])[
            None, :, :
        ]
        category = np.where(known, self.artifacts.static.category[safe] + 1, 0).astype(np.int64)[
            None, :
        ]
        subcategory = np.where(known, self.artifacts.static.subcategory[safe] + 1, 0).astype(
            np.int64
        )[None, :]
        return np.ascontiguousarray(text), category, subcategory, mask[None, :]

    def user_vector(self, history_indices: NDArray[np.int64]) -> tuple[NDArray[np.float32], bool]:
        key = tuple(
            int(index)
            for index in history_indices[history_indices >= 0][-self.settings.max_history :]
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached, True
        text, category, subcategory, mask = self.history_tensors(history_indices)
        vector = run_user_tower(self.artifacts.user_tower, text, category, subcategory, mask)
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        self.cache.put(key, vector)
        return vector, False

    # -- the request --------------------------------------------------------
    def recommend(
        self,
        user_id: str,
        k: int,
        *,
        now: float | None = None,
        ef_search: int | None = None,
        n_candidates: int | None = None,
    ) -> RecommendResult:
        timings: dict[str, float] = {}
        started = time.perf_counter()

        def stage(name: str, mark: float) -> float:
            current = time.perf_counter()
            timings[name] = (current - mark) * 1000.0
            return current

        as_of = now if now is not None else self.artifacts.snapshot_as_of
        candidates_wanted = n_candidates or self.settings.retrieval_candidates

        mark = time.perf_counter()
        history_indices = self.store.fetch_history(user_id)
        mark = stage("history_fetch", mark)

        user_vector, cache_hit = self.user_vector(history_indices)
        mark = stage("user_embedding", mark)

        if ef_search is not None:
            self.artifacts.index.ef_search = ef_search
        search = self.artifacts.index.search(user_vector, candidates_wanted)
        candidate_indices = search.indices[0]
        retrieval_scores = search.scores[0]
        valid = candidate_indices >= 0
        candidate_indices = candidate_indices[valid]
        retrieval_scores = retrieval_scores[valid]
        mark = stage("retrieval", mark)

        candidates, user_block = self.store.fetch_counters(
            user_id, candidate_indices, as_of, history_indices
        )
        mark = stage("counter_fetch", mark)

        features = compute_features(candidates, user_block, as_of, self.settings)
        mark = stage("feature_build", mark)

        category_ids = np.where(candidates.category >= 0, candidates.category + 1, 0).astype(
            np.int64
        )
        subcategory_ids = np.where(
            candidates.subcategory >= 0, candidates.subcategory + 1, 0
        ).astype(np.int64)
        history_text, history_category, history_subcategory, history_mask = self.history_tensors(
            history_indices
        )
        ranker_history = self.settings.ranker_max_history
        logits = run_ranker(
            self.artifacts.ranker,
            features,
            np.ascontiguousarray(candidates.embeddings),
            category_ids,
            subcategory_ids,
            np.ascontiguousarray(history_text[:, -ranker_history:, :]),
            np.ascontiguousarray(history_category[:, -ranker_history:]),
            np.ascontiguousarray(history_subcategory[:, -ranker_history:]),
            np.ascontiguousarray(history_mask[:, -ranker_history:]),
        )
        mark = stage("ranking", mark)

        probabilities = self.artifacts.calibrator.transform(logits.astype(np.float64))
        top = np.argsort(-logits)[:k]
        cold_column = _cold_column(self.settings)
        items = [
            Recommendation(
                news_id=self.artifacts.news_ids[int(candidate_indices[row])],
                title=self.artifacts.titles[int(candidate_indices[row])],
                category=self.artifacts.categories[int(candidate_indices[row])],
                score=float(logits[row]),
                probability=float(probabilities[row]),
                retrieval_score=float(retrieval_scores[row]),
                is_cold=bool(features[row, cold_column] > 0.5),
            )
            for row in top
        ]
        stage("postprocess", mark)

        total = (time.perf_counter() - started) * 1000.0
        return RecommendResult(
            user_id=user_id,
            k=k,
            as_of=as_of,
            items=items,
            timings_ms=timings,
            candidates=int(candidate_indices.size),
            history_length=int((history_indices >= 0).sum()),
            cache_hit=cache_hit,
            total_ms=total,
        )

    def feature_matrix(
        self, user_id: str, candidate_indices: NDArray[np.int64], now: float
    ) -> NDArray[np.float32]:
        """The online feature matrix for a fixed candidate set (used by the skew test)."""
        history_indices = self.store.fetch_history(user_id)
        candidates, user_block = self.store.fetch_counters(
            user_id, candidate_indices, now, history_indices
        )
        return compute_features(candidates, user_block, now, self.settings)


def _cold_column(settings: Any) -> int:
    from news_recsys.features.time_aware import feature_names

    return feature_names(settings).index("art_is_cold")
