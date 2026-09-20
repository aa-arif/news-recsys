"""Time-aware features, computed by exactly one implementation.

This module is the contract between offline training and online serving. The same
function, :func:`compute_features`, produces the feature matrix in both worlds; only the
*source* of the counters differs:

* offline, :class:`TimeAwareFeatureStore` replays the event log in timestamp order;
* online, the serving path reads the same counters out of Redis (see
  ``news_recsys.serving``) and packs them into the same :class:`CandidateBlock`.

``tests/test_train_serve_skew.py`` asserts the two paths produce bitwise-identical
float32 rows. Duplicating this logic in a "serving version" is the classic way to ship a
model that scores differently in production than it did offline, so it is deliberately
impossible here: there is only one copy.

**No-leakage rule.** Features for an impression at time ``t`` are read *before* the
impression is applied to the counters, so every counter reflects events strictly earlier
than ``t``. The replay is single-pass and ordered, which makes that rule structural
rather than something a reviewer has to check feature by feature.

**Feedback delay.** The replay applies clicks at impression time, i.e. it assumes a
zero-latency feedback loop. A real system sees clicks after a delay (and after a join
window), so these counters are marginally fresher than production's would be. The
serving path reads the same counters, so the two agree; what would change in production
is the counter pipeline, not this code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from news_recsys.config import Settings, get_settings
from news_recsys.features.vocab import UNKNOWN, Vocabulary

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SECONDS_PER_HOUR = 3600.0


def decay_factors(now: float, decay_ts: FloatArray, half_lives: FloatArray) -> FloatArray:
    """Exponential decay multipliers, shape ``(n, len(half_lives))``.

    Shared by the offline replay and the online Redis reader so the two cannot drift:
    a counter last touched at ``decay_ts`` is worth ``2 ** (-elapsed_hours / half_life)``
    of its stored value now.
    """
    elapsed_hours = (now - decay_ts) / SECONDS_PER_HOUR
    return np.exp2(-np.maximum(elapsed_hours, 0.0)[:, None] / half_lives[None, :])


def feature_names(settings: Settings) -> tuple[str, ...]:
    """Ordered feature names. The single source of truth for column order."""
    names: list[str] = ["art_impr_log1p", "art_click_log1p", "art_ctr_smooth"]
    for half_life in settings.popularity_half_lives_hours:
        tag = _half_life_tag(half_life)
        names += [f"art_impr_decay_{tag}_log1p", f"art_ctr_decay_{tag}"]
    names += [
        "art_age_hours_log1p",
        "art_hours_since_last_impr_log1p",
        "art_is_cold",
        "cat_ctr_smooth",
        "cat_impr_log1p",
        "subcat_ctr_smooth",
        "subcat_impr_log1p",
        "user_impr_log1p",
        "user_click_log1p",
        "user_ctr_smooth",
        "user_hours_since_last_impr_log1p",
        "user_history_len_log1p",
        "user_is_cold",
        "user_cat_affinity",
        "text_sim_hist_mean",
        "text_sim_hist_max",
        "text_sim_hist_last",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "title_len_log1p",
        "abstract_len_log1p",
        "n_title_entities_log1p",
        "n_abstract_entities_log1p",
    ]
    return tuple(names)


def _half_life_tag(half_life_hours: float) -> str:
    return f"{half_life_hours:g}h"


# ---------------------------------------------------------------------------
# Blocks: the data compute_features() consumes, whatever produced it.
# ---------------------------------------------------------------------------


@dataclass
class CandidateBlock:
    """Per-candidate counter state, already decayed to the request timestamp.

    Offline this is gathered from the replay's arrays; online it is built from Redis
    values. Shapes: ``(n_candidates,)`` except the decay arrays, which are
    ``(n_candidates, n_half_lives)``.
    """

    impressions: FloatArray
    clicks: FloatArray
    decay_impressions: FloatArray
    decay_clicks: FloatArray
    first_seen: FloatArray
    last_seen: FloatArray
    is_known: NDArray[np.bool_]
    category: IntArray
    subcategory: IntArray
    category_impressions: FloatArray
    category_clicks: FloatArray
    subcategory_impressions: FloatArray
    subcategory_clicks: FloatArray
    title_chars: FloatArray
    abstract_chars: FloatArray
    title_entities: FloatArray
    abstract_entities: FloatArray
    embeddings: NDArray[np.float32]


@dataclass
class UserBlock:
    """Per-request user state."""

    impressions: float
    clicks: float
    last_seen: float
    is_known: bool
    category_clicks: FloatArray  # shape (n_categories,)
    history_length: int
    history_embeddings: NDArray[np.float32]  # shape (h, dim); h may be 0


def compute_features(
    candidates: CandidateBlock,
    user: UserBlock,
    now: float,
    settings: Settings,
) -> NDArray[np.float32]:
    """Build the ``(n_candidates, n_features)`` matrix for one request.

    ``now`` is POSIX seconds. Every counter in ``candidates`` must already be decayed to
    ``now`` by whoever produced the block.
    """
    n = candidates.impressions.shape[0]
    columns: list[FloatArray] = []

    prior_clicks = settings.ctr_prior_clicks
    prior_impressions = settings.ctr_prior_impressions

    def smoothed_ctr(clicks: FloatArray, impressions: FloatArray) -> FloatArray:
        return (clicks + prior_clicks) / (impressions + prior_clicks + prior_impressions)

    # -- article counters ---------------------------------------------------
    columns.append(np.log1p(candidates.impressions))
    columns.append(np.log1p(candidates.clicks))
    columns.append(smoothed_ctr(candidates.clicks, candidates.impressions))

    for index in range(len(settings.popularity_half_lives_hours)):
        decayed_impressions = candidates.decay_impressions[:, index]
        decayed_clicks = candidates.decay_clicks[:, index]
        columns.append(np.log1p(decayed_impressions))
        columns.append(smoothed_ctr(decayed_clicks, decayed_impressions))

    known = candidates.is_known & (candidates.impressions > 0)
    age_hours = np.where(known, (now - candidates.first_seen) / SECONDS_PER_HOUR, 0.0)
    since_last = np.where(known, (now - candidates.last_seen) / SECONDS_PER_HOUR, 0.0)
    columns.append(np.log1p(np.maximum(age_hours, 0.0)))
    columns.append(np.log1p(np.maximum(since_last, 0.0)))
    columns.append((~known).astype(np.float64))

    # -- category / subcategory --------------------------------------------
    columns.append(smoothed_ctr(candidates.category_clicks, candidates.category_impressions))
    columns.append(np.log1p(candidates.category_impressions))
    columns.append(smoothed_ctr(candidates.subcategory_clicks, candidates.subcategory_impressions))
    columns.append(np.log1p(candidates.subcategory_impressions))

    # -- user ---------------------------------------------------------------
    user_known = user.is_known and user.impressions > 0
    columns.append(np.full(n, np.log1p(user.impressions)))
    columns.append(np.full(n, np.log1p(user.clicks)))
    columns.append(
        np.full(n, float(smoothed_ctr(np.array([user.clicks]), np.array([user.impressions]))[0]))
    )
    user_since_last = (now - user.last_seen) / SECONDS_PER_HOUR if user_known else 0.0
    columns.append(np.full(n, np.log1p(max(user_since_last, 0.0))))
    columns.append(np.full(n, np.log1p(float(user.history_length))))
    columns.append(np.full(n, 0.0 if user_known else 1.0))

    # -- user x content -----------------------------------------------------
    total_category_clicks = float(user.category_clicks.sum())
    n_categories = user.category_clicks.shape[0]
    safe_category = np.where(candidates.category >= 0, candidates.category, 0)
    category_clicks = np.where(candidates.category >= 0, user.category_clicks[safe_category], 0.0)
    columns.append((category_clicks + 1.0) / (total_category_clicks + n_categories))

    similarity_mean, similarity_max, similarity_last = _history_similarity(
        candidates.embeddings, user.history_embeddings
    )
    columns.append(similarity_mean)
    columns.append(similarity_max)
    columns.append(similarity_last)

    # -- request context ----------------------------------------------------
    # Derived from the request timestamp only: available offline and online alike.
    hour_of_day = (now % 86_400.0) / 3600.0
    # 1970-01-01 was a Thursday, hence the offset.
    day_of_week = ((now // 86_400.0) + 4) % 7
    columns.append(np.full(n, np.sin(2 * np.pi * hour_of_day / 24.0)))
    columns.append(np.full(n, np.cos(2 * np.pi * hour_of_day / 24.0)))
    columns.append(np.full(n, np.sin(2 * np.pi * day_of_week / 7.0)))
    columns.append(np.full(n, np.cos(2 * np.pi * day_of_week / 7.0)))
    columns.append(np.full(n, 1.0 if day_of_week >= 5 else 0.0))

    # -- static article attributes -----------------------------------------
    columns.append(np.log1p(candidates.title_chars))
    columns.append(np.log1p(candidates.abstract_chars))
    columns.append(np.log1p(candidates.title_entities))
    columns.append(np.log1p(candidates.abstract_entities))

    matrix = np.stack(columns, axis=1)
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _history_similarity(
    candidate_embeddings: NDArray[np.float32], history_embeddings: NDArray[np.float32]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Cosine similarity of each candidate to the user's history (mean / max / latest).

    Embeddings are L2-normalised on write, so a dot product is the cosine. The mean
    history vector is re-normalised; an empty history yields zeros, which is the honest
    answer for a user we have never seen.
    """
    n = candidate_embeddings.shape[0]
    if history_embeddings.shape[0] == 0:
        zeros = np.zeros(n, dtype=np.float64)
        return zeros, zeros.copy(), zeros.copy()

    candidates64 = candidate_embeddings.astype(np.float64)
    history64 = history_embeddings.astype(np.float64)

    mean_vector = history64.mean(axis=0)
    norm = np.linalg.norm(mean_vector)
    mean_vector = mean_vector / norm if norm > 0 else mean_vector

    similarities = candidates64 @ history64.T  # (n_candidates, history)
    return (
        candidates64 @ mean_vector,
        similarities.max(axis=1),
        similarities[:, -1].copy(),
    )


# ---------------------------------------------------------------------------
# Offline: the replay that produces training features and the serving snapshot.
# ---------------------------------------------------------------------------


class TimeAwareFeatureStore:
    """Counter state over the article/user/category vocabulary, advanced in time order.

    Read with :meth:`gather`, then advance with :meth:`update`. Doing it in that order is
    the no-leakage rule; :meth:`gather` never looks at the impression being scored.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        embeddings: NDArray[np.float32],
        settings: Settings | None = None,
        *,
        article_static: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vocabulary = vocabulary
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.half_lives = np.asarray(self.settings.popularity_half_lives_hours, dtype=np.float64)
        self.n_half_lives = self.half_lives.shape[0]

        n_news = vocabulary.n_news
        n_users = vocabulary.n_users
        n_categories = vocabulary.n_categories
        n_subcategories = vocabulary.n_subcategories

        self.article_impressions = np.zeros(n_news, dtype=np.float64)
        self.article_clicks = np.zeros(n_news, dtype=np.float64)
        self.article_decay_impressions = np.zeros((n_news, self.n_half_lives), dtype=np.float64)
        self.article_decay_clicks = np.zeros((n_news, self.n_half_lives), dtype=np.float64)
        self.article_decay_ts = np.zeros(n_news, dtype=np.float64)
        self.article_first_seen = np.zeros(n_news, dtype=np.float64)
        self.article_last_seen = np.zeros(n_news, dtype=np.float64)

        self.category_impressions = np.zeros(n_categories, dtype=np.float64)
        self.category_clicks = np.zeros(n_categories, dtype=np.float64)
        self.subcategory_impressions = np.zeros(n_subcategories, dtype=np.float64)
        self.subcategory_clicks = np.zeros(n_subcategories, dtype=np.float64)

        self.user_impressions = np.zeros(n_users, dtype=np.float64)
        self.user_clicks = np.zeros(n_users, dtype=np.float64)
        self.user_last_seen = np.zeros(n_users, dtype=np.float64)
        self.user_category_clicks = np.zeros((n_users, n_categories), dtype=np.float64)

        static = article_static or {}
        self.title_chars = static.get("title_chars", np.zeros(n_news, dtype=np.float64))
        self.abstract_chars = static.get("abstract_chars", np.zeros(n_news, dtype=np.float64))
        self.title_entities = static.get("title_entities", np.zeros(n_news, dtype=np.float64))
        self.abstract_entities = static.get("abstract_entities", np.zeros(n_news, dtype=np.float64))

    # -- reads --------------------------------------------------------------
    def decayed(self, indices: IntArray, now: float) -> tuple[FloatArray, FloatArray]:
        """Exponentially decayed (impressions, clicks) for ``indices`` as of ``now``."""
        factors = decay_factors(now, self.article_decay_ts[indices], self.half_lives)
        return (
            self.article_decay_impressions[indices] * factors,
            self.article_decay_clicks[indices] * factors,
        )

    def candidate_block(self, article_indices: IntArray, now: float) -> CandidateBlock:
        known = article_indices >= 0
        safe = np.where(known, article_indices, 0)
        decay_impressions, decay_clicks = self.decayed(safe, now)
        category = np.where(known, self.vocabulary.news_category[safe], UNKNOWN).astype(np.int64)
        subcategory = np.where(known, self.vocabulary.news_subcategory[safe], UNKNOWN).astype(
            np.int64
        )
        safe_category = np.where(category >= 0, category, 0)
        safe_subcategory = np.where(subcategory >= 0, subcategory, 0)
        zero = np.zeros(article_indices.shape[0], dtype=np.float64)

        return CandidateBlock(
            impressions=np.where(known, self.article_impressions[safe], 0.0),
            clicks=np.where(known, self.article_clicks[safe], 0.0),
            decay_impressions=np.where(known[:, None], decay_impressions, 0.0),
            decay_clicks=np.where(known[:, None], decay_clicks, 0.0),
            first_seen=np.where(known, self.article_first_seen[safe], 0.0),
            last_seen=np.where(known, self.article_last_seen[safe], 0.0),
            is_known=known,
            category=category,
            subcategory=subcategory,
            category_impressions=np.where(
                category >= 0, self.category_impressions[safe_category], 0.0
            ),
            category_clicks=np.where(category >= 0, self.category_clicks[safe_category], 0.0),
            subcategory_impressions=np.where(
                subcategory >= 0, self.subcategory_impressions[safe_subcategory], 0.0
            ),
            subcategory_clicks=np.where(
                subcategory >= 0, self.subcategory_clicks[safe_subcategory], 0.0
            ),
            title_chars=np.where(known, self.title_chars[safe], zero),
            abstract_chars=np.where(known, self.abstract_chars[safe], zero),
            title_entities=np.where(known, self.title_entities[safe], zero),
            abstract_entities=np.where(known, self.abstract_entities[safe], zero),
            embeddings=np.where(known[:, None], self.embeddings[safe], 0.0).astype(np.float32),
        )

    def user_block(self, user_index: int, history_indices: IntArray) -> UserBlock:
        history = history_indices[history_indices >= 0][-self.settings.max_history :]
        history_embeddings = (
            self.embeddings[history]
            if history.size
            else np.zeros((0, self.embeddings.shape[1]), np.float32)
        )
        known = user_index >= 0
        n_categories = self.vocabulary.n_categories
        return UserBlock(
            impressions=float(self.user_impressions[user_index]) if known else 0.0,
            clicks=float(self.user_clicks[user_index]) if known else 0.0,
            last_seen=float(self.user_last_seen[user_index]) if known else 0.0,
            is_known=bool(known),
            category_clicks=(
                self.user_category_clicks[user_index]
                if known
                else np.zeros(n_categories, dtype=np.float64)
            ),
            history_length=int(history.size),
            history_embeddings=history_embeddings,
        )

    def features_for_impression(
        self, article_indices: IntArray, user_index: int, history_indices: IntArray, now: float
    ) -> NDArray[np.float32]:
        return compute_features(
            self.candidate_block(article_indices, now),
            self.user_block(user_index, history_indices),
            now,
            self.settings,
        )

    # -- writes -------------------------------------------------------------
    def update(
        self, article_indices: IntArray, labels: NDArray[Any], user_index: int, now: float
    ) -> None:
        """Apply one impression to the counters. Call *after* reading features for it."""
        known = article_indices >= 0
        indices = article_indices[known]
        if indices.size == 0:
            self._update_user(
                user_index, np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64), now
            )
            return
        labels64 = np.asarray(labels, dtype=np.float64)[known]

        # Decay first so counters written now are on the same clock as counters read now.
        factors = decay_factors(now, self.article_decay_ts[indices], self.half_lives)
        self.article_decay_impressions[indices] *= factors
        self.article_decay_clicks[indices] *= factors
        self.article_decay_ts[indices] = now

        np.add.at(self.article_impressions, indices, 1.0)
        np.add.at(self.article_clicks, indices, labels64)
        np.add.at(
            self.article_decay_impressions, indices, np.ones((indices.size, self.n_half_lives))
        )
        np.add.at(
            self.article_decay_clicks,
            indices,
            np.repeat(labels64[:, None], self.n_half_lives, axis=1),
        )

        unseen = self.article_first_seen[indices] == 0.0
        if unseen.any():
            self.article_first_seen[indices[unseen]] = now
        self.article_last_seen[indices] = now

        categories = self.vocabulary.news_category[indices].astype(np.int64)
        subcategories = self.vocabulary.news_subcategory[indices].astype(np.int64)
        np.add.at(self.category_impressions, categories, 1.0)
        np.add.at(self.category_clicks, categories, labels64)
        np.add.at(self.subcategory_impressions, subcategories, 1.0)
        np.add.at(self.subcategory_clicks, subcategories, labels64)

        self._update_user(user_index, labels64, categories, now)

    def _update_user(
        self, user_index: int, labels: FloatArray, categories: IntArray, now: float
    ) -> None:
        if user_index < 0:
            return
        self.user_impressions[user_index] += 1.0
        self.user_clicks[user_index] += float(labels.sum()) if labels.size else 0.0
        self.user_last_seen[user_index] = now
        if labels.size:
            clicked = labels > 0
            if clicked.any():
                np.add.at(self.user_category_clicks[user_index], categories[clicked], 1.0)
