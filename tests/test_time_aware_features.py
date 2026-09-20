"""The no-leakage contract of the shared feature module.

These tests are the reason the offline numbers can be trusted: they check that a feature
for an impression at time t cannot see the impression itself, that counters decay the way
the config says, and that unknown ids degrade to "cold" instead of raising or silently
reading someone else's row.
"""

from __future__ import annotations

import numpy as np
import pytest

from news_recsys.config import Settings
from news_recsys.features.build import build_features
from news_recsys.features.time_aware import (
    SECONDS_PER_HOUR,
    TimeAwareFeatureStore,
    feature_names,
)
from news_recsys.features.vocab import build_vocabulary


@pytest.fixture(scope="module")
def store_bits(synthetic_settings: Settings):
    vocabulary = build_vocabulary(synthetic_settings)
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(vocabulary.n_news, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return vocabulary, embeddings


def make_store(synthetic_settings: Settings, store_bits) -> TimeAwareFeatureStore:
    vocabulary, embeddings = store_bits
    return TimeAwareFeatureStore(vocabulary, embeddings, synthetic_settings)


def column(settings: Settings, name: str) -> int:
    return feature_names(settings).index(name)


def test_feature_names_match_matrix_width(synthetic_settings: Settings, store_bits) -> None:
    store = make_store(synthetic_settings, store_bits)
    matrix = store.features_for_impression(
        np.array([0, 1, 2]), 0, np.array([], dtype=np.int64), 1_000_000.0
    )
    assert matrix.shape == (3, len(feature_names(synthetic_settings)))
    assert matrix.dtype == np.float32


def test_first_sighting_is_cold_then_warm(synthetic_settings: Settings, store_bits) -> None:
    store = make_store(synthetic_settings, store_bits)
    articles = np.array([5, 6])
    cold_col = column(synthetic_settings, "art_is_cold")
    impressions_col = column(synthetic_settings, "art_impr_log1p")
    now = 1_600_000_000.0

    first = store.features_for_impression(articles, 0, np.array([], dtype=np.int64), now)
    assert first[:, cold_col].tolist() == [1.0, 1.0]
    assert first[:, impressions_col].tolist() == [0.0, 0.0]

    store.update(articles, np.array([1, 0]), 0, now)

    later = store.features_for_impression(articles, 0, np.array([], dtype=np.int64), now + 60)
    assert later[:, cold_col].tolist() == [0.0, 0.0]
    np.testing.assert_allclose(later[:, impressions_col], np.log1p([1.0, 1.0]), rtol=1e-6)


def test_update_after_read_is_what_prevents_leakage(
    synthetic_settings: Settings, store_bits
) -> None:
    """A click recorded at time t must not appear in the features read at time t."""
    store = make_store(synthetic_settings, store_bits)
    articles = np.array([9])
    clicks_col = column(synthetic_settings, "art_click_log1p")
    now = 1_600_000_000.0

    before = store.features_for_impression(articles, 1, np.array([], dtype=np.int64), now)
    store.update(articles, np.array([1]), 1, now)
    after = store.features_for_impression(articles, 1, np.array([], dtype=np.int64), now)

    assert before[0, clicks_col] == 0.0
    assert after[0, clicks_col] == pytest.approx(np.log1p(1.0), rel=1e-6)


def test_decayed_popularity_halves_over_one_half_life(
    synthetic_settings: Settings, store_bits
) -> None:
    store = make_store(synthetic_settings, store_bits)
    half_life = synthetic_settings.popularity_half_lives_hours[0]
    articles = np.array([3])
    now = 1_600_000_000.0
    store.update(articles, np.array([1]), 0, now)

    immediately, _ = store.decayed(articles, now)
    one_half_life_later, _ = store.decayed(articles, now + half_life * SECONDS_PER_HOUR)
    assert immediately[0, 0] == pytest.approx(1.0)
    assert one_half_life_later[0, 0] == pytest.approx(0.5)


def test_unknown_ids_are_cold_rather_than_wrong(synthetic_settings: Settings, store_bits) -> None:
    store = make_store(synthetic_settings, store_bits)
    store.update(np.array([4]), np.array([1]), 0, 1_600_000_000.0)

    matrix = store.features_for_impression(
        np.array([-1, 4]), -1, np.array([-1], dtype=np.int64), 1_600_000_100.0
    )
    cold_col = column(synthetic_settings, "art_is_cold")
    user_cold_col = column(synthetic_settings, "user_is_cold")
    assert matrix[0, cold_col] == 1.0
    assert matrix[1, cold_col] == 0.0
    assert matrix[0, user_cold_col] == 1.0  # unknown user
    assert np.isfinite(matrix).all()


def test_history_similarity_is_bounded_and_self_consistent(
    synthetic_settings: Settings, store_bits
) -> None:
    store = make_store(synthetic_settings, store_bits)
    mean_col = column(synthetic_settings, "text_sim_hist_mean")
    max_col = column(synthetic_settings, "text_sim_hist_max")
    last_col = column(synthetic_settings, "text_sim_hist_last")

    empty = store.features_for_impression(np.array([7]), 0, np.array([], dtype=np.int64), 1e9)
    assert empty[0, mean_col] == 0.0 and empty[0, max_col] == 0.0 and empty[0, last_col] == 0.0

    # An article that is itself in the history must have max similarity 1.
    with_history = store.features_for_impression(
        np.array([7, 8]), 0, np.array([7, 11], dtype=np.int64), 1e9
    )
    assert with_history[0, max_col] == pytest.approx(1.0, abs=1e-6)
    assert -1.0 - 1e-6 <= float(with_history[1, max_col]) <= 1.0 + 1e-6
    # "last" compares against the most recent history entry (index 11, not 7).
    assert with_history[0, last_col] != pytest.approx(1.0, abs=1e-6)


def test_replay_marks_every_article_cold_on_its_first_appearance(
    synthetic_artifacts: Settings,
) -> None:
    """End-to-end: in the full replay, an article is cold exactly until it is first shown."""
    synthetic_settings = synthetic_artifacts
    folds = build_features(synthetic_settings, vocabulary=build_vocabulary(synthetic_settings))
    cold_col = column(synthetic_settings, "art_is_cold")

    features = np.concatenate([folds[fold].features for fold in ("train", "val", "test")])
    news_index = np.concatenate([folds[fold].news_index for fold in ("train", "val", "test")])
    timestamps = np.concatenate([folds[fold].timestamp for fold in ("train", "val", "test")])
    order = np.argsort(timestamps, kind="stable")
    features, news_index = features[order], news_index[order]

    seen: set[int] = set()
    for row in range(features.shape[0]):
        article = int(news_index[row])
        expected_cold = 0.0 if article in seen else 1.0
        assert features[row, cold_col] == expected_cold
        seen.add(article)
