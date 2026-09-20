"""Training/serving skew: the online feature path must reproduce the offline one exactly.

The offline pipeline replays the event log and writes a snapshot of its counter state.
Serving seeds Redis from that snapshot and rebuilds features from it. If the two paths
agree bit for bit on the same request, then any difference in online behaviour has to
come from the *data* (staleness, a missing counter update), never from two diverging
implementations of the same feature.

"Bit for bit" is the right bar and not an unreasonable one: both paths call the same
``compute_features`` on float64 inputs and cast once at the end, and Redis values are
packed as raw float64. A tolerance here would hide exactly the bugs worth catching - a
feature computed from the wrong column, a decay applied twice, a category id off by one.
"""

from __future__ import annotations

import numpy as np
import pytest

from news_recsys.config import Settings
from news_recsys.data.sequences import build_impression_histories
from news_recsys.data.splits import load_events, load_impressions
from news_recsys.features.build import load_snapshot
from news_recsys.features.time_aware import compute_features
from news_recsys.features.vocab import Vocabulary, load_vocabulary
from news_recsys.serving.redis_store import RedisFeatureStore
from news_recsys.serving.seed import article_static, seed_counters, seed_histories

fakeredis = pytest.importorskip("fakeredis")


@pytest.fixture(scope="module")
def seeded(synthetic_artifacts: Settings):
    """A fake Redis seeded from the snapshot, plus the equivalent offline store."""
    settings = synthetic_artifacts
    vocabulary = load_vocabulary(settings)
    store, as_of = load_snapshot(settings, vocabulary=vocabulary)

    client = fakeredis.FakeStrictRedis()
    seed_counters(client, store, vocabulary, settings)
    seed_histories(client, settings, fold="test")

    online = RedisFeatureStore(
        client,
        vocabulary,
        np.asarray(store.embeddings),
        article_static(settings, vocabulary),
        settings,
    )
    return settings, vocabulary, store, online, as_of


def sample_requests(settings: Settings, vocabulary: Vocabulary, limit: int = 25):
    """Real test-fold impressions: a user, their history, and the articles they saw."""
    impressions = load_impressions("test", settings)
    events = load_events("test", settings, columns=["impression_key", "news_id"])
    histories = build_impression_histories("test", settings, vocabulary=vocabulary)
    history_by_key = {
        int(key): histories.history[row][histories.mask[row] > 0]
        for row, key in enumerate(histories.impression_key)
    }

    grouped: dict[int, list[str]] = {}
    for key, news_id in zip(
        events["impression_key"].to_list(), events["news_id"].to_list(), strict=True
    ):
        grouped.setdefault(int(key), []).append(news_id)

    requests = []
    seen_users: set[str] = set()
    for key, user_id in zip(
        impressions["impression_key"].to_list(), impressions["user_id"].to_list(), strict=True
    ):
        candidates = grouped.get(int(key))
        if not candidates:
            continue
        # Redis holds one history per user - the one seeded from their first impression of
        # the fold - so those are the requests whose offline state the seed reproduces.
        # A later impression of the same user has a longer history offline, and comparing
        # against it would be testing the seeding policy, not the feature code.
        if user_id in seen_users:
            continue
        seen_users.add(user_id)
        requests.append(
            {
                "user_id": user_id,
                "candidates": vocabulary.news_indices(candidates),
                "history": history_by_key.get(int(key), np.empty(0, dtype=np.int64)),
            }
        )
        if len(requests) >= limit:
            break
    return requests


def test_online_features_match_offline_bit_for_bit(seeded) -> None:
    settings, vocabulary, store, online, as_of = seeded
    requests = sample_requests(settings, vocabulary)
    assert requests, "no sampled requests - the synthetic fixture is broken"

    mismatches = []
    for request in requests:
        user_index = vocabulary.user_index(request["user_id"])
        offline = store.features_for_impression(
            request["candidates"], user_index, request["history"], as_of
        )

        candidates, user_block = online.fetch_counters(
            request["user_id"],
            request["candidates"],
            as_of,
            online.fetch_history(request["user_id"]),
        )
        served = compute_features(candidates, user_block, as_of, settings)

        if not np.array_equal(offline, served):
            differing = np.argwhere(offline != served)
            mismatches.append((request["user_id"], differing[:5].tolist()))

    assert not mismatches, (
        f"online/offline feature mismatch for {len(mismatches)} requests: {mismatches[:3]}"
    )


def test_history_survives_the_redis_round_trip(seeded) -> None:
    settings, vocabulary, _, online, _ = seeded
    histories = build_impression_histories("test", settings, vocabulary=vocabulary)
    impressions = load_impressions("test", settings)

    checked = 0
    seen_users: set[str] = set()
    for row, user_id in enumerate(impressions["user_id"].to_list()):
        if user_id in seen_users:
            continue  # seeding keeps the first impression's history per user
        seen_users.add(user_id)
        expected = histories.history[row][histories.mask[row] > 0]
        actual = online.fetch_history(user_id)
        assert actual.tolist() == expected.tolist()
        checked += 1
        if checked >= 20:
            break
    assert checked > 0


def test_unknown_user_serves_cold_features_instead_of_failing(seeded) -> None:
    settings, _vocabulary, _, online, as_of = seeded
    candidates = np.arange(5, dtype=np.int64)
    history = online.fetch_history("U-does-not-exist")
    assert history.size == 0

    candidates_block, user_block = online.fetch_counters(
        "U-does-not-exist", candidates, as_of, history
    )
    served = compute_features(candidates_block, user_block, as_of, settings)
    assert np.isfinite(served).all()
    assert user_block.is_known is False


def test_articles_absent_from_redis_read_as_zero_counters(seeded) -> None:
    settings, vocabulary, store, online, as_of = seeded
    # An article the replay never saw has no key in Redis; offline it has zero counters.
    never_seen = np.flatnonzero(store.article_impressions == 0)[:3].astype(np.int64)
    if never_seen.size == 0:
        pytest.skip("every article was shown at least once in the synthetic dataset")

    user_id = "U0"
    history = online.fetch_history(user_id)
    candidates_block, user_block = online.fetch_counters(user_id, never_seen, as_of, history)
    served = compute_features(candidates_block, user_block, as_of, settings)
    offline = store.features_for_impression(
        never_seen, vocabulary.user_index(user_id), history, as_of
    )
    np.testing.assert_array_equal(served, offline)
