"""Load the offline feature-store snapshot into Redis.

Lives in the package rather than in the script so that the training/serving skew test
seeds Redis through exactly the same code the deployment uses. If seeding and serving
disagreed about, say, the order of the packed decay counters, a test that re-implemented
seeding would happily pass while production served garbage.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from news_recsys.config import Settings
from news_recsys.data.splits import load_impressions, load_news
from news_recsys.features.time_aware import TimeAwareFeatureStore
from news_recsys.features.vocab import Vocabulary
from news_recsys.logging_utils import get_logger
from news_recsys.serving.redis_store import (
    ARTICLE_PREFIX,
    CATEGORY_PREFIX,
    HISTORY_PREFIX,
    POPULAR_KEY,
    SUBCATEGORY_PREFIX,
    USER_PREFIX,
    ArticleStatic,
    RedisFeatureStore,
)

logger = get_logger("serving.seed")


def article_static(settings: Settings, vocabulary: Vocabulary) -> ArticleStatic:
    news = load_news(settings)
    return ArticleStatic(
        category=vocabulary.news_category.astype(np.int64),
        subcategory=vocabulary.news_subcategory.astype(np.int64),
        title_chars=news["title"].str.len_chars().to_numpy().astype(np.float64),
        abstract_chars=news["abstract"].str.len_chars().to_numpy().astype(np.float64),
        title_entities=news["n_title_entities"].to_numpy().astype(np.float64),
        abstract_entities=news["n_abstract_entities"].to_numpy().astype(np.float64),
    )


def seed_counters(
    client: Any,
    store: TimeAwareFeatureStore,
    vocabulary: Vocabulary,
    settings: Settings,
    *,
    batch: int = 5000,
) -> dict[str, Any]:
    """Write article, category, subcategory and user counters."""
    writer = RedisFeatureStore(
        client,
        vocabulary,
        np.zeros((1, 1), dtype=np.float32),
        article_static(settings, vocabulary),
        settings,
    )

    articles = np.column_stack(
        [
            store.article_impressions,
            store.article_clicks,
            store.article_decay_impressions,
            store.article_decay_clicks,
            store.article_decay_ts,
            store.article_first_seen,
            store.article_last_seen,
        ]
    )
    # Unseen articles read back as zeros anyway, so only write the ones carrying counts.
    active = np.flatnonzero(store.article_impressions > 0)
    pipeline = client.pipeline(transaction=False)
    for position, index in enumerate(active, start=1):
        pipeline.set(f"{ARTICLE_PREFIX}{int(index)}", writer.pack_article(articles[index]))
        if position % batch == 0:
            pipeline.execute()
    pipeline.execute()

    pipeline = client.pipeline(transaction=False)
    for index in range(vocabulary.n_categories):
        pipeline.set(
            f"{CATEGORY_PREFIX}{index}",
            writer.pack_pair(
                float(store.category_impressions[index]), float(store.category_clicks[index])
            ),
        )
    for index in range(vocabulary.n_subcategories):
        pipeline.set(
            f"{SUBCATEGORY_PREFIX}{index}",
            writer.pack_pair(
                float(store.subcategory_impressions[index]), float(store.subcategory_clicks[index])
            ),
        )
    pipeline.execute()

    users = np.column_stack(
        [
            store.user_impressions,
            store.user_clicks,
            store.user_last_seen,
            store.user_category_clicks,
        ]
    )
    active_users = np.flatnonzero(store.user_impressions > 0)
    pipeline = client.pipeline(transaction=False)
    for position, index in enumerate(active_users, start=1):
        pipeline.set(f"{USER_PREFIX}{int(index)}", writer.pack_user(users[index]))
        if position % batch == 0:
            pipeline.execute()
    pipeline.execute()

    return {
        "articles": int(active.size),
        "users": int(active_users.size),
        "categories": vocabulary.n_categories,
        "subcategories": vocabulary.n_subcategories,
    }


USERS_FILENAME = "serving_users.txt"


def seed_popularity(
    client: Any,
    store: TimeAwareFeatureStore,
    settings: Settings,
    as_of: float,
    *,
    limit: int | None = None,
) -> int:
    """Write the trending list: articles ranked by smoothed, decayed CTR at ``as_of``.

    M3 measured that this user-independent source retrieves the clicked article far more
    often than the two-tower does on MIND-small, so the serving path blends both rather
    than pretending the learned retriever is enough. Only articles active in the last 24h
    are eligible - an article nobody has seen for days is not "trending".
    """
    limit = limit or settings.popularity_list_size
    active = np.flatnonzero(
        (store.article_last_seen > 0) & ((as_of - store.article_last_seen) <= 24 * 3600)
    ).astype(np.int64)
    if active.size == 0:
        return 0

    decayed_impressions, decayed_clicks = store.decayed(active, as_of)
    smoothed = (decayed_clicks[:, 0] + settings.ctr_prior_clicks) / (
        decayed_impressions[:, 0] + settings.ctr_prior_clicks + settings.ctr_prior_impressions
    )
    ranked = active[np.argsort(-smoothed)][:limit]

    client.delete(POPULAR_KEY)
    if ranked.size:
        client.rpush(POPULAR_KEY, *[int(index) for index in ranked])
    return int(ranked.size)


def seed_histories(
    client: Any, settings: Settings, *, fold: str = "test", batch: int = 5000
) -> int:
    """Write each user's click history as it stood at their first impression of ``fold``.

    Also writes the list of seeded user ids next to the model artifacts: the load test
    needs a population of users that actually exist in Redis, and reading it from a file
    keeps Locust from having to import the data stack.
    """
    impressions = load_impressions(fold, settings)
    seen: set[str] = set()
    pipeline = client.pipeline(transaction=False)
    written = 0
    for user_id, history in zip(
        impressions["user_id"].to_list(), impressions["history"].to_list(), strict=True
    ):
        if user_id in seen:
            continue
        seen.add(user_id)
        key = f"{HISTORY_PREFIX}{user_id}"
        pipeline.delete(key)
        if history:
            pipeline.rpush(key, *history)
        written += 1
        if written % batch == 0:
            pipeline.execute()
    pipeline.execute()

    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    users_path = settings.artifact_dir / USERS_FILENAME
    users_path.write_text("\n".join(sorted(seen)), encoding="utf-8")
    return written
