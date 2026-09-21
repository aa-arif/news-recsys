"""Retrieval recall over a *fresh* candidate pool, rebuilt for every impression.

The Recall@K in M3 searches the whole 65,238-article catalogue, most of which is stale by
the test day - articles nobody has been shown for days are still competing for the top-K.
That is the right number for "can the tower find the needle anywhere", and the wrong number
for "would this retriever work in a news feed", where the live pool is a few thousand
articles that were circulating in the last day.

This script rebuilds the pool per impression: the candidates are the articles that appeared
in **at least one impression during the previous ``window`` hours**, judged only from events
strictly earlier than the impression being scored. Because the replay is ordered and applies
an impression's events only after scoring it, that constraint is structural rather than a
claim (the same discipline as ``features/build.py``).

Three retrievers are scored under each pool, at a fixed budget:

* **two-tower** - dot product against the item vectors;
* **trending** - smoothed, exponentially decayed CTR evaluated *at the impression's
  timestamp*, so it is a live signal rather than a static list;
* **blend** - half the budget from each, de-duplicated, which is what the serving path does.

Recall is reported unconditionally (an article missing from the pool counts as a miss, which
is what a user experiences) and conditioned on reachability (which isolates the retriever
from the pool's coverage).
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from news_recsys.config import Settings, get_settings, seed_everything
from news_recsys.data.sequences import build_impression_histories
from news_recsys.data.splits import load_all_events, load_news
from news_recsys.features.text import load_embeddings
from news_recsys.features.time_aware import TimeAwareFeatureStore
from news_recsys.features.vocab import Vocabulary, load_vocabulary
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger, timed
from news_recsys.models.two_tower import TwoTowerModel
from news_recsys.models.two_tower_train import SequenceBatcher, encode_users

logger = get_logger("scripts.eval_retrieval_fresh")

CUTOFFS = (50, 200)
HOUR = 3600.0


class RecallTally:
    """Hit counts per retriever, cutoff and pool, plus reachability."""

    def __init__(self, pools: list[str], retrievers: list[str], cutoffs: tuple[int, ...]) -> None:
        self.cutoffs = cutoffs
        self.hits = {
            pool: {name: dict.fromkeys(cutoffs, 0) for name in retrievers} for pool in pools
        }
        self.reachable = dict.fromkeys(pools, 0)
        self.clicks = 0

    def add(self, pool: str, retriever: str, ranks: dict[int, bool]) -> None:
        for cutoff, hit in ranks.items():
            if hit:
                self.hits[pool][retriever][cutoff] += 1

    def report(self) -> dict[str, Any]:
        total = max(self.clicks, 1)
        out: dict[str, Any] = {"clicks_scored": self.clicks, "pools": {}}
        for pool, retrievers in self.hits.items():
            reachable = self.reachable[pool]
            out["pools"][pool] = {
                "reachable_clicks": reachable,
                "reachable_share": reachable / total,
                "retrievers": {
                    name: {
                        **{f"recall@{cutoff}": hits[cutoff] / total for cutoff in self.cutoffs},
                        **{
                            f"recall@{cutoff}_given_reachable": (
                                hits[cutoff] / reachable if reachable else float("nan")
                            )
                            for cutoff in self.cutoffs
                        },
                    }
                    for name, hits in retrievers.items()
                },
            }
        return out


def trending_scores(
    store: TimeAwareFeatureStore, now: float, settings: Settings
) -> NDArray[np.float64]:
    """Smoothed, decayed CTR for every article as of ``now`` (shortest half-life).

    Only the shortest half-life is needed, so the decay is applied to that column directly
    rather than through ``store.decayed``, which would do three times the work per
    impression and this loop runs 73,152 times.
    """
    half_life = store.half_lives[0]
    elapsed_hours = np.maximum((now - store.article_decay_ts) / HOUR, 0.0)
    factor = np.exp2(-elapsed_hours / half_life)
    impressions = store.article_decay_impressions[:, 0] * factor
    clicks = store.article_decay_clicks[:, 0] * factor
    return (clicks + settings.ctr_prior_clicks) / (
        impressions + settings.ctr_prior_clicks + settings.ctr_prior_impressions
    )


def pool_rank(scores: NDArray[np.float64], pool_mask: NDArray[np.bool_], target: int) -> int | None:
    """0-based rank of ``target`` among the pool, or None when it is not in the pool.

    Counting how many pool articles outscore the target is equivalent to ranking the pool
    and much cheaper than sorting it once per impression.
    """
    if not pool_mask[target]:
        return None
    return int(np.count_nonzero(pool_mask & (scores > scores[target])))


def hits_from_rank(rank: int | None, cutoffs: tuple[int, ...]) -> dict[int, bool]:
    return {cutoff: rank is not None and rank < cutoff for cutoff in cutoffs}


def blend_from_ranks(
    tower: int | None, trending: int | None, cutoffs: tuple[int, ...]
) -> dict[int, bool]:
    """Half the budget from each source, de-duplicated - the serving path's candidate set.

    The union is at most ``cutoff`` items and often fewer once duplicates are removed,
    which is exactly what the server sends to the ranker.
    """
    return {
        cutoff: (tower is not None and tower < cutoff // 2)
        or (trending is not None and trending < cutoff // 2)
        for cutoff in cutoffs
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--windows", default="24,48", help="fresh-pool windows in hours")
    parser.add_argument("--max-impressions", type=int, default=0, help="0 scores every impression")
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    seed_everything(settings.seed)

    vocabulary: Vocabulary = load_vocabulary(settings)
    embeddings = np.asarray(load_embeddings(settings, mmap=False), dtype=np.float32)
    model, _ = TwoTowerModel.load(settings.artifact_dir)
    item_vectors = np.ascontiguousarray(np.load(settings.artifact_dir / "item_vectors.npy"))
    windows = [float(value) for value in args.windows.split(",") if value.strip()]

    with timed(logger, "encode test user vectors"):
        histories = build_impression_histories("test", settings, vocabulary=vocabulary)
        batcher = SequenceBatcher(embeddings, vocabulary, np.zeros(vocabulary.n_news))
        user_vectors = encode_users(model, batcher, histories.history, histories.mask)
    user_row = {int(key): row for row, key in enumerate(histories.impression_key)}

    events = load_all_events(settings)
    news_map = {news_id: index for index, news_id in enumerate(vocabulary.news_ids)}
    event_news = np.asarray(
        [news_map.get(news_id, -1) for news_id in events["news_id"].to_list()], dtype=np.int64
    )
    event_label = events["label"].to_numpy().astype(np.int8)
    event_key = events["impression_key"].to_numpy()
    event_ts = (events["time"].dt.epoch("ms").to_numpy() / 1000.0).astype(np.float64)
    event_fold = events["fold"].to_numpy()

    news = load_news(settings)
    store = TimeAwareFeatureStore(
        vocabulary,
        embeddings,
        settings,
        article_static={
            "title_chars": news["title"].str.len_chars().to_numpy().astype(np.float64),
            "abstract_chars": news["abstract"].str.len_chars().to_numpy().astype(np.float64),
            "title_entities": news["n_title_entities"].to_numpy().astype(np.float64),
            "abstract_entities": news["n_abstract_entities"].to_numpy().astype(np.float64),
        },
    )

    pools = ["full", *[f"fresh_{window:g}h" for window in windows]]
    retrievers = ["two_tower", "trending", "blend"]
    tally = RecallTally(pools, retrievers, CUTOFFS)
    pool_sizes: dict[str, list[int]] = {pool: [] for pool in pools}

    boundaries = np.flatnonzero(event_key[1:] != event_key[:-1]) + 1
    boundaries = np.concatenate(([0], boundaries, [event_key.size]))
    scored_impressions = 0

    with timed(logger, f"replay {boundaries.size - 1} impressions"):
        for start, end in pairwise(boundaries):
            now = float(event_ts[start])
            indices = event_news[start:end]
            labels = event_label[start:end]

            is_test = str(event_fold[start]) == "test"
            budget_reached = args.max_impressions and scored_impressions >= args.max_impressions
            clicked = indices[(labels == 1) & (indices >= 0)]

            if is_test and clicked.size and not budget_reached:
                scored_impressions += 1
                row = user_row.get(int(event_key[start]))
                if row is not None:
                    tower_scores = (item_vectors @ user_vectors[row]).astype(np.float64)
                    trend_scores = trending_scores(store, now, settings)
                    seen = store.article_last_seen > 0
                    # "full" is the M3 setting: every article in the index competes.
                    masks = {"full": np.ones(seen.shape, dtype=bool)}
                    for window in windows:
                        masks[f"fresh_{window:g}h"] = seen & (
                            (now - store.article_last_seen) <= window * HOUR
                        )

                    for pool, mask in masks.items():
                        pool_sizes[pool].append(int(np.count_nonzero(mask)))
                        for target in clicked:
                            tower_rank = pool_rank(tower_scores, mask, int(target))
                            trend_rank = pool_rank(trend_scores, mask, int(target))
                            tally.add(pool, "two_tower", hits_from_rank(tower_rank, CUTOFFS))
                            tally.add(pool, "trending", hits_from_rank(trend_rank, CUTOFFS))
                            tally.add(
                                pool, "blend", blend_from_ranks(tower_rank, trend_rank, CUTOFFS)
                            )
                            if tower_rank is not None:
                                tally.reachable[pool] += 1
                    tally.clicks += int(clicked.size)

            store.update(indices, labels, -1, now)

    report = tally.report()
    for pool in pools:
        sizes = np.asarray(pool_sizes[pool], dtype=np.float64)
        report["pools"][pool]["mean_pool_size"] = float(sizes.mean()) if sizes.size else 0.0
        report["pools"][pool]["median_pool_size"] = float(np.median(sizes)) if sizes.size else 0.0

    payload = {
        "dataset": settings.dataset,
        "catalogue": vocabulary.n_news,
        "impressions_scored": scored_impressions,
        "cutoffs": list(CUTOFFS),
        "windows_hours": windows,
        "protocol": (
            "Per-impression candidate pool: articles with at least one impression in the "
            "preceding window, judged only from events strictly earlier than the impression. "
            "Trending is scored with live decayed CTR at the impression timestamp; blend takes "
            "half the budget from each source."
        ),
        **report,
    }
    path = write_json(settings.metrics_dir / f"retrieval_fresh_{settings.dataset}.json", payload)

    for pool, block in report["pools"].items():
        logger.info(
            "%-12s pool~%5.0f articles | reachable %.3f | tower R@200 %.4f | trending R@200 %.4f | blend R@200 %.4f",
            pool,
            block["mean_pool_size"],
            block["reachable_share"],
            block["retrievers"]["two_tower"]["recall@200"],
            block["retrievers"]["trending"]["recall@200"],
            block["retrievers"]["blend"]["recall@200"],
        )
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
