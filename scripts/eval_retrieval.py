"""M3: retrieval quality and the recall/latency curve.

Four things are measured, because "Recall@K" on its own hides which part of the system is
responsible for a miss:

1. **Recall@K over the full catalogue** - every article in the index is a candidate. This
   is the honest headline number and the hardest setting.
2. **The index-freshness ceiling** - what share of clicked articles even existed in the
   index when it was built. A static index built at midnight cannot retrieve an article
   published at 09:00, however good the model is.
3. **Recall@K over the live pool, conditioned on reachability** - restricted to articles
   active in the 24h before the index build, and to clicks on those articles. This
   separates *model* quality from *index coverage*.
4. **ANN fidelity (overlap@K vs exact search) against latency**, swept over ``efSearch``.

A user-independent "most popular right now" retriever is scored alongside, because on
news that is a genuinely strong candidate generator, and a two-tower model that cannot
beat it is not earning its place.

``efSearch`` is chosen on validation and only then applied to test.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from news_recsys.config import Settings, get_settings, seed_everything
from news_recsys.data.sequences import build_click_sequences
from news_recsys.features.build import load_snapshot
from news_recsys.features.text import load_embeddings
from news_recsys.features.time_aware import TimeAwareFeatureStore
from news_recsys.features.vocab import load_vocabulary
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger, timed
from news_recsys.models.two_tower import TwoTowerModel
from news_recsys.models.two_tower_train import SequenceBatcher, encode_users
from news_recsys.plots import plot_recall_latency
from news_recsys.retrieval.faiss_index import ItemIndex, exact_search, set_search_threads

logger = get_logger("scripts.eval_retrieval")

EF_SWEEP = (16, 32, 64, 128, 256, 512)
LIVE_WINDOW_HOURS = 24.0


def recall_at_k(
    ranked: NDArray[np.int64], targets: NDArray[np.int64], cutoffs: tuple[int, ...]
) -> dict[str, float]:
    return {
        f"recall@{cutoff}": float((ranked[:, :cutoff] == targets[:, None]).any(axis=1).mean())
        for cutoff in cutoffs
    }


def overlap_at_k(approximate: NDArray[np.int64], exact: NDArray[np.int64], k: int) -> float:
    """Share of the exact top-k that the ANN search also returned."""
    matches = [
        len(set(approximate[row, :k].tolist()) & set(exact[row, :k].tolist()))
        for row in range(exact.shape[0])
    ]
    return float(np.mean(matches) / k)


def per_query_latency(
    index: ItemIndex, queries: NDArray[np.float32], k: int, ef_search: int
) -> dict[str, float]:
    """Single-query latency, which is what serving actually experiences."""
    index.ef_search = ef_search
    timings = np.empty(queries.shape[0], dtype=np.float64)
    for row in range(queries.shape[0]):
        start = time.perf_counter()
        index.index.search(queries[row : row + 1], k)
        timings[row] = (time.perf_counter() - start) * 1000.0
    return {
        "mean_ms": float(timings.mean()),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "queries": int(queries.shape[0]),
    }


def popularity_ranking(
    store: TimeAwareFeatureStore,
    as_of: float,
    pool: NDArray[np.int64],
    k: int,
    settings: Settings,
) -> NDArray[np.int64]:
    """The same top-k for every user: highest smoothed decayed CTR at ``as_of``."""
    decayed_impressions, decayed_clicks = store.decayed(pool, as_of)
    smoothed = (decayed_clicks[:, 0] + settings.ctr_prior_clicks) / (
        decayed_impressions[:, 0] + settings.ctr_prior_clicks + settings.ctr_prior_impressions
    )
    return pool[np.argsort(-smoothed)[:k]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--latency-queries", type=int, default=2000)
    parser.add_argument("--max-eval-clicks", type=int, default=0, help="0 uses every click")
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    seed_everything(settings.seed)
    set_search_threads(1)  # per-request latency, not batch throughput

    vocabulary = load_vocabulary(settings)
    embeddings = np.asarray(load_embeddings(settings, mmap=False), dtype=np.float32)
    model, _ = TwoTowerModel.load(settings.artifact_dir)
    item_vectors = np.load(settings.artifact_dir / "item_vectors.npy")
    index = ItemIndex.load(settings.artifact_dir, settings)
    store, as_of = load_snapshot(settings, vocabulary=vocabulary)

    batcher = SequenceBatcher(embeddings, vocabulary, np.zeros(vocabulary.n_news))
    n_items = int(item_vectors.shape[0])
    cutoffs = tuple(cutoff for cutoff in settings.recall_cutoffs if cutoff <= n_items) or (n_items,)
    largest = max(cutoffs)
    candidates_k = min(settings.retrieval_candidates, n_items)
    rng = np.random.default_rng(settings.seed)

    # The pool an index built at snapshot time would hold: articles shown at least once in
    # the LIVE_WINDOW_HOURS before it. Uses no information from the test day.
    active = store.article_last_seen > 0
    fresh = (as_of - store.article_last_seen) <= LIVE_WINDOW_HOURS * 3600
    live_pool = np.flatnonzero(active & fresh).astype(np.int64)
    logger.info(
        "live pool: %d of %d articles active in the %.0fh before as_of",
        live_pool.size,
        n_items,
        LIVE_WINDOW_HOURS,
    )

    results: dict[str, Any] = {
        "dataset": settings.dataset,
        "index": {
            "vectors": index.size,
            "hnsw_m": settings.faiss_hnsw_m,
            "ef_construction": settings.faiss_ef_construction,
        },
        "live_pool": {
            "window_hours": LIVE_WINDOW_HOURS,
            "articles": int(live_pool.size),
            "catalogue": n_items,
            "as_of": as_of,
        },
        "protocol": (
            "Recall@K of the clicked article over the full catalogue, and separately over "
            "the live pool restricted to reachable clicks; efSearch chosen on val."
        ),
        "folds": {},
    }

    for fold in ("val", "test"):
        clicks = build_click_sequences(fold, settings, vocabulary=vocabulary)
        if args.max_eval_clicks and len(clicks) > args.max_eval_clicks:
            sample = rng.choice(len(clicks), size=args.max_eval_clicks, replace=False)
            history, mask = clicks.history[sample], clicks.mask[sample]
            positive = clicks.positive[sample]
        else:
            history, mask, positive = clicks.history, clicks.mask, clicks.positive

        with timed(logger, f"encode {fold} users ({history.shape[0]} clicks)"):
            user_vectors = encode_users(model, batcher, history, mask)

        with timed(logger, f"exact search {fold} (full catalogue)"):
            exact_indices, _ = exact_search(item_vectors, user_vectors, largest)

        fold_result: dict[str, Any] = {
            "clicks_scored": int(positive.size),
            "full_catalogue": {
                "articles": n_items,
                "exact": recall_at_k(exact_indices, positive, cutoffs),
            },
            "ef_sweep": [],
        }

        # -- index freshness ceiling and live-pool recall --------------------
        in_pool = np.isin(positive, live_pool)
        fold_result["live_pool"] = {
            "reachable_click_share": float(in_pool.mean()),
            "clicks_reachable": int(in_pool.sum()),
        }
        if in_pool.any():
            with timed(logger, f"exact search {fold} (live pool)"):
                pool_k = int(min(largest, live_pool.size))
                pool_indices, _ = exact_search(
                    item_vectors[live_pool], user_vectors[in_pool], pool_k
                )
                mapped = live_pool[pool_indices]
            pool_cutoffs = tuple(cutoff for cutoff in cutoffs if cutoff <= pool_k)
            given_reachable = recall_at_k(mapped, positive[in_pool], pool_cutoffs)
            fold_result["live_pool"]["exact_given_reachable"] = given_reachable
            fold_result["live_pool"]["exact_unconditional"] = {
                key: value * float(in_pool.mean()) for key, value in given_reachable.items()
            }

        # -- user-independent popularity retriever ---------------------------
        popular = popularity_ranking(store, as_of, live_pool, largest, settings)
        fold_result["popularity_retriever"] = recall_at_k(
            np.tile(popular, (positive.size, 1)),
            positive,
            tuple(cutoff for cutoff in cutoffs if cutoff <= popular.size),
        )

        # -- HNSW sweep ------------------------------------------------------
        latency_sample = rng.choice(
            user_vectors.shape[0],
            size=min(args.latency_queries, user_vectors.shape[0]),
            replace=False,
        )
        for ef_search in EF_SWEEP:
            index.ef_search = ef_search
            with timed(logger, f"{fold} HNSW search efSearch={ef_search}"):
                result = index.search(user_vectors, largest)
            latency = per_query_latency(
                index, user_vectors[latency_sample], candidates_k, ef_search
            )
            point = {
                "ef_search": ef_search,
                **recall_at_k(result.indices, positive, cutoffs),
                "overlap@100_vs_exact": overlap_at_k(
                    result.indices, exact_indices, min(100, largest)
                ),
                "batch_seconds": result.seconds,
                **latency,
                "exact_recall": fold_result["full_catalogue"]["exact"][f"recall@{largest}"],
            }
            fold_result["ef_sweep"].append(point)
            logger.info(
                "%s ef=%3d recall@%d=%.4f overlap@100=%.4f p95=%.3fms",
                fold,
                ef_search,
                largest,
                point[f"recall@{largest}"],
                point["overlap@100_vs_exact"],
                point["p95_ms"],
            )
        results["folds"][fold] = fold_result

    # -- choose efSearch on validation --------------------------------------
    target_metric = f"recall@{candidates_k}" if candidates_k in cutoffs else f"recall@{largest}"
    val_points = results["folds"]["val"]["ef_sweep"]
    val_exact = results["folds"]["val"]["full_catalogue"]["exact"][target_metric]
    within_one_percent = [point for point in val_points if point[target_metric] >= 0.99 * val_exact]
    chosen = min(within_one_percent or val_points, key=lambda point: point["p95_ms"])
    results["selected_ef_search"] = {
        "ef_search": chosen["ef_search"],
        "rule": f"smallest p95 latency whose val {target_metric} is within 1% of exact search",
        "val_recall": chosen[target_metric],
        "val_exact_recall": val_exact,
        "val_p95_ms": chosen["p95_ms"],
    }
    test_point = next(
        point
        for point in results["folds"]["test"]["ef_sweep"]
        if point["ef_search"] == chosen["ef_search"]
    )
    results["test_at_selected_ef"] = test_point
    logger.info(
        "selected efSearch=%d | test %s=%.4f (exact %.4f) p95=%.3f ms",
        chosen["ef_search"],
        target_metric,
        test_point[target_metric],
        results["folds"]["test"]["full_catalogue"]["exact"][target_metric],
        test_point["p95_ms"],
    )

    figure = plot_recall_latency(
        results["folds"]["test"]["ef_sweep"],
        settings.figures_dir / f"recall_latency_{settings.dataset}.png",
        title=f"Retrieval recall vs latency (MIND-{settings.dataset}, test)",
        k=candidates_k if candidates_k in cutoffs else largest,
    )
    results["figures"] = {"recall_latency": str(figure.relative_to(settings.root_dir))}
    write_json(settings.metrics_dir / f"retrieval_{settings.dataset}.json", results)


if __name__ == "__main__":
    main()
