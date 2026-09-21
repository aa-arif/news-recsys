"""M7: MMR re-ranking - what diversity costs in ranking quality.

The ranker's scores on the logged slate are re-ordered with MMR at several values of
lambda, and both sides of the trade are measured on the same impressions: nDCG@10 (does
the user still get what they wanted) and intra-list diversity (how much of the list is
the same story twice).

Lambda is *not* tuned here - the point of the milestone is the curve, and the operating
point is a product decision, not a validation-set decision.
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from typing import Any

import numpy as np

from news_recsys.config import get_settings, seed_everything
from news_recsys.data.sequences import build_impression_histories
from news_recsys.eval.metrics import evaluate_ranking, group_boundaries
from news_recsys.features.build import load_fold_features
from news_recsys.features.text import load_embeddings
from news_recsys.features.vocab import load_vocabulary
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger, timed
from news_recsys.models.ranker_train import RankerBatcher, build_ranker_dataset, predict
from news_recsys.plots import plot_diversity_tradeoff
from news_recsys.serving.rerank import intra_list_diversity, mean_pairwise_distance, mmr_scores

logger = get_logger("scripts.eval_rerank")

LAMBDAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--fold", default="test", choices=["val", "test"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-impressions", type=int, default=0, help="0 scores every impression")
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    seed_everything(settings.seed)

    from news_recsys.models.ranker import DinDcnRanker

    vocabulary = load_vocabulary(settings)
    embeddings = np.asarray(load_embeddings(settings, mmap=False), dtype=np.float32)
    fold = load_fold_features(args.fold, settings)
    histories = build_impression_histories(args.fold, settings, vocabulary=vocabulary)
    dataset = build_ranker_dataset(fold, histories, settings)
    model, _ = DinDcnRanker.load(settings.artifact_dir)

    with timed(logger, f"score {args.fold} with the ranker"):
        batcher = RankerBatcher(dataset, embeddings, vocabulary)
        logits, rows = predict(
            model, batcher, impressions_per_batch=settings.ranker_impressions_per_batch
        )
        order = np.argsort(rows)
        scores = logits[order]

    bounds = group_boundaries(fold.impression_key)
    slices = list(pairwise(bounds))
    if args.max_impressions:
        slices = slices[: args.max_impressions]

    categories = vocabulary.news_category.astype(np.int64)
    results: list[dict[str, Any]] = []

    for lambda_ in LAMBDAS:
        reranked = np.empty_like(scores)
        diversity: list[float] = []
        distance: list[float] = []
        labels_kept: list[np.ndarray] = []
        scores_kept: list[np.ndarray] = []
        groups_kept: list[np.ndarray] = []

        for start, end in slices:
            slate_scores = scores[start:end]
            slate_news = fold.news_index[start:end].astype(np.int64)
            safe = np.where(slate_news >= 0, slate_news, 0)
            slate_embeddings = embeddings[safe]

            new_scores = (
                slate_scores
                if lambda_ >= 1.0
                else mmr_scores(slate_scores, slate_embeddings, lambda_)
            )
            reranked[start:end] = new_scores

            top = np.argsort(-new_scores)[: args.k]
            diversity.append(intra_list_diversity(categories[safe[top]], args.k))
            distance.append(mean_pairwise_distance(slate_embeddings[top]))

            labels_kept.append(fold.labels[start:end])
            scores_kept.append(new_scores)
            groups_kept.append(fold.impression_key[start:end])

        report = evaluate_ranking(
            np.concatenate(labels_kept).astype(np.float64),
            np.concatenate(scores_kept),
            np.concatenate(groups_kept),
            cutoffs=settings.ndcg_cutoffs,
        )
        means = report.means()
        point = {
            "lambda": lambda_,
            **means,
            "diversity": float(np.mean(diversity)),
            "mean_pairwise_distance": float(np.mean(distance)),
            "impressions": report.n_impressions,
        }
        results.append(point)
        logger.info(
            "lambda=%.1f nDCG@10=%.4f diversity=%.4f pairwise-distance=%.4f",
            lambda_,
            means["ndcg@10"],
            point["diversity"],
            point["mean_pairwise_distance"],
        )

    baseline = results[0]
    payload = {
        "dataset": settings.dataset,
        "fold": args.fold,
        "k": args.k,
        "points": results,
        "baseline_lambda_1": baseline,
        "note": (
            "MMR is applied to the logged slate, so nDCG stays comparable with the ranker's "
            "own numbers; lambda is a product decision and is deliberately not tuned here."
        ),
    }
    figure = plot_diversity_tradeoff(
        results,
        settings.figures_dir / f"diversity_tradeoff_{settings.dataset}.png",
        title=f"MMR: nDCG@10 vs category diversity (MIND-{settings.dataset}, {args.fold})",
    )
    payload["figures"] = {"diversity_tradeoff": str(figure.relative_to(settings.root_dir))}
    write_json(settings.metrics_dir / f"rerank_{settings.dataset}.json", payload)


if __name__ == "__main__":
    main()
