"""MMR re-ranking: it must be a no-op at lambda=1 and actually diversify below it."""

from __future__ import annotations

import numpy as np
import pytest

from news_recsys.serving.rerank import (
    intra_list_diversity,
    mean_pairwise_distance,
    mmr_scores,
    mmr_select,
    normalise,
)


def clustered_candidates(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three tight topical clusters; relevance decreasing within the first cluster."""
    rng = np.random.default_rng(seed)
    centres = np.eye(3, 8, dtype=np.float32)
    embeddings = []
    categories = []
    for cluster in range(3):
        for _ in range(6):
            vector = centres[cluster] + 0.02 * rng.normal(size=8).astype(np.float32)
            embeddings.append(vector / np.linalg.norm(vector))
            categories.append(cluster)
    embeddings_array = np.asarray(embeddings, dtype=np.float32)
    # The most relevant items are all in cluster 0, so a pure-relevance top-5 is degenerate.
    scores = np.concatenate([np.linspace(1.0, 0.8, 6), np.linspace(0.5, 0.3, 6), np.linspace(0.4, 0.2, 6)])
    return scores, embeddings_array, np.asarray(categories, dtype=np.int64)


def test_lambda_one_is_exactly_the_relevance_order() -> None:
    scores, embeddings, _ = clustered_candidates()
    selected = mmr_select(scores, embeddings, k=5, lambda_=1.0)
    assert selected.tolist() == np.argsort(-scores)[:5].tolist()


def test_lower_lambda_increases_category_diversity() -> None:
    scores, embeddings, categories = clustered_candidates()
    relevance_only = mmr_select(scores, embeddings, k=5, lambda_=1.0)
    diversified = mmr_select(scores, embeddings, k=5, lambda_=0.5)

    assert intra_list_diversity(categories[relevance_only], 5) < intra_list_diversity(
        categories[diversified], 5
    )
    assert mean_pairwise_distance(embeddings[relevance_only]) < mean_pairwise_distance(
        embeddings[diversified]
    )


def test_selection_has_no_duplicates_and_respects_k() -> None:
    scores, embeddings, _ = clustered_candidates()
    for lambda_ in (1.0, 0.8, 0.5, 0.0):
        selected = mmr_select(scores, embeddings, k=7, lambda_=lambda_)
        assert selected.size == 7
        assert len(set(selected.tolist())) == 7


def test_k_larger_than_the_candidate_list_is_clamped() -> None:
    scores, embeddings, _ = clustered_candidates()
    selected = mmr_select(scores, embeddings, k=100, lambda_=0.7)
    assert selected.size == scores.size


def test_mmr_scores_reproduce_the_mmr_order_under_sorting() -> None:
    scores, embeddings, _ = clustered_candidates()
    order = mmr_select(scores, embeddings, k=len(scores), lambda_=0.6)
    reordered = mmr_scores(scores, embeddings, 0.6)
    assert np.argsort(-reordered).tolist() == order.tolist()


def test_normalise_handles_a_constant_score_column() -> None:
    assert normalise(np.full(4, 3.0)).tolist() == [0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(normalise(np.array([0.0, 5.0])), [0.0, 1.0])


def test_diversity_metrics_are_bounded() -> None:
    _, embeddings, categories = clustered_candidates()
    assert 0.0 < intra_list_diversity(categories[:5], 5) <= 1.0
    assert mean_pairwise_distance(embeddings[:1]) == pytest.approx(0.0)
    assert 0.0 <= mean_pairwise_distance(embeddings[:6]) <= 2.0
