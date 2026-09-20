"""Maximal Marginal Relevance re-ranking for topical diversity.

A pure relevance ranking on news collapses: the top 10 becomes ten versions of the same
story, because the features that make one story attractive make all of its near-duplicates
attractive too. MMR trades a little relevance for spread:

    pick argmax_i  lambda * relevance(i) - (1 - lambda) * max_{j in selected} sim(i, j)

``lambda = 1`` is the untouched ranker. Lower values push the list apart. Similarity is
the cosine between article text embeddings, so "similar" means *about the same thing*,
not merely "same category" - two different stories filed under `sports` are allowed to
co-exist, two takes on the same game are not.

Relevance is min-max normalised per impression so that ``lambda`` means the same thing
across requests with different score scales.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalise(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Min-max to [0, 1] within one candidate list."""
    low, high = float(scores.min()), float(scores.max())
    if high - low < 1e-12:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def mmr_select(
    scores: NDArray[np.float64],
    embeddings: NDArray[np.float32],
    k: int,
    lambda_: float,
) -> NDArray[np.int64]:
    """Return the indices of the ``k`` selected candidates, in presentation order."""
    n = scores.shape[0]
    k = min(k, n)
    if lambda_ >= 1.0:
        return np.argsort(-scores)[:k].astype(np.int64)

    relevance = normalise(np.asarray(scores, dtype=np.float64))
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    similarity = vectors @ vectors.T

    selected: list[int] = []
    remaining = np.ones(n, dtype=bool)
    max_similarity = np.zeros(n, dtype=np.float64)

    for _ in range(k):
        objective = lambda_ * relevance - (1.0 - lambda_) * max_similarity
        objective[~remaining] = -np.inf
        choice = int(np.argmax(objective))
        selected.append(choice)
        remaining[choice] = False
        max_similarity = np.maximum(max_similarity, similarity[choice].astype(np.float64))

    return np.asarray(selected, dtype=np.int64)


def mmr_scores(
    scores: NDArray[np.float64],
    embeddings: NDArray[np.float32],
    lambda_: float,
) -> NDArray[np.float64]:
    """Scores that reproduce the MMR order under a plain descending sort.

    Re-ranking has to be expressible as a score so the existing evaluation path (which
    ranks by score) can measure it without a second, parallel implementation.
    """
    order = mmr_select(scores, embeddings, len(scores), lambda_)
    reordered = np.empty_like(scores, dtype=np.float64)
    reordered[order] = np.arange(len(order), 0, -1, dtype=np.float64)
    return reordered


def intra_list_diversity(categories: NDArray[np.int64], k: int) -> float:
    """Distinct categories in the top-k, divided by k."""
    top = categories[:k]
    if top.size == 0:
        return 0.0
    return float(len(set(top.tolist())) / min(k, top.size))


def mean_pairwise_distance(embeddings: NDArray[np.float32]) -> float:
    """1 - mean pairwise cosine similarity of the presented list."""
    if embeddings.shape[0] < 2:
        return 0.0
    vectors = np.asarray(embeddings, dtype=np.float64)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    similarity = vectors @ vectors.T
    n = similarity.shape[0]
    off_diagonal = (similarity.sum() - np.trace(similarity)) / (n * (n - 1))
    return float(1.0 - off_diagonal)
