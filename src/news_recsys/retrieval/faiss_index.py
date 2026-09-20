"""FAISS HNSW index over the item tower's output vectors.

HNSW rather than IVF because the catalogue is small enough to hold in memory, the graph
gives a smooth recall/latency dial (``efSearch``) that can be tuned per deployment, and
it needs no training step - which matters for a corpus where most articles are hours old.

Item vectors are L2-normalised, so inner product is cosine similarity and the index is
built with ``METRIC_INNER_PRODUCT``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np
from numpy.typing import NDArray

from news_recsys.config import Settings, get_settings
from news_recsys.logging_utils import get_logger

logger = get_logger("retrieval.faiss")

INDEX_FILENAME = "items_hnsw.faiss"
IDMAP_FILENAME = "items_index_ids.npy"


@dataclass
class SearchResult:
    indices: NDArray[np.int64]
    scores: NDArray[np.float32]
    seconds: float


class ItemIndex:
    """HNSW index plus the row -> article-index mapping."""

    def __init__(self, index: faiss.Index, item_ids: NDArray[np.int64]) -> None:
        self.index = index
        self.item_ids = item_ids

    # -- construction -------------------------------------------------------
    @classmethod
    def build(
        cls,
        vectors: NDArray[np.float32],
        *,
        item_ids: NDArray[np.int64] | None = None,
        m: int = 32,
        ef_construction: int = 200,
    ) -> ItemIndex:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        start = time.perf_counter()
        index.add(vectors)
        logger.info(
            "built HNSW (M=%d, efConstruction=%d) over %d vectors in %.1fs",
            m,
            ef_construction,
            vectors.shape[0],
            time.perf_counter() - start,
        )
        ids = item_ids if item_ids is not None else np.arange(vectors.shape[0], dtype=np.int64)
        return cls(index, ids)

    @property
    def _hnsw(self) -> Any:
        # faiss's python stubs type read_index() as the base Index, which has no `hnsw`.
        return cast(Any, self.index).hnsw

    @property
    def ef_search(self) -> int:
        return int(self._hnsw.efSearch)

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        self._hnsw.efSearch = int(value)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    # -- search -------------------------------------------------------------
    def search(
        self, queries: NDArray[np.float32], k: int, *, ef_search: int | None = None
    ) -> SearchResult:
        if ef_search is not None:
            self.ef_search = ef_search
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        start = time.perf_counter()
        scores, rows = self.index.search(queries, k)
        elapsed = time.perf_counter() - start
        # -1 marks "fewer than k neighbours found"; map it through without corrupting ids.
        indices = np.where(rows >= 0, self.item_ids[np.maximum(rows, 0)], -1)
        return SearchResult(indices=indices, scores=scores, seconds=elapsed)

    # -- persistence --------------------------------------------------------
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / INDEX_FILENAME
        faiss.write_index(self.index, str(path))
        np.save(directory / IDMAP_FILENAME, self.item_ids)
        logger.info("wrote %s (%.1f MiB)", path, path.stat().st_size / 1024 / 1024)
        return path

    @classmethod
    def load(cls, directory: Path, settings: Settings | None = None) -> ItemIndex:
        settings = settings or get_settings()
        index = faiss.read_index(str(directory / INDEX_FILENAME))
        item_ids = np.load(directory / IDMAP_FILENAME)
        instance = cls(index, item_ids)
        instance.ef_search = settings.faiss_ef_search
        return instance


def exact_search(
    vectors: NDArray[np.float32], queries: NDArray[np.float32], k: int, *, chunk: int = 512
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Brute-force top-k, used as the ground truth the HNSW recall is measured against."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    indices = np.empty((queries.shape[0], k), dtype=np.int64)
    scores = np.empty((queries.shape[0], k), dtype=np.float32)
    for start in range(0, queries.shape[0], chunk):
        stop = min(start + chunk, queries.shape[0])
        block = queries[start:stop] @ vectors.T
        top = np.argpartition(-block, kth=k - 1, axis=1)[:, :k]
        block_scores = np.take_along_axis(block, top, axis=1)
        order = np.argsort(-block_scores, axis=1)
        indices[start:stop] = np.take_along_axis(top, order, axis=1)
        scores[start:stop] = np.take_along_axis(block_scores, order, axis=1)
    return indices, scores


def set_search_threads(threads: int) -> None:
    """FAISS parallelises a *batch* of queries; serving issues one query at a time."""
    faiss.omp_set_num_threads(threads)
