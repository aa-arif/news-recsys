"""M3: build the FAISS HNSW index over the item tower's vectors."""

from __future__ import annotations

import argparse
import time

import numpy as np

from news_recsys.config import get_settings
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger
from news_recsys.retrieval.faiss_index import ItemIndex

logger = get_logger("scripts.build_index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--m", type=int, default=None, help="HNSW graph degree")
    parser.add_argument("--ef-construction", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()

    vectors = np.load(settings.artifact_dir / "item_vectors.npy")
    m = args.m or settings.faiss_hnsw_m
    ef_construction = args.ef_construction or settings.faiss_ef_construction

    start = time.perf_counter()
    index = ItemIndex.build(vectors, m=m, ef_construction=ef_construction)
    build_seconds = time.perf_counter() - start
    path = index.save(settings.artifact_dir)

    write_json(
        settings.metrics_dir / f"index_{settings.dataset}.json",
        {
            "dataset": settings.dataset,
            "vectors": int(vectors.shape[0]),
            "dim": int(vectors.shape[1]),
            "hnsw_m": m,
            "ef_construction": ef_construction,
            "build_seconds": build_seconds,
            "index_megabytes": round(path.stat().st_size / 1024 / 1024, 2),
        },
    )
    logger.info("index of %d vectors built in %.1fs", vectors.shape[0], build_seconds)


if __name__ == "__main__":
    main()
