"""Everything the server loads once at startup.

Loading is explicit and eager: if an artifact is missing, the process fails at boot
rather than on the first request. The embedding matrix is memory-mapped, so N workers on
one box share one copy of the 100 MB of article vectors instead of each paying for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from news_recsys.config import Settings, get_settings
from news_recsys.data.splits import load_news
from news_recsys.features.build import SNAPSHOT_FILENAME
from news_recsys.features.text import load_embeddings
from news_recsys.features.vocab import Vocabulary, load_vocabulary
from news_recsys.logging_utils import get_logger
from news_recsys.models.calibration import PlattCalibrator, PriorCorrection
from news_recsys.models.onnx_export import RANKER_FILENAME, USER_TOWER_FILENAME, make_session
from news_recsys.retrieval.faiss_index import ItemIndex, set_search_threads
from news_recsys.serving.redis_store import ArticleStatic

logger = get_logger("serving.state")


@dataclass
class ServingArtifacts:
    """Immutable, process-wide serving state."""

    settings: Settings
    vocabulary: Vocabulary
    embeddings: NDArray[np.float32]
    static: ArticleStatic
    index: ItemIndex
    user_tower: ort.InferenceSession
    ranker: ort.InferenceSession
    calibrator: PlattCalibrator
    prior: PriorCorrection
    snapshot_as_of: float
    news_ids: list[str]
    titles: list[str]
    categories: list[str]

    @classmethod
    def load(cls, settings: Settings | None = None, *, search_threads: int = 1) -> ServingArtifacts:
        settings = settings or get_settings()
        directory = settings.artifact_dir
        vocabulary = load_vocabulary(settings)
        embeddings = load_embeddings(settings, mmap=True)
        news = load_news(settings)

        static = ArticleStatic(
            category=vocabulary.news_category.astype(np.int64),
            subcategory=vocabulary.news_subcategory.astype(np.int64),
            title_chars=news["title"].str.len_chars().to_numpy().astype(np.float64),
            abstract_chars=news["abstract"].str.len_chars().to_numpy().astype(np.float64),
            title_entities=news["n_title_entities"].to_numpy().astype(np.float64),
            abstract_entities=news["n_abstract_entities"].to_numpy().astype(np.float64),
        )

        # FAISS parallelises across queries in a batch; a server wants each request to
        # use one thread and concurrency to come from requests, not from inside one.
        set_search_threads(search_threads)
        index = ItemIndex.load(directory, settings)

        snapshot = np.load(directory / SNAPSHOT_FILENAME)
        calibrator_path = directory / "ranker_calibrator.json"
        calibrator = (
            PlattCalibrator.load(calibrator_path) if calibrator_path.exists() else PlattCalibrator()
        )

        artifacts = cls(
            settings=settings,
            vocabulary=vocabulary,
            embeddings=embeddings,
            static=static,
            index=index,
            user_tower=make_session(directory / USER_TOWER_FILENAME, settings),
            ranker=make_session(directory / RANKER_FILENAME, settings),
            calibrator=calibrator,
            prior=PriorCorrection(negative_keep_rate=settings.ranker_negative_sample_rate),
            snapshot_as_of=float(snapshot["as_of"][0]),
            news_ids=vocabulary.news_ids,
            titles=news["title"].to_list(),
            categories=news["category"].to_list(),
        )
        logger.info(
            "serving artifacts: %d articles, %d index vectors, snapshot as_of=%.0f",
            vocabulary.n_news,
            index.size,
            artifacts.snapshot_as_of,
        )
        return artifacts

    def describe(self) -> dict[str, Any]:
        return {
            "dataset": self.settings.dataset,
            "articles": self.vocabulary.n_news,
            "index_vectors": self.index.size,
            "ef_search": self.index.ef_search,
            "retrieval_candidates": self.settings.retrieval_candidates,
            "snapshot_as_of": self.snapshot_as_of,
            "ort_intra_op_threads": self.settings.ort_intra_op_threads,
            "ort_inter_op_threads": self.settings.ort_inter_op_threads,
            "user_embedding_cache_size": self.settings.user_embedding_cache_size,
        }


def artifact_paths(settings: Settings) -> dict[str, Path]:
    directory = settings.artifact_dir
    return {
        "vocabulary": directory / "vocab.json",
        "embeddings": directory / "news_embeddings.npy",
        "index": directory / "items_hnsw.faiss",
        "user_tower_onnx": directory / USER_TOWER_FILENAME,
        "ranker_onnx": directory / RANKER_FILENAME,
        "snapshot": directory / SNAPSHOT_FILENAME,
    }


def missing_artifacts(settings: Settings) -> list[str]:
    return [name for name, path in artifact_paths(settings).items() if not path.exists()]
