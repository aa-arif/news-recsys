"""Article text embeddings.

Articles are represented by the sentence-transformer embedding of ``title. abstract``.
This is the single most important modelling choice in the project: measured in M1, 71% of
the articles in the test fold never appear in a training impression, so an ID-embedding
model would be guessing on most of the test set, while a text embedding exists for an
article the moment it is published.

The embedding matrix is row-aligned with :class:`~news_recsys.features.vocab.Vocabulary`
and stored as float32 ``.npy`` so both training and the ONNX serving path memory-map the
exact same vectors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from news_recsys.config import Settings, get_settings
from news_recsys.data.splits import load_news
from news_recsys.features.vocab import Vocabulary
from news_recsys.logging_utils import get_logger, timed

logger = get_logger("features.text")

EMBEDDING_FILENAME = "news_embeddings.npy"


def article_texts(news: pl.DataFrame) -> list[str]:
    """``title. abstract`` per article, in catalogue order."""
    return news.select(
        pl.when(pl.col("abstract").str.len_chars() > 0)
        .then(pl.col("title") + pl.lit(". ") + pl.col("abstract"))
        .otherwise(pl.col("title"))
        .alias("text")
    )["text"].to_list()


def embed_articles(
    settings: Settings | None = None,
    *,
    vocabulary: Vocabulary | None = None,
    normalize: bool = True,
) -> NDArray[np.float32]:
    """Embed every article in the catalogue. Returns a ``(n_news, text_dim)`` matrix."""
    from sentence_transformers import SentenceTransformer  # imported lazily: heavy

    settings = settings or get_settings()
    news = load_news(settings)
    if vocabulary is not None and news["news_id"].to_list() != vocabulary.news_ids:
        raise ValueError("news catalogue and vocabulary are out of sync; rebuild the vocabulary")

    texts = article_texts(news)
    model = SentenceTransformer(settings.text_model, device="cpu")
    model.max_seq_length = settings.text_max_tokens

    with timed(logger, f"embedding {len(texts)} articles with {settings.text_model}"):
        embeddings = model.encode(
            texts,
            batch_size=settings.embed_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def save_embeddings(embeddings: NDArray[np.float32], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / EMBEDDING_FILENAME
    np.save(path, embeddings)
    logger.info("wrote %s %s (%.1f MiB)", path, embeddings.shape, path.stat().st_size / 1024 / 1024)
    return path


def load_embeddings(settings: Settings | None = None, *, mmap: bool = True) -> NDArray[np.float32]:
    settings = settings or get_settings()
    path = settings.artifact_dir / EMBEDDING_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/embed_news.py first")
    return np.load(path, mmap_mode="r" if mmap else None)
