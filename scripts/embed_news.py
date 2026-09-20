"""M2: build the vocabulary and embed every article's text."""

from __future__ import annotations

import argparse

from news_recsys.config import get_settings, seed_everything
from news_recsys.features.text import embed_articles, save_embeddings
from news_recsys.features.vocab import build_vocabulary
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger, timed

logger = get_logger("scripts.embed_news")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    seed_everything(settings.seed)

    with timed(logger, "build vocabulary") as vocab_timing:
        vocabulary = build_vocabulary(settings)
        vocabulary.save(settings.artifact_dir)

    with timed(logger, "embed articles") as embed_timing:
        embeddings = embed_articles(settings, vocabulary=vocabulary)
        save_embeddings(embeddings, settings.artifact_dir)

    write_json(
        settings.metrics_dir / f"embeddings_{settings.dataset}.json",
        {
            "model": settings.text_model,
            "dim": int(embeddings.shape[1]),
            "articles": int(embeddings.shape[0]),
            "max_tokens": settings.text_max_tokens,
            "batch_size": settings.embed_batch_size,
            "device": "cpu",
            "vocab_seconds": vocab_timing["seconds"],
            "embed_seconds": embed_timing["seconds"],
            "articles_per_second": float(embeddings.shape[0]) / max(embed_timing["seconds"], 1e-9),
            "users": vocabulary.n_users,
            "categories": vocabulary.n_categories,
            "subcategories": vocabulary.n_subcategories,
        },
    )


if __name__ == "__main__":
    main()
