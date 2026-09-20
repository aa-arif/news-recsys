"""M3: train the two-tower retrieval model (in-batch softmax + logQ correction)."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from news_recsys.config import get_settings, seed_everything
from news_recsys.data.sequences import build_click_sequences
from news_recsys.features.text import load_embeddings
from news_recsys.features.vocab import load_vocabulary
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger, timed
from news_recsys.models.two_tower import TwoTowerConfig
from news_recsys.models.two_tower_train import (
    SequenceBatcher,
    encode_all_items,
    log_sampling_probabilities,
    train_two_tower,
)

logger = get_logger("scripts.train_two_tower")

ITEM_VECTORS_FILENAME = "item_vectors.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-sample", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=0, help="0 keeps torch's default")
    args = parser.parse_args()

    overrides = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.epochs:
        overrides["two_tower_epochs"] = args.epochs
    if args.batch_size:
        overrides["two_tower_batch_size"] = args.batch_size
    settings = get_settings(**overrides)
    settings.ensure_dirs()
    seed_everything(settings.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    vocabulary = load_vocabulary(settings)
    embeddings = np.asarray(load_embeddings(settings, mmap=False), dtype=np.float32)

    with timed(logger, "build click sequences"):
        train = build_click_sequences("train", settings, vocabulary=vocabulary)
        validation = build_click_sequences("val", settings, vocabulary=vocabulary)

    config = TwoTowerConfig(text_dim=embeddings.shape[1], output_dim=settings.two_tower_dim)
    with timed(logger, "train two-tower") as timing:
        model, history, selection = train_two_tower(
            train,
            validation,
            embeddings,
            vocabulary,
            settings,
            config=config,
            val_sample=args.val_sample,
        )

    model.save(settings.artifact_dir, vocabulary, settings)

    batcher = SequenceBatcher(
        embeddings, vocabulary, log_sampling_probabilities(train.positive, vocabulary.n_news)
    )
    with timed(logger, "encode catalogue"):
        item_vectors = encode_all_items(model, batcher, vocabulary.n_news)
    np.save(settings.artifact_dir / ITEM_VECTORS_FILENAME, item_vectors)

    write_json(
        settings.metrics_dir / f"two_tower_{settings.dataset}.json",
        {
            "dataset": settings.dataset,
            "config": config.to_dict(),
            "epochs": settings.two_tower_epochs,
            "batch_size": settings.two_tower_batch_size,
            "learning_rate": settings.two_tower_lr,
            "max_history": settings.max_history,
            "train_clicks": len(train),
            "val_clicks": len(validation),
            "train_seconds": timing["seconds"],
            "torch_threads": torch.get_num_threads(),
            "selection": selection,
            "training_curve": history.epochs,
            "item_vectors": {
                "path": str(
                    (settings.artifact_dir / ITEM_VECTORS_FILENAME).relative_to(settings.root_dir)
                ),
                "shape": list(item_vectors.shape),
            },
        },
    )
    logger.info(
        "best epoch %s (val recall@100 %.4f)",
        selection["best_epoch"],
        selection["best_val_recall@100"],
    )


if __name__ == "__main__":
    main()
