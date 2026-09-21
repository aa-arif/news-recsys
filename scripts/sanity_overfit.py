"""Can the two-tower overfit a tiny training subset?

Before concluding anything from a low Recall@K, rule out the boring explanations: a broken
target, a scrambled index, a loss that cannot drive the objective. A model that *cannot*
memorise a few hundred examples has a bug; a model that can, but generalises poorly, has a
capacity, data or training-budget problem. Those need completely different fixes, and the
difference is one cheap experiment.

The subset is trained with regularisation off (dropout 0) because the question is capacity,
not generalisation, and recall is measured on the *same* examples the model trained on,
retrieving from the full 65,238-article catalogue with exact search. Chance level for
Recall@10 is 10/65,238 = 0.015%.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from news_recsys.config import get_settings, seed_everything
from news_recsys.data.sequences import ClickSequences, build_click_sequences
from news_recsys.features.text import load_embeddings
from news_recsys.features.vocab import load_vocabulary
from news_recsys.io_utils import write_json
from news_recsys.logging_utils import get_logger, timed
from news_recsys.models.two_tower import TwoTowerConfig, TwoTowerModel
from news_recsys.models.two_tower_train import (
    SequenceBatcher,
    encode_all_items,
    encode_users,
    exact_recall_at_k,
    log_sampling_probabilities,
)

logger = get_logger("scripts.sanity_overfit")


def subset(sequences: ClickSequences, rows: np.ndarray) -> ClickSequences:
    return ClickSequences(
        history=sequences.history[rows],
        mask=sequences.mask[rows],
        positive=sequences.positive[rows],
        user_index=sequences.user_index[rows],
        impression_key=sequences.impression_key[rows],
        timestamp=sequences.timestamp[rows],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=["small", "large", "synthetic"])
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=25)
    args = parser.parse_args()

    settings = get_settings(dataset=args.dataset) if args.dataset else get_settings()
    settings.ensure_dirs()
    seed_everything(settings.seed)
    rng = np.random.default_rng(settings.seed)

    vocabulary = load_vocabulary(settings)
    embeddings = np.asarray(load_embeddings(settings, mmap=False), dtype=np.float32)
    train = build_click_sequences("train", settings, vocabulary=vocabulary)

    rows = rng.choice(len(train), size=min(args.examples, len(train)), replace=False)
    tiny = subset(train, rows)
    logger.info(
        "overfitting %d clicks (%d distinct target articles) out of a %d-article catalogue",
        len(tiny),
        int(np.unique(tiny.positive).size),
        vocabulary.n_news,
    )

    config = TwoTowerConfig(
        text_dim=embeddings.shape[1], output_dim=settings.two_tower_dim, dropout=args.dropout
    )
    model = TwoTowerModel(config, vocabulary.n_categories, vocabulary.n_subcategories)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    batcher = SequenceBatcher(
        embeddings, vocabulary, log_sampling_probabilities(tiny.positive, vocabulary.n_news)
    )

    curve: list[dict[str, float]] = []
    started = time.perf_counter()

    with timed(logger, f"overfit {len(tiny)} examples for {args.epochs} epochs"):
        for epoch in range(args.epochs + 1):
            if epoch % args.eval_every == 0:
                item_vectors = encode_all_items(model, batcher, vocabulary.n_news)
                user_vectors = encode_users(model, batcher, tiny.history, tiny.mask)
                recalls = exact_recall_at_k(user_vectors, item_vectors, tiny.positive, (1, 10, 100))
                point = {
                    "epoch": epoch,
                    "seconds": time.perf_counter() - started,
                    **{f"train_recall@{k}": value for k, value in recalls.items()},
                }
                if curve:
                    point["loss"] = curve[-1].get("loss", float("nan"))
                curve.append(point)
                logger.info(
                    "epoch %3d | train recall@1 %.3f @10 %.3f @100 %.3f",
                    epoch,
                    recalls[1],
                    recalls[10],
                    recalls[100],
                )
            if epoch == args.epochs:
                break

            model.train()
            order = rng.permutation(len(tiny))
            running, steps = 0.0, 0
            for start in range(0, order.size, args.batch_size):
                batch_rows = order[start : start + args.batch_size]
                if batch_rows.size < 8:
                    continue
                text, category, subcategory, mask = batcher.history_tensors(
                    tiny.history[batch_rows], tiny.mask[batch_rows]
                )
                item_text, item_category, item_subcategory = batcher.item_tensors(
                    tiny.positive[batch_rows]
                )
                user_vectors, item_vectors = model(
                    text, category, subcategory, mask, item_text, item_category, item_subcategory
                )
                loss = model.in_batch_loss(
                    user_vectors,
                    item_vectors,
                    batcher.batch_log_probabilities(tiny.positive[batch_rows]),
                )
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()
                running += float(loss.detach())
                steps += 1
            if curve:
                curve[-1]["loss"] = running / max(steps, 1)

    final = curve[-1]
    chance = 10 / vocabulary.n_news
    verdict = (
        "model can memorise the subset: the architecture, loss and index plumbing are sound, "
        "so low validation recall is a generalisation/capacity/training-budget problem"
        if final["train_recall@10"] > 0.5
        else "model cannot memorise even a tiny subset: suspect a bug in the target, the "
        "batching or the loss before spending more compute on training"
    )
    logger.info("%s", verdict)

    write_json(
        settings.metrics_dir / f"two_tower_overfit_{settings.dataset}.json",
        {
            "dataset": settings.dataset,
            "examples": len(tiny),
            "distinct_targets": int(np.unique(tiny.positive).size),
            "catalogue": vocabulary.n_news,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "dropout": args.dropout,
            "chance_recall@10": chance,
            "final": final,
            "curve": curve,
            "verdict": verdict,
        },
    )


if __name__ == "__main__":
    main()
