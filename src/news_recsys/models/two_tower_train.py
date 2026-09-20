"""Training loop for the two-tower retrieval model.

Kept out of the script so the smoke test, the unit tests and the real run all drive the
same code. Model selection is by validation Recall@100 computed with exact search, so the
retrieval quality of the *model* is never confused with the approximation error of the
ANN index (which M3 measures separately).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from news_recsys.config import Settings
from news_recsys.data.sequences import ClickSequences
from news_recsys.features.vocab import Vocabulary
from news_recsys.logging_utils import get_logger
from news_recsys.models.two_tower import TwoTowerConfig, TwoTowerModel, shifted_category_ids

logger = get_logger("models.two_tower_train")


@dataclass
class TrainingHistory:
    epochs: list[dict[str, float]] = field(default_factory=list)

    def add(self, **values: float) -> None:
        self.epochs.append(values)


def log_sampling_probabilities(positives: NDArray[np.int64], n_items: int) -> NDArray[np.float64]:
    """``log P(item)`` estimated from click frequency, for the logQ correction.

    In-batch negatives are other rows' positives, so an item's chance of appearing as a
    negative is its share of all clicks. Items never clicked in training get the floor,
    which is the right prior for an item the sampler has never produced.
    """
    counts = np.bincount(positives, minlength=n_items).astype(np.float64)
    total = counts.sum()
    floor = 1.0 / max(total, 1.0)
    probabilities = np.maximum(counts / max(total, 1.0), floor)
    return np.log(probabilities)


class SequenceBatcher:
    """Assembles tensors for a batch of click sequences."""

    def __init__(
        self,
        embeddings: NDArray[np.float32],
        vocabulary: Vocabulary,
        log_probabilities: NDArray[np.float64],
    ) -> None:
        self.embeddings = embeddings
        self.vocabulary = vocabulary
        self.log_probabilities = log_probabilities

    def history_tensors(
        self, history: NDArray[np.int64], mask: NDArray[np.float32]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        safe = np.where(history >= 0, history, 0)
        text = self.embeddings[safe] * mask[..., None]
        category, subcategory = shifted_category_ids(history, self.vocabulary)
        return (
            torch.from_numpy(np.ascontiguousarray(text, dtype=np.float32)),
            torch.from_numpy(category),
            torch.from_numpy(subcategory),
            torch.from_numpy(mask),
        )

    def item_tensors(self, items: NDArray[np.int64]) -> tuple[Tensor, Tensor, Tensor]:
        safe = np.where(items >= 0, items, 0)
        category, subcategory = shifted_category_ids(items, self.vocabulary)
        return (
            torch.from_numpy(np.ascontiguousarray(self.embeddings[safe], dtype=np.float32)),
            torch.from_numpy(category),
            torch.from_numpy(subcategory),
        )

    def batch_log_probabilities(self, items: NDArray[np.int64]) -> Tensor:
        return torch.from_numpy(self.log_probabilities[items].astype(np.float32))


@torch.no_grad()
def encode_all_items(
    model: TwoTowerModel, batcher: SequenceBatcher, n_items: int, *, batch_size: int = 4096
) -> NDArray[np.float32]:
    """Item vectors for the whole catalogue (what the FAISS index is built from)."""
    model.eval()
    vectors = np.empty((n_items, model.config.output_dim), dtype=np.float32)
    for start in range(0, n_items, batch_size):
        items = np.arange(start, min(start + batch_size, n_items), dtype=np.int64)
        text, category, subcategory = batcher.item_tensors(items)
        vectors[start : start + items.size] = model.item_tower(text, category, subcategory).numpy()
    return vectors


@torch.no_grad()
def encode_users(
    model: TwoTowerModel,
    batcher: SequenceBatcher,
    history: NDArray[np.int64],
    mask: NDArray[np.float32],
    *,
    batch_size: int = 1024,
) -> NDArray[np.float32]:
    model.eval()
    vectors = np.empty((history.shape[0], model.config.output_dim), dtype=np.float32)
    for start in range(0, history.shape[0], batch_size):
        stop = min(start + batch_size, history.shape[0])
        text, category, subcategory, batch_mask = batcher.history_tensors(
            history[start:stop], mask[start:stop]
        )
        vectors[start:stop] = model.user_tower(text, category, subcategory, batch_mask).numpy()
    return vectors


def exact_recall_at_k(
    user_vectors: NDArray[np.float32],
    item_vectors: NDArray[np.float32],
    positives: NDArray[np.int64],
    cutoffs: tuple[int, ...],
    *,
    chunk: int = 512,
) -> dict[int, float]:
    """Recall@k over the **full catalogue** with exact search (the retrieval ceiling)."""
    largest = max(cutoffs)
    hits = dict.fromkeys(cutoffs, 0)
    for start in range(0, user_vectors.shape[0], chunk):
        stop = min(start + chunk, user_vectors.shape[0])
        scores = user_vectors[start:stop] @ item_vectors.T
        top = np.argpartition(-scores, kth=largest - 1, axis=1)[:, :largest]
        ordered_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-ordered_scores, axis=1)
        ranked = np.take_along_axis(top, order, axis=1)
        target = positives[start:stop][:, None]
        for cutoff in cutoffs:
            hits[cutoff] += int((ranked[:, :cutoff] == target).any(axis=1).sum())
    total = max(user_vectors.shape[0], 1)
    return {cutoff: hits[cutoff] / total for cutoff in cutoffs}


def train_two_tower(
    train: ClickSequences,
    validation: ClickSequences,
    embeddings: NDArray[np.float32],
    vocabulary: Vocabulary,
    settings: Settings,
    *,
    config: TwoTowerConfig | None = None,
    val_sample: int = 5000,
    log_every: int = 100,
) -> tuple[TwoTowerModel, TrainingHistory, dict[str, Any]]:
    """Train with in-batch softmax + logQ correction; select the epoch on val Recall@100."""
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)

    config = config or TwoTowerConfig(
        text_dim=embeddings.shape[1], output_dim=settings.two_tower_dim
    )
    model = TwoTowerModel(config, vocabulary.n_categories, vocabulary.n_subcategories)
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.two_tower_lr, weight_decay=1e-5)

    log_probabilities = log_sampling_probabilities(train.positive, vocabulary.n_news)
    batcher = SequenceBatcher(embeddings, vocabulary, log_probabilities)

    # A fixed validation subsample keeps per-epoch monitoring cheap and comparable.
    sample = rng.choice(len(validation), size=min(val_sample, len(validation)), replace=False)
    history = TrainingHistory()
    best_state: dict[str, Tensor] | None = None
    best_recall = -1.0
    best_epoch = -1

    n_samples = len(train)
    batch_size = settings.two_tower_batch_size
    for epoch in range(settings.two_tower_epochs):
        model.train()
        order = rng.permutation(n_samples)
        running_loss, seen_batches = 0.0, 0

        for step, start in enumerate(range(0, n_samples, batch_size)):
            rows = order[start : start + batch_size]
            if rows.size < 8:  # a degenerate last batch makes in-batch softmax meaningless
                continue
            text, category, subcategory, mask = batcher.history_tensors(
                train.history[rows], train.mask[rows]
            )
            item_text, item_category, item_subcategory = batcher.item_tensors(train.positive[rows])

            user_vectors, item_vectors = model(
                text, category, subcategory, mask, item_text, item_category, item_subcategory
            )
            loss = model.in_batch_loss(
                user_vectors, item_vectors, batcher.batch_log_probabilities(train.positive[rows])
            )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            running_loss += float(loss.detach())
            seen_batches += 1
            if log_every and step % log_every == 0:
                logger.info("epoch %d step %4d loss %.4f", epoch, step, float(loss.detach()))

        item_vectors_all = encode_all_items(model, batcher, vocabulary.n_news)
        user_vectors_val = encode_users(
            model, batcher, validation.history[sample], validation.mask[sample]
        )
        recalls = exact_recall_at_k(
            user_vectors_val, item_vectors_all, validation.positive[sample], (10, 100)
        )
        epoch_loss = running_loss / max(seen_batches, 1)
        history.add(
            epoch=epoch, loss=epoch_loss, **{f"val_recall@{k}": v for k, v in recalls.items()}
        )
        logger.info(
            "epoch %d | loss %.4f | val recall@10 %.4f | val recall@100 %.4f",
            epoch,
            epoch_loss,
            recalls[10],
            recalls[100],
        )

        if recalls[100] > best_recall:
            best_recall = recalls[100]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    selection = {
        "best_epoch": best_epoch,
        "best_val_recall@100": best_recall,
        "val_subsample": int(sample.size),
        "epochs": history.epochs,
    }
    return model, history, selection
