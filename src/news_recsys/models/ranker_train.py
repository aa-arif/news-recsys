"""Training loop and batching for the DIN + DCN-v2 ranker.

Batches are formed from whole **impressions**, not shuffled rows. Three reasons:

1. the history only has to be encoded once per impression instead of once per candidate
   (~37x less work on MIND-small), which is what makes CPU training feasible;
2. negative downsampling is then a per-impression decision, so every impression keeps its
   positives and the label distribution inside a slate stays interpretable;
3. it matches serving, where one request carries one history and many candidates.

Negatives are subsampled at ``ranker_negative_sample_rate`` and resampled every epoch, so
across epochs the model still sees most of the negative pool. The resulting probability
bias is corrected in closed form at inference time by
:class:`~news_recsys.models.calibration.PriorCorrection`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from news_recsys.config import Settings
from news_recsys.data.sequences import ImpressionHistories
from news_recsys.eval.metrics import evaluate_ranking, group_boundaries
from news_recsys.features.build import FoldFeatures
from news_recsys.features.vocab import Vocabulary
from news_recsys.logging_utils import get_logger
from news_recsys.models.ranker import DinDcnRanker, RankerConfig

logger = get_logger("models.ranker_train")


@dataclass
class RankerDataset:
    """A fold arranged for impression-grouped batching."""

    dense: NDArray[np.float32]
    labels: NDArray[np.float32]
    news_index: NDArray[np.int64]
    impression_key: NDArray[np.int64]
    group_of_row: NDArray[np.int64]
    group_starts: NDArray[np.int64]
    group_sizes: NDArray[np.int64]
    history: NDArray[np.int64]
    history_mask: NDArray[np.float32]

    @property
    def n_groups(self) -> int:
        return int(self.group_starts.size)

    @property
    def n_rows(self) -> int:
        return int(self.labels.size)


def build_ranker_dataset(
    fold: FoldFeatures, histories: ImpressionHistories, settings: Settings
) -> RankerDataset:
    """Align a fold's rows with the per-impression history matrices."""
    bounds = group_boundaries(fold.impression_key)
    starts = bounds[:-1]
    sizes = np.diff(bounds)
    group_keys = fold.impression_key[starts]

    history_row = {int(key): row for row, key in enumerate(histories.impression_key)}
    missing = [int(key) for key in group_keys if int(key) not in history_row]
    if missing:
        raise KeyError(f"{len(missing)} impressions have no history row (first: {missing[0]})")
    rows = np.asarray([history_row[int(key)] for key in group_keys], dtype=np.int64)

    max_history = settings.ranker_max_history
    history = histories.history[rows][:, -max_history:]
    mask = histories.mask[rows][:, -max_history:]

    group_of_row = np.repeat(np.arange(starts.size, dtype=np.int64), sizes)
    return RankerDataset(
        dense=fold.features,
        labels=fold.labels.astype(np.float32),
        news_index=fold.news_index.astype(np.int64),
        impression_key=fold.impression_key,
        group_of_row=group_of_row,
        group_starts=starts,
        group_sizes=sizes,
        history=history,
        history_mask=mask,
    )


class RankerBatcher:
    """Turns a set of impressions into the tensors the model consumes."""

    def __init__(
        self, dataset: RankerDataset, embeddings: NDArray[np.float32], vocabulary: Vocabulary
    ) -> None:
        self.dataset = dataset
        self.embeddings = embeddings
        self.vocabulary = vocabulary

    def _category_ids(
        self, news_index: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        known = news_index >= 0
        safe = np.where(known, news_index, 0)
        category = np.where(known, self.vocabulary.news_category[safe].astype(np.int64) + 1, 0)
        subcategory = np.where(
            known, self.vocabulary.news_subcategory[safe].astype(np.int64) + 1, 0
        )
        return category, subcategory

    def batch(
        self,
        groups: NDArray[np.int64],
        *,
        negative_rate: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Tensor]:
        data = self.dataset
        row_blocks = []
        for group in groups:
            start = int(data.group_starts[group])
            stop = start + int(data.group_sizes[group])
            rows = np.arange(start, stop, dtype=np.int64)
            if negative_rate < 1.0:
                labels = data.labels[rows]
                keep = labels > 0
                if rng is not None:
                    keep = keep | (rng.random(rows.size) < negative_rate)
                if not keep.any():
                    continue
                rows = rows[keep]
            row_blocks.append(rows)

        rows = np.concatenate(row_blocks) if row_blocks else np.empty(0, dtype=np.int64)
        group_positions = {int(group): position for position, group in enumerate(groups)}
        group_index = np.asarray(
            [group_positions[int(group)] for group in data.group_of_row[rows]], dtype=np.int64
        )

        news_index = data.news_index[rows]
        category, subcategory = self._category_ids(news_index)
        history = data.history[groups]
        history_category, history_subcategory = self._category_ids(history)
        history_text = (
            self.embeddings[np.where(history >= 0, history, 0)]
            * data.history_mask[groups][..., None]
        )

        return {
            "dense": torch.from_numpy(np.ascontiguousarray(data.dense[rows])),
            "candidate_text": torch.from_numpy(
                np.ascontiguousarray(self.embeddings[np.where(news_index >= 0, news_index, 0)])
            ),
            "candidate_category": torch.from_numpy(category),
            "candidate_subcategory": torch.from_numpy(subcategory),
            "history_text": torch.from_numpy(np.ascontiguousarray(history_text, dtype=np.float32)),
            "history_category": torch.from_numpy(history_category),
            "history_subcategory": torch.from_numpy(history_subcategory),
            "history_mask": torch.from_numpy(np.ascontiguousarray(data.history_mask[groups])),
            "group_index": torch.from_numpy(group_index),
            "labels": torch.from_numpy(np.ascontiguousarray(data.labels[rows])),
            "rows": torch.from_numpy(rows),
        }


@torch.no_grad()
def predict(
    model: DinDcnRanker,
    batcher: RankerBatcher,
    *,
    groups: NDArray[np.int64] | None = None,
    impressions_per_batch: int = 64,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Score every row of the given impressions. Returns (logits, row indices)."""
    model.eval()
    dataset = batcher.dataset
    groups = np.arange(dataset.n_groups, dtype=np.int64) if groups is None else groups
    logits: list[NDArray[np.float64]] = []
    row_ids: list[NDArray[np.int64]] = []

    for start in range(0, groups.size, impressions_per_batch):
        chunk = groups[start : start + impressions_per_batch]
        batch = batcher.batch(chunk)
        if batch["labels"].numel() == 0:
            continue
        output = model(
            batch["dense"],
            batch["candidate_text"],
            batch["candidate_category"],
            batch["candidate_subcategory"],
            batch["history_text"],
            batch["history_category"],
            batch["history_subcategory"],
            batch["history_mask"],
            batch["group_index"],
        )
        logits.append(output.numpy().astype(np.float64))
        row_ids.append(batch["rows"].numpy())

    return np.concatenate(logits), np.concatenate(row_ids)


def evaluate_subset(
    model: DinDcnRanker,
    batcher: RankerBatcher,
    groups: NDArray[np.int64],
    settings: Settings,
    *,
    impressions_per_batch: int = 64,
) -> dict[str, float]:
    logits, rows = predict(
        model, batcher, groups=groups, impressions_per_batch=impressions_per_batch
    )
    dataset = batcher.dataset
    report = evaluate_ranking(
        dataset.labels[rows], logits, dataset.impression_key[rows], cutoffs=settings.ndcg_cutoffs
    )
    return report.means()


def train_ranker(
    train: RankerDataset,
    validation: RankerDataset,
    embeddings: NDArray[np.float32],
    vocabulary: Vocabulary,
    settings: Settings,
    *,
    val_impressions: int = 5000,
    log_every: int = 200,
) -> tuple[DinDcnRanker, dict[str, Any]]:
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)

    config = RankerConfig(
        dense_dim=int(train.dense.shape[1]),
        text_dim=int(embeddings.shape[1]),
        item_dim=settings.ranker_item_dim,
        attention_dim=settings.ranker_attention_dim,
        cross_layers=settings.ranker_cross_layers,
        mlp_dims=tuple(settings.ranker_mlp_dims),
        dropout=settings.ranker_dropout,
        n_categories=vocabulary.n_categories,
        n_subcategories=vocabulary.n_subcategories,
    )
    model = DinDcnRanker(config)
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.ranker_lr, weight_decay=1e-5)
    loss_function = torch.nn.BCEWithLogitsLoss()

    train_batcher = RankerBatcher(train, embeddings, vocabulary)
    val_batcher = RankerBatcher(validation, embeddings, vocabulary)
    val_groups = np.sort(
        rng.choice(
            validation.n_groups, size=min(val_impressions, validation.n_groups), replace=False
        )
    )

    curve: list[dict[str, float]] = []
    best_state: dict[str, Tensor] | None = None
    best_score, best_epoch = -np.inf, -1
    batch_size = settings.ranker_impressions_per_batch

    for epoch in range(settings.ranker_epochs):
        model.train()
        order = rng.permutation(train.n_groups)
        running_loss, seen, rows_seen = 0.0, 0, 0

        for step, start in enumerate(range(0, order.size, batch_size)):
            chunk = np.sort(order[start : start + batch_size])
            batch = train_batcher.batch(
                chunk, negative_rate=settings.ranker_negative_sample_rate, rng=rng
            )
            if batch["labels"].numel() == 0:
                continue
            logits = model(
                batch["dense"],
                batch["candidate_text"],
                batch["candidate_category"],
                batch["candidate_subcategory"],
                batch["history_text"],
                batch["history_category"],
                batch["history_subcategory"],
                batch["history_mask"],
                batch["group_index"],
            )
            loss = loss_function(logits, batch["labels"])

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            running_loss += float(loss.detach())
            rows_seen += int(batch["labels"].numel())
            seen += 1
            if log_every and step % log_every == 0:
                logger.info(
                    "epoch %d step %5d loss %.4f (%d rows)",
                    epoch,
                    step,
                    float(loss.detach()),
                    rows_seen,
                )

        means = evaluate_subset(
            model, val_batcher, val_groups, settings, impressions_per_batch=batch_size
        )
        epoch_loss = running_loss / max(seen, 1)
        curve.append({"epoch": epoch, "loss": epoch_loss, "rows_seen": rows_seen, **means})
        logger.info(
            "epoch %d | loss %.4f | val AUC %.4f | val nDCG@10 %.4f",
            epoch,
            epoch_loss,
            means["auc"],
            means["ndcg@10"],
        )

        if means["ndcg@10"] > best_score:
            best_score, best_epoch = means["ndcg@10"], epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    selection = {
        "best_epoch": best_epoch,
        "best_val_ndcg@10_subsample": best_score,
        "val_impressions_monitored": int(val_groups.size),
        "training_curve": curve,
        "config": config.to_dict(),
        "negative_sample_rate": settings.ranker_negative_sample_rate,
    }
    return model, selection
