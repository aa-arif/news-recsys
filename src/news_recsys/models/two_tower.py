"""Two-tower retrieval model.

* **Item tower** - the article's text embedding concatenated with learned category and
  subcategory embeddings, through an MLP, L2-normalised.
* **User tower** - the same item tower applied to every article in the click history,
  pooled with additive attention, through an MLP, L2-normalised.

Two properties of MIND drove this design (both measured in M1):

* 71% of test articles never appear in a training impression, so the item tower takes
  *content* and never an article-id embedding;
* 89% of test users never appear in training, so the user tower takes the click *history*
  arriving with the request and never a user-id embedding.

Training uses a sampled softmax over in-batch negatives with the **logQ correction**:
popular articles appear as negatives far more often than chance, so the raw in-batch
softmax systematically penalises them. Subtracting ``log P(item)`` from each logit
removes that bias, which is the difference between a retrieval model that is merely
trained and one whose scores mean something.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from news_recsys.config import Settings
from news_recsys.features.vocab import Vocabulary

MODEL_FILENAME = "two_tower.pt"


@dataclass
class TwoTowerConfig:
    text_dim: int = 384
    category_dim: int = 32
    subcategory_dim: int = 32
    hidden_dim: int = 256
    output_dim: int = 128
    attention_dim: int = 64
    dropout: float = 0.1
    temperature: float = 0.05

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


class ItemTower(nn.Module):
    """Content -> normalised item vector."""

    def __init__(self, config: TwoTowerConfig, n_categories: int, n_subcategories: int) -> None:
        super().__init__()
        self.config = config
        # +1 row for "unknown", which is where an article published after the last index
        # build lands. Index 0 is reserved for it and real ids are shifted by one.
        self.category = nn.Embedding(n_categories + 1, config.category_dim, padding_idx=0)
        self.subcategory = nn.Embedding(n_subcategories + 1, config.subcategory_dim, padding_idx=0)
        input_dim = config.text_dim + config.category_dim + config.subcategory_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, text: Tensor, category: Tensor, subcategory: Tensor) -> Tensor:
        features = torch.cat([text, self.category(category), self.subcategory(subcategory)], dim=-1)
        return nn.functional.normalize(self.mlp(features), dim=-1)


class AdditiveAttention(nn.Module):
    """The attention pooling used by NAML/NRMS: a learned query over history vectors."""

    def __init__(self, input_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, attention_dim)
        self.query = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        scores = self.query(torch.tanh(self.projection(values))).squeeze(-1)
        scores = scores.masked_fill(mask <= 0, float("-inf"))
        # A user with an empty history would otherwise softmax over all -inf and produce
        # NaNs; fall back to a zero vector, which the MLP can learn a prior for.
        empty = mask.sum(dim=-1, keepdim=True) <= 0
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(empty, torch.zeros_like(weights), weights)
        return torch.bmm(weights.unsqueeze(1), values).squeeze(1)


class UserTower(nn.Module):
    """Click history -> normalised user vector (shares the item tower)."""

    def __init__(self, config: TwoTowerConfig, item_tower: ItemTower) -> None:
        super().__init__()
        self.config = config
        self.item_tower = item_tower
        self.attention = AdditiveAttention(config.output_dim, config.attention_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.output_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(
        self,
        history_text: Tensor,
        history_category: Tensor,
        history_subcategory: Tensor,
        mask: Tensor,
    ) -> Tensor:
        batch, length, _ = history_text.shape
        flat = self.item_tower(
            history_text.reshape(batch * length, -1),
            history_category.reshape(-1),
            history_subcategory.reshape(-1),
        ).reshape(batch, length, -1)
        flat = flat * mask.unsqueeze(-1)
        pooled = self.attention(flat, mask)
        return nn.functional.normalize(self.mlp(pooled), dim=-1)


class TwoTowerModel(nn.Module):
    """Both towers plus the logQ-corrected in-batch softmax loss."""

    def __init__(self, config: TwoTowerConfig, n_categories: int, n_subcategories: int) -> None:
        super().__init__()
        self.config = config
        self.item_tower = ItemTower(config, n_categories, n_subcategories)
        self.user_tower = UserTower(config, self.item_tower)

    def forward(
        self,
        history_text: Tensor,
        history_category: Tensor,
        history_subcategory: Tensor,
        mask: Tensor,
        item_text: Tensor,
        item_category: Tensor,
        item_subcategory: Tensor,
    ) -> tuple[Tensor, Tensor]:
        user_vectors = self.user_tower(history_text, history_category, history_subcategory, mask)
        item_vectors = self.item_tower(item_text, item_category, item_subcategory)
        return user_vectors, item_vectors

    def in_batch_loss(
        self, user_vectors: Tensor, item_vectors: Tensor, log_sampling_probability: Tensor
    ) -> Tensor:
        """Sampled softmax with the logQ correction.

        ``log_sampling_probability`` is ``log P(item)`` for each in-batch positive, i.e.
        the probability that this item shows up as a negative for someone else.
        """
        logits = (user_vectors @ item_vectors.T) / self.config.temperature
        logits = logits - log_sampling_probability.unsqueeze(0)
        targets = torch.arange(user_vectors.shape[0], device=user_vectors.device)
        return nn.functional.cross_entropy(logits, targets)

    def save(self, directory: Path, vocabulary: Vocabulary, settings: Settings) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MODEL_FILENAME
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config.to_dict(),
                "n_categories": vocabulary.n_categories,
                "n_subcategories": vocabulary.n_subcategories,
                "max_history": settings.max_history,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, directory: Path, *, map_location: str = "cpu") -> tuple[TwoTowerModel, dict]:
        payload = torch.load(
            directory / MODEL_FILENAME, map_location=map_location, weights_only=False
        )
        config = TwoTowerConfig(**payload["config"])
        model = cls(config, payload["n_categories"], payload["n_subcategories"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload


def shifted_category_ids(
    news_index: NDArray[np.int64], vocabulary: Vocabulary
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Category / subcategory ids shifted by one so 0 means "unknown article"."""
    known = news_index >= 0
    safe = np.where(known, news_index, 0)
    category = np.where(known, vocabulary.news_category[safe].astype(np.int64) + 1, 0)
    subcategory = np.where(known, vocabulary.news_subcategory[safe].astype(np.int64) + 1, 0)
    return category, subcategory
