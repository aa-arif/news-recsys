"""The ranking model: DIN-style target attention over history + a DCN-v2 cross network.

Structure, and why each half is there:

* **DIN target attention.** Which part of a user's history matters depends on the article
  being scored - a sports history is evidence for a sports candidate and noise for a
  finance one. The local activation unit scores every history item against the candidate
  from ``[h, e, h-e, h*e]`` and pools the history with those weights, so the user
  representation is computed *per candidate* rather than once per request.
* **DCN-v2 cross network.** The dense side is 35 hand-built time-aware features plus the
  content vectors; feature crosses (recency x user affinity, popularity x freshness) are
  exactly what a ranker needs and what a plain MLP spends capacity rediscovering. The
  cross layers build them explicitly and cheaply.

The two are combined in DCN-v2's parallel arrangement: cross output and deep output are
concatenated into the final logit.

History is encoded **once per impression** (``group_index`` maps each candidate row to its
impression), which is what makes CPU training tractable here and mirrors serving, where
one user's history is encoded once per request and reused for all 200 candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

MODEL_FILENAME = "ranker.pt"


@dataclass
class RankerConfig:
    dense_dim: int = 35
    text_dim: int = 384
    category_dim: int = 32
    subcategory_dim: int = 32
    item_dim: int = 128
    attention_dim: int = 64
    cross_layers: int = 3
    mlp_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    n_categories: int = 18
    n_subcategories: int = 270

    def to_dict(self) -> dict[str, object]:
        payload = self.__dict__.copy()
        payload["mlp_dims"] = list(self.mlp_dims)
        return payload


class ItemEncoder(nn.Module):
    """Article content -> a dense item vector (id 0 is the unknown-article row)."""

    def __init__(self, config: RankerConfig) -> None:
        super().__init__()
        self.category = nn.Embedding(config.n_categories + 1, config.category_dim, padding_idx=0)
        self.subcategory = nn.Embedding(
            config.n_subcategories + 1, config.subcategory_dim, padding_idx=0
        )
        self.projection = nn.Sequential(
            nn.Linear(
                config.text_dim + config.category_dim + config.subcategory_dim, config.item_dim
            ),
            nn.GELU(),
        )

    def forward(self, text: Tensor, category: Tensor, subcategory: Tensor) -> Tensor:
        return self.projection(
            torch.cat([text, self.category(category), self.subcategory(subcategory)], dim=-1)
        )


class TargetAttention(nn.Module):
    """DIN local activation unit: candidate-aware pooling over the history."""

    def __init__(self, item_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(item_dim * 4, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, candidate: Tensor, history: Tensor, mask: Tensor) -> Tensor:
        """``candidate`` (N, D), ``history`` (N, H, D), ``mask`` (N, H) -> (N, D)."""
        expanded = candidate.unsqueeze(1).expand_as(history)
        pair = torch.cat([history, expanded, history - expanded, history * expanded], dim=-1)
        scores = self.scorer(pair).squeeze(-1)
        scores = scores.masked_fill(mask <= 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        # Users with no history must contribute a zero interest vector, not NaNs.
        weights = weights * (mask.sum(dim=-1, keepdim=True) > 0)
        return torch.bmm(weights.unsqueeze(1), history).squeeze(1)


class CrossNetworkV2(nn.Module):
    """DCN-v2 cross layers: ``x_{l+1} = x_0 * (W_l x_l + b_l) + x_l``."""

    def __init__(self, input_dim: int, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(n_layers)])

    def forward(self, x0: Tensor) -> Tensor:
        x = x0
        for layer in self.layers:
            x = x0 * layer(x) + x
        return x


class DinDcnRanker(nn.Module):
    """Impression-log ranker: clicked vs shown-but-not-clicked."""

    def __init__(self, config: RankerConfig) -> None:
        super().__init__()
        self.config = config
        self.item_encoder = ItemEncoder(config)
        self.attention = TargetAttention(config.item_dim, config.attention_dim)
        self.dense_norm = nn.BatchNorm1d(config.dense_dim)

        cross_input = config.dense_dim + config.item_dim * 3
        self.cross = CrossNetworkV2(cross_input, config.cross_layers)

        layers: list[nn.Module] = []
        previous = cross_input
        for size in config.mlp_dims:
            layers += [nn.Linear(previous, size), nn.GELU(), nn.Dropout(config.dropout)]
            previous = size
        self.deep = nn.Sequential(*layers)
        self.head = nn.Linear(cross_input + previous, 1)

    def encode_history(
        self, history_text: Tensor, history_category: Tensor, history_subcategory: Tensor
    ) -> Tensor:
        groups, length, _ = history_text.shape
        encoded = self.item_encoder(
            history_text.reshape(groups * length, -1),
            history_category.reshape(-1),
            history_subcategory.reshape(-1),
        )
        return encoded.reshape(groups, length, -1)

    def score(
        self,
        dense: Tensor,
        candidate_text: Tensor,
        candidate_category: Tensor,
        candidate_subcategory: Tensor,
        history_encoded: Tensor,
        history_mask: Tensor,
        group_index: Tensor,
    ) -> Tensor:
        candidate = self.item_encoder(candidate_text, candidate_category, candidate_subcategory)
        history = history_encoded.index_select(0, group_index)
        mask = history_mask.index_select(0, group_index)
        interest = self.attention(candidate, history, mask)

        features = torch.cat(
            [self.dense_norm(dense), candidate, interest, candidate * interest], dim=-1
        )
        crossed = self.cross(features)
        deep = self.deep(features)
        return self.head(torch.cat([crossed, deep], dim=-1)).squeeze(-1)

    def forward(
        self,
        dense: Tensor,
        candidate_text: Tensor,
        candidate_category: Tensor,
        candidate_subcategory: Tensor,
        history_text: Tensor,
        history_category: Tensor,
        history_subcategory: Tensor,
        history_mask: Tensor,
        group_index: Tensor,
    ) -> Tensor:
        history_encoded = self.encode_history(history_text, history_category, history_subcategory)
        return self.score(
            dense,
            candidate_text,
            candidate_category,
            candidate_subcategory,
            history_encoded,
            history_mask,
            group_index,
        )

    def save(self, directory: Path, extra: dict | None = None) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MODEL_FILENAME
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config.to_dict(),
                "extra": extra or {},
            },
            path,
        )
        return path

    @classmethod
    def load(cls, directory: Path, *, map_location: str = "cpu") -> tuple[DinDcnRanker, dict]:
        payload = torch.load(
            directory / MODEL_FILENAME, map_location=map_location, weights_only=False
        )
        config_payload = dict(payload["config"])
        config_payload["mlp_dims"] = tuple(config_payload["mlp_dims"])
        config = RankerConfig(**config_payload)
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload.get("extra", {})


class RankerForServing(nn.Module):
    """ONNX-friendly wrapper: one user's history, N candidates, one logit each."""

    def __init__(self, ranker: DinDcnRanker) -> None:
        super().__init__()
        self.ranker = ranker

    def forward(
        self,
        dense: Tensor,
        candidate_text: Tensor,
        candidate_category: Tensor,
        candidate_subcategory: Tensor,
        history_text: Tensor,
        history_category: Tensor,
        history_subcategory: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        group_index = torch.zeros(dense.shape[0], dtype=torch.int64, device=dense.device)
        return self.ranker(
            dense,
            candidate_text,
            candidate_category,
            candidate_subcategory,
            history_text,
            history_category,
            history_subcategory,
            history_mask,
            group_index,
        )
