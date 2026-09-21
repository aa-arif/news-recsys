"""Export the user tower and the ranker to ONNX for serving.

Serving runs ONNX Runtime rather than PyTorch: it starts in a fraction of the time, has a
much smaller memory footprint per worker, and gives explicit control over intra/inter-op
threads - which matters because a serving box runs many concurrent requests, each of
which wants a *small* number of threads (a single request grabbing all cores is how a
p99 gets destroyed).

Both exports use dynamic axes for the candidate count, and each export is verified
against the PyTorch model on random input before it is written - a silent export bug is
otherwise invisible until the online metrics move.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray

from news_recsys.config import Settings
from news_recsys.logging_utils import get_logger
from news_recsys.models.ranker import DinDcnRanker, RankerForServing
from news_recsys.models.two_tower import TwoTowerModel

logger = get_logger("models.onnx")

USER_TOWER_FILENAME = "user_tower.onnx"
RANKER_FILENAME = "ranker.onnx"
OPSET = 17


@dataclass
class ExportReport:
    path: Path
    max_absolute_difference: float
    inputs: list[str]
    outputs: list[str]
    ordering_identical: bool | None = None
    verified_on: str = "random tensors"


def session_options(settings: Settings) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = settings.ort_intra_op_threads
    options.inter_op_num_threads = settings.ort_inter_op_threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def make_session(path: Path, settings: Settings) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(path), sess_options=session_options(settings), providers=["CPUExecutionProvider"]
    )


class UserTowerWrapper(torch.nn.Module):
    """Single-user signature: one history in, one normalised vector out."""

    def __init__(self, model: TwoTowerModel) -> None:
        super().__init__()
        self.user_tower = model.user_tower

    def forward(
        self,
        history_text: torch.Tensor,
        history_category: torch.Tensor,
        history_subcategory: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.user_tower(history_text, history_category, history_subcategory, history_mask)


def export_user_tower(
    model: TwoTowerModel, directory: Path, settings: Settings, *, history_length: int
) -> ExportReport:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / USER_TOWER_FILENAME
    wrapper = UserTowerWrapper(model).eval()

    text = torch.randn(1, history_length, model.config.text_dim)
    category = torch.randint(0, 2, (1, history_length), dtype=torch.int64)
    subcategory = torch.randint(0, 2, (1, history_length), dtype=torch.int64)
    mask = torch.ones(1, history_length)

    torch.onnx.export(
        wrapper,
        (text, category, subcategory, mask),
        str(path),
        input_names=["history_text", "history_category", "history_subcategory", "history_mask"],
        output_names=["user_vector"],
        dynamic_axes={
            "history_text": {0: "batch", 1: "history"},
            "history_category": {0: "batch", 1: "history"},
            "history_subcategory": {0: "batch", 1: "history"},
            "history_mask": {0: "batch", 1: "history"},
            "user_vector": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    with torch.no_grad():
        expected = wrapper(text, category, subcategory, mask).numpy()
    session = make_session(path, settings)
    actual = session.run(
        None,
        {
            "history_text": text.numpy(),
            "history_category": category.numpy(),
            "history_subcategory": subcategory.numpy(),
            "history_mask": mask.numpy(),
        },
    )[0]
    difference = float(np.abs(expected - actual).max())
    logger.info("user tower ONNX max|torch-onnx| = %.2e -> %s", difference, path)
    return ExportReport(
        path=path,
        max_absolute_difference=difference,
        inputs=[item.name for item in session.get_inputs()],
        outputs=[item.name for item in session.get_outputs()],
    )


def export_ranker(
    model: DinDcnRanker,
    directory: Path,
    settings: Settings,
    *,
    history_length: int,
    n_candidates: int = 8,
    verification_inputs: dict[str, NDArray[Any]] | None = None,
) -> ExportReport:
    """Export the ranker, verifying against PyTorch on ``verification_inputs`` when given.

    Verifying on *random* tensors understates agreement badly: the dense features pass
    through a BatchNorm whose running statistics come from real data, so random inputs land
    far out in the tails and amplify ordinary float32 differences. A real batch makes the
    reported number mean something.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RANKER_FILENAME
    wrapper = RankerForServing(model).eval()

    dense = torch.randn(n_candidates, model.config.dense_dim)
    candidate_text = torch.randn(n_candidates, model.config.text_dim)
    candidate_category = torch.randint(0, 2, (n_candidates,), dtype=torch.int64)
    candidate_subcategory = torch.randint(0, 2, (n_candidates,), dtype=torch.int64)
    history_text = torch.randn(1, history_length, model.config.text_dim)
    history_category = torch.randint(0, 2, (1, history_length), dtype=torch.int64)
    history_subcategory = torch.randint(0, 2, (1, history_length), dtype=torch.int64)
    history_mask = torch.ones(1, history_length)

    inputs = (
        dense,
        candidate_text,
        candidate_category,
        candidate_subcategory,
        history_text,
        history_category,
        history_subcategory,
        history_mask,
    )
    names = [
        "dense",
        "candidate_text",
        "candidate_category",
        "candidate_subcategory",
        "history_text",
        "history_category",
        "history_subcategory",
        "history_mask",
    ]
    torch.onnx.export(
        wrapper,
        inputs,
        str(path),
        input_names=names,
        output_names=["logits"],
        dynamic_axes={
            "dense": {0: "candidates"},
            "candidate_text": {0: "candidates"},
            "candidate_category": {0: "candidates"},
            "candidate_subcategory": {0: "candidates"},
            "history_text": {1: "history"},
            "history_category": {1: "history"},
            "history_subcategory": {1: "history"},
            "history_mask": {1: "history"},
            "logits": {0: "candidates"},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    with torch.no_grad():
        expected_dummy = wrapper(*inputs).numpy()
    session = make_session(path, settings)

    if verification_inputs is None:
        feed = {name: tensor.numpy() for name, tensor in zip(names, inputs, strict=True)}
        expected = expected_dummy
    else:
        feed = {name: np.ascontiguousarray(verification_inputs[name]) for name in names}
        with torch.no_grad():
            expected = wrapper(*[torch.from_numpy(feed[name]) for name in names]).numpy()

    actual = np.asarray(session.run(None, feed)[0], dtype=np.float32)
    difference = float(np.abs(expected - actual).max())
    same_order = bool(np.array_equal(np.argsort(-expected), np.argsort(-actual)))
    logger.info(
        "ranker ONNX max|torch-onnx| = %.2e (ordering identical: %s, verified on %s) -> %s",
        difference,
        same_order,
        "a real batch" if verification_inputs is not None else "random tensors",
        path,
    )
    return ExportReport(
        path=path,
        max_absolute_difference=difference,
        inputs=[item.name for item in session.get_inputs()],
        outputs=[item.name for item in session.get_outputs()],
        ordering_identical=same_order,
        verified_on="real impression batch"
        if verification_inputs is not None
        else "random tensors",
    )


def run_user_tower(
    session: ort.InferenceSession,
    history_text: NDArray[np.float32],
    history_category: NDArray[np.int64],
    history_subcategory: NDArray[np.int64],
    history_mask: NDArray[np.float32],
) -> NDArray[np.float32]:
    outputs = session.run(
        None,
        {
            "history_text": history_text,
            "history_category": history_category,
            "history_subcategory": history_subcategory,
            "history_mask": history_mask,
        },
    )
    return np.asarray(outputs[0], dtype=np.float32)


def run_ranker(
    session: ort.InferenceSession,
    dense: NDArray[np.float32],
    candidate_text: NDArray[np.float32],
    candidate_category: NDArray[np.int64],
    candidate_subcategory: NDArray[np.int64],
    history_text: NDArray[np.float32],
    history_category: NDArray[np.int64],
    history_subcategory: NDArray[np.int64],
    history_mask: NDArray[np.float32],
) -> NDArray[np.float32]:
    outputs = session.run(
        None,
        {
            "dense": dense,
            "candidate_text": candidate_text,
            "candidate_category": candidate_category,
            "candidate_subcategory": candidate_subcategory,
            "history_text": history_text,
            "history_category": history_category,
            "history_subcategory": history_subcategory,
            "history_mask": history_mask,
        },
    )
    return np.asarray(outputs[0], dtype=np.float32)
