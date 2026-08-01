"""Training and evaluation behind one compact generator-facing interface."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import random
from collections.abc import Callable, Iterable
from typing import Any, Mapping

import numpy as np
import torch

from ginn_v2.evidence import evidence_loss
from ginn_v2.generator import ConditionalGenerator


CheckpointCallback = Callable[
    [int, ConditionalGenerator, Mapping[str, Any], bool],
    None,
]


@dataclass(frozen=True)
class LearningConfig:
    epochs: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    log_every_batches: int = 20
    gradient_clip_norm: float = 5.0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.log_every_batches <= 0:
            raise ValueError("epochs and log_every_batches must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer controls are invalid.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")


def _to_device(
    batch: Mapping[str, np.ndarray | torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    boolean = {"observed_valid", "lateral_valid", "support"}
    for key, value in batch.items():
        result[key] = torch.as_tensor(
            value,
            dtype=torch.bool if key in boolean else torch.float32,
            device=device,
        )
    return result


def _forward_loss(
    generator: ConditionalGenerator,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    output = generator.network(
        batch["seismic"],
        batch["lfm"],
        batch["observed_valid"],
        batch["lateral_m"],
        batch["lateral_valid"],
    )
    return evidence_loss(output, batch)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def evaluate_generator(
    generator: ConditionalGenerator,
    batches: Iterable[Mapping[str, np.ndarray | torch.Tensor]],
) -> dict[str, float]:
    generator.network.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for source in batches:
            loss = _forward_loss(generator, _to_device(source, generator.device))
            count += 1
            for key, value in loss.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
    if count == 0:
        raise ValueError("evaluation received no batches.")
    return {key: value / count for key, value in totals.items()}


def train_generator(
    generator: ConditionalGenerator,
    training_batches: (
        Iterable[Mapping[str, np.ndarray | torch.Tensor]]
        | Callable[[], Iterable[Mapping[str, np.ndarray | torch.Tensor]]]
    ),
    tuning_batches: (
        Iterable[Mapping[str, np.ndarray | torch.Tensor]]
        | Callable[[], Iterable[Mapping[str, np.ndarray | torch.Tensor]]]
    ),
    *,
    config: LearningConfig = LearningConfig(),
    logger: logging.Logger | None = None,
    resume_state: Mapping[str, Any] | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> dict[str, object]:
    """Train the evidence bottleneck with epoch-boundary recovery hooks.

    ``resume_state`` is the runtime payload from a previous ``last.pt``
    checkpoint.  The callback is invoked only after a complete epoch and is
    responsible for durable checkpoint publication; the training loop does
    not keep a second copy of the best model in memory.
    """
    log = logger or logging.getLogger(__name__)
    optimizer = torch.optim.AdamW(
        generator.network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    resume = dict(resume_state or {})
    start_epoch = int(resume.get("epoch", 0)) + 1
    if start_epoch < 1:
        raise ValueError("resume epoch must be non-negative.")
    if resume_state is not None and "optimizer_state" not in resume:
        raise ValueError("resume checkpoint lacks optimizer_state.")
    if "optimizer_state" in resume:
        optimizer.load_state_dict(resume["optimizer_state"])
    if "rng_state" in resume:
        _restore_rng_state(resume["rng_state"])

    def batches(source):
        return source() if callable(source) else source

    history: list[dict[str, float]] = [
        {str(key): float(value) for key, value in dict(row).items()}
        for row in resume.get("history", [])
    ]
    best_loss = float(resume.get("best_tuning_loss", float("inf")))
    best_epoch = int(resume.get("best_epoch", 0))
    if start_epoch > config.epochs:
        log.info(
            "checkpoint already reached target epochs | epoch=%d | target=%d",
            start_epoch - 1,
            config.epochs,
        )
        return {
            "best_tuning_loss": best_loss,
            "best_epoch": best_epoch,
            "history": history,
            "epochs_completed": start_epoch - 1,
            "start_epoch": start_epoch,
        }

    for epoch in range(start_epoch, config.epochs + 1):
        generator.network.train()
        running = 0.0
        batch_count = 0
        for batch_index, source in enumerate(batches(training_batches), start=1):
            batch_count = batch_index
            optimizer.zero_grad(set_to_none=True)
            loss = _forward_loss(generator, _to_device(source, generator.device))
            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                generator.network.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            running += float(loss["loss"].detach().cpu())
            if batch_index % config.log_every_batches == 0:
                log.info(
                    "epoch %d/%d | batch %d | train_loss=%.6f",
                    epoch,
                    config.epochs,
                    batch_index,
                    running / batch_index,
                )
        if batch_count == 0:
            raise ValueError("training received no batches.")
        tuning = evaluate_generator(generator, batches(tuning_batches))
        row = {
            "epoch": float(epoch),
            "train_loss": running / batch_count,
            **{f"tuning_{key}": value for key, value in tuning.items()},
        }
        history.append(row)
        log.info(
            "epoch %d/%d complete | train_loss=%.6f | tuning_loss=%.6f",
            epoch,
            config.epochs,
            row["train_loss"],
            row["tuning_loss"],
        )

        is_best = tuning["loss"] < best_loss
        if is_best:
            best_loss = tuning["loss"]
            best_epoch = epoch

        runtime_state = {
            "epoch": epoch,
            "best_tuning_loss": float(best_loss),
            "best_epoch": best_epoch,
            "history": [dict(item) for item in history],
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _capture_rng_state(),
        }
        if checkpoint_callback is not None:
            checkpoint_callback(epoch, generator, runtime_state, is_best)

    return {
        "best_tuning_loss": best_loss,
        "best_epoch": best_epoch,
        "history": history,
        "epochs_completed": config.epochs,
        "start_epoch": start_epoch,
    }


__all__ = [
    "CheckpointCallback",
    "LearningConfig",
    "evaluate_generator",
    "train_generator",
]
