"""Training and evaluation behind one compact generator-facing interface."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from collections.abc import Callable, Iterable
from typing import Mapping

import numpy as np
import torch

from ginn_v2.evidence import evidence_loss
from ginn_v2.generator import ConditionalGenerator


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
) -> dict[str, object]:
    """Train only the explicit band-limited evidence bottleneck."""
    log = logger or logging.getLogger(__name__)
    optimizer = torch.optim.AdamW(
        generator.network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    def batches(source):
        return source() if callable(source) else source

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
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
        if tuning["loss"] < best_loss:
            best_loss = tuning["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in generator.network.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint candidate.")
    generator.network.load_state_dict(best_state, strict=True)
    return {
        "best_tuning_loss": best_loss,
        "history": history,
        "epochs_completed": config.epochs,
    }


__all__ = [
    "LearningConfig",
    "evaluate_generator",
    "train_generator",
]
