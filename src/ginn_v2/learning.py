"""Training seams for the section-level event-track generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from ginn_v2.contracts import GenerationPolicy, ObservationTile
from ginn_v2.generator import ConditionalGenerator


@dataclass(frozen=True)
class LearningConfig:
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1.0e-3
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")


def validate_observation_contract(
    generator: ConditionalGenerator,
    tiles: Iterable[ObservationTile],
    *,
    vp_model_mps_by_identity: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run the cheap stage-0 evidence seam check over supplied tiles."""

    count = 0
    domains: set[str] = set()
    widths: list[int] = []
    for tile in tiles:
        vp_model_mps = None
        if tile.sample_domain == "depth":
            if vp_model_mps_by_identity is None or tile.identity not in vp_model_mps_by_identity:
                raise ValueError(
                    f"depth tile {tile.identity!r} requires explicit model-grid Vp."
                )
            vp_model_mps = vp_model_mps_by_identity[tile.identity]
        evidence = generator.observe(tile, vp_model_mps=vp_model_mps)
        count += 1
        domains.add(evidence.model_axis.sample_domain)
        widths.append(tile.width)
    if count == 0:
        raise ValueError("contract validation received no tiles.")
    return {
        "tile_count": count,
        "domains": sorted(domains),
        "width_min": min(widths),
        "width_max": max(widths),
    }


def train_generator(
    generator: ConditionalGenerator,
    training_tiles: Iterable[Mapping[str, object]],
    *,
    config: LearningConfig = LearningConfig(),
) -> dict[str, Any]:
    """Stage-1 seam reserved for ordered event-track training."""

    del generator, training_tiles, config
    raise NotImplementedError(
        "ordered event-track training is implemented in Structured GINN V2 Stage 1."
    )


def evaluate_generator(
    generator: ConditionalGenerator,
    tiles: Iterable[ObservationTile],
    *,
    policy: GenerationPolicy = GenerationPolicy(),
) -> dict[str, Any]:
    """Stage-1 seam reserved for section-level event-track evaluation."""

    del generator, tiles, policy
    raise NotImplementedError(
        "section-level event-track evaluation is implemented in Structured GINN V2 Stage 1."
    )


__all__ = [
    "LearningConfig",
    "evaluate_generator",
    "train_generator",
    "validate_observation_contract",
]
