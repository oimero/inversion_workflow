"""Small inference seams around the section generator and forward diagnostic."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from ginn_v2.contracts import GenerationPolicy, ObservationTile, StructuredEnsemble
from ginn_v2.generator import ConditionalGenerator


def infer_section(
    generator: ConditionalGenerator,
    tile: ObservationTile,
    *,
    policy: GenerationPolicy = GenerationPolicy(),
    vp_model_mps: np.ndarray | None = None,
) -> StructuredEnsemble:
    """Generate one complete section through the generator seam."""

    return generator.generate(tile, policy, vp_model_mps=vp_model_mps)


def forward_diagnostic(
    prediction: StructuredEnsemble,
    forward: Callable[[np.ndarray], np.ndarray],
    observed_seismic: np.ndarray,
    support: np.ndarray,
) -> Mapping[str, float]:
    """Evaluate a selected result without training or changing its ranking."""

    representative = prediction.representative.get("projected_log_ai")
    if representative is None:
        raise ValueError("representative result has no projected_log_ai field.")
    predicted = np.asarray(forward(np.asarray(representative)), dtype=np.float64)
    observed = np.asarray(observed_seismic, dtype=np.float64)
    valid = np.asarray(support, dtype=bool)
    if predicted.shape != observed.shape or valid.shape != observed.shape:
        raise ValueError("forward diagnostic arrays must share one shape.")
    valid &= np.isfinite(predicted) & np.isfinite(observed)
    if np.count_nonzero(valid) < 2:
        raise ValueError("forward diagnostic has fewer than two valid samples.")
    residual = predicted[valid] - observed[valid]
    correlation = (
        float(np.corrcoef(predicted[valid], observed[valid])[0, 1])
        if np.std(predicted[valid]) > 0.0 and np.std(observed[valid]) > 0.0
        else float("nan")
    )
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "correlation": correlation,
        "valid_sample_count": float(np.count_nonzero(valid)),
    }


__all__ = ["forward_diagnostic", "infer_section"]
