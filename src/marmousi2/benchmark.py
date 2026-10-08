"""Adapters for evaluating saved Marmousi2 PIAI predictions.

The main workflow owns model execution. This module only loads prediction
artifacts, converts current v3 prediction objects to a reviewable NPZ, and
delegates scientific scoring to ``evaluation.evaluate_marmousi2_prediction``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import evaluate_marmousi2_prediction


def load_prediction_array(
    path: str | Path,
    *,
    preferred: tuple[str, ...] = ("log_ai", "prediction"),
) -> np.ndarray:
    """Load one saved prediction array without changing its physical meaning."""

    values, _valid_mask = _load_prediction_artifact(path, preferred=preferred)
    return values


def _load_prediction_artifact(
    path: str | Path,
    *,
    preferred: tuple[str, ...] = ("log_ai", "prediction"),
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a prediction and its explicit support mask, if present."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        values: np.ndarray | None = None
        for key in preferred:
            if key in data.files:
                values = np.asarray(data[key], dtype=np.float64)
                break
        if values is None:
            raise ValueError(f"{source} does not contain one of {preferred}.")
        valid_mask = None
        if "valid_mask" in data.files:
            valid_mask = np.asarray(data["valid_mask"])
            if valid_mask.dtype != np.bool_:
                raise ValueError("Prediction valid_mask must have a boolean dtype.")
    if valid_mask is not None and valid_mask.shape != values.shape:
        raise ValueError(f"Prediction valid mask shape {valid_mask.shape} differs from log-AI shape {values.shape}.")
    return values, valid_mask


def _prediction_log_ai(prediction: Any) -> np.ndarray:
    """Extract log-AI from a v3 prediction object or a saved volume result."""

    path = getattr(prediction, "log_ai_path", None)
    if path is not None:
        return np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float32)
    value = getattr(prediction, "log_ai", None)
    if value is None:
        raise TypeError("Prediction must expose log_ai or log_ai_path.")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _prediction_valid_mask(prediction: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    path = getattr(prediction, "valid_mask_path", None)
    if path is None:
        value = getattr(prediction, "valid_mask", None)
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        mask = np.asarray(value)
    else:
        mask = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False))
    if mask.dtype != np.bool_:
        raise ValueError("Prediction valid mask must have a boolean dtype.")
    if mask.shape != shape:
        raise ValueError(f"Prediction valid mask shape {mask.shape} differs from log-AI shape {shape}.")
    return mask


def save_prediction_npz(path: str | Path, prediction: Any) -> np.ndarray:
    """Save a v3 prediction object as an NPZ accepted by the evaluator.

    The evaluator consumes log-AI.  Unsupported v3 volume samples remain
    represented by the prediction's existing mask and are not filled here.
    """

    destination = Path(path)
    values = _prediction_log_ai(prediction)
    valid_mask = _prediction_valid_mask(prediction, tuple(values.shape))
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"log_ai": values}
    if valid_mask is not None:
        payload["valid_mask"] = valid_mask
    np.savez_compressed(destination, **payload)
    return values.astype(np.float64, copy=False)


def evaluate_prediction(
    prepared_dir: str | Path,
    prediction_path: str | Path,
    *,
    lfm_path: str | Path | None = None,
    lfm_log_ai: np.ndarray | None = None,
    scope: str = "final",
    support_relative_threshold: float = 0.25,
    cutoff_hz: float = 5.0,
    shortwave_cutoff_hz: float = 20.0,
) -> dict[str, Any]:
    """Evaluate one saved prediction under the existing leakage-aware protocol."""

    prediction, prediction_valid_mask = _load_prediction_artifact(prediction_path)
    if lfm_path is not None:
        lfm = load_prediction_array(lfm_path, preferred=("log_ai", "lfm_log_ai", "prediction"))
    elif lfm_log_ai is not None:
        lfm = np.asarray(lfm_log_ai, dtype=np.float64)
    else:
        raise ValueError("An LFM array is required; pass lfm_path or lfm_log_ai.")
    result = evaluate_marmousi2_prediction(
        Path(prepared_dir),
        prediction,
        lfm,
        scope=scope,
        cutoff_hz=cutoff_hz,
        shortwave_cutoff_hz=shortwave_cutoff_hz,
        support_relative_threshold=support_relative_threshold,
        prediction_valid_mask=prediction_valid_mask,
    )
    result["prediction_path"] = str(Path(prediction_path))
    return result


__all__ = [
    "evaluate_prediction",
    "load_prediction_array",
    "save_prediction_npz",
]
