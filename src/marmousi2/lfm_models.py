"""Numerical low-frequency-model construction for the Marmousi2 benchmark.

Builders operate in natural-log acoustic-impedance space and return an
:class:`LfmArray` so the wrapper can persist both the numerical array and the
provenance needed to distinguish training-only and oracle baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _log_ai_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [well/profile, time].")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must have at least one row and one time sample.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite logAI values.")
    return array


def _axis(value: Any, *, name: str, require_strict: bool = True) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if require_strict and np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing and unique.")
    return array


def _twt_axis(value: Any, expected_size: int) -> np.ndarray:
    twt = _axis(value, name="twt_s")
    if twt.size != expected_size:
        raise ValueError(f"twt_s length {twt.size} does not match time dimension {expected_size}.")
    return twt


@dataclass(frozen=True)
class LfmArray:
    """A log-AI LFM array plus provenance and optional derived fields."""

    log_ai: np.ndarray
    metadata: Mapping[str, Any]
    fields: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.log_ai, dtype=np.float64)
        if values.ndim not in {0, 1, 2}:
            raise ValueError("log_ai must be a scalar, [time] vector, or [profile, time] matrix.")
        if not np.all(np.isfinite(values)):
            raise ValueError("log_ai must contain only finite values.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        if not isinstance(self.fields, Mapping):
            raise TypeError("fields must be a mapping.")
        fields = {}
        for name, field_value in self.fields.items():
            field_array = np.asarray(field_value, dtype=np.float64)
            if not np.all(np.isfinite(field_array)):
                raise ValueError(f"fields[{name!r}] must contain only finite values.")
            fields[str(name)] = field_array
        object.__setattr__(self, "log_ai", values)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "fields", fields)

    def __iter__(self):
        """Allow ``log_ai, metadata = build_*`` for wrapper compatibility."""

        yield self.log_ai
        yield self.metadata


def build_truth_huber_trend(truth_log_ai: np.ndarray, twt_s: np.ndarray, *, f_scale: float = 0.1) -> LfmArray:
    """Fit one robust Huber logAI-vs-TWT line per truth profile.

    This is an explicit full-truth/oracle baseline.  A Huber loss keeps the line
    from being dragged by the isolated high-contrast events that least squares
    would chase, and it is the same robust loss the workflow's ``trend`` baseline
    uses.  The API accepts only the truth matrix and its TWT axis; callers must
    keep the result out of model selection and final-test tuning.
    """

    from scipy.optimize import least_squares

    truth = _log_ai_matrix(truth_log_ai, name="truth_log_ai")
    twt = _twt_axis(twt_s, truth.shape[1])
    scale = _positive_float(f_scale, name="f_scale")
    centered = twt - float(np.mean(twt))
    denominator = float(np.dot(centered, centered))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("twt_s must contain at least two distinct samples for a trend fit.")
    centered_truth = truth - np.mean(truth, axis=1, keepdims=True)
    # The closed-form least-squares line is the warm start for the robust fit.
    starts = (centered_truth @ centered) / denominator
    intercepts = np.empty(truth.shape[0], dtype=np.float64)
    slopes = np.empty(truth.shape[0], dtype=np.float64)
    for profile in range(truth.shape[0]):
        start = (float(np.mean(truth[profile])) - float(starts[profile]) * float(np.mean(twt)), float(starts[profile]))
        fit = least_squares(
            lambda parameters: parameters[0] + parameters[1] * twt - truth[profile],
            np.asarray(start, dtype=np.float64),
            loss="huber",
            f_scale=scale,
        )
        if not fit.success or np.any(~np.isfinite(fit.x)):
            raise ValueError(f"Huber trend fit failed for profile {profile}: {fit.message}")
        intercepts[profile], slopes[profile] = (float(fit.x[0]), float(fit.x[1]))
    lfm = intercepts[:, None] + slopes[:, None] * twt[None, :]
    metadata = {
        "schema": "marmousi2_lfm_truth_huber_trend_v1",
        "method": "huber_least_squares",
        "robust_f_scale_log_ai": scale,
        "fit_space": "logAI",
        "source": "full_truth_oracle",
        "oracle": True,
        "label_leakage": "full truth; evaluation/oracle reference only",
        "profile_count": int(truth.shape[0]),
        "time_count": int(truth.shape[1]),
        "twt_start_s": float(twt[0]),
        "twt_end_s": float(twt[-1]),
        "twt_sampling": "strictly_increasing",
        "line_form": "logAI(t) = intercept + slope * t",
        "units": {"log_ai": "natural_log_of_m/s*g/cc", "twt": "s"},
    }
    return LfmArray(
        lfm,
        metadata,
        fields={"intercept_log_ai": intercepts, "slope_log_ai_per_s": slopes},
    )


__all__ = ["LfmArray", "build_truth_huber_trend"]
