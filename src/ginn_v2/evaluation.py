"""Frozen metrics and acceptance gates for body inversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class GateThresholds:
    pretrain_masked_corr_improvement: float = 0.01
    pretrain_masked_shape_ratio: float = 0.99
    masked_corr_drop_tolerance: float = 0.01
    masked_shape_ratio: float = 1.05
    visible_corr_drop_tolerance: float = 0.01
    visible_shape_ratio: float = 1.05
    well_lfm_rmse_ratio: float = 1.50
    well_fraction_improved: float = 0.60
    lfm_drift_rmse: float = 0.03
    short_wave_energy_fraction: float = 0.01
    roughness_ratio: float = 1.25
    degradation_ratio: float = 1.05
    platform_improvement: float = 0.01

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"Gate threshold {name} must be finite and non-negative.")
        if not 0.0 <= self.well_fraction_improved <= 1.0:
            raise ValueError("well_fraction_improved must be within [0, 1].")
        if self.masked_shape_ratio <= 0.0 or self.visible_shape_ratio <= 0.0 or self.well_lfm_rmse_ratio <= 0.0:
            raise ValueError("Shape and RMSE gate ratios must be positive.")
        if self.roughness_ratio <= 0.0:
            raise ValueError("roughness_ratio must be positive.")


@dataclass(frozen=True)
class EvaluationMetrics:
    masked_correlation: np.ndarray
    masked_shape_loss: np.ndarray
    visible_correlation: np.ndarray
    visible_shape_loss: np.ndarray
    well_rmse_by_well: Mapping[str, float]
    well_bias_by_well: Mapping[str, float]
    well_pooled_rmse: float
    well_pooled_bias: float
    lfm_drift_rmse: float
    short_wave_energy_fraction: float
    roughness_ratio: float
    analytic_gain_mean: float
    raw_amplitude_residual_mean: float
    support_contiguous_fraction: float
    orientation_disagreement_rms_ratio: float
    sample_count: int

    def __post_init__(self) -> None:
        arrays = (self.masked_correlation, self.masked_shape_loss, self.visible_correlation, self.visible_shape_loss)
        if any(np.asarray(value).ndim != 1 or np.asarray(value).size == 0 for value in arrays):
            raise ValueError("Validation trace metrics must be non-empty one-dimensional arrays.")
        for value in arrays:
            if np.any(~np.isfinite(value)):
                raise ValueError("Validation trace metrics must be finite.")
        for name in (
            "well_pooled_rmse",
            "well_pooled_bias",
            "lfm_drift_rmse",
            "short_wave_energy_fraction",
            "roughness_ratio",
            "analytic_gain_mean",
            "raw_amplitude_residual_mean",
            "support_contiguous_fraction",
            "orientation_disagreement_rms_ratio",
        ):
            _finite_scalar(getattr(self, name), name=name)
        if not self.well_rmse_by_well or set(self.well_rmse_by_well) != set(self.well_bias_by_well):
            raise ValueError("Well metrics must contain matching non-empty well identities.")
        for name, values in (
            ("well_rmse_by_well", self.well_rmse_by_well),
            ("well_bias_by_well", self.well_bias_by_well),
        ):
            if any(not np.isfinite(float(value)) for value in values.values()):
                raise ValueError(f"{name} must contain only finite values.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive.")

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("masked_correlation", "masked_shape_loss", "visible_correlation", "visible_shape_loss"):
            payload[key] = np.asarray(payload[key], dtype=np.float64).tolist()
        payload["well_rmse_by_well"] = dict(self.well_rmse_by_well)
        payload["well_bias_by_well"] = dict(self.well_bias_by_well)
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "EvaluationMetrics":
        values = dict(payload)
        for key in ("masked_correlation", "masked_shape_loss", "visible_correlation", "visible_shape_loss"):
            values[key] = np.asarray(values[key], dtype=np.float64)
        values["well_rmse_by_well"] = dict(values["well_rmse_by_well"])  # type: ignore[arg-type]
        values["well_bias_by_well"] = dict(values["well_bias_by_well"])  # type: ignore[arg-type]
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class GateReport:
    passed: bool
    failed_gates: tuple[str, ...]
    first_failed_gate: str | None
    details: Mapping[str, float | int | bool]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failed_gates": list(self.failed_gates),
            "first_failed_gate": self.first_failed_gate,
            "details": dict(self.details),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "GateReport":
        return cls(
            passed=bool(payload["passed"]),
            failed_gates=tuple(str(item) for item in payload["failed_gates"]),  # type: ignore[union-attr]
            first_failed_gate=(
                None if payload.get("first_failed_gate") is None else str(payload["first_failed_gate"])
            ),
            details=dict(payload["details"]),  # type: ignore[arg-type]
        )


def _relative_improvement(current: float, previous: float) -> float:
    denominator = max(abs(float(previous)), 1e-12)
    return (float(previous) - float(current)) / denominator


def _median(array: np.ndarray) -> float:
    return float(np.median(np.asarray(array, dtype=np.float64)))


def evaluate_gates(
    metrics: EvaluationMetrics,
    lfm_baseline: EvaluationMetrics,
    pretrain_baseline: EvaluationMetrics,
    *,
    thresholds: GateThresholds | None = None,
) -> GateReport:
    """Apply the frozen HANDOFF gates in their documented order."""

    threshold = thresholds or GateThresholds()
    masked_corr_change = np.asarray(metrics.masked_correlation) - np.asarray(pretrain_baseline.masked_correlation)
    masked_shape_ratio = _median(metrics.masked_shape_loss) / max(_median(pretrain_baseline.masked_shape_loss), 1e-12)
    visible_corr_change = np.asarray(metrics.visible_correlation) - np.asarray(pretrain_baseline.visible_correlation)
    visible_shape_ratio = _median(metrics.visible_shape_loss) / max(_median(pretrain_baseline.visible_shape_loss), 1e-12)
    baseline_wells = dict(pretrain_baseline.well_rmse_by_well)
    lfm_wells = dict(lfm_baseline.well_rmse_by_well)
    current_wells = dict(metrics.well_rmse_by_well)
    common_wells = sorted(set(baseline_wells) & set(current_wells))
    if common_wells != sorted(baseline_wells) or common_wells != sorted(lfm_wells):
        raise ValueError("LFM, pretrain, and current well metric identities differ.")
    pretrain_well_ratios = np.asarray(
        [current_wells[name] / max(abs(baseline_wells[name]), 1e-12) for name in common_wells], dtype=np.float64
    )
    lfm_well_ratios = np.asarray(
        [current_wells[name] / max(abs(lfm_wells[name]), 1e-12) for name in common_wells], dtype=np.float64
    )
    failed: list[str] = []
    details: dict[str, float | int | bool] = {
        "masked_corr_change_from_pretrain_median": _median(masked_corr_change),
        "masked_shape_ratio": float(masked_shape_ratio),
        "visible_corr_median": _median(metrics.visible_correlation),
        "visible_corr_change_from_pretrain_median": _median(visible_corr_change),
        "visible_shape_ratio": float(visible_shape_ratio),
        "well_pooled_rmse_ratio_to_pretrain": metrics.well_pooled_rmse / max(abs(pretrain_baseline.well_pooled_rmse), 1e-12),
        "well_pooled_rmse_ratio_to_lfm": metrics.well_pooled_rmse / max(abs(lfm_baseline.well_pooled_rmse), 1e-12),
        "well_pooled_abs_bias": abs(metrics.well_pooled_bias),
        "lfm_well_pooled_abs_bias": abs(lfm_baseline.well_pooled_bias),
        "well_fraction_improved_from_pretrain": float(np.mean(pretrain_well_ratios < 1.0)),
        "well_fraction_not_degraded_from_lfm": float(np.mean(lfm_well_ratios <= 1.0)),
        "lfm_drift_rmse": float(metrics.lfm_drift_rmse),
        "short_wave_energy_fraction": float(metrics.short_wave_energy_fraction),
        "roughness_ratio": float(metrics.roughness_ratio),
        "support_contiguous_fraction": float(metrics.support_contiguous_fraction),
        "orientation_disagreement_rms_ratio": float(metrics.orientation_disagreement_rms_ratio),
    }
    if details["masked_corr_change_from_pretrain_median"] < -threshold.masked_corr_drop_tolerance or masked_shape_ratio > threshold.masked_shape_ratio:
        failed.append("masked_shape")
    if details["visible_corr_change_from_pretrain_median"] < -threshold.visible_corr_drop_tolerance or visible_shape_ratio > threshold.visible_shape_ratio:
        failed.append("visible_center_shape")
    if (
        details["well_pooled_rmse_ratio_to_pretrain"] >= 1.0
        or details["well_pooled_rmse_ratio_to_lfm"] > threshold.well_lfm_rmse_ratio
        or details["well_fraction_improved_from_pretrain"] < threshold.well_fraction_improved
    ):
        failed.append("trusted_well_body")
    if metrics.lfm_drift_rmse > threshold.lfm_drift_rmse:
        failed.append("lfm_drift")
    if metrics.short_wave_energy_fraction > threshold.short_wave_energy_fraction or metrics.roughness_ratio > threshold.roughness_ratio:
        failed.append("roughness")
    if metrics.support_contiguous_fraction < 1.0 or not np.isfinite(metrics.orientation_disagreement_rms_ratio):
        failed.append("orientation_support")
    return GateReport(
        passed=not failed,
        failed_gates=tuple(failed),
        first_failed_gate=failed[0] if failed else None,
        details=details,
    )


def gate_improved(current: EvaluationMetrics, previous: EvaluationMetrics, gate: str, *, threshold: float = 0.01) -> bool:
    """Return whether the named failed gate improved by at least one percent."""

    if gate == "masked_shape":
        current_value = _median(current.masked_shape_loss)
        previous_value = _median(previous.masked_shape_loss)
    elif gate == "visible_center_shape":
        current_value = _median(current.visible_shape_loss)
        previous_value = _median(previous.visible_shape_loss)
    elif gate == "trusted_well_body":
        current_value = current.well_pooled_rmse
        previous_value = previous.well_pooled_rmse
    elif gate == "lfm_drift":
        current_value = current.lfm_drift_rmse
        previous_value = previous.lfm_drift_rmse
    elif gate == "roughness":
        current_value = max(current.short_wave_energy_fraction, current.roughness_ratio)
        previous_value = max(previous.short_wave_energy_fraction, previous.roughness_ratio)
    else:
        return False
    return _relative_improvement(current_value, previous_value) >= float(threshold)


__all__ = [
    "EvaluationMetrics",
    "GateReport",
    "GateThresholds",
    "evaluate_gates",
    "gate_improved",
]
