"""Gap-aware time-domain reference conditioning and three-band log-AI split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cup.seismic.lfm_time import lowpass_twt_log
from cup.utils.masks import true_runs
from cup.well.gaps import fill_short_joint_gaps
from cup.well.las import StandardVpRhoLogs
from wtie.processing import grid


@dataclass(frozen=True)
class ReferenceConditioningConfig:
    max_short_gap_s: float = 0.010
    hampel_window_samples: int = 7
    hampel_sigma: float = 4.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_short_gap_s) or self.max_short_gap_s < 0.0:
            raise ValueError("max_short_gap_s must be finite and non-negative.")
        if self.hampel_window_samples < 3 or self.hampel_window_samples % 2 == 0:
            raise ValueError("hampel_window_samples must be an odd integer >= 3.")
        if not np.isfinite(self.hampel_sigma) or self.hampel_sigma <= 0.0:
            raise ValueError("hampel_sigma must be positive and finite.")


@dataclass(frozen=True)
class ThreeBandSplitConfig:
    lfm_cutoff_hz: float
    ginn_cutoff_hz: float
    reference_cutoff_hz: float
    filter_order: int = 6
    buffer_seconds: float | None = None
    buffer_mode: str = "reflect"

    def validate(self, dt_s: float) -> None:
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive and finite, got {dt_s}.")
        nyquist = 0.5 / float(dt_s)
        cutoffs = (
            float(self.lfm_cutoff_hz),
            float(self.ginn_cutoff_hz),
            float(self.reference_cutoff_hz),
        )
        if not (0.0 < cutoffs[0] < cutoffs[1] < cutoffs[2] < nyquist):
            raise ValueError(
                "Frequency cutoffs must satisfy "
                f"0 < lfm < ginn < reference < Nyquist ({nyquist:g} Hz), got {cutoffs}."
            )
        if int(self.filter_order) < 2 or int(self.filter_order) % 2 != 0:
            raise ValueError("filter_order must be a positive even integer for zero-phase filtering.")
        if self.buffer_seconds is not None and (
            not np.isfinite(self.buffer_seconds) or self.buffer_seconds < 0.0
        ):
            raise ValueError("buffer_seconds must be finite and non-negative when provided.")
        if self.buffer_mode not in {"reflect", "edge"}:
            raise ValueError("buffer_mode must be 'reflect' or 'edge'.")


@dataclass(frozen=True)
class ConditionedWellLog:
    log_ai: grid.Log
    observed_mask: np.ndarray
    interpolation_mask: np.ndarray
    conditioned_mask: np.ndarray

    def __post_init__(self) -> None:
        shape = np.asarray(self.log_ai.values).shape
        for name in ("observed_mask", "interpolation_mask", "conditioned_mask"):
            value = np.asarray(getattr(self, name), dtype=bool)
            if value.shape != shape:
                raise ValueError(f"{name} must match log_ai values.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class WellFrequencyBands:
    reference_log_ai: grid.Log
    lfm_log_ai: grid.Log
    ginn_target_log_ai: grid.Log
    ginn_band_log_ai: grid.Log
    enhance_residual_log_ai: grid.Log
    observed_mask: np.ndarray
    interpolation_mask: np.ndarray
    conditioned_mask: np.ndarray
    valid_band_mask: np.ndarray

    def __post_init__(self) -> None:
        logs = (
            self.reference_log_ai,
            self.lfm_log_ai,
            self.ginn_target_log_ai,
            self.ginn_band_log_ai,
            self.enhance_residual_log_ai,
        )
        reference_basis = np.asarray(self.reference_log_ai.basis, dtype=np.float64)
        reference_shape = np.asarray(self.reference_log_ai.values).shape
        for log in logs[1:]:
            if np.asarray(log.values).shape != reference_shape or not np.allclose(
                log.basis,
                reference_basis,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("All frequency-band logs must share one TWT axis.")
        for name in (
            "observed_mask",
            "interpolation_mask",
            "conditioned_mask",
            "valid_band_mask",
        ):
            value = np.asarray(getattr(self, name), dtype=bool)
            if value.shape != reference_shape:
                raise ValueError(f"{name} must match frequency-band log values.")
            object.__setattr__(self, name, value)

    @property
    def reference_ai(self) -> grid.Log:
        values = np.full(self.reference_log_ai.values.shape, np.nan, dtype=np.float64)
        values[self.valid_band_mask] = np.exp(self.reference_log_ai.values[self.valid_band_mask])
        return grid.Log(
            values,
            self.reference_log_ai.basis,
            "twt",
            name="reference_AI",
            unit="m/s*g/cm3",
            allow_nan=True,
        )

    @property
    def lfm_ai(self) -> grid.Log:
        values = np.full(self.lfm_log_ai.values.shape, np.nan, dtype=np.float64)
        values[self.valid_band_mask] = np.exp(self.lfm_log_ai.values[self.valid_band_mask])
        return grid.Log(
            values,
            self.lfm_log_ai.basis,
            "twt",
            name="lfm_AI",
            unit="m/s*g/cm3",
            allow_nan=True,
        )

    @property
    def ginn_target_ai(self) -> grid.Log:
        values = np.full(self.ginn_target_log_ai.values.shape, np.nan, dtype=np.float64)
        values[self.valid_band_mask] = np.exp(self.ginn_target_log_ai.values[self.valid_band_mask])
        return grid.Log(
            values,
            self.ginn_target_log_ai.basis,
            "twt",
            name="ginn_target_AI",
            unit="m/s*g/cm3",
            allow_nan=True,
        )


def _hampel(values: np.ndarray, window: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(values, dtype=np.float64).copy()
    replaced = np.zeros(out.shape, dtype=bool)
    radius = int(window) // 2
    for start, stop in true_runs(np.isfinite(out)):
        if stop - start < window:
            continue
        source = out[start:stop].copy()
        for local in range(source.size):
            lo = max(0, local - radius)
            hi = min(source.size, local + radius + 1)
            sample = source[lo:hi]
            median = float(np.median(sample))
            mad = float(np.median(np.abs(sample - median)))
            scale = 1.4826 * mad
            if scale > 0.0 and abs(float(source[local]) - median) > float(sigma) * scale:
                out[start + local] = median
                replaced[start + local] = True
    return out, replaced


def _segmented_interp(
    x: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    out = np.full(target.shape, np.nan, dtype=np.float64)
    for start, stop in true_runs(np.isfinite(values)):
        if stop - start < 2:
            continue
        inside = (target >= x[start]) & (target <= x[stop - 1])
        out[inside] = np.interp(target[inside], x[start:stop], values[start:stop])
    return out


def _project_support_mask(
    md: np.ndarray,
    source_mask: np.ndarray,
    target_md: np.ndarray,
) -> np.ndarray:
    contamination = _segmented_interp(md, np.asarray(source_mask, dtype=float), target_md)
    return np.isfinite(contamination) & (contamination > 1e-9)


def build_conditioned_reference_log(
    standard: StandardVpRhoLogs,
    table: grid.TimeDepthTable,
    target_twt_s: np.ndarray,
    cfg: ReferenceConditioningConfig,
) -> ConditionedWellLog:
    """Condition standard LAS AI and sample it on a regular seismic TWT axis."""
    if not table.is_md_domain:
        raise ValueError("Reference conditioning requires an MD-domain TimeDepthTable.")
    target_twt = np.asarray(target_twt_s, dtype=np.float64).reshape(-1)
    if target_twt.size < 2 or np.any(~np.isfinite(target_twt)) or np.any(np.diff(target_twt) <= 0.0):
        raise ValueError("target_twt_s must be a finite, strictly increasing axis.")

    logs, interpolation_md = fill_short_joint_gaps(
        standard,
        table,
        max_short_gap_s=cfg.max_short_gap_s,
    )
    md = np.asarray(logs.basis, dtype=np.float64)
    ai = np.asarray(logs.AI.values, dtype=np.float64)
    valid = np.isfinite(ai) & (ai > 0.0)
    log_ai_md = np.full(ai.shape, np.nan, dtype=np.float64)
    log_ai_md[valid] = np.log(ai[valid])
    conditioned_md, hampel_md = _hampel(
        log_ai_md,
        cfg.hampel_window_samples,
        cfg.hampel_sigma,
    )

    target_md = np.interp(target_twt, table.twt, table.md, left=np.nan, right=np.nan)
    values = _segmented_interp(md, conditioned_md, target_md)
    interpolation_mask = _project_support_mask(md, interpolation_md, target_md)
    conditioned_mask = _project_support_mask(md, hampel_md, target_md)
    valid_target = np.isfinite(values) & np.isfinite(target_md)
    observed_mask = valid_target & ~interpolation_mask & ~conditioned_mask
    log = grid.Log(
        values,
        target_twt,
        "twt",
        name="conditioned_log_AI",
        unit="ln(m/s*g/cm3)",
        allow_nan=True,
    )
    return ConditionedWellLog(
        log_ai=log,
        observed_mask=observed_mask,
        interpolation_mask=interpolation_mask,
        conditioned_mask=conditioned_mask,
    )


def project_standard_log_ai_to_twt(
    standard: StandardVpRhoLogs,
    table: grid.TimeDepthTable,
    target_twt_s: np.ndarray,
    *,
    name: str = "projected_log_AI",
) -> grid.Log:
    """Project standard LAS AI to TWT without conditioning or filtering."""
    if not table.is_md_domain:
        raise ValueError("Filtered-LAS projection requires an MD-domain TimeDepthTable.")
    target_twt = np.asarray(target_twt_s, dtype=np.float64).reshape(-1)
    if target_twt.size < 2 or np.any(~np.isfinite(target_twt)) or np.any(np.diff(target_twt) <= 0.0):
        raise ValueError("target_twt_s must be a finite, strictly increasing axis.")

    md = np.asarray(standard.logs.basis, dtype=np.float64)
    ai = np.asarray(standard.logs.AI.values, dtype=np.float64)
    log_ai_md = np.full(ai.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ai) & (ai > 0.0)
    log_ai_md[valid] = np.log(ai[valid])
    target_md = np.interp(target_twt, table.twt, table.md, left=np.nan, right=np.nan)
    values = _segmented_interp(md, log_ai_md, target_md)
    values[~np.isfinite(target_md)] = np.nan
    return grid.Log(
        values,
        target_twt,
        "twt",
        name=name,
        unit="ln(m/s*g/cm3)",
        allow_nan=True,
    )


def segmented_lowpass(log: grid.Log, cutoff_hz: float, cfg: ThreeBandSplitConfig) -> grid.Log:
    """Apply one Butterworth low-pass independently to each finite run."""
    values = np.asarray(log.values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    min_samples = max(8, 3 * int(cfg.filter_order) + 2)
    for start, stop in true_runs(np.isfinite(values)):
        if stop - start < min_samples:
            continue
        segment = grid.Log(
            values[start:stop],
            log.basis[start:stop],
            "twt",
            name=log.name,
            unit=log.unit,
            allow_nan=False,
        )
        filtered = lowpass_twt_log(
            segment,
            cutoff_hz=float(cutoff_hz),
            order=int(cfg.filter_order),
            buffer_seconds=cfg.buffer_seconds,
            buffer_mode=cfg.buffer_mode,
        )
        out[start:stop] = filtered.values
    return grid.Log(
        out,
        log.basis,
        "twt",
        name=log.name,
        unit=log.unit,
        allow_nan=True,
    )


def build_frequency_bands(conditioned: ConditionedWellLog, cfg: ThreeBandSplitConfig) -> WellFrequencyBands:
    dt_s = float(conditioned.log_ai.sampling_rate)
    cfg.validate(dt_s)
    reference = segmented_lowpass(conditioned.log_ai, cfg.reference_cutoff_hz, cfg)
    lfm = segmented_lowpass(conditioned.log_ai, cfg.lfm_cutoff_hz, cfg)
    ginn_target = segmented_lowpass(conditioned.log_ai, cfg.ginn_cutoff_hz, cfg)
    valid = np.isfinite(reference.values) & np.isfinite(lfm.values) & np.isfinite(ginn_target.values)
    ginn_band = np.full(reference.values.shape, np.nan, dtype=np.float64)
    enhance = np.full(reference.values.shape, np.nan, dtype=np.float64)
    ginn_band[valid] = ginn_target.values[valid] - lfm.values[valid]
    enhance[valid] = reference.values[valid] - ginn_target.values[valid]

    def make(values: np.ndarray, name: str) -> grid.Log:
        return grid.Log(
            values,
            reference.basis,
            "twt",
            name=name,
            unit="ln(m/s*g/cm3)",
            allow_nan=True,
        )

    return WellFrequencyBands(
        reference_log_ai=make(reference.values, "reference_log_AI"),
        lfm_log_ai=make(lfm.values, "lfm_log_AI"),
        ginn_target_log_ai=make(ginn_target.values, "ginn_target_log_AI"),
        ginn_band_log_ai=make(ginn_band, "ginn_band_log_AI"),
        enhance_residual_log_ai=make(enhance, "enhance_residual_log_AI"),
        observed_mask=conditioned.observed_mask,
        interpolation_mask=conditioned.interpolation_mask,
        conditioned_mask=conditioned.conditioned_mask,
        valid_band_mask=valid,
    )


def replace_ginn_target(
    bands: WellFrequencyBands,
    ginn_target_log_ai: grid.Log,
) -> WellFrequencyBands:
    """Replace the GINN target while preserving reference, LFM, and provenance."""
    basis = np.asarray(bands.reference_log_ai.basis, dtype=np.float64)
    target_basis = np.asarray(ginn_target_log_ai.basis, dtype=np.float64)
    if target_basis.shape != basis.shape or not np.allclose(target_basis, basis, rtol=0.0, atol=1e-9):
        raise ValueError("Replacement GINN target must use the frequency-band TWT axis.")

    reference_values = np.asarray(bands.reference_log_ai.values, dtype=np.float64)
    lfm_values = np.asarray(bands.lfm_log_ai.values, dtype=np.float64)
    target_values = np.asarray(ginn_target_log_ai.values, dtype=np.float64)
    valid = np.isfinite(reference_values) & np.isfinite(lfm_values) & np.isfinite(target_values)
    ginn_band = np.full(reference_values.shape, np.nan, dtype=np.float64)
    enhance = np.full(reference_values.shape, np.nan, dtype=np.float64)
    ginn_band[valid] = target_values[valid] - lfm_values[valid]
    enhance[valid] = reference_values[valid] - target_values[valid]

    def make(values: np.ndarray, name: str) -> grid.Log:
        return grid.Log(
            values,
            basis,
            "twt",
            name=name,
            unit="ln(m/s*g/cm3)",
            allow_nan=True,
        )

    return WellFrequencyBands(
        reference_log_ai=bands.reference_log_ai,
        lfm_log_ai=bands.lfm_log_ai,
        ginn_target_log_ai=make(target_values, "ginn_target_log_AI"),
        ginn_band_log_ai=make(ginn_band, "ginn_band_log_AI"),
        enhance_residual_log_ai=make(enhance, "enhance_residual_log_AI"),
        observed_mask=bands.observed_mask,
        interpolation_mask=bands.interpolation_mask,
        conditioned_mask=bands.conditioned_mask,
        valid_band_mask=valid,
    )


def ginn_cutoff_candidates(
    right_half_amplitude_hz: float,
    *,
    minimum_ratio: float = 0.4,
    maximum_ratio: float = 1.3,
    step_hz: float = 5.0,
) -> list[float]:
    if (
        not np.isfinite(right_half_amplitude_hz)
        or right_half_amplitude_hz <= 0.0
        or not np.isfinite(minimum_ratio)
        or not np.isfinite(maximum_ratio)
        or not 0.0 < minimum_ratio < maximum_ratio
    ):
        raise ValueError("Candidate ratios and right half-amplitude frequency are invalid.")
    if not np.isfinite(step_hz) or step_hz <= 0.0:
        raise ValueError("step_hz must be positive and finite.")
    lower = np.floor(float(right_half_amplitude_hz) * float(minimum_ratio) / step_hz) * step_hz
    upper = np.ceil(float(right_half_amplitude_hz) * float(maximum_ratio) / step_hz) * step_hz
    return [float(value) for value in np.arange(lower, upper + 0.5 * step_hz, step_hz)]
