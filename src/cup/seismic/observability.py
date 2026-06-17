"""Forward-observability analysis for time-domain acoustic impedance.

The module contains the numerical core for the research gate described in
``docs/spec/forward-observability-gate.md``.  It deliberately has no run
directory discovery or project-data I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.signal.windows import tukey


FFT_CONVENTION = "numpy_forward_exp_minus_i_2pi_kn_over_n"
DIFFERENCE_CONVENTION = "r[j]=tanh((x[j]-x[j-1])/2), basis=t[j]"
CONVOLUTION_CONVENTION = "numpy.convolve(wavelet, reflectivity, mode='same')"


@dataclass(frozen=True)
class WaveletScenario:
    """One normalized wavelet uncertainty scenario."""

    name: str
    kind: str
    time_s: np.ndarray
    amplitude: np.ndarray
    source_well: str = ""


@dataclass(frozen=True)
class ObservabilityWindow:
    """One whole-target or adjacent-zone time window."""

    window_id: str
    window_type: str
    top_horizon: str
    bottom_horizon: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class WeightedAmplitudeFit:
    """Positive nuisance-amplitude fit in standardized observed units."""

    observed_standardized: np.ndarray
    synthetic_centered: np.ndarray
    residual: np.ndarray
    scale: float
    observed_rms: float
    synthetic_rms: float
    mismatch_rms: float
    corr: float
    nmae: float


@dataclass(frozen=True)
class PhaseBasis:
    """Weighted-orthonormal sine/cosine basis for one frequency."""

    values: np.ndarray
    weights: np.ndarray
    condition_number: float


def regular_dt(time_s: np.ndarray, *, label: str = "time") -> float:
    """Return the regular positive sampling interval of a 1-D time axis."""
    time = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if time.size < 2:
        raise ValueError(f"{label} must contain at least two samples.")
    diffs = np.diff(time)
    if np.any(~np.isfinite(diffs)) or np.any(diffs <= 0.0):
        raise ValueError(f"{label} must be finite and strictly increasing.")
    dt = float(np.median(diffs))
    if not np.allclose(diffs, dt, rtol=1e-5, atol=1e-9):
        raise ValueError(f"{label} must be regularly sampled.")
    return dt


def l2_normalize(values: np.ndarray) -> np.ndarray:
    """Return a finite, positive-energy vector with unit Euclidean norm."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("Cannot normalize an empty or non-finite vector.")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Cannot normalize a zero-energy vector.")
    return array / norm


def acoustic_reflectivity_from_log_ai(log_ai: np.ndarray) -> np.ndarray:
    """Compute exact vertical acoustic reflectivity on the lower sample."""
    values = np.asarray(log_ai, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("log_ai must contain at least two samples.")
    if np.any(~np.isfinite(values)):
        raise ValueError("log_ai must be finite.")
    return np.tanh(0.5 * np.diff(values))


def forward_log_ai(log_ai: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
    """Apply the project Robinson forward model to regularly sampled log(AI)."""
    reflectivity = acoustic_reflectivity_from_log_ai(log_ai)
    wavelet_values = np.asarray(wavelet, dtype=np.float64).reshape(-1)
    if wavelet_values.size < 3 or wavelet_values.size % 2 == 0:
        raise ValueError("wavelet must have odd length >= 3.")
    if np.any(~np.isfinite(wavelet_values)):
        raise ValueError("wavelet must be finite.")
    trace = np.convolve(wavelet_values, reflectivity, mode="same")
    if wavelet_values.size > reflectivity.size:
        difference = trace.size - reflectivity.size
        start = difference // 2
        trace = trace[start : start + reflectivity.size]
    return trace


def _zero_padded_length(size: int) -> int:
    target = max(256, int(size) * 8)
    return 1 << int(np.ceil(np.log2(target)))


def _center_pad(values: np.ndarray, n_fft: int) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if n_fft < array.size:
        raise ValueError("n_fft must not be shorter than the input.")
    start = (n_fft - array.size) // 2
    padded = np.zeros(n_fft, dtype=np.float64)
    padded[start : start + array.size] = array
    return padded, start


def constant_phase_rotate(values: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate constant phase without circular wrap-around."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    n_fft = _zero_padded_length(array.size)
    padded, start = _center_pad(array, n_fft)
    analytic = hilbert(padded)
    rotated = np.real(analytic * np.exp(1j * np.deg2rad(float(degrees))))
    return l2_normalize(rotated[start : start + array.size])


def fractional_time_shift(values: np.ndarray, *, dt_s: float, shift_s: float) -> np.ndarray:
    """Delay a wavelet by ``shift_s`` using a zero-padded Fourier phase ramp."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    n_fft = _zero_padded_length(array.size)
    padded, start = _center_pad(array, n_fft)
    frequency = np.fft.fftfreq(n_fft, d=float(dt_s))
    shifted = np.fft.ifft(
        np.fft.fft(padded) * np.exp(-2j * np.pi * frequency * float(shift_s))
    ).real
    return l2_normalize(shifted[start : start + array.size])


def make_artificial_wavelet_scenarios(
    nominal: WaveletScenario,
    *,
    phase_degrees: float = 10.0,
    fractional_shift_samples: float = 0.5,
) -> list[WaveletScenario]:
    """Create the four controlled phase/time perturbations of nominal."""
    dt_s = regular_dt(nominal.time_s, label="nominal wavelet time")
    scenarios: list[WaveletScenario] = []
    for sign in (-1.0, 1.0):
        degrees = sign * float(phase_degrees)
        scenarios.append(
            WaveletScenario(
                name=f"nominal_phase_{degrees:+g}deg",
                kind="artificial_phase",
                time_s=np.asarray(nominal.time_s, dtype=np.float64).copy(),
                amplitude=constant_phase_rotate(nominal.amplitude, degrees),
            )
        )
    for sign in (-1.0, 1.0):
        samples = sign * float(fractional_shift_samples)
        scenarios.append(
            WaveletScenario(
                name=f"nominal_shift_{samples:+g}dt",
                kind="artificial_shift",
                time_s=np.asarray(nominal.time_s, dtype=np.float64).copy(),
                amplitude=fractional_time_shift(
                    nominal.amplitude,
                    dt_s=dt_s,
                    shift_s=samples * dt_s,
                ),
            )
        )
    return scenarios


def frequency_grid(
    *,
    dt_s: float,
    start_hz: float,
    step_hz: float,
    configured_max_hz: float,
) -> np.ndarray:
    """Build the explicit Hz grid capped at 0.45 Nyquist."""
    values = [float(start_hz), float(step_hz), float(configured_max_hz), float(dt_s)]
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Frequency-grid values and dt_s must be positive finite numbers.")
    hard_max = 0.45 * (0.5 / float(dt_s))
    maximum = min(float(configured_max_hz), hard_max)
    if maximum + 1e-12 < float(start_hz):
        raise ValueError("Frequency-grid maximum is below start_hz.")
    count = int(np.floor((maximum - float(start_hz)) / float(step_hz) + 1e-10)) + 1
    return float(start_hz) + float(step_hz) * np.arange(count, dtype=np.float64)


def operator_transfer_rows(
    scenarios: Sequence[WaveletScenario],
    frequencies_hz: np.ndarray,
) -> pd.DataFrame:
    """Evaluate the exact discrete linearized operator at requested frequencies."""
    records: list[dict[str, Any]] = []
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    for scenario in scenarios:
        dt_s = regular_dt(scenario.time_s, label=f"{scenario.name} wavelet time")
        amplitude = np.asarray(scenario.amplitude, dtype=np.float64).reshape(-1)
        wavelet_response = np.asarray(
            [
                np.sum(amplitude * np.exp(-2j * np.pi * frequency * scenario.time_s))
                for frequency in frequencies
            ],
            dtype=np.complex128,
        )
        difference_response = 0.5 * (1.0 - np.exp(-2j * np.pi * frequencies * dt_s))
        combined = wavelet_response * difference_response
        absolute = np.abs(combined)
        maximum = float(np.max(absolute)) if absolute.size else 0.0
        normalized = absolute / maximum if maximum > 0.0 else np.zeros_like(absolute)
        for index, frequency in enumerate(frequencies):
            support = support_class(float(normalized[index]))
            records.append(
                {
                    "wavelet_scenario": scenario.name,
                    "wavelet_scenario_kind": scenario.kind,
                    "source_well": scenario.source_well,
                    "frequency_hz": float(frequency),
                    "wavelet_magnitude": float(abs(wavelet_response[index])),
                    "wavelet_phase_rad": float(np.angle(wavelet_response[index])),
                    "difference_magnitude": float(abs(difference_response[index])),
                    "difference_phase_rad": float(np.angle(difference_response[index])),
                    "combined_magnitude_absolute": float(absolute[index]),
                    "combined_magnitude_normalized": float(normalized[index]),
                    "combined_phase_rad": float(np.angle(combined[index])),
                    "operator_support_class": support,
                    "fft_convention": FFT_CONVENTION,
                    "difference_convention": DIFFERENCE_CONVENTION,
                    "convolution_convention": CONVOLUTION_CONVENTION,
                    "wavelet_center_convention": "odd_length_zero_time_center_sample",
                }
            )
    return pd.DataFrame.from_records(records)


def support_class(normalized_magnitude: float) -> str:
    """Map normalized operator magnitude to the documented support class."""
    value = float(normalized_magnitude)
    if not np.isfinite(value):
        return "unsupported"
    if value >= 0.5:
        return "core"
    if value >= 0.1:
        return "weak"
    return "unsupported"


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean with strictly positive total finite weight."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if array.shape != weight.shape:
        raise ValueError("values and weights must have matching shapes.")
    valid = np.isfinite(array) & np.isfinite(weight) & (weight >= 0.0)
    total = float(np.sum(weight[valid]))
    if total <= 0.0:
        raise ValueError("weights have no positive finite mass.")
    return float(np.sum(weight[valid] * array[valid]) / total)


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted root mean square."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    mean_square = weighted_mean(array * array, weights)
    return float(np.sqrt(max(mean_square, 0.0)))


def weighted_inner(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    """Weighted inner product normalized by total weight."""
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("inner-product arrays must have matching shapes.")
    return weighted_mean(a * b, weights)


def weighted_center(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Subtract the weighted mean."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array - weighted_mean(array, weights)


def make_phase_basis(
    time_s: np.ndarray,
    *,
    frequency_hz: float,
    alpha: float = 0.5,
    max_condition_number: float = 1e6,
) -> PhaseBasis:
    """Build a weighted-orthonormal two-column sine/cosine phase basis."""
    time = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if time.size < 2:
        raise ValueError("phase basis requires at least two samples.")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("Tukey alpha must be within [0, 1].")
    weights = tukey(time.size, alpha=float(alpha), sym=True).astype(np.float64)
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("Tukey weights have zero mass.")
    phase = 2.0 * np.pi * float(frequency_hz) * (time - float(time[0]))
    raw = np.column_stack([np.sin(phase), np.cos(phase)])
    normalized_weights = weights / float(np.sum(weights))
    gram = raw.T @ (normalized_weights[:, None] * raw)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if np.any(~np.isfinite(eigenvalues)) or float(np.min(eigenvalues)) <= 0.0:
        raise ValueError("ill_conditioned_phase_basis")
    condition_number = float(np.max(eigenvalues) / np.min(eigenvalues))
    if condition_number > float(max_condition_number):
        raise ValueError("ill_conditioned_phase_basis")
    inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    basis = raw @ inverse_sqrt
    return PhaseBasis(values=basis, weights=weights, condition_number=condition_number)


def weighted_amplitude_fit(
    observed: np.ndarray,
    synthetic: np.ndarray,
    weights: np.ndarray,
    *,
    min_synthetic_rms: float = 1e-6,
) -> WeightedAmplitudeFit:
    """Fit positive synthetic scale after weighted centering/standardization."""
    observed_values = np.asarray(observed, dtype=np.float64).reshape(-1)
    synthetic_values = np.asarray(synthetic, dtype=np.float64).reshape(-1)
    if observed_values.shape != synthetic_values.shape:
        raise ValueError("observed and synthetic must have matching shapes.")
    observed_centered = weighted_center(observed_values, weights)
    observed_std = weighted_rms(observed_centered, weights)
    if not np.isfinite(observed_std) or observed_std <= 0.0:
        raise ValueError("invalid_observed_energy")
    standardized = observed_centered / observed_std
    synthetic_centered = weighted_center(synthetic_values, weights)
    synthetic_rms = weighted_rms(synthetic_centered, weights)
    if not np.isfinite(synthetic_rms) or synthetic_rms < float(min_synthetic_rms):
        raise ValueError("invalid_low_synthetic_energy")
    denominator = weighted_inner(synthetic_centered, synthetic_centered, weights)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("invalid_low_synthetic_energy")
    scale = weighted_inner(standardized, synthetic_centered, weights) / denominator
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("invalid_nonpositive_scale")
    estimate = scale * synthetic_centered
    residual = standardized - estimate
    denom_corr = weighted_rms(standardized, weights) * weighted_rms(estimate, weights)
    corr = (
        weighted_inner(standardized, estimate, weights) / denom_corr
        if denom_corr > 0.0
        else float("nan")
    )
    weight = np.asarray(weights, dtype=np.float64)
    nmae_denominator = weighted_mean(np.abs(standardized), weight)
    nmae = (
        weighted_mean(np.abs(residual), weight) / nmae_denominator
        if nmae_denominator > 0.0
        else float("nan")
    )
    return WeightedAmplitudeFit(
        observed_standardized=standardized,
        synthetic_centered=synthetic_centered,
        residual=residual,
        scale=float(scale),
        observed_rms=float(observed_std),
        synthetic_rms=float(synthetic_rms),
        mismatch_rms=weighted_rms(residual, weights),
        corr=float(corr),
        nmae=float(nmae),
    )


def _weighted_singular_values(response: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(response, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != weight.size:
        raise ValueError("response must be [n_samples, n_basis] and match weights.")
    normalized = weight / float(np.sum(weight))
    return np.linalg.svd(np.sqrt(normalized)[:, None] * matrix, compute_uv=False)


def finite_difference_response(
    baseline_log_ai: np.ndarray,
    *,
    basis_full: np.ndarray,
    output_indices: np.ndarray,
    wavelet: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Return two finite-difference response columns on selected output samples."""
    baseline = np.asarray(baseline_log_ai, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis_full, dtype=np.float64)
    indices = np.asarray(output_indices, dtype=np.int64).reshape(-1)
    if basis.shape != (baseline.size, 2):
        raise ValueError("basis_full must have shape [len(log_ai), 2].")
    if not np.isfinite(epsilon) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive and finite.")
    columns = []
    for column in range(2):
        perturbation = basis[:, column]
        upper = forward_log_ai(baseline + float(epsilon) * perturbation, wavelet)
        lower = forward_log_ai(baseline - float(epsilon) * perturbation, wavelet)
        derivative = (upper - lower) / (2.0 * float(epsilon))
        columns.append(derivative[indices - 1])
    return np.column_stack(columns)


def response_sensitivities(
    response: np.ndarray,
    *,
    fit: WeightedAmplitudeFit,
    weights: np.ndarray,
) -> dict[str, float]:
    """Compute raw, fixed-scale and scale-marginalized conservative sensitivities."""
    raw = np.asarray(response, dtype=np.float64)
    centered = np.column_stack([weighted_center(raw[:, col], weights) for col in range(raw.shape[1])])
    denominator = weighted_inner(fit.synthetic_centered, fit.synthetic_centered, weights)
    perpendicular = np.empty_like(centered)
    for column in range(centered.shape[1]):
        projection = (
            weighted_inner(centered[:, column], fit.synthetic_centered, weights) / denominator
        )
        perpendicular[:, column] = centered[:, column] - projection * fit.synthetic_centered
    raw_sv = _weighted_singular_values(centered, weights)
    fixed_sv = _weighted_singular_values(fit.scale * centered, weights)
    marginalized_sv = _weighted_singular_values(fit.scale * perpendicular, weights)
    return {
        "sensitivity_raw": float(np.min(raw_sv)),
        "sensitivity_fixed_scale": float(np.min(fixed_sv)),
        "sensitivity_scale_marginalized": float(np.min(marginalized_sv)),
    }


def band_projection_rms(
    values: np.ndarray,
    *,
    phase_basis: PhaseBasis,
) -> float:
    """Weighted least-squares RMS of a signal projected onto the phase basis."""
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    basis = np.asarray(phase_basis.values, dtype=np.float64)
    if basis.shape[0] != signal.size:
        raise ValueError("phase basis and values must have matching sample counts.")
    weights = phase_basis.weights / float(np.sum(phase_basis.weights))
    gram = basis.T @ (weights[:, None] * basis)
    rhs = basis.T @ (weights * signal)
    coefficients = np.linalg.solve(gram, rhs)
    projected = basis @ coefficients
    return weighted_rms(projected, phase_basis.weights)


def analyze_frequency_scenario(
    *,
    time_s: np.ndarray,
    filtered_log_ai: np.ndarray,
    preprocessed_log_ai: np.ndarray,
    observed: np.ndarray,
    output_indices: np.ndarray,
    frequency_hz: float,
    scenario: WaveletScenario,
    epsilon: float = 1e-3,
    tukey_alpha: float = 0.5,
    max_basis_condition_number: float = 1e6,
    min_synthetic_rms: float = 1e-6,
) -> dict[str, Any]:
    """Analyze one well/window/frequency/wavelet scenario."""
    time = np.asarray(time_s, dtype=np.float64).reshape(-1)
    filtered = np.asarray(filtered_log_ai, dtype=np.float64).reshape(-1)
    preprocessed = np.asarray(preprocessed_log_ai, dtype=np.float64).reshape(-1)
    observed_values = np.asarray(observed, dtype=np.float64).reshape(-1)
    indices = np.asarray(output_indices, dtype=np.int64).reshape(-1)
    if filtered.shape != time.shape or preprocessed.shape != time.shape:
        raise ValueError("time and both log_ai arrays must have matching shapes.")
    if observed_values.size != time.size - 1:
        raise ValueError("observed must align with time_s[1:].")
    if indices.size == 0 or np.any(indices < 1) or np.any(indices >= time.size):
        raise ValueError("output_indices must select samples from time_s[1:].")
    if np.any(np.diff(indices) != 1):
        raise ValueError("output_indices must be one continuous run.")

    local_time = time[indices]
    phase_basis = make_phase_basis(
        local_time,
        frequency_hz=float(frequency_hz),
        alpha=float(tukey_alpha),
        max_condition_number=float(max_basis_condition_number),
    )
    basis_full = np.zeros((time.size, 2), dtype=np.float64)
    basis_full[indices, :] = phase_basis.values
    synthetic_full = forward_log_ai(filtered, scenario.amplitude)
    synthetic = synthetic_full[indices - 1]
    observed_window = observed_values[indices - 1]
    fit = weighted_amplitude_fit(
        observed_window,
        synthetic,
        phase_basis.weights,
        min_synthetic_rms=float(min_synthetic_rms),
    )

    filtered_response = finite_difference_response(
        filtered,
        basis_full=basis_full,
        output_indices=indices,
        wavelet=scenario.amplitude,
        epsilon=float(epsilon),
    )
    preprocessed_response = finite_difference_response(
        preprocessed,
        basis_full=basis_full,
        output_indices=indices,
        wavelet=scenario.amplitude,
        epsilon=float(epsilon),
    )
    filtered_sensitivity = response_sensitivities(
        filtered_response,
        fit=fit,
        weights=phase_basis.weights,
    )
    preprocessed_sensitivity = response_sensitivities(
        preprocessed_response,
        fit=fit,
        weights=phase_basis.weights,
    )
    filtered_band_rms = band_projection_rms(
        filtered[indices],
        phase_basis=phase_basis,
    )
    preprocessed_band_rms = band_projection_rms(
        preprocessed[indices],
        phase_basis=phase_basis,
    )
    residual_band_rms = band_projection_rms(
        fit.residual,
        phase_basis=phase_basis,
    )
    sensitivity = float(filtered_sensitivity["sensitivity_scale_marginalized"])
    if not np.isfinite(sensitivity) or sensitivity <= 0.0:
        raise ValueError("invalid_sensitivity")
    noise_equivalent = float(fit.mismatch_rms / sensitivity)
    detectability = (
        float(preprocessed_band_rms / noise_equivalent)
        if np.isfinite(noise_equivalent) and noise_equivalent > 0.0
        else float("nan")
    )
    return {
        **filtered_sensitivity,
        "preprocessed_sensitivity_raw": preprocessed_sensitivity["sensitivity_raw"],
        "preprocessed_sensitivity_fixed_scale": preprocessed_sensitivity["sensitivity_fixed_scale"],
        "preprocessed_sensitivity_scale_marginalized": preprocessed_sensitivity[
            "sensitivity_scale_marginalized"
        ],
        "phase_basis_condition_number": phase_basis.condition_number,
        "baseline_observed_rms": fit.observed_rms,
        "baseline_synthetic_rms": fit.synthetic_rms,
        "baseline_scale": fit.scale,
        "baseline_corr": fit.corr,
        "baseline_nmae": fit.nmae,
        "mismatch_rms": fit.mismatch_rms,
        "residual_band_rms": residual_band_rms,
        "filtered_log_ai_band_rms": filtered_band_rms,
        "preprocessed_log_ai_band_rms": preprocessed_band_rms,
        "conditioning_amplitude_ratio": (
            float(filtered_band_rms / preprocessed_band_rms)
            if preprocessed_band_rms > 0.0
            else float("nan")
        ),
        "conditioning_sensitivity_ratio": (
            float(sensitivity / preprocessed_sensitivity["sensitivity_scale_marginalized"])
            if preprocessed_sensitivity["sensitivity_scale_marginalized"] > 0.0
            else float("nan")
        ),
        "noise_equivalent_log_ai": noise_equivalent,
        "detectability_ratio": detectability,
        "n_valid_samples": int(indices.size),
        "n_cycles": float((local_time[-1] - local_time[0] + regular_dt(time)) * frequency_hz),
    }


def lower_empirical_quantile(values: Iterable[float], quantile: float) -> float:
    """Return NumPy's non-interpolating inverted-CDF empirical quantile."""
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, float(quantile), method="inverted_cdf"))


def aggregate_well_scenarios(
    rows: pd.DataFrame,
    *,
    admitted_candidate_count: int,
    required_artificial_count: int = 3,
) -> pd.DataFrame:
    """Aggregate scenario-level rows to conservative per-well evidence."""
    group_columns = [
        "well_name",
        "route",
        "spatial_cluster_id",
        "window_id",
        "window_type",
        "top_horizon",
        "bottom_horizon",
        "window_start_s",
        "window_end_s",
        "frequency_hz",
    ]
    records: list[dict[str, Any]] = []
    if rows.empty:
        return pd.DataFrame()
    if "detectability_ratio" not in rows.columns:
        rows = rows.copy()
        rows["detectability_ratio"] = np.nan
    for group_key, group in rows.groupby(group_columns, dropna=False):
        base = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
        valid = group[
            group["status"].eq("ok")
            & pd.to_numeric(group["detectability_ratio"], errors="coerce").notna()
        ]
        nominal_count = int(valid["wavelet_scenario_kind"].eq("nominal").sum())
        candidate_count = int(valid["wavelet_scenario_kind"].eq("candidate").sum())
        artificial_count = int(valid["wavelet_scenario_kind"].str.startswith("artificial_").sum())
        candidate_required = max(3, int(np.ceil(0.5 * int(admitted_candidate_count))))
        scenarios_ok = (
            nominal_count == 1
            and candidate_count >= candidate_required
            and artificial_count >= int(required_artificial_count)
        )
        detectability = (
            lower_empirical_quantile(valid["detectability_ratio"], 0.25)
            if scenarios_ok
            else float("nan")
        )
        records.append(
            {
                **base,
                "detectability_ratio": detectability,
                "scenario_status": "ok" if scenarios_ok else "insufficient_wavelet_scenarios",
                "valid_wavelet_scenario_count": int(len(valid)),
                "valid_candidate_wavelet_count": candidate_count,
                "valid_artificial_perturbation_count": artificial_count,
                "admitted_candidate_count": int(admitted_candidate_count),
                "required_candidate_wavelet_count": candidate_required,
            }
        )
    return pd.DataFrame.from_records(records)


def aggregate_frequency_evidence(
    well_rows: pd.DataFrame,
    operator_rows: pd.DataFrame,
    *,
    min_wells: int = 5,
    min_clusters: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-well evidence within clusters and then across clusters."""
    if well_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    valid = well_rows[
        well_rows["scenario_status"].eq("ok")
        & pd.to_numeric(well_rows["detectability_ratio"], errors="coerce").notna()
    ].copy()
    cluster_group = [
        "window_id",
        "window_type",
        "top_horizon",
        "bottom_horizon",
        "frequency_hz",
        "spatial_cluster_id",
    ]
    cluster_records: list[dict[str, Any]] = []
    for key, group in valid.groupby(cluster_group, dropna=False):
        cluster_records.append(
            {
                **dict(zip(cluster_group, key if isinstance(key, tuple) else (key,))),
                "valid_well_count": int(group["well_name"].nunique()),
                "cluster_detectability_ratio": float(np.median(group["detectability_ratio"])),
                "window_start_s_median": float(np.median(group["window_start_s"])),
                "window_end_s_median": float(np.median(group["window_end_s"])),
                "status": "ok",
                "reasons": "",
            }
        )
    clusters = pd.DataFrame.from_records(cluster_records)

    operator_summary = conservative_operator_support(operator_rows)
    evidence_group = cluster_group[:-1]
    evidence_records: list[dict[str, Any]] = []
    all_groups = well_rows.groupby(evidence_group, dropna=False)
    for key, all_well_group in all_groups:
        base = dict(zip(evidence_group, key if isinstance(key, tuple) else (key,)))
        matching_wells = valid
        for column, value in base.items():
            matching_wells = matching_wells[matching_wells[column].eq(value)]
        if clusters.empty:
            group = clusters
        else:
            group = clusters
            for column, value in base.items():
                group = group[group[column].eq(value)]
        n_wells = int(matching_wells["well_name"].nunique())
        n_clusters = int(group["spatial_cluster_id"].nunique()) if not group.empty else 0
        values = (
            group["cluster_detectability_ratio"].to_numpy(dtype=np.float64)
            if not group.empty
            else np.asarray([], dtype=np.float64)
        )
        median = float(np.median(values)) if values.size else float("nan")
        p25 = lower_empirical_quantile(values, 0.25)
        if n_wells < int(min_wells) or n_clusters < int(min_clusters):
            status = "insufficient_evidence"
        elif p25 >= 1.0:
            status = "robust_detectable"
        elif median >= 1.0:
            status = "conditional"
        else:
            status = "not_detectable"
        frequency = float(base["frequency_hz"])
        operator_match = operator_summary[
            np.isclose(operator_summary["frequency_hz"].to_numpy(dtype=float), frequency)
        ]
        operator_info = operator_match.iloc[0].to_dict() if len(operator_match) == 1 else {}
        evidence_records.append(
            {
                **base,
                "window_start_s_min": float(pd.to_numeric(all_well_group["window_start_s"]).min()),
                "window_start_s_max": float(pd.to_numeric(all_well_group["window_start_s"]).max()),
                "window_end_s_min": float(pd.to_numeric(all_well_group["window_end_s"]).min()),
                "window_end_s_max": float(pd.to_numeric(all_well_group["window_end_s"]).max()),
                "valid_well_count": n_wells,
                "valid_cluster_count": n_clusters,
                "cluster_median_detectability_ratio": median,
                "cluster_p25_detectability_ratio": p25,
                "evidence_status": status,
                "nominal_operator_support": operator_info.get(
                    "nominal_operator_support", "unsupported"
                ),
                "conservative_operator_support": operator_info.get(
                    "conservative_operator_support", "unsupported"
                ),
                "operator_normalized_magnitude_p25": operator_info.get(
                    "operator_normalized_magnitude_p25", float("nan")
                ),
            }
        )
    return clusters, pd.DataFrame.from_records(evidence_records)


def conservative_operator_support(operator_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize nominal and lower-P25 operator support across scenarios."""
    records = []
    for frequency, group in operator_rows.groupby("frequency_hz", dropna=False):
        values = group["combined_magnitude_normalized"].to_numpy(dtype=np.float64)
        p25 = lower_empirical_quantile(values, 0.25)
        nominal = group[group["wavelet_scenario_kind"].eq("nominal")]
        nominal_value = (
            float(nominal["combined_magnitude_normalized"].iloc[0])
            if len(nominal) == 1
            else float("nan")
        )
        records.append(
            {
                "frequency_hz": float(frequency),
                "nominal_operator_support": support_class(nominal_value),
                "conservative_operator_support": support_class(p25),
                "operator_normalized_magnitude_p25": p25,
            }
        )
    return pd.DataFrame.from_records(records)


def experiment_class(evidence_status: str, operator_support: str) -> str:
    """Map empirical evidence and conservative support to experiment class."""
    if operator_support == "unsupported":
        return "unsupported_or_unresolved"
    if evidence_status == "robust_detectable" and operator_support == "core":
        return "must_recover"
    if evidence_status in {"robust_detectable", "conditional"} and operator_support in {
        "core",
        "weak",
    }:
        return "stress_test"
    return "unsupported_or_unresolved"


def contiguous_experiment_ranges(
    evidence: pd.DataFrame,
    *,
    frequency_step_hz: float,
) -> list[dict[str, Any]]:
    """Collapse whole-target frequency rows into contiguous experiment ranges."""
    if evidence.empty:
        return []
    whole = evidence[evidence["window_type"].eq("whole_target")].copy()
    if whole.empty:
        return []
    whole["experiment_class"] = [
        experiment_class(status, support)
        for status, support in zip(
            whole["evidence_status"],
            whole["conservative_operator_support"],
        )
    ]
    whole = whole.sort_values("frequency_hz")
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in whole.to_dict(orient="records"):
        frequency = float(row["frequency_hz"])
        category = str(row["experiment_class"])
        if (
            current is None
            or category != current["experiment_class"]
            or not np.isclose(
                frequency - float(current["end_hz"]),
                float(frequency_step_hz),
                rtol=0.0,
                atol=1e-9,
            )
        ):
            current = {
                "experiment_class": category,
                "start_hz": frequency,
                "end_hz": frequency,
                "frequencies_hz": [frequency],
                "evidence_statuses": [str(row["evidence_status"])],
                "operator_supports": [str(row["conservative_operator_support"])],
            }
            records.append(current)
        else:
            current["end_hz"] = frequency
            current["frequencies_hz"].append(frequency)
            current["evidence_statuses"].append(str(row["evidence_status"]))
            current["operator_supports"].append(str(row["conservative_operator_support"]))
    return records


def dataclass_row(value: Any) -> dict[str, Any]:
    """Return ``asdict`` for public dataclasses without exposing NumPy arrays."""
    row = asdict(value)
    for key, item in list(row.items()):
        if isinstance(item, np.ndarray):
            row[key] = item.tolist()
    return row
