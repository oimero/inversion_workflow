"""Leakage-aware offline evaluation for Marmousi2 benchmark predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from wtie.processing import grid

from cup.lfm.math import apply_lfm_lowpass, parse_lowpass_spec
from cup.physics.numpy_backend import forward_time
from cup.seismic.geometry import SampleAxis
from cup.well.evaluation_support import load_evaluation_support_manifest


EVALUATION_SCOPES = ("validation", "final")
_KNOWN_ROLES = ("train", "validation", "test")


def _profile_time_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [profile,time] (or [1,profile,time]).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values; evaluation requires continuous finite support.")
    return array


def _safe_corr(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference.size < 2 or candidate.size != reference.size:
        return None
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        return None
    ref_centered = reference - float(np.mean(reference))
    cand_centered = candidate - float(np.mean(candidate))
    ref_norm = float(np.linalg.norm(ref_centered))
    cand_norm = float(np.linalg.norm(cand_centered))
    if ref_norm <= np.finfo(np.float64).eps or cand_norm <= np.finfo(np.float64).eps:
        return None
    value = float(np.dot(ref_centered, cand_centered) / (ref_norm * cand_norm))
    return value if np.isfinite(value) else None


def _error_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    source: str,
    role: str,
    units: str,
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate metric arrays must have matching shapes.")
    valid = np.isfinite(reference) & np.isfinite(candidate)
    ref = reference[valid]
    pred = candidate[valid]
    if ref.size == 0:
        return {
            "source": source,
            "role": role,
            "units": units,
            "support_count": 0,
            "rmse": None,
            "bias": None,
            "corr": None,
        }
    error = pred - ref
    return {
        "source": source,
        "role": role,
        "units": units,
        "support_count": int(ref.size),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "bias": float(np.mean(error)),
        "corr": _safe_corr(ref, pred),
    }


def _stat_summary(values: Iterable[float], *, source: str, role: str, units: str) -> dict[str, Any]:
    values_array = np.asarray(list(values), dtype=np.float64)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size == 0:
        return {
            "source": source,
            "role": role,
            "units": units,
            "support_count": 0,
            "mean": None,
            "median": None,
            "p95": None,
        }
    return {
        "source": source,
        "role": role,
        "units": units,
        "support_count": int(values_array.size),
        "mean": float(np.mean(values_array)),
        "median": float(np.median(values_array)),
        "p95": float(np.percentile(values_array, 95.0)),
    }


def _role_table(prepared_dir: Path, profile_count: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    path = prepared_dir / "evaluation" / "well_roles.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    required = {"well_name", "role", "profile_index"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"well_roles.csv is missing columns: {missing}")
    if table.empty:
        raise ValueError("well_roles.csv is empty.")
    normalized_roles: list[str] = []
    profile_indices: list[int] = []
    for row in table.itertuples(index=False):
        role = str(getattr(row, "role")).strip().casefold()
        if role not in _KNOWN_ROLES:
            raise ValueError(f"Unsupported well role: {role!r}")
        raw_index = float(getattr(row, "profile_index"))
        index = int(raw_index)
        if not np.isfinite(raw_index) or raw_index != index or not (0 <= index < profile_count):
            raise ValueError(f"Invalid profile_index in well_roles.csv: {raw_index!r}")
        normalized_roles.append(role)
        profile_indices.append(index)
    if len(set(profile_indices)) != len(profile_indices):
        raise ValueError("well_roles.csv contains duplicate profile_index values.")
    table = table.copy()
    table["role"] = normalized_roles
    table["profile_index"] = profile_indices
    by_role = {
        role: table.loc[table["role"].eq(role), "profile_index"].to_numpy(dtype=int)
        for role in _KNOWN_ROLES
    }
    return table, by_role


def _load_prepared_inputs(prepared_dir: Path) -> dict[str, Any]:
    truth_path = prepared_dir / "evaluation" / "truth.npz"
    if not truth_path.is_file():
        raise FileNotFoundError(truth_path)
    with np.load(truth_path, allow_pickle=False) as saved:
        missing = sorted({"log_ai", "x_m", "twt_s"} - set(saved.files))
        if missing:
            raise ValueError(f"truth.npz is missing arrays: {missing}")
        truth = _profile_time_array(saved["log_ai"], name="truth log_ai")
        x_m = np.asarray(saved["x_m"], dtype=np.float64)
        twt_s = np.asarray(saved["twt_s"], dtype=np.float64)
    if x_m.ndim != 1 or x_m.size != truth.shape[0] or not np.all(np.isfinite(x_m)):
        raise ValueError("truth x_m must be finite and match the profile axis.")
    if twt_s.ndim != 1 or twt_s.size != truth.shape[1] or not np.all(np.isfinite(twt_s)):
        raise ValueError("truth twt_s must be finite and match the time axis.")
    if np.any(np.diff(twt_s) <= 0.0):
        raise ValueError("truth twt_s must be strictly increasing.")
    dt_s = np.diff(twt_s)
    if dt_s.size == 0 or not np.allclose(dt_s, dt_s[0], rtol=1.0e-6, atol=1.0e-12):
        raise ValueError("Marmousi2 evaluation requires a regularly sampled TWT axis.")

    seismic_path = prepared_dir / "seismic.npz"
    if not seismic_path.is_file():
        raise FileNotFoundError(seismic_path)
    with np.load(seismic_path, allow_pickle=False) as saved:
        if "seismic" not in saved.files:
            raise ValueError("seismic.npz does not contain a seismic array.")
        seismic = np.asarray(saved["seismic"], dtype=np.float64)
    if seismic.ndim == 3 and seismic.shape[0] == 1:
        seismic = seismic[0]
    if seismic.shape != truth.shape or not np.all(np.isfinite(seismic)):
        raise ValueError("Prepared seismic must be finite and match truth [profile,time].")

    wavelet_path = prepared_dir / "wavelet" / "selected_wavelet.csv"
    if not wavelet_path.is_file():
        raise FileNotFoundError(wavelet_path)
    wavelet = pd.read_csv(wavelet_path)
    if not {"time_s", "amplitude"}.issubset(wavelet.columns):
        raise ValueError("selected_wavelet.csv must contain time_s and amplitude columns.")
    wavelet_time = wavelet["time_s"].to_numpy(dtype=np.float64)
    wavelet_amp = wavelet["amplitude"].to_numpy(dtype=np.float64)
    if (
        wavelet_time.ndim != 1
        or wavelet_time.size < 3
        or wavelet_amp.shape != wavelet_time.shape
        or not np.all(np.isfinite(wavelet_time))
        or not np.all(np.isfinite(wavelet_amp))
        or np.any(np.diff(wavelet_time) <= 0.0)
    ):
        raise ValueError("selected wavelet has an invalid finite time/amplitude axis.")
    if not np.allclose(np.diff(wavelet_time), dt_s[0], rtol=1.0e-6, atol=1.0e-12):
        raise ValueError("selected wavelet sample spacing differs from prepared TWT spacing.")

    roles, by_role = _role_table(prepared_dir, truth.shape[0])
    return {
        "truth": truth,
        "x_m": x_m,
        "twt_s": twt_s,
        "dt_s": float(dt_s[0]),
        "seismic": seismic,
        "wavelet_time_s": wavelet_time,
        "wavelet_amp": wavelet_amp,
        "roles": roles,
        "by_role": by_role,
        "truth_path": truth_path,
        "roles_path": prepared_dir / "evaluation" / "well_roles.csv",
        "seismic_path": seismic_path,
        "wavelet_path": wavelet_path,
    }


def _load_well_targets(prepared_dir: Path, truth: np.ndarray, twt_s: np.ndarray) -> dict[str, Any]:
    """Load the two well-target semantics emitted by ``prepare.py``."""
    path = prepared_dir / "evaluation" / "well_targets.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        required = {"profile_index", "fixed_log_ai", "body_log_ai", "twt_s"}
        missing = sorted(required - set(saved.files))
        if missing:
            raise ValueError(f"well_targets.npz is missing arrays: {missing}")
        indices = np.asarray(saved["profile_index"], dtype=np.int64)
        fixed = _profile_time_array(saved["fixed_log_ai"], name="fixed well targets")
        body = _profile_time_array(saved["body_log_ai"], name="body well targets")
        target_axis = np.asarray(saved["twt_s"], dtype=np.float64)
    if indices.ndim != 1 or indices.size != fixed.shape[0] or fixed.shape != body.shape:
        raise ValueError("well_targets.npz profile and target shapes do not agree.")
    if np.any(indices < 0) or np.any(indices >= truth.shape[0]):
        raise ValueError("well_targets.npz profile indices are outside truth support.")
    if target_axis.shape != twt_s.shape or not np.array_equal(target_axis, twt_s):
        raise ValueError("well_targets.npz TWT axis differs from truth.npz.")
    return {"path": path, "profile_index": indices, "fixed": fixed, "body": body}


def _well_target_metrics(
    prediction: np.ndarray,
    targets: Mapping[str, Any],
    *,
    source: str,
    well_indices: np.ndarray,
    sample_mask: np.ndarray,
) -> dict[str, Any]:
    selected = np.isin(targets["profile_index"], well_indices)
    indices = np.asarray(targets["profile_index"], dtype=int)[selected]
    fixed = np.where(sample_mask[indices], np.asarray(targets["fixed"])[selected], np.nan)
    body = np.where(sample_mask[indices], np.asarray(targets["body"])[selected], np.nan)
    predicted = np.where(sample_mask[indices], prediction[indices], np.nan)
    return {
        "profile_indices": [int(value) for value in indices],
        "fixed_original_well": _error_metrics(
            fixed,
            predicted,
            source=source,
            role="fixed_original_well_curve",
            units="natural_log(AI)",
        ),
        "body_target": _error_metrics(
            body,
            predicted,
            source=source,
            role="smoothed_body_target",
            units="natural_log(AI)",
        ),
    }


def _rows_for_scope(by_role: Mapping[str, np.ndarray], scope: str, profile_count: int) -> tuple[list[str], np.ndarray]:
    scope = str(scope).casefold()
    if scope not in EVALUATION_SCOPES:
        raise ValueError(f"scope must be one of {EVALUATION_SCOPES}, got {scope!r}.")
    roles = ["train", "validation"]
    if scope == "final":
        roles.append("test")
    indices = np.concatenate([np.asarray(by_role.get(role, []), dtype=int) for role in roles])
    indices = np.unique(indices)
    if indices.size == 0:
        raise ValueError(f"No well profiles are available for scope={scope!r}.")
    if np.any(indices < 0) or np.any(indices >= profile_count):
        raise ValueError("Selected role profile indices are outside the prepared profile axis.")
    return roles, indices


def _support_mask(
    seismic: np.ndarray,
    *,
    relative_threshold: float = 0.25,
) -> tuple[np.ndarray, float]:
    """Flag the profiles whose observed trace carries signal.

    PostM keeps muted edges (trace RMS about 1 % of a live trace) that hold
    truth but no data, so a section-wide RMSE over them measures extrapolation
    instead of inversion.  The threshold is relative to the median trace RMS of
    the whole prepared section, which keeps it independent of the requested
    scope and leaves a uniformly illuminated section untouched.
    """
    if not np.isfinite(relative_threshold) or relative_threshold <= 0.0:
        raise ValueError("support_relative_threshold must be finite and positive.")
    trace_rms = np.sqrt(np.mean(np.asarray(seismic, dtype=np.float64) ** 2, axis=1))
    median_trace_rms = float(np.median(trace_rms))
    return trace_rms >= relative_threshold * median_trace_rms, median_trace_rms


def _impedance_metrics(
    truth: np.ndarray,
    candidate: np.ndarray,
    *,
    indices: np.ndarray,
    source: str,
    label: str,
    sample_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    selected_truth = truth[indices]
    selected_candidate = candidate[indices]
    if sample_mask is not None:
        selected_truth = np.where(sample_mask[indices], selected_truth, np.nan)
        selected_candidate = np.where(sample_mask[indices], selected_candidate, np.nan)
    log_metrics = _error_metrics(
        selected_truth,
        selected_candidate,
        source=source,
        role=label,
        units="natural_log(AI)",
    )
    truth_ai = np.exp(selected_truth)
    candidate_ai = np.exp(selected_candidate)
    if np.any(np.isinf(truth_ai)) or np.any(np.isinf(candidate_ai)):
        raise ValueError("Exponentiating logAI produced non-finite linear AI metrics.")
    ai_metrics = _error_metrics(
        truth_ai,
        candidate_ai,
        source=source,
        role=label,
        units="m/s*g/cm3",
    )
    return {"log_ai": log_metrics, "linear_ai": ai_metrics}


def _by_role_metrics(
    truth: np.ndarray,
    candidate: np.ndarray,
    by_role: Mapping[str, np.ndarray],
    included_roles: Iterable[str],
    *,
    source: str,
    label: str,
    sample_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    roles = list(included_roles)
    indices = np.unique(np.concatenate([np.asarray(by_role.get(role, []), dtype=int) for role in roles]))
    result: dict[str, Any] = {
        "overall": _impedance_metrics(truth, candidate, indices=indices, source=source, label=label, sample_mask=sample_mask),
        "by_role": {},
    }
    for role in roles:
        role_indices = np.asarray(by_role.get(role, []), dtype=int)
        if role_indices.size:
            result["by_role"][role] = _impedance_metrics(
                truth,
                candidate,
                indices=role_indices,
                source=source,
                label=role,
                sample_mask=sample_mask,
            )
    return result


def _waveform_metrics(
    observed: np.ndarray,
    synthetic: np.ndarray,
    *,
    indices: np.ndarray,
    source: str,
    label: str,
    sample_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=np.float64)
    synthetic = np.asarray(synthetic, dtype=np.float64)
    if observed.shape != synthetic.shape:
        raise ValueError("Observed and synthetic seismic arrays must have matching shapes.")
    selected_observed = observed[indices]
    selected_synthetic = synthetic[indices]
    valid = np.isfinite(selected_observed) & np.isfinite(selected_synthetic)
    if sample_mask is not None:
        valid &= sample_mask[indices]
    values_observed = selected_observed[valid]
    values_synthetic = selected_synthetic[valid]
    profile_corrs = [
        _safe_corr(selected_observed[row][valid[row]], selected_synthetic[row][valid[row]])
        for row in range(selected_observed.shape[0])
        if np.all(np.isfinite(selected_observed[row])) and np.all(np.isfinite(selected_synthetic[row]))
    ]
    profile_corrs = [value for value in profile_corrs if value is not None]
    error = values_synthetic - values_observed
    return {
        "source": source,
        "role": label,
        "units": "seismic amplitude (relative)",
        "support_count": int(values_observed.size),
        "profile_count": int(indices.size),
        "rmse": None if values_observed.size == 0 else float(np.sqrt(np.mean(error * error))),
        "bias": None if values_observed.size == 0 else float(np.mean(error)),
        "waveform_corr": _safe_corr(values_observed, values_synthetic),
        "mean_waveform_corr": None if not profile_corrs else float(np.mean(profile_corrs)),
        "median_waveform_corr": None if not profile_corrs else float(np.median(profile_corrs)),
        "corr_profile_support_count": int(len(profile_corrs)),
    }


def _by_role_waveform_metrics(
    observed: np.ndarray,
    synthetic: np.ndarray,
    by_role: Mapping[str, np.ndarray],
    included_roles: Iterable[str],
    *,
    source: str,
    label: str,
    sample_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    roles = list(included_roles)
    indices = np.unique(np.concatenate([np.asarray(by_role.get(role, []), dtype=int) for role in roles]))
    result = {
        "overall": _waveform_metrics(observed, synthetic, indices=indices, source=source, label=label, sample_mask=sample_mask),
        "by_role": {},
    }
    for role in roles:
        role_indices = np.asarray(by_role.get(role, []), dtype=int)
        if role_indices.size:
            result["by_role"][role] = _waveform_metrics(
                observed,
                synthetic,
                indices=role_indices,
                source=source,
                label=role,
                sample_mask=sample_mask,
            )
    return result


def _lowpass_profiles(values: np.ndarray, twt_s: np.ndarray, cutoff_hz: float) -> np.ndarray:
    cutoff = float(cutoff_hz)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff_hz must be finite and positive.")
    axis = SampleAxis(values=twt_s, domain="time", unit="s")
    config = {
        "enabled": True,
        "cutoff_hz": cutoff,
        "order": 4,
        "buffer_mode": "reflect",
        "buffer_axis_units": 0.2,
    }
    spec = parse_lowpass_spec(config, axis)
    output = np.empty_like(values, dtype=np.float64)
    for index, row in enumerate(values):
        log = grid.Log(row, twt_s, "twt", name="evaluation", allow_nan=False)
        filtered = np.asarray(apply_lfm_lowpass(log, spec).values, dtype=np.float64)
        if not np.all(np.isfinite(filtered)):
            raise ValueError("Low-pass evaluation produced non-finite samples.")
        output[index] = filtered
    return output


def _roughness_metrics(values: np.ndarray, *, indices: np.ndarray, source: str, label: str, dt_s: float) -> dict[str, Any]:
    derivative = np.diff(values[indices], axis=-1) / float(dt_s)
    rms = np.sqrt(np.mean(derivative * derivative, axis=-1))
    return _stat_summary(rms, source=source, role=label, units="natural_log(AI)/s")


def _highfrequency_metrics(
    values: np.ndarray,
    lowpassed: np.ndarray,
    *,
    indices: np.ndarray,
    source: str,
    label: str,
    cutoff_hz: float,
) -> dict[str, Any]:
    residual = values[indices] - lowpassed[indices]
    rms = np.sqrt(np.mean(residual * residual, axis=-1))
    fractions: list[float] = []
    for row, row_low in zip(values[indices], lowpassed[indices]):
        denominator = float(np.sum((row - np.mean(row)) ** 2))
        numerator = float(np.sum((row - row_low) ** 2))
        if denominator > np.finfo(np.float64).eps:
            fractions.append(numerator / denominator)
    result = _stat_summary(rms, source=source, role=label, units="natural_log(AI) RMS")
    result["relative_energy_fraction"] = _stat_summary(
        fractions,
        source=source,
        role=label,
        units="dimensionless high-frequency energy fraction",
    )
    result["cutoff_hz"] = float(cutoff_hz)
    return result


def _shortwave_metrics(
    values: np.ndarray,
    lowpassed: np.ndarray,
    *,
    indices: np.ndarray,
    source: str,
    label: str,
    cutoff_hz: float,
) -> dict[str, Any]:
    """Report residual energy above the explicit short-wave cutoff.

    The 5 Hz diagnostic is intentionally kept separate from this metric.  A
    sample above 5 Hz is not automatically called an oscillation; this metric
    only labels the residual above ``shortwave_cutoff_hz`` as short-wave.
    """

    result = _highfrequency_metrics(
        values,
        lowpassed,
        indices=indices,
        source=source,
        label=label,
        cutoff_hz=cutoff_hz,
    )
    result["units"] = "natural_log(AI) short-wave residual RMS"
    shortwave_fraction = dict(result["relative_energy_fraction"])
    shortwave_fraction["units"] = "dimensionless short-wave energy fraction"
    result["relative_energy_fraction"] = shortwave_fraction
    summary_keys = ("source", "role", "units", "support_count", "mean", "median", "p95")
    result["shortwave_rms"] = {key: result[key] for key in summary_keys}
    result["shortwave_energy_fraction"] = result["relative_energy_fraction"]
    result["description"] = "RMS and relative energy of the residual above shortwave_cutoff_hz"
    return result


def _structure_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    lfm: np.ndarray,
    *,
    indices: np.ndarray,
    twt_s: np.ndarray,
    dt_s: float,
    cutoff_hz: float,
    label: str,
    shortwave_cutoff_hz: float = 20.0,
) -> dict[str, Any]:
    # Restrict the low-pass work to the scope rows first.  Besides making the
    # validation report cheaper, this ensures that default diagnostics never
    # need to process held-out truth at unrelated profiles.
    selected_indices = np.asarray(indices, dtype=int)
    local_indices = np.arange(selected_indices.size, dtype=int)
    selected_truth = np.asarray(truth, dtype=np.float64)[selected_indices]
    selected_prediction = np.asarray(prediction, dtype=np.float64)[selected_indices]
    selected_lfm = np.asarray(lfm, dtype=np.float64)[selected_indices]
    pred_low = _lowpass_profiles(selected_prediction, twt_s, cutoff_hz)
    lfm_low = _lowpass_profiles(selected_lfm, twt_s, cutoff_hz)
    truth_low = _lowpass_profiles(selected_truth, twt_s, cutoff_hz)
    pred_short_low = _lowpass_profiles(selected_prediction, twt_s, shortwave_cutoff_hz)
    lfm_short_low = _lowpass_profiles(selected_lfm, twt_s, shortwave_cutoff_hz)
    truth_short_low = _lowpass_profiles(selected_truth, twt_s, shortwave_cutoff_hz)
    drift_source = "prediction-vs-LFM low-frequency drift"
    return {
        "prediction": _roughness_metrics(selected_prediction, indices=local_indices, source="prediction", label=label, dt_s=dt_s),
        "lfm_baseline": _roughness_metrics(selected_lfm, indices=local_indices, source="lfm_baseline", label=label, dt_s=dt_s),
        "truth_reference": _roughness_metrics(selected_truth, indices=local_indices, source="truth_reference_offline", label=label, dt_s=dt_s),
        "low_frequency_drift": {
            "against_lfm": _error_metrics(
                lfm_low,
                pred_low,
                source=drift_source,
                role=label,
                units="natural_log(AI)",
            ),
            "against_truth": _error_metrics(
                truth_low,
                pred_low,
                source="prediction-vs-truth low-frequency comparison (offline)",
                role=label,
                units="natural_log(AI)",
            ),
        },
        "highfrequency_energy": {
            "prediction": _highfrequency_metrics(
                selected_prediction,
                pred_low,
                indices=local_indices,
                source="prediction",
                label=label,
                cutoff_hz=cutoff_hz,
            ),
            "lfm_baseline": _highfrequency_metrics(
                selected_lfm,
                lfm_low,
                indices=local_indices,
                source="lfm_baseline",
                label=label,
                cutoff_hz=cutoff_hz,
            ),
            "truth_reference": _highfrequency_metrics(
                selected_truth,
                truth_low,
                indices=local_indices,
                source="truth_reference_offline",
                label=label,
                cutoff_hz=cutoff_hz,
            ),
        },
        "shortwave_energy": {
            "prediction": _shortwave_metrics(
                selected_prediction,
                pred_short_low,
                indices=local_indices,
                source="prediction",
                label=label,
                cutoff_hz=shortwave_cutoff_hz,
            ),
            "lfm_baseline": _shortwave_metrics(
                selected_lfm,
                lfm_short_low,
                indices=local_indices,
                source="lfm_baseline",
                label=label,
                cutoff_hz=shortwave_cutoff_hz,
            ),
            "truth_reference": _shortwave_metrics(
                selected_truth,
                truth_short_low,
                indices=local_indices,
                source="truth_reference_offline",
                label=label,
                cutoff_hz=shortwave_cutoff_hz,
            ),
        },
    }


def _profile_diagnostics(
    truth: np.ndarray,
    prediction: np.ndarray,
    lfm: np.ndarray,
    observed: np.ndarray,
    physical_prediction: np.ndarray,
    physical_lfm: np.ndarray,
    physical_truth: np.ndarray | None = None,
    *,
    indices: np.ndarray,
    twt_s: np.ndarray,
    dt_s: float,
    cutoff_hz: float,
    shortwave_cutoff_hz: float,
    label: str,
) -> dict[str, Any]:
    """Section diagnostics over an explicit set of profiles."""
    structure = _structure_metrics(
        truth,
        prediction,
        lfm,
        indices=indices,
        twt_s=twt_s,
        dt_s=dt_s,
        cutoff_hz=cutoff_hz,
        shortwave_cutoff_hz=shortwave_cutoff_hz,
        label=label,
    )
    source = "prepared/evaluation/truth.npz"
    physical_metrics = {
        "prediction": _waveform_metrics(
            observed,
            physical_prediction,
            indices=indices,
            source="prepared/seismic.npz + selected_wavelet.csv",
            label=label,
        ),
        "lfm_baseline": _waveform_metrics(
            observed,
            physical_lfm,
            indices=indices,
            source="prepared/seismic.npz + selected_wavelet.csv",
            label=label,
        ),
    }
    if physical_truth is not None:
        physical_metrics["truth_reference"] = _waveform_metrics(
            observed,
            physical_truth,
            indices=indices,
            source="prepared/evaluation/truth.npz + selected_wavelet.csv (offline truth forward)",
            label=label,
        )
    return {
        "support_count": int(np.asarray(indices, dtype=int).size * truth.shape[1]),
        "impedance_metrics": {
            "prediction": _impedance_metrics(
                truth,
                prediction,
                indices=indices,
                source=source,
                label=label,
            ),
            "lfm_baseline": _impedance_metrics(
                truth,
                lfm,
                indices=indices,
                source=source,
                label=label,
            ),
        },
        "physical_metrics": physical_metrics,
        "low_frequency_drift": structure["low_frequency_drift"],
        "roughness": {
            "prediction": structure["prediction"],
            "lfm_baseline": structure["lfm_baseline"],
            "truth_reference": structure["truth_reference"],
        },
        "highfrequency_energy": structure["highfrequency_energy"],
        "shortwave_energy": structure["shortwave_energy"],
    }


def evaluate_marmousi2_prediction(
    prepared_dir: Path,
    prediction_log_ai: Any,
    lfm_log_ai: Any,
    *,
    scope: str = "validation",
    cutoff_hz: float = 5.0,
    shortwave_cutoff_hz: float = 20.0,
    support_relative_threshold: float = 0.25,
    fixed_well_log_ai: Any | None = None,
    body_target_log_ai: Any | None = None,
) -> dict[str, Any]:
    """Evaluate one prediction against prepared truth under an explicit scope.

    The default ``validation`` scope includes only train and validation pseudo-
    wells.  Test-well and full-profile truth metrics are constructed only by
    explicitly requesting ``scope='final'``.  ``shortwave_cutoff_hz`` controls
    a separate short-wave energy diagnostic and defaults to 20 Hz.
    ``support_relative_threshold`` sets the observed-amplitude floor used by
    ``supported_profile`` / ``data_support``: a profile counts as supported when
    its trace RMS reaches that fraction of the section's median trace RMS.
    """

    prepared_dir = Path(prepared_dir).resolve()
    scope = str(scope).casefold()
    prepared = _load_prepared_inputs(prepared_dir)
    truth = prepared["truth"]
    prediction = _profile_time_array(prediction_log_ai, name="prediction_log_ai")
    lfm = _profile_time_array(lfm_log_ai, name="lfm_log_ai")
    if prediction.shape != truth.shape or lfm.shape != truth.shape:
        raise ValueError(
            f"prediction and LFM shapes must match truth {truth.shape}; got {prediction.shape} and {lfm.shape}."
        )
    included_roles, well_indices = _rows_for_scope(prepared["by_role"], scope, truth.shape[0])
    axis = SampleAxis(prepared["twt_s"], "time", "s")
    support_path = prepared_dir / "well_controls" / "qc" / "evaluation_support.json"
    supports = load_evaluation_support_manifest(support_path, sample_axis=axis)
    well_sample_mask = np.zeros(truth.shape, dtype=bool)
    support_windows = {}
    for row in prepared["roles"].itertuples(index=False):
        if int(row.profile_index) not in well_indices:
            continue
        support = supports[str(row.well_name)]
        well_sample_mask[int(row.profile_index), support.start_index:support.stop_index] = True
        support_windows[str(row.well_name)] = support.to_mapping()
    physical_prediction = forward_time(
        prediction,
        prepared["wavelet_time_s"],
        prepared["wavelet_amp"],
        sample_step_s=prepared["dt_s"],
    )
    physical_lfm = forward_time(
        lfm,
        prepared["wavelet_time_s"],
        prepared["wavelet_amp"],
        sample_step_s=prepared["dt_s"],
    )
    physical_truth = forward_time(
        truth,
        prepared["wavelet_time_s"],
        prepared["wavelet_amp"],
        sample_step_s=prepared["dt_s"],
    )
    observed = prepared["seismic"]
    if physical_prediction.shape != observed.shape or not np.all(np.isfinite(physical_prediction)):
        raise ValueError("Prediction physical forward model returned invalid seismic support.")
    if not np.all(np.isfinite(physical_lfm)) or not np.all(np.isfinite(physical_truth)):
        raise ValueError("LFM/truth physical forward model returned invalid seismic support.")

    well_targets = _load_well_targets(prepared_dir, truth, prepared["twt_s"])
    if fixed_well_log_ai is not None or body_target_log_ai is not None:
        if fixed_well_log_ai is None or body_target_log_ai is None:
            raise ValueError("fixed_well_log_ai and body_target_log_ai must be supplied together.")
        fixed = _profile_time_array(fixed_well_log_ai, name="fixed_well_log_ai")
        body = _profile_time_array(body_target_log_ai, name="body_target_log_ai")
        if well_targets is None:
            raise ValueError("Explicit well targets require evaluation/well_targets.npz profile indices.")
        if fixed.shape != body.shape or fixed.shape != np.asarray(well_targets["fixed"]).shape:
            raise ValueError("Explicit well target arrays do not match prepared well target shape.")
        well_targets = {**well_targets, "fixed": fixed, "body": body}

    prediction_impedance = _by_role_metrics(
        truth,
        prediction,
        prepared["by_role"],
        included_roles,
        source="prepared/evaluation/truth.npz",
        label="train+validation" if scope == "validation" else "train+validation+test",
        sample_mask=well_sample_mask,
    )
    lfm_impedance = _by_role_metrics(
        truth,
        lfm,
        prepared["by_role"],
        included_roles,
        source="prepared/evaluation/truth.npz",
        label="LFM baseline",
        sample_mask=well_sample_mask,
    )
    prediction_waveform = _by_role_waveform_metrics(
        observed,
        physical_prediction,
        prepared["by_role"],
        included_roles,
        source="prepared/seismic.npz + selected_wavelet.csv",
        label="prediction",
        sample_mask=well_sample_mask,
    )
    lfm_waveform = _by_role_waveform_metrics(
        observed,
        physical_lfm,
        prepared["by_role"],
        included_roles,
        source="prepared/seismic.npz + selected_wavelet.csv",
        label="LFM baseline",
        sample_mask=well_sample_mask,
    )
    truth_waveform = _by_role_waveform_metrics(
        observed,
        physical_truth,
        prepared["by_role"],
        included_roles,
        source="prepared/evaluation/truth.npz + selected_wavelet.csv (offline truth forward)",
        label="truth reference",
        sample_mask=well_sample_mask,
    )
    training_indices = np.asarray(prepared["by_role"].get("train", []), dtype=int)
    unlabeled_seismic_indices = np.setdiff1d(np.arange(observed.shape[0]), training_indices, assume_unique=True)
    unlabeled_seismic = {
        "profile_indices": [int(value) for value in unlabeled_seismic_indices],
        "uses_impedance_truth": False,
        "prediction": _waveform_metrics(
            observed,
            physical_prediction,
            indices=unlabeled_seismic_indices,
            source="prepared/seismic.npz + selected_wavelet.csv (unlabeled seismic only)",
            label="unlabeled_seismic",
        ),
        "lfm_baseline": _waveform_metrics(
            observed,
            physical_lfm,
            indices=unlabeled_seismic_indices,
            source="prepared/seismic.npz + selected_wavelet.csv (unlabeled seismic only)",
            label="unlabeled_seismic",
        ),
    }
    if scope == "final":
        unlabeled_seismic["truth_reference"] = _waveform_metrics(
            observed, physical_truth, indices=unlabeled_seismic_indices,
            source="prepared/evaluation/truth.npz + selected_wavelet.csv (offline truth only)",
            label="unlabeled_seismic",
        )
    structure = _structure_metrics(
        truth,
        prediction,
        lfm,
        indices=well_indices,
        twt_s=prepared["twt_s"],
        dt_s=prepared["dt_s"],
        cutoff_hz=cutoff_hz,
        shortwave_cutoff_hz=shortwave_cutoff_hz,
        label="train+validation" if scope == "validation" else "train+validation+test",
    )
    support_mask, median_trace_rms = _support_mask(
        observed,
        relative_threshold=support_relative_threshold,
    )
    result: dict[str, Any] = {
        "schema_version": "marmousi2_benchmark_evaluation_v1",
        "status": "ok",
        "prepared_dir": str(prepared_dir),
        "scope": scope,
        "evaluation_only": True,
        "model_selection_default": scope == "validation",
        "test_truth_used": scope == "final" and bool(prepared["by_role"].get("test", np.array([])).size),
        "full_truth_used": scope == "final",
        "roles_included": included_roles,
        "roles_excluded": [role for role in _KNOWN_ROLES if role not in included_roles],
        "well_profile_indices": [int(value) for value in well_indices],
        "support_count": int(well_sample_mask.sum()),
        "well_evaluation_support": {"manifest": str(support_path), "wells": support_windows},
        "section_evaluation_support_s": [float(axis.values[0]), float(axis.values[-1])],
        "data_support": {
            "definition": "profile RMS >= support_relative_threshold * median profile RMS of prepared/seismic.npz",
            "relative_threshold": float(support_relative_threshold),
            "median_trace_rms": median_trace_rms,
            "supported_trace_count": int(support_mask.sum()),
            "muted_trace_count": int((~support_mask).sum()),
            "well_count": int(well_indices.size),
            "supported_well_count": int(support_mask[well_indices].sum()),
            "applies_to": "section-level diagnostics only; well scopes keep their declared rows",
        },
        "unlabeled_seismic_profile_indices": unlabeled_seismic["profile_indices"],
        "sources": {
            "truth": str(prepared["truth_path"]),
            "well_roles": str(prepared["roles_path"]),
            "seismic": str(prepared["seismic_path"]),
            "wavelet": str(prepared["wavelet_path"]),
            "truth_is_offline_only": True,
        },
        "impedance_metrics": {
            "prediction": prediction_impedance,
            "lfm_baseline": lfm_impedance,
        },
        # A concise alias retained for report consumers.
        "well_metrics": {
            "prediction": prediction_impedance,
            "lfm_baseline": lfm_impedance,
        },
        "physical_metrics": {
            "prediction": prediction_waveform,
            "lfm_baseline": lfm_waveform,
            "truth_reference": truth_waveform,
            "unlabeled_seismic": unlabeled_seismic,
        },
        "waveform_metrics": {
            "prediction": prediction_waveform,
            "lfm_baseline": lfm_waveform,
            "truth_reference": truth_waveform,
        },
        "truth_forward_correlation": {
            "prediction_vs_observed": prediction_waveform,
            "lfm_vs_observed": lfm_waveform,
            "truth_vs_observed": truth_waveform,
            "prediction_vs_truth_forward": _by_role_waveform_metrics(
                physical_truth,
                physical_prediction,
                prepared["by_role"],
                included_roles,
                source="prepared/evaluation/truth.npz + selected_wavelet.csv",
                label="prediction-vs-truth-forward",
                sample_mask=well_sample_mask,
            ),
        },
        "well_target_metrics": _well_target_metrics(
            prediction,
            well_targets,
            source="prepared/evaluation/well_targets.npz",
            well_indices=well_indices,
            sample_mask=well_sample_mask,
        ),
        "low_frequency_drift": structure["low_frequency_drift"],
        "roughness": {
            "prediction": structure["prediction"],
            "lfm_baseline": structure["lfm_baseline"],
            "truth_reference": structure["truth_reference"],
        },
        "highfrequency_energy": structure["highfrequency_energy"],
        "high_frequency_energy": structure["highfrequency_energy"],
        "shortwave_energy": structure["shortwave_energy"],
        "shortwave_metrics": structure["shortwave_energy"],
        "filter": {
            "operator": "Step-7 Butterworth low-pass",
            "cutoff_hz": float(cutoff_hz),
            "shortwave_cutoff_hz": float(shortwave_cutoff_hz),
            "purpose": "evaluation diagnostics only; does not change training or prediction",
        },
        "unlabeled_seismic": unlabeled_seismic,
    }
    if scope == "final":
        full_indices = np.arange(truth.shape[0], dtype=int)
        result["full_profile"] = _profile_diagnostics(
            truth,
            prediction,
            lfm,
            observed,
            physical_prediction,
            physical_lfm,
            physical_truth,
            indices=full_indices,
            twt_s=prepared["twt_s"],
            dt_s=prepared["dt_s"],
            cutoff_hz=cutoff_hz,
            shortwave_cutoff_hz=shortwave_cutoff_hz,
            label="full_profile_truth_final",
        )
        supported_indices = full_indices[support_mask]
        if supported_indices.size == 0:
            raise ValueError(
                "No prepared profile reaches the data-support threshold; "
                "the observed section carries no signal to score against."
            )
        result["supported_profile"] = {
            **_profile_diagnostics(
                truth,
                prediction,
                lfm,
                observed,
                physical_prediction,
                physical_lfm,
                physical_truth,
                indices=supported_indices,
                twt_s=prepared["twt_s"],
                dt_s=prepared["dt_s"],
                cutoff_hz=cutoff_hz,
                shortwave_cutoff_hz=shortwave_cutoff_hz,
                label="supported_profile_truth_final",
            ),
            "excluded_trace_count": int(full_indices.size - supported_indices.size),
            "relative_threshold": float(support_relative_threshold),
        }
    return result


__all__ = ["EVALUATION_SCOPES", "evaluate_marmousi2_prediction"]
