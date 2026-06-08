"""Shared seismic well-QC plotting helpers.

The functions in this module are intentionally data-only: callers prepare the
well traces, masks, and output paths; this module computes light metrics and
renders consistent figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

WaveformQcMode = Literal["shape", "amplitude"]
ImpedanceResidualReference = Literal["low", "full"]


def _as_float_1d(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _finite_pair(reference: np.ndarray, estimate: np.ndarray, mask: Any | None = None) -> np.ndarray:
    reference = _as_float_1d(reference)
    estimate = _as_float_1d(estimate)
    valid = np.isfinite(reference) & np.isfinite(estimate)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)
    return valid


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(np.nanstd(left)) <= 0.0 or float(np.nanstd(right)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _weighted_or_plain(values: np.ndarray, weights: np.ndarray | None, reducer: str) -> float:
    if values.size == 0:
        return float("nan")
    if weights is None:
        if reducer == "mean":
            return float(np.mean(values))
        if reducer == "rms":
            return float(np.sqrt(np.mean(values**2)))
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
        denom = float(np.sum(w))
        if denom <= 0.0:
            return float("nan")
        if reducer == "mean":
            return float(np.sum(values * w) / denom)
        if reducer == "rms":
            return float(np.sqrt(np.sum((values**2) * w) / denom))
    raise ValueError(f"Unsupported reducer={reducer!r}.")


def sample_volume_at_points(
    volume: np.ndarray,
    *,
    ilines: np.ndarray,
    xlines: np.ndarray,
    inline_values: np.ndarray,
    xline_values: np.ndarray,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """Sample a regular 3-D volume at floating inline/xline points.

    Parameters
    ----------
    volume
        Volume with shape ``(n_inline, n_xline, n_sample)``.
    ilines, xlines
        Regular line coordinate axes for the first two volume dimensions.
    inline_values, xline_values
        Floating line coordinates to sample.
    sample_indices
        Integer sample indices along the last volume dimension.

    Returns
    -------
    numpy.ndarray
        One sampled value per input point. Out-of-range or non-finite inputs
        produce ``NaN``.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {vol.shape}.")
    il_axis = _as_float_1d(ilines)
    xl_axis = _as_float_1d(xlines)
    if il_axis.size != vol.shape[0] or xl_axis.size != vol.shape[1]:
        raise ValueError(
            "Line axes do not match volume shape: "
            f"ilines={il_axis.size}, xlines={xl_axis.size}, volume={vol.shape}."
        )
    inline_arr = _as_float_1d(inline_values)
    xline_arr = _as_float_1d(xline_values)
    sample_arr = np.asarray(sample_indices, dtype=np.float64).reshape(-1)
    if not (inline_arr.size == xline_arr.size == sample_arr.size):
        raise ValueError("inline_values, xline_values, and sample_indices must have the same length.")

    il_pos = np.interp(inline_arr, il_axis, np.arange(il_axis.size, dtype=np.float64), left=np.nan, right=np.nan)
    xl_pos = np.interp(xline_arr, xl_axis, np.arange(xl_axis.size, dtype=np.float64), left=np.nan, right=np.nan)
    out = np.full(inline_arr.shape, np.nan, dtype=np.float64)
    for idx, (i_f, j_f, k_f) in enumerate(zip(il_pos, xl_pos, sample_arr)):
        if not (np.isfinite(i_f) and np.isfinite(j_f) and np.isfinite(k_f)):
            continue
        if not np.isclose(k_f, round(float(k_f)), rtol=0.0, atol=1e-6):
            raise ValueError(f"sample_indices must be integer sample indices, got {k_f!r}.")
        k = int(round(float(k_f)))
        if k < 0 or k >= vol.shape[2]:
            continue
        i0 = int(np.floor(i_f))
        j0 = int(np.floor(j_f))
        i1 = min(i0 + 1, vol.shape[0] - 1)
        j1 = min(j0 + 1, vol.shape[1] - 1)
        if i0 < 0 or j0 < 0 or i0 >= vol.shape[0] or j0 >= vol.shape[1]:
            continue
        wi = float(i_f - i0)
        wj = float(j_f - j0)
        v00 = vol[i0, j0, k]
        v01 = vol[i0, j1, k]
        v10 = vol[i1, j0, k]
        v11 = vol[i1, j1, k]
        out[idx] = (1.0 - wi) * (1.0 - wj) * v00 + (1.0 - wi) * wj * v01 + wi * (1.0 - wj) * v10 + wi * wj * v11
    return out


def impedance_qc_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    mask: Any | None = None,
    weights: Any | None = None,
    prefix: str = "",
) -> dict[str, float | int]:
    """Compute standard well-impedance QC metrics."""
    ref = _as_float_1d(reference)
    est = _as_float_1d(estimate)
    valid = _finite_pair(ref, est, mask=mask)
    if weights is not None:
        w_all = _as_float_1d(weights)
        if w_all.size != ref.size:
            raise ValueError("weights must have the same length as reference.")
        w = w_all[valid]
    else:
        w = None
    ref_v = ref[valid]
    est_v = est[valid]
    diff = est_v - ref_v
    stem = f"{prefix}_" if prefix else ""
    mae = _weighted_or_plain(np.abs(diff), w, "mean")
    rmse = _weighted_or_plain(diff, w, "rms")
    bias = _weighted_or_plain(diff, w, "mean")
    denom = _weighted_or_plain(np.abs(ref_v), w, "mean")
    return {
        f"{stem}n_samples": int(valid.sum()),
        f"{stem}corr": _safe_corr(ref_v, est_v),
        f"{stem}mae": mae,
        f"{stem}rmse": rmse,
        f"{stem}bias": bias,
        f"{stem}nmae": float(mae / denom) if np.isfinite(mae) and np.isfinite(denom) and denom > 0.0 else float("nan"),
    }


def waveform_qc_metrics(
    observed: np.ndarray,
    synthetic: np.ndarray,
    *,
    mask: Any | None = None,
    prefix: str = "",
) -> dict[str, float | int]:
    """Compute standard waveform QC metrics for a shared amplitude domain."""
    obs = _as_float_1d(observed)
    syn = _as_float_1d(synthetic)
    valid = _finite_pair(obs, syn, mask=mask)
    obs_v = obs[valid]
    syn_v = syn[valid]
    diff = obs_v - syn_v
    stem = f"{prefix}_" if prefix else ""
    obs_rms = float(np.sqrt(np.mean(obs_v**2))) if obs_v.size else float("nan")
    syn_rms = float(np.sqrt(np.mean(syn_v**2))) if syn_v.size else float("nan")
    rmse = float(np.sqrt(np.mean(diff**2))) if diff.size else float("nan")
    mae = float(np.mean(np.abs(diff))) if diff.size else float("nan")
    nmae_denom = float(np.sum(np.abs(obs_v))) if obs_v.size else float("nan")
    return {
        f"{stem}n_samples": int(valid.sum()),
        f"{stem}corr": _safe_corr(obs_v, syn_v),
        f"{stem}mae": mae,
        f"{stem}rmse": rmse,
        f"{stem}bias": float(np.mean(syn_v - obs_v)) if diff.size else float("nan"),
        f"{stem}nmae": float(np.sum(np.abs(diff)) / nmae_denom)
        if np.isfinite(nmae_denom) and nmae_denom > 0.0
        else float("nan"),
        f"{stem}observed_rms": obs_rms,
        f"{stem}synthetic_rms": syn_rms,
        f"{stem}rms_ratio": float(syn_rms / obs_rms) if np.isfinite(obs_rms) and obs_rms > 0.0 else float("nan"),
    }


def reflectivity_from_ai(ai: np.ndarray) -> np.ndarray:
    """Compute normal-incidence reflectivity from an AI trace."""
    values = _as_float_1d(ai)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    upper = values[:-1]
    lower = values[1:]
    valid = np.isfinite(upper) & np.isfinite(lower)
    refl = np.full(upper.shape, np.nan, dtype=np.float64)
    refl[valid] = (lower[valid] - upper[valid]) / (lower[valid] + upper[valid] + 1e-10)
    out[:-1] = refl
    return out


def _normalized_xcorr(
    observed: np.ndarray,
    synthetic: np.ndarray,
    sample_step: float,
    *,
    mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    obs = _as_float_1d(observed)
    syn = _as_float_1d(synthetic)
    valid = _finite_pair(obs, syn, mask=mask)
    if int(valid.sum()) < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), float("nan")
    obs_v = obs[valid] - float(np.mean(obs[valid]))
    syn_v = syn[valid] - float(np.mean(syn[valid]))
    denom = float(np.linalg.norm(obs_v) * np.linalg.norm(syn_v))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.asarray([], dtype=float), np.asarray([], dtype=float), float("nan")
    corr = np.correlate(syn_v, obs_v, mode="full") / denom
    lags = np.arange(-obs_v.size + 1, obs_v.size, dtype=np.float64) * float(sample_step)
    return lags, corr, float(corr[int(np.nanargmax(np.abs(corr)))])


def _style_trace_axis(ax: Any, y_values: np.ndarray, *, xlabel: str, title: str, show_ylabel: bool = False) -> None:
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("TWT (ms)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linestyle=":")


def plot_well_waveform_qc(
    path: str | Path,
    *,
    samples_s: np.ndarray,
    observed: np.ndarray,
    synthetic: np.ndarray,
    impedance: np.ndarray | None = None,
    reflectivity: np.ndarray | None = None,
    mode: WaveformQcMode = "shape",
    title: str = "Well waveform QC",
    observed_label: str = "Seismic",
    synthetic_label: str = "Synthetic",
    amplitude_label: str = "Amplitude",
    mask: Any | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Render a well waveform QC figure.

    ``mode='shape'`` keeps synthetic and observed waveforms in separate tracks.
    ``mode='amplitude'`` overlays them on the same axis so their amplitudes are
    visually comparable.
    """
    if mode not in {"shape", "amplitude"}:
        raise ValueError(f"Unsupported waveform QC mode={mode!r}.")
    sample_axis = _as_float_1d(samples_s)
    obs = _as_float_1d(observed)
    syn = _as_float_1d(synthetic)
    if not (sample_axis.size == obs.size == syn.size):
        raise ValueError("samples_s, observed, and synthetic must have the same length.")
    ai = None if impedance is None else _as_float_1d(impedance)
    refl = reflectivity_from_ai(ai) if reflectivity is None and ai is not None else None
    if reflectivity is not None:
        refl = _as_float_1d(reflectivity)
        if refl.size != sample_axis.size:
            raise ValueError("reflectivity must have the same length as samples_s.")
    if ai is not None and ai.size != sample_axis.size:
        raise ValueError("impedance must have the same length as samples_s.")
    qc_mask = None if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
    if qc_mask is not None and qc_mask.size != sample_axis.size:
        raise ValueError("mask must have the same length as samples_s.")

    residual = obs - syn
    y_ms = sample_axis * 1000.0
    sample_step = float(np.nanmedian(np.diff(sample_axis))) if sample_axis.size > 1 else 1.0
    lags_s, xcorr, peak_xcorr = _normalized_xcorr(obs, syn, sample_step, mask=qc_mask)

    n_leading = int(ai is not None) + int(refl is not None)
    n_waveform = 4 if mode == "shape" else 3
    ncols = n_leading + n_waveform
    width = max(10.0, 2.55 * ncols)
    fig, axes = plt.subplots(1, ncols, figsize=(width, 7.0), sharey=False)
    axes = np.asarray(axes).reshape(-1)
    axis_idx = 0

    if ai is not None:
        axes[axis_idx].plot(ai, y_ms, color="tab:blue", lw=1.0)
        _style_trace_axis(axes[axis_idx], y_ms, xlabel="AI", title="Impedance", show_ylabel=True)
        axis_idx += 1

    if refl is not None:
        axes[axis_idx].plot(refl, y_ms, color="tab:purple", lw=0.8)
        axes[axis_idx].axvline(0.0, color="black", lw=0.6, alpha=0.4)
        _style_trace_axis(axes[axis_idx], y_ms, xlabel="Reflectivity", title="Reflectivity", show_ylabel=axis_idx == 0)
        axis_idx += 1

    if mode == "shape":
        axes[axis_idx].plot(syn, y_ms, color="tab:red", lw=0.9)
        axes[axis_idx].axvline(0.0, color="black", lw=0.6, alpha=0.4)
        _style_trace_axis(axes[axis_idx], y_ms, xlabel=amplitude_label, title=synthetic_label, show_ylabel=axis_idx == 0)
        axis_idx += 1

        axes[axis_idx].plot(obs, y_ms, color="black", lw=0.9)
        axes[axis_idx].axvline(0.0, color="black", lw=0.6, alpha=0.4)
        _style_trace_axis(axes[axis_idx], y_ms, xlabel=amplitude_label, title=observed_label, show_ylabel=axis_idx == 0)
        axis_idx += 1
        residual_axis = axes[axis_idx]
        axis_idx += 1
        xcorr_axis = axes[axis_idx]
    else:
        axes[axis_idx].plot(obs, y_ms, color="black", lw=0.95, label=observed_label)
        axes[axis_idx].plot(syn, y_ms, color="tab:red", lw=0.95, alpha=0.9, label=synthetic_label)
        axes[axis_idx].axvline(0.0, color="black", lw=0.6, alpha=0.4)
        _style_trace_axis(axes[axis_idx], y_ms, xlabel=amplitude_label, title="Waveform overlay", show_ylabel=axis_idx == 0)
        axes[axis_idx].legend(loc="best", fontsize=8)
        axis_idx += 1
        residual_axis = axes[axis_idx]
        axis_idx += 1
        xcorr_axis = axes[axis_idx]

    residual_axis.plot(residual, y_ms, color="tab:gray", lw=0.9)
    residual_axis.axvline(0.0, color="black", lw=0.6, alpha=0.4)
    _style_trace_axis(residual_axis, y_ms, xlabel=amplitude_label, title="Residual", show_ylabel=residual_axis is axes[0])

    if xcorr.size:
        xcorr_axis.plot(xcorr, lags_s * 1000.0, color="tab:green", lw=1.0)
        xcorr_axis.axhline(0.0, color="black", lw=0.6, alpha=0.4)
        xcorr_axis.axvline(0.0, color="black", lw=0.6, alpha=0.4)
        xcorr_axis.invert_yaxis()
    xcorr_axis.set_xlabel("Correlation")
    xcorr_axis.set_ylabel("Lag (ms)")
    xcorr_axis.set_title(f"Cross-corr peak={peak_xcorr:.3f}" if np.isfinite(peak_xcorr) else "Cross-corr")
    xcorr_axis.grid(True, alpha=0.25, linestyle=":")

    metric_text = ""
    if metrics:
        parts = []
        for key in ("corr", "nmae", "rmse", "rms_ratio"):
            value = metrics.get(key)
            if value is not None and np.isfinite(float(value)):
                parts.append(f"{key}={float(value):.3g}")
        metric_text = " | " + ", ".join(parts) if parts else ""
    fig.suptitle(f"{title} | mode={mode}{metric_text}", y=0.995)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_well_impedance_qc(
    path: str | Path,
    *,
    samples_s: np.ndarray,
    model_ai: np.ndarray,
    low_ai: np.ndarray | None = None,
    full_ai: np.ndarray | None = None,
    title: str = "Well impedance QC",
    model_label: str = "Model AI",
    low_label: str = "Low-frequency well AI",
    full_label: str = "Full-frequency well AI",
    residual_reference: ImpedanceResidualReference = "low",
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Render a gray/blue/red well impedance QC figure."""
    sample_axis = _as_float_1d(samples_s)
    model = _as_float_1d(model_ai)
    if model.size != sample_axis.size:
        raise ValueError("model_ai must have the same length as samples_s.")
    low = None if low_ai is None else _as_float_1d(low_ai)
    full = None if full_ai is None else _as_float_1d(full_ai)
    if low is not None and low.size != sample_axis.size:
        raise ValueError("low_ai must have the same length as samples_s.")
    if full is not None and full.size != sample_axis.size:
        raise ValueError("full_ai must have the same length as samples_s.")

    if residual_reference not in {"low", "full"}:
        raise ValueError(f"Unsupported residual_reference={residual_reference!r}.")
    if residual_reference == "low" and low is not None:
        residual_ref = low
        residual_label = "Model - low reference"
    elif full is not None:
        residual_ref = full
        residual_label = "Model - full reference"
    else:
        residual_ref = low
        residual_label = "Model - low reference"
    residual = model - residual_ref if residual_ref is not None else np.full(model.shape, np.nan)
    y_ms = sample_axis * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 8.5), width_ratios=[2.3, 1.0], sharey=True)
    if full is not None:
        axes[0].plot(full, y_ms, label=full_label, lw=0.8, alpha=0.38, color="gray")
    if low is not None:
        axes[0].plot(low, y_ms, label=low_label, lw=1.7, color="tab:blue")
    axes[0].plot(model, y_ms, label=model_label, lw=1.9, color="tab:red")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("AI")
    axes[0].set_ylabel("TWT (ms)")
    axes[0].set_title("Impedance")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.28, linestyle=":")

    axes[1].plot(residual, y_ms, color="tab:gray", lw=0.9)
    axes[1].axvline(0.0, color="black", lw=0.7, alpha=0.45)
    axes[1].set_xlabel(residual_label)
    axes[1].set_title("Residual")
    axes[1].grid(True, alpha=0.28, linestyle=":")

    metric_text = ""
    if metrics:
        parts = []
        for key in ("low_corr", "low_rmse", "full_corr", "full_rmse", "corr", "rmse"):
            value = metrics.get(key)
            if value is not None and np.isfinite(float(value)):
                parts.append(f"{key}={float(value):.3g}")
        metric_text = " | " + ", ".join(parts[:4]) if parts else ""
    fig.suptitle(f"{title}{metric_text}", y=0.995)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


__all__ = [
    "ImpedanceResidualReference",
    "WaveformQcMode",
    "impedance_qc_metrics",
    "plot_well_impedance_qc",
    "plot_well_waveform_qc",
    "reflectivity_from_ai",
    "sample_volume_at_points",
    "waveform_qc_metrics",
]
