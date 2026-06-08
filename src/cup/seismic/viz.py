"""Seismic well-QC plotting helpers.

The waveform QC style is intentionally copied from
``wtie.utils.viz.plot_tie_window`` so workflow scripts can depend on a
``cup.seismic`` entry point without changing the figure aesthetics.
"""

from __future__ import annotations

from typing import Literal, Tuple

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

from wtie.processing import grid

WaveformQcMode = Literal["shape", "amplitude"]


def _plot_trace(
    trace: grid.BaseTrace,
    figsize: Tuple[int, int] = (3, 5),
    fig_axes: tuple | None = None,
    plot_params: dict | None = None,
) -> tuple:
    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    if plot_params is None:
        plot_params = {}

    ax.plot(trace.values, trace.basis, **plot_params)
    ax.set_xlabel(trace.name)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(trace.basis_type)

    ax.set_ylim((trace.basis[0], trace.basis[-1]))
    ax.invert_yaxis()

    if trace.is_twt or trace.is_tlag:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax


def _plot_reflectivity(
    reflectivity: grid.Reflectivity,
    figsize: tuple = (3, 5),
    fig_axes: tuple | None = None,
) -> tuple:
    x_picks = np.where(reflectivity.values != 0)[0]
    y_picks = reflectivity.values[reflectivity.values != 0.0]

    basis = reflectivity.basis

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    ax.plot(np.zeros_like(basis), basis, color="k", alpha=0.5, lw=0.5)
    for i in range(len(x_picks)):
        ax.hlines(basis[x_picks[i]], xmin=0, xmax=y_picks[i], color="r", lw=1.0)

    ax.set_xlabel(reflectivity.name)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(reflectivity.basis_type)

    ax.set_ylim((basis[0], basis[-1]))
    ax.invert_yaxis()

    ax.yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax


def _plot_wiggle_trace(
    trace: grid.BaseTrace,
    figsize: Tuple[int, int] = (4, 4),
    repeat_n_times: int = 7,
    scaling: float = 1.0,
    fig_axes: tuple | None = None,
) -> tuple:
    if fig_axes is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig, ax = fig_axes

    for offset in np.linspace(0.0, scaling * 1.0, num=repeat_n_times):
        if repeat_n_times == 1:
            # trace in the middle
            offset = scaling / 2.0

        _x = trace.values + offset
        ax.plot(_x, trace.basis, color="k", lw=0.5)
        ax.plot(offset * np.ones_like(_x), trace.basis, color="k", lw=0.2)

        ax.fill_betweenx(trace.basis, offset, _x, where=(_x >= offset), color="b", alpha=0.6, interpolate=True)  # type: ignore
        ax.fill_betweenx(trace.basis, offset, _x, where=(_x < offset), color="r", alpha=0.6, interpolate=True)  # type: ignore

    ax.set_xlabel(trace.name)
    ax.set_ylim((trace.basis[0], trace.basis[-1]))
    ax.invert_yaxis()
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(trace.basis_type)

    _space = 0.05 * np.abs(trace.values.max())
    ax.set_xlim((trace.values.min() - _space, (trace.values.max() + scaling + _space)))
    ax.set_xticks([])

    if trace.is_twt:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax


def _plot_dynamic_xcorr(
    dxcorr: grid.DynamicXCorr,
    figsize: Tuple[int, int] = (4, 5),
    fig_axes: tuple | None = None,
    im_params: dict | None = None,
) -> tuple:
    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    # basis in seconds, lags in ms
    extent = [1000 * dxcorr.lags_basis[0], 1000 * dxcorr.lags_basis[-1], dxcorr.basis[-1], dxcorr.basis[0]]

    if im_params is None:
        im_params = dict(cmap="RdYlBu_r", vmin=-0.85, vmax=0.85)

    im = ax.imshow(dxcorr.values, extent=extent, interpolation="bilinear", aspect="auto", **im_params)  # type: ignore

    ax.set_xlabel(dxcorr.name)  # type: ignore
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(dxcorr.basis_type)
    if dxcorr.is_twt or dxcorr.is_tlag:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, orientation="vertical", pad=0.05, aspect=30)
    cbar.set_label("", fontsize=11, rotation=90, y=0.5)

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax, cbar


def plot_well_waveform_qc(
    logset: grid.LogSet | grid.Log,
    reflectivity: grid.Reflectivity,
    synthetic_seismic: grid.Seismic,
    real_seismic: grid.Seismic,
    xcorr: grid.XCorr,
    dxcorr: grid.DynamicXCorr,
    figsize: Tuple[int, int] = (7, 4),
    wiggle_scale_syn: float = 1.0,
    wiggle_scale_real: float = 1.0,
    mode: WaveformQcMode = "shape",
    adaptive_wiggle_scale: bool = False,
    compact_x_ticks: bool = False,
) -> tuple:
    """Draw the post-stack well-tie waveform QC window.

    ``mode="shape"`` is a local, workflow-owned copy of
    ``wtie.utils.viz.plot_tie_window``. ``mode="amplitude"`` keeps the same
    six-panel style but preserves the caller's amplitude domain. The optional
    display knobs are deliberately explicit so the fourth-step 1:1 style remains
    unchanged unless a caller asks for compact/adaptive rendering.
    """
    if mode not in {"shape", "amplitude"}:
        raise ValueError(f"Unsupported waveform QC mode={mode!r}.")
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 8)

    axes = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[2:4]),
        fig.add_subplot(gs[4:6]),
        fig.add_subplot(gs[6:7]),
        fig.add_subplot(gs[7:]),
    ]

    # logs
    ai_trace = logset.AI if isinstance(logset, grid.LogSet) else logset
    _plot_trace(ai_trace, fig_axes=(fig, axes[0]))
    _plot_reflectivity(reflectivity, fig_axes=(fig, axes[1]))

    plot_wiggle_scale_syn = float(wiggle_scale_syn)
    plot_wiggle_scale_real = float(wiggle_scale_real)
    if mode == "amplitude" or adaptive_wiggle_scale:
        finite_for_scale = np.concatenate(
            [
                np.asarray(synthetic_seismic.values, dtype=float).reshape(-1),
                np.asarray(real_seismic.values, dtype=float).reshape(-1),
                np.asarray(real_seismic.values - synthetic_seismic.values, dtype=float).reshape(-1),
            ]
        )
        finite_for_scale = finite_for_scale[np.isfinite(finite_for_scale)]
        if finite_for_scale.size:
            value_span = float(np.max(finite_for_scale) - np.min(finite_for_scale))
            value_absmax = float(np.max(np.abs(finite_for_scale)))
            adaptive_scale = max(value_span, value_absmax, 1e-12)
            plot_wiggle_scale_syn = adaptive_scale
            plot_wiggle_scale_real = adaptive_scale

    # seismic
    _plot_wiggle_trace(synthetic_seismic, scaling=plot_wiggle_scale_syn, repeat_n_times=5, fig_axes=(fig, axes[2]))
    _plot_wiggle_trace(real_seismic, scaling=plot_wiggle_scale_real, repeat_n_times=5, fig_axes=(fig, axes[3]))

    residual = grid.Seismic(real_seismic.values - synthetic_seismic.values, real_seismic.basis, "twt", name="Residual")

    _plot_wiggle_trace(residual, scaling=plot_wiggle_scale_real, repeat_n_times=1, fig_axes=(fig, axes[4]))

    if mode == "amplitude" or compact_x_ticks:
        axes[0].xaxis.set_major_locator(MaxNLocator(nbins=3))
        axes[0].tick_params(axis="x", labelsize=8)
        axes[1].xaxis.set_major_locator(MaxNLocator(nbins=3))
        axes[1].tick_params(axis="x", labelsize=8)

    if mode == "amplitude" or adaptive_wiggle_scale:
        finite = np.concatenate(
            [
                np.asarray(synthetic_seismic.values, dtype=float).reshape(-1),
                np.asarray(real_seismic.values, dtype=float).reshape(-1),
                np.asarray(residual.values, dtype=float).reshape(-1),
            ]
        )
        finite = finite[np.isfinite(finite)]
        if finite.size:
            value_min = float(np.min(finite))
            value_max = float(np.max(finite))
            common_scale = max(float(plot_wiggle_scale_syn), float(plot_wiggle_scale_real))
            space = 0.05 * max(abs(value_min), abs(value_max), common_scale, 1e-12)
            common_xlim = (value_min - space, value_max + common_scale + space)
            for ax in axes[2:5]:
                ax.set_xlim(common_xlim)

    # dxcoor
    _plot_dynamic_xcorr(dxcorr, fig_axes=(fig, axes[5]))
    axes[5].set_xlabel("Correlation")

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.set_yticklabels("")

    for ax in axes:
        ax.locator_params(axis="y", nbins=28)

    fig.suptitle("Max correlation of %.2f at a lag of %.3f s (Rc = %.2f)" % (xcorr.R, xcorr.lag, xcorr.Rc))

    fig.tight_layout()
    return fig, axes


def _as_bool_mask(mask: np.ndarray | None, size: int) -> np.ndarray:
    if mask is None:
        return np.ones(size, dtype=bool)
    out = np.asarray(mask, dtype=bool).reshape(-1)
    if out.size != size:
        raise ValueError(f"mask size {out.size} does not match trace size {size}.")
    return out


def _validate_same_basis(*traces: grid.Log | None) -> np.ndarray:
    basis: np.ndarray | None = None
    for trace in traces:
        if trace is None:
            continue
        values = np.asarray(trace.basis, dtype=np.float64)
        if basis is None:
            basis = values
        elif values.shape != basis.shape or not np.allclose(values, basis, equal_nan=True):
            raise ValueError("All impedance QC traces must share the same basis.")
    if basis is None:
        raise ValueError("At least one impedance QC trace is required.")
    return basis


def _metric_pair(reference: np.ndarray, model: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(reference)
        & np.isfinite(model)
    )
    if not np.any(valid):
        return {
            "n_samples": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "corr": float("nan"),
            "nmae": float("nan"),
        }
    diff = model[valid] - reference[valid]
    denom = max(float(np.mean(np.abs(reference[valid]))), 1e-12)
    corr = float(np.corrcoef(reference[valid], model[valid])[0, 1]) if int(np.sum(valid)) > 1 else float("nan")
    return {
        "n_samples": int(np.sum(valid)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "corr": corr,
        "nmae": float(np.mean(np.abs(diff)) / denom),
    }


def impedance_qc_metrics(
    *,
    model_ai: grid.Log,
    low_ai: grid.Log | None = None,
    full_ai: grid.Log | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute local well-side impedance QC metrics."""
    basis = _validate_same_basis(model_ai, low_ai, full_ai)
    valid_mask = _as_bool_mask(mask, basis.size)
    model_values = np.asarray(model_ai.values, dtype=np.float64)
    metrics: dict[str, float | int] = {}
    if low_ai is not None:
        low_metrics = _metric_pair(np.asarray(low_ai.values, dtype=np.float64), model_values, valid_mask)
        metrics.update({f"vs_low_{key}": value for key, value in low_metrics.items()})
    if full_ai is not None:
        full_metrics = _metric_pair(np.asarray(full_ai.values, dtype=np.float64), model_values, valid_mask)
        metrics.update({f"vs_full_{key}": value for key, value in full_metrics.items()})
    return metrics


def _trace_label(trace: grid.Log | None, fallback: str) -> str:
    if trace is None:
        return fallback
    return str(trace.name or fallback)


def plot_well_impedance_qc(
    *,
    model_ai: grid.Log,
    low_ai: grid.Log | None = None,
    full_ai: grid.Log | None = None,
    mask: np.ndarray | None = None,
    title: str | None = None,
    model_label: str | None = None,
    figsize: tuple[float, float] = (7.5, 8.5),
) -> tuple:
    """Draw well-side impedance QC in the depth-inversion three-line style."""
    basis = _validate_same_basis(model_ai, low_ai, full_ai)
    valid_mask = _as_bool_mask(mask, basis.size)
    model_values = np.asarray(model_ai.values, dtype=np.float64)
    low_values = None if low_ai is None else np.asarray(low_ai.values, dtype=np.float64)
    full_values = None if full_ai is None else np.asarray(full_ai.values, dtype=np.float64)

    reference_trace = low_ai if low_ai is not None else full_ai
    reference_values = low_values if low_values is not None else full_values
    has_residual = reference_values is not None
    ncols = 2 if has_residual else 1
    width_ratios = [3.0, 1.4] if has_residual else [1.0]
    fig, axes_arr = plt.subplots(
        1,
        ncols,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes_arr).tolist()
    ax = axes[0]

    if full_values is not None:
        ax.plot(full_values, basis, label=_trace_label(full_ai, "Full-band well AI"), lw=0.8, alpha=0.35, color="gray")
    if low_values is not None:
        ax.plot(low_values, basis, label=_trace_label(low_ai, "Low-frequency well AI"), lw=1.6, color="blue")
    ax.plot(model_values, basis, label=model_label or _trace_label(model_ai, "Model AI"), lw=1.9, color="red")
    if np.any(valid_mask):
        ymin = float(np.nanmin(basis[valid_mask]))
        ymax = float(np.nanmax(basis[valid_mask]))
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(float(basis[0]), float(basis[-1]))
    ax.invert_yaxis()
    ax.set_xlabel("AI")
    ax.set_ylabel("TWT (s)" if model_ai.is_twt else model_ai.basis_type)
    ax.set_title(title or "Well impedance QC")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=":")

    finite_for_xlim = [model_values[np.isfinite(model_values)]]
    if low_values is not None:
        finite_for_xlim.append(low_values[np.isfinite(low_values)])
    if full_values is not None:
        finite_for_xlim.append(full_values[np.isfinite(full_values)])
    finite_concat = np.concatenate([arr for arr in finite_for_xlim if arr.size])
    if finite_concat.size:
        lo, hi = np.percentile(finite_concat, [1.0, 99.0])
        if np.isclose(lo, hi):
            pad = max(abs(float(lo)) * 0.02, 1.0)
        else:
            pad = 0.06 * float(hi - lo)
        ax.set_xlim(float(lo) - pad, float(hi) + pad)

    if has_residual and reference_values is not None:
        res_ax = axes[1]
        residual = model_values - reference_values
        res_ax.plot(residual, basis, lw=0.9, color="tab:purple")
        res_ax.axvline(0.0, color="black", lw=0.8, alpha=0.55)
        res_ax.set_xlabel("Model - Ref")
        res_ax.set_title("Residual")
        res_ax.grid(True, alpha=0.3, linestyle=":")
        finite_res = residual[np.isfinite(residual) & valid_mask]
        if finite_res.size:
            limit = float(np.percentile(np.abs(finite_res), 99.0))
            limit = max(limit, 1.0)
            res_ax.set_xlim(-1.08 * limit, 1.08 * limit)

    return fig, axes


def sample_volume_at_points(
    volume: np.ndarray,
    geometry: dict,
    inline_values: np.ndarray,
    xline_values: np.ndarray,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample a volume at line coordinates and integer sample indices."""
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape={vol.shape}.")
    inline_values = np.asarray(inline_values, dtype=np.float64).reshape(-1)
    xline_values = np.asarray(xline_values, dtype=np.float64).reshape(-1)
    sample_indices = np.asarray(sample_indices).reshape(-1)
    if not (inline_values.size == xline_values.size == sample_indices.size):
        raise ValueError("inline_values, xline_values and sample_indices must have the same size.")

    inline_min = float(geometry["inline_min"])
    inline_step = float(geometry["inline_step"])
    xline_min = float(geometry["xline_min"])
    xline_step = float(geometry["xline_step"])
    i_float = (inline_values - inline_min) / inline_step
    j_float = (xline_values - xline_min) / xline_step
    i_float = np.clip(i_float, 0.0, vol.shape[0] - 1.0)
    j_float = np.clip(j_float, 0.0, vol.shape[1] - 1.0)
    i0 = np.floor(i_float).astype(np.int64)
    j0 = np.floor(j_float).astype(np.int64)
    i1 = np.ceil(i_float).astype(np.int64)
    j1 = np.ceil(j_float).astype(np.int64)
    wi = i_float - i0
    wj = j_float - j0

    sample_float = sample_indices.astype(np.float64)
    if not np.all(np.isfinite(sample_float)):
        raise ValueError("sample_indices must be finite.")
    if np.any(np.abs(sample_float - np.round(sample_float)) > 1e-6):
        raise ValueError("sample_indices must be integer-valued for point QC sampling.")
    k = np.clip(np.round(sample_float).astype(np.int64), 0, vol.shape[2] - 1)

    t00 = vol[i0, j0, k]
    t01 = vol[i0, j1, k]
    t10 = vol[i1, j0, k]
    t11 = vol[i1, j1, k]
    return (1.0 - wi) * (1.0 - wj) * t00 + (1.0 - wi) * wj * t01 + wi * (1.0 - wj) * t10 + wi * wj * t11


__all__ = [
    "impedance_qc_metrics",
    "plot_well_impedance_qc",
    "plot_well_waveform_qc",
    "sample_volume_at_points",
]
