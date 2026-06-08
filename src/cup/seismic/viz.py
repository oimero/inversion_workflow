"""Seismic well-QC plotting helpers.

The waveform QC style is intentionally copied from
``wtie.utils.viz.plot_tie_window`` so workflow scripts can depend on a
``cup.seismic`` entry point without changing the figure aesthetics.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from wtie.processing import grid


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
    logset: grid.LogSet,
    reflectivity: grid.Reflectivity,
    synthetic_seismic: grid.Seismic,
    real_seismic: grid.Seismic,
    xcorr: grid.XCorr,
    dxcorr: grid.DynamicXCorr,
    figsize: Tuple[int, int] = (7, 4),
    wiggle_scale_syn: float = 1.0,
    wiggle_scale_real: float = 1.0,
) -> tuple:
    """Draw the post-stack well-tie waveform QC window.

    This function is a local, workflow-owned copy of
    ``wtie.utils.viz.plot_tie_window``.  Keep this implementation visually
    identical unless the workflow deliberately defines a new QC style.
    """
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
    _plot_trace(logset.AI, fig_axes=(fig, axes[0]))
    _plot_reflectivity(reflectivity, fig_axes=(fig, axes[1]))

    # seismic
    _plot_wiggle_trace(synthetic_seismic, scaling=wiggle_scale_syn, repeat_n_times=5, fig_axes=(fig, axes[2]))
    _plot_wiggle_trace(real_seismic, scaling=wiggle_scale_real, repeat_n_times=5, fig_axes=(fig, axes[3]))

    residual = grid.Seismic(real_seismic.values - synthetic_seismic.values, real_seismic.basis, "twt", name="Residual")

    _plot_wiggle_trace(residual, scaling=wiggle_scale_real, repeat_n_times=1, fig_axes=(fig, axes[4]))

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


__all__ = ["plot_well_waveform_qc"]
