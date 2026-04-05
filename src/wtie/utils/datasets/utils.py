"""Small collections to facilitate the manipulation of the input/output
of the wtie.optimize.tie functions."""

# from collections import namedtuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient

from wtie import grid
from wtie.utils import viz as _viz

# InputSet = namedtuple('InputSet',('logset_md','seismic', 'wellpath', 'table'))


@dataclass
class InputSet:
    """ """

    logset_md: grid.LogSet
    seismic: grid.seismic_t
    table: grid.TimeDepthTable

    wellpath: grid.WellPath = None  # type: ignore

    def __post_init__(self):
        assert self.logset_md.is_md

    def plot_inputs(self, figsize: tuple = (9, 4), scale: float = 1.0):

        fig, axes = plt.subplots(1, 4, gridspec_kw={"width_ratios": [1, 3, 3, 2]}, figsize=figsize)
        _viz.plot_trace(self.logset_md.AI, fig_axes=(fig, axes[0]))
        _viz.plot_wellpath(self.wellpath, fig_axes=(fig, axes[1]))
        _viz.plot_td_table(self.table, fig_axes=(fig, axes[2]))

        if self.seismic.is_prestack:
            _viz.plot_prestack_trace(self.seismic, scale=scale, fig_axes=(fig, axes[3]))
        else:
            _viz.plot_trace(self.seismic, fig_axes=(fig, axes[3]))  # type: ignore

        return fig, axes


@dataclass
class OutputSet:
    """ """

    wavelet: grid.wlt_t
    logset_twt: grid.LogSet
    seismic: grid.seismic_t
    synth_seismic: grid.seismic_t
    table: grid.TimeDepthTable
    r: grid.ref_t

    wellpath: grid.WellPath = None  # type: ignore

    xcorr: grid.xcorr_t = None  # type: ignore
    dlags: grid.DynamicLag = None  # type: ignore
    dxcorr: grid.DynamicXCorr = None  # type: ignore
    ax_client: AxClient = None  # type: ignore

    def plot_tie_window(self, wiggle_scale: float = None, figsize: tuple = (9, 5), **kwargs) -> plt.subplots:  # type: ignore

        if self.seismic.is_prestack:
            fig, axes = _viz.plot_prestack_tie_window(
                self.logset_twt,
                self.r,  # type: ignore
                self.synth_seismic,  # type: ignore
                self.seismic,  # type: ignore
                self.xcorr,  # type: ignore
                figsize=figsize,
                wiggle_scale_syn=wiggle_scale,
                wiggle_scale_real=wiggle_scale,
                **kwargs,
            )
        else:
            fig, axes = _viz.plot_tie_window(
                self.logset_twt,
                self.r,  # type: ignore
                self.synth_seismic,  # type: ignore
                self.seismic,  # type: ignore
                self.xcorr,  # type: ignore
                self.dxcorr,
                figsize=figsize,
                wiggle_scale_syn=wiggle_scale,
                wiggle_scale_real=wiggle_scale,
                **kwargs,
            )
        axes[0].locator_params(axis="y", nbins=16)

        return fig, axes

    def plot_wavelet(self, **kwargs):
        if self.wavelet.is_prestack:
            return _viz.plot_prestack_wavelet(self.wavelet, **kwargs)  # type: ignore
        else:
            return _viz.plot_wavelet(self.wavelet, **kwargs)  # type: ignore

    def plot_optimization_objective(self, **kwargs):
        return _viz.plot_optimization_objective(self.ax_client, **kwargs)
