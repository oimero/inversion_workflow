"""wtie.utils.viz: 井震对齐结果与中间量可视化工具。

本模块提供叠后/叠前地震道、反射系数、子波、相关性矩阵、
测井曲线与井轨迹的 Matplotlib 绘图函数，
用于井震对齐过程中的质检、对比分析与结果展示。

边界说明
--------
- 本模块不负责反演优化、动态校正求解或模型训练。
- 本模块仅执行绘图与坐标轴组织，不做输入数据质量控制与业务判优。

核心公开对象
------------
1. plot_trace / plot_prestack_trace: 1D 轨迹与叠前道集基础曲线图。
2. plot_wiggle_trace / plot_prestack_wiggle_trace: wiggle 填充显示。
3. plot_tie_window / plot_prestack_tie_window: 叠后与叠前井震对齐综合窗口。
4. plot_wavelet / plot_prestack_wavelet: 子波时域与频域展示。
5. plot_dynamic_xcorr / plot_trace_as_pixels: 相关矩阵与像素化显示。

Examples
--------
>>> import matplotlib.pyplot as plt
>>> fig, ax = plot_trace(trace)
>>> fig2, axes2 = plot_tie_window(logset, reflectivity, synthetic_seismic, real_seismic, xcorr, dxcorr)
>>> plt.show()
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FormatStrFormatter

from wtie.processing import grid
from wtie.processing.spectral import compute_spectrum
from wtie.utils.types_ import Tuple


def plot_seismics(
    real_seismic: grid.Seismic,
    pred_seismic: grid.Seismic,
    reflectivity: grid.Reflectivity,
    normalize: bool = True,
    figsize: Tuple[int, int] = (7, 4),
) -> tuple:
    """对比绘制真实地震、合成地震与反射系数脉冲。

    Parameters
    ----------
    real_seismic : grid.Seismic
        真实地震道，``values`` shape 为 ``(n_samples,)``，采样基准为 ``basis``。
    pred_seismic : grid.Seismic
        合成地震道，要求与 ``real_seismic`` 使用相同 ``basis``。
    reflectivity : grid.Reflectivity
        反射系数序列，要求与地震道共用相同 ``basis``。
    normalize : bool, default=True
        是否将三条曲线按各自最大绝对振幅归一化到约 ``[-1, 1]``。
    figsize : Tuple[int, int], default=(7, 4)
        图幅尺寸，单位为英寸。

    Returns
    -------
    tuple
        ``(fig, axes)``，其中 ``axes`` 长度为 2：上图为真实地震，下图为合成地震。

    Raises
    ------
    AssertionError
        当三者 ``basis`` 在 ``rtol=1e-3`` 下不一致时触发。

    Examples
    --------
    >>> fig, axes = plot_seismics(real_seismic, pred_seismic, reflectivity)
    >>> plt.show()
    """
    assert np.allclose(real_seismic.basis, pred_seismic.basis, rtol=1e-3)
    assert np.allclose(real_seismic.basis, reflectivity.basis, rtol=1e-3)
    basis = real_seismic.basis

    if normalize:

        def f(x):
            return x / max(x.max(), -x.min())
    else:

        def f(x):
            return x

    x_picks = np.where(reflectivity.values != 0)[0]
    y_picks = f(reflectivity.values[reflectivity.values != 0.0])

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    axes[0].plot(basis, np.zeros_like(basis), color="r", alpha=0.5, label="Reflectivity")
    axes[0].plot(basis, f(real_seismic.values), lw=1.5, label="Real seismic")
    # axes[0].plot(ref_t, noise, 'g', alpha=0.3, lw=1.)
    for i in range(len(x_picks)):
        axes[0].vlines(basis[x_picks[i]], ymin=0, ymax=y_picks[i], color="r", lw=1.0)

    axes[1].plot(basis, np.zeros_like(basis), color="r", alpha=0.5)
    axes[1].plot(basis, f(pred_seismic.values), lw=1.5, label="Synthetic seismic")
    for i in range(len(x_picks)):
        axes[1].vlines(basis[x_picks[i]], ymin=0, ymax=y_picks[i], color="r", lw=1.0)

    for ax in axes:
        ax.set_xlim([basis[0], basis[-1]])
        if normalize:
            ax.set_ylim([-1.05, 1.05])
        ax.legend(loc="best")

    axes[0].set_xlabel(reflectivity.basis_type)

    plt.tight_layout()

    return fig, axes


def plot_seismic_and_reflectivity(
    seismic: grid.Seismic,
    reflectivity: grid.Reflectivity,
    normalize: bool = False,
    figsize: Tuple[int, int] = (3, 5),
    fig_axes: tuple = None,  # type: ignore
    title: str = None,  # type: ignore
) -> tuple:
    """在同一坐标轴叠加绘制地震道与反射系数。

    Parameters
    ----------
    seismic : grid.Seismic
        地震道，``values`` shape 为 ``(n_samples,)``。
    reflectivity : grid.Reflectivity
        反射系数序列，``values`` shape 为 ``(n_samples,)``。
    normalize : bool, default=False
        是否分别对地震道与反射系数做最大绝对值归一化。
    figsize : Tuple[int, int], default=(3, 5)
        新建图窗时的图幅尺寸（英寸）。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)`` 时在该坐标轴上作图。
    title : str, optional
        若提供则写入 x 轴标签位置，用作标题文本。

    Returns
    -------
    tuple
        ``(fig, ax)``。

    Raises
    ------
    AssertionError
        当 ``seismic`` 与 ``reflectivity`` 的 ``basis`` 不一致时触发。

    Examples
    --------
    >>> fig, ax = plot_seismic_and_reflectivity(seismic, reflectivity, normalize=True)
    >>> plt.show()
    """
    assert np.allclose(seismic.basis, reflectivity.basis, rtol=1e-3)

    if normalize:
        seis_ = np.copy(seismic.values)
        seis_ /= np.abs(seis_).max()
        seismic = grid.update_trace_values(seis_, seismic)  # type: ignore

        ref_ = np.copy(reflectivity.values)
        ref_ /= np.abs(ref_).max()
        reflectivity = grid.update_trace_values(ref_, reflectivity)  # type: ignore

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    plot_reflectivity(reflectivity, fig_axes=(fig, ax))
    plot_trace(seismic, fig_axes=(fig, ax))

    if title is not None:
        ax.set_xlabel(title)
    else:
        ax.set_xlabel("")

    plt.tight_layout()

    return fig, ax


def plot_tie_window(
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
    """绘制叠后井震标定综合窗口。

    图中包含测井曲线、反射系数、合成/真实地震 wiggle、残差道以及动态相关矩阵。

    Parameters
    ----------
    logset : grid.LogSet
        测井集合，至少使用 ``AI`` 曲线。
    reflectivity : grid.Reflectivity
        反射系数序列，shape 为 ``(n_samples,)``。
    synthetic_seismic : grid.Seismic
        合成地震道，shape 为 ``(n_samples,)``。
    real_seismic : grid.Seismic
        真实地震道，shape 为 ``(n_samples,)``。
    xcorr : grid.XCorr
        全局互相关结果，仅用于标题显示 ``R``、``lag``、``Rc``。
    dxcorr : grid.DynamicXCorr
        动态互相关矩阵，``values`` 通常为 ``(n_lags, n_samples)``。
    figsize : Tuple[int, int], default=(7, 4)
        图幅尺寸（英寸）。
    wiggle_scale_syn : float, default=1.0
        合成地震 wiggle 的水平偏移尺度（无量纲）。
    wiggle_scale_real : float, default=1.0
        真实/残差地震 wiggle 的水平偏移尺度（无量纲）。

    Returns
    -------
    tuple
        ``(fig, axes)``，``axes`` 长度为 6。

    Examples
    --------
    >>> fig, axes = plot_tie_window(logset, reflectivity, synthetic_seismic, real_seismic, xcorr, dxcorr)
    >>> plt.show()
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
    plot_trace(logset.AI, fig_axes=(fig, axes[0]))
    plot_reflectivity(reflectivity, fig_axes=(fig, axes[1]))

    # seismic
    plot_wiggle_trace(synthetic_seismic, scaling=wiggle_scale_syn, repeat_n_times=5, fig_axes=(fig, axes[2]))
    plot_wiggle_trace(real_seismic, scaling=wiggle_scale_real, repeat_n_times=5, fig_axes=(fig, axes[3]))

    residual = grid.Seismic(real_seismic.values - synthetic_seismic.values, real_seismic.basis, "twt", name="Residual")

    plot_wiggle_trace(residual, scaling=wiggle_scale_real, repeat_n_times=1, fig_axes=(fig, axes[4]))

    # for ax in axes[2:5]:
    #    _space = 0.05*np.abs(real_seismic.values.max())
    #    ax.set_xlim((real_seismic.values.min() - _space,
    #                 (real_seismic.values.max() + wiggle_scale_real + _space)))
    #    ax.set_xticks([])

    # dxcoor
    # _,_,cbar = plot_trace_as_pixels(xcorr, fig_axes=(fig, axes[4]))
    _, _, cbar = plot_dynamic_xcorr(dxcorr, fig_axes=(fig, axes[5]))
    axes[5].set_xlabel("Correlation")

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.set_yticklabels("")

    for ax in axes:
        ax.locator_params(axis="y", nbins=28)
        # ax.locator_params(axis='x', nbins=8)

    # axes[4].yaxis.tick_right()

    fig.suptitle("Max correlation of %.2f at a lag of %.3f s (Rc = %.2f)" % (xcorr.R, xcorr.lag, xcorr.Rc))

    fig.tight_layout()
    return fig, axes


def TMPplot_tie_window(
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
    """绘制简化版叠后井震标定窗口（临时接口）。

    与 :func:`plot_tie_window` 相比，本函数不绘制残差道，仅展示测井、反射系数、
    合成/真实地震及动态相关矩阵。

    Parameters
    ----------
    logset : grid.LogSet
        测井集合，至少使用 ``AI`` 曲线。
    reflectivity : grid.Reflectivity
        反射系数序列，shape 为 ``(n_samples,)``。
    synthetic_seismic : grid.Seismic
        合成地震道，shape 为 ``(n_samples,)``。
    real_seismic : grid.Seismic
        真实地震道，shape 为 ``(n_samples,)``。
    xcorr : grid.XCorr
        全局互相关结果，仅用于标题信息。
    dxcorr : grid.DynamicXCorr
        动态互相关矩阵。
    figsize : Tuple[int, int], default=(7, 4)
        图幅尺寸（英寸）。
    wiggle_scale_syn : float, default=1.0
        合成地震 wiggle 水平偏移尺度（无量纲）。
    wiggle_scale_real : float, default=1.0
        真实地震 wiggle 水平偏移尺度（无量纲）。

    Returns
    -------
    tuple
        ``(fig, axes)``，``axes`` 长度为 5。

    Examples
    --------
    >>> fig, axes = TMPplot_tie_window(logset, reflectivity, synthetic_seismic, real_seismic, xcorr, dxcorr)
    >>> plt.show()
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 7)

    axes = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[2:4]),
        fig.add_subplot(gs[4:6]),
        fig.add_subplot(gs[6:]),
    ]

    # logs
    plot_trace(logset.AI, fig_axes=(fig, axes[0]))
    plot_reflectivity(reflectivity, fig_axes=(fig, axes[1]))

    # seismic
    plot_wiggle_trace(synthetic_seismic, scaling=wiggle_scale_syn, repeat_n_times=5, fig_axes=(fig, axes[2]))
    plot_wiggle_trace(real_seismic, scaling=wiggle_scale_real, repeat_n_times=5, fig_axes=(fig, axes[3]))

    # dxcoor
    # _,_,cbar = plot_trace_as_pixels(xcorr, fig_axes=(fig, axes[4]))
    _, _, cbar = plot_dynamic_xcorr(dxcorr, fig_axes=(fig, axes[4]))
    axes[4].set_xlabel("Correlation")

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.set_yticklabels("")

    for ax in axes:
        ax.locator_params(axis="y", nbins=28)
        # ax.locator_params(axis='x', nbins=8)

    # axes[4].yaxis.tick_right()

    fig.suptitle("Max correlation of %.2f at a lag of %.3f s (Rc = %.2f)" % (xcorr.R, xcorr.lag, xcorr.Rc))

    fig.tight_layout()
    return fig, axes


def plot_prestack_tie_window(
    logset: grid.LogSet,
    reflectivity: grid.PreStackReflectivity,
    synthetic_seismic: grid.PreStackSeismic,
    real_seismic: grid.PreStackSeismic,
    xcorr: grid.PreStackXCorr,
    figsize: Tuple[int, int] = (7, 4),
    decimate_wiggles: int = 2,
    wiggle_scale_syn: float = 1.0,
    wiggle_scale_real: float = 1.0,
    reflectivity_scale: float = 1.0,
) -> tuple:
    """绘制叠前井震标定综合窗口。

    包含 AI、Vp/Vs、叠前合成/真实道集（gather）以及叠前相关矩阵可视化。

    Parameters
    ----------
    logset : grid.LogSet
        测井集合，使用 ``AI`` 与 ``Vp_Vs_ratio`` 曲线。
    reflectivity : grid.PreStackReflectivity
        叠前反射系数，``values`` shape 为 ``(n_traces, n_samples)``。
    synthetic_seismic : grid.PreStackSeismic
        叠前合成地震，``values`` shape 为 ``(n_traces, n_samples)``。
    real_seismic : grid.PreStackSeismic
        叠前真实地震，``values`` shape 为 ``(n_traces, n_samples)``。
    xcorr : grid.PreStackXCorr
        叠前互相关结果，用于像素图与标题统计。
    figsize : Tuple[int, int], default=(7, 4)
        图幅尺寸（英寸）。
    decimate_wiggles : int, default=2
        角度方向抽样步长；``n`` 表示每隔 ``n`` 条道绘制一条。
    wiggle_scale_syn : float, default=1.0
        合成道集 wiggle 水平偏移尺度（无量纲）。
    wiggle_scale_real : float, default=1.0
        真实道集 wiggle 水平偏移尺度（无量纲）。
    reflectivity_scale : float, default=1.0
        反射系数叠加到道集时的振幅缩放系数（无量纲）。

    Returns
    -------
    tuple
        ``(fig, axes)``，``axes`` 长度为 5。

    Examples
    --------
    >>> fig, axes = plot_prestack_tie_window(logset, reflectivity, synthetic_seismic, real_seismic, xcorr)
    >>> plt.show()
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 7)

    axes = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[2:4]),
        fig.add_subplot(gs[4:6]),
        fig.add_subplot(gs[6:]),
    ]

    # logs
    plot_trace(logset.AI, fig_axes=(fig, axes[0]))
    plot_trace(logset.Vp_Vs_ratio, fig_axes=(fig, axes[1]))

    # seismic
    plot_prestack_wiggle_trace(
        synthetic_seismic, scaling=wiggle_scale_syn, decimate_every_n=decimate_wiggles, fig_axes=(fig, axes[2])
    )
    axes[2].set_title("Synthetic gather")
    plot_prestack_wiggle_trace(
        real_seismic, scaling=wiggle_scale_real, decimate_every_n=decimate_wiggles, fig_axes=(fig, axes[3])
    )
    axes[3].set_title("Real gather")

    for ax in [axes[2], axes[3]]:
        plot_prestack_reflectivity(
            reflectivity,
            scaling=reflectivity_scale,
            decimate_every_n=decimate_wiggles,
            hline_params={"color": "k", "lw": 1.0},
            fig_axes=(fig, ax),
        )

    # xcoor
    _, _, cbar = plot_prestack_trace_as_pixels(xcorr, fig_axes=(fig, axes[4]), decimate_wiggles=decimate_wiggles)

    for ax in axes[1:-1]:
        ax.set_ylabel("")
        ax.set_yticklabels("")

    for ax in axes[:-1]:
        ax.locator_params(axis="y", nbins=28)
        # ax.locator_params(axis='x', nbins=8)

    axes[4].yaxis.tick_right()

    # fig.suptitle("Max correlation of %.2f at a lag of %.3f s (Rc = %.2f)" % \
    #             (xcorr.R, xcorr.lag, xcorr.Rc))

    fig.suptitle(
        "Prestack well-tie. \nMean max correlation of %.2f at a mean lag of %.3f s (Mean Rc = %.2f)"
        % (xcorr.R.mean(), xcorr.lag.mean(), xcorr.Rc.mean())
    )

    fig.tight_layout()
    return fig, axes


def NONOplot_prestack_tie_window(
    logset: grid.LogSet,
    reflectivity: grid.PreStackReflectivity,
    synthetic_seismic: grid.PreStackSeismic,
    real_seismic: grid.PreStackSeismic,
    xcorr: grid.PreStackXCorr,
    figsize: Tuple[int, int] = (7, 4),
    decimate_wiggles: int = 2,
    wiggle_scale_syn: float = 1.0,
    wiggle_scale_real: float = 1.0,
    reflectivity_scale: float = 1.0,
) -> tuple:
    """绘制包含残差道集的叠前井震标定窗口（试验接口）。

    该函数在 :func:`plot_prestack_tie_window` 基础上增加了残差道集子图。

    Parameters
    ----------
    logset : grid.LogSet
        测井集合，使用 ``AI`` 与 ``Vp_Vs_ratio``。
    reflectivity : grid.PreStackReflectivity
        叠前反射系数，shape 为 ``(n_traces, n_samples)``。
    synthetic_seismic : grid.PreStackSeismic
        叠前合成地震，shape 为 ``(n_traces, n_samples)``。
    real_seismic : grid.PreStackSeismic
        叠前真实地震，shape 为 ``(n_traces, n_samples)``。
    xcorr : grid.PreStackXCorr
        叠前互相关结果。
    figsize : Tuple[int, int], default=(7, 4)
        图幅尺寸（英寸）。
    decimate_wiggles : int, default=2
        角度方向抽样步长。
    wiggle_scale_syn : float, default=1.0
        合成道集 wiggle 水平偏移尺度（无量纲）。
    wiggle_scale_real : float, default=1.0
        真实与残差道集 wiggle 水平偏移尺度（无量纲）。
    reflectivity_scale : float, default=1.0
        反射系数叠加缩放系数（无量纲）。

    Returns
    -------
    tuple
        ``(fig, axes)``，``axes`` 长度为 6。

    Examples
    --------
    >>> fig, axes = NONOplot_prestack_tie_window(logset, reflectivity, synthetic_seismic, real_seismic, xcorr)
    >>> plt.show()
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 9)

    axes = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[2:4]),
        fig.add_subplot(gs[4:6]),
        fig.add_subplot(gs[6:8]),
        fig.add_subplot(gs[8:]),
    ]

    # logs
    plot_trace(logset.AI, fig_axes=(fig, axes[0]))
    plot_trace(logset.Vp_Vs_ratio, fig_axes=(fig, axes[1]))

    # seismic
    plot_prestack_wiggle_trace(
        synthetic_seismic, scaling=wiggle_scale_syn, decimate_every_n=decimate_wiggles, fig_axes=(fig, axes[2])
    )
    axes[2].set_title("Synthetic gather")
    plot_prestack_wiggle_trace(
        real_seismic, scaling=wiggle_scale_real, decimate_every_n=decimate_wiggles, fig_axes=(fig, axes[3])
    )
    axes[3].set_title("Real gather")

    # residual
    residual = []
    for theta in real_seismic.angles:
        residual.append(
            grid.Seismic(
                real_seismic[theta].values - synthetic_seismic[theta].values,
                real_seismic.basis,
                "twt",
                name="Residual",
                theta=theta,
            )
        )
    residual = grid.PreStackSeismic(residual, name="Residual")  # type: ignore

    plot_prestack_wiggle_trace(
        residual, scaling=wiggle_scale_real, decimate_every_n=decimate_wiggles, fig_axes=(fig, axes[4])
    )
    axes[4].set_title("Residual")

    for ax in [axes[2], axes[3], axes[4]]:
        plot_prestack_reflectivity(
            reflectivity,
            scaling=reflectivity_scale,
            decimate_every_n=decimate_wiggles,
            hline_params={"color": "k", "lw": 1.0},
            fig_axes=(fig, ax),
        )

    # xcoor
    _, _, cbar = plot_prestack_trace_as_pixels(xcorr, fig_axes=(fig, axes[5]), decimate_wiggles=decimate_wiggles)

    for ax in axes[1:-1]:
        ax.set_ylabel("")
        ax.set_yticklabels("")

    for ax in axes[:-1]:
        ax.locator_params(axis="y", nbins=28)
        # ax.locator_params(axis='x', nbins=8)

    axes[4].yaxis.tick_right()

    # fig.suptitle("Max correlation of %.2f at a lag of %.3f s (Rc = %.2f)" % \
    #             (xcorr.R, xcorr.lag, xcorr.Rc))

    fig.suptitle(
        "Prestack well-tie. \nMean max correlation of %.2f at a mean lag of %.3f s (Mean Rc = %.2f)"
        % (xcorr.R.mean(), xcorr.lag.mean(), xcorr.Rc.mean())
    )

    fig.tight_layout()
    return fig, axes


def plot_wavelet(
    wavelet: grid.Wavelet,
    figsize: Tuple[int, int] = None,  # type: ignore
    title: str = "Predicted wavelet",
    plot_params: dict = None,  # type: ignore
    fmax: float = None,  # type: ignore
    phi_max: float = 100,
    abs_t_max: float = None,  # type: ignore
    fig_axes: tuple = None,  # type: ignore
) -> tuple:
    """绘制单道子波时域波形及频谱（振幅/相位）。

    Parameters
    ----------
    wavelet : grid.Wavelet
        子波对象，``values`` shape 为 ``(n_samples,)``，采样间隔为 ``dt = sampling_rate``（s）。
    figsize : Tuple[int, int], optional
        新建图窗尺寸（英寸）。当 ``fig_axes`` 提供时忽略。
    title : str, default="Predicted wavelet"
        待确认：当前实现未使用该参数设置标题。
    plot_params : dict, optional
        待确认：当前实现未使用该参数。
    fmax : float, optional
        频率坐标上限（Hz）。为空时使用 Nyquist 端点。
    phi_max : float, default=100
        相位图 y 轴显示范围为 ``[-abs(phi_max), abs(phi_max)]``（度）。
    abs_t_max : float, optional
        时域图 x 轴绝对时间上限（s），显示区间为 ``[-abs_t_max, abs_t_max]``。
    fig_axes : tuple, optional
        传入已有 ``(fig, axes)``，其中 ``axes`` 需包含 3 个子图轴。

    Returns
    -------
    tuple
        ``(fig, axes)``，三个子图依次为时域波形、归一化振幅谱、相位谱。

    Raises
    ------
    AssertionError
        当不确定性频率轴 ``wavelet.uncertainties.ff`` 与计算频率轴不一致时触发。

    Examples
    --------
    >>> fig, axes = plot_wavelet(wavelet, fmax=60)
    >>> plt.show()
    """
    if plot_params is None:
        plot_params = {}

    if fig_axes is None:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2)

        axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    else:
        fig, axes = fig_axes

    ff, ampl, _, phase = compute_spectrum(wavelet.values, wavelet.sampling_rate, to_degree=True)

    ampl /= ampl.max()

    if fmax is None:
        fmax = ff[-1]

    axes[0].plot(wavelet.basis, wavelet.values, color="k", alpha=0.5, lw=0.5)
    axes[0].fill_between(
        wavelet.basis,
        wavelet.values,
        where=(wavelet.values >= 0.0),  # type: ignore
        color="b",
        alpha=0.8,
        interpolate=True,  # type: ignore
    )
    axes[0].fill_between(
        wavelet.basis,
        wavelet.values,
        where=(wavelet.values < 0.0),  # type: ignore
        color="r",
        alpha=0.8,
        interpolate=True,  # type: ignore
    )

    axes[0].plot(wavelet.basis, np.zeros_like(wavelet.basis), color="k", lw=0.5)
    axes[0].set_xlim((wavelet.basis[0], wavelet.basis[-1]))
    axes[0].set_ylim((2.0 * wavelet.values.min(), 1.15 * wavelet.values.max()))

    if abs_t_max is not None:
        axes[0].set_xlim((-abs_t_max, abs_t_max))

    # amplitude
    axes[1].plot(ff, ampl, "-")
    axes[1].set_ylim((0.0, 1.1 * ampl.max()))
    axes[1].set_xlim((ff[0], fmax))

    # phase
    axes[2].plot(ff, phase, "+")
    axes[2].plot(ff, np.zeros_like(phase), color="k", lw=0.5, alpha=0.5, linestyle="--")
    axes[2].set_xlim((ff[0], fmax))
    axes[2].set_ylim((-abs(phi_max), abs(phi_max)))

    # uncertainities
    if wavelet.uncertainties is not None:
        assert np.allclose(ff, wavelet.uncertainties.ff)

        # amplitude
        ampl_mean = wavelet.uncertainties.ampl_mean
        ampl_std = wavelet.uncertainties.ampl_std
        max_ = ampl_mean.max()
        ampl_mean /= max_
        ampl_std /= max_
        axes[1].fill_between(ff, ampl_mean - ampl_std, ampl_mean + ampl_std, color="gray", alpha=0.7)

        # phase
        axes[2].plot(ff, wavelet.uncertainties.phase_mean, color="gray", alpha=0.9, lw=0.9)
        axes[2].fill_between(
            ff,
            wavelet.uncertainties.phase_mean - wavelet.uncertainties.phase_std,
            wavelet.uncertainties.phase_mean + wavelet.uncertainties.phase_std,
            color="gray",
            alpha=0.6,
        )

    axes[0].set_xlabel("Time [s]")
    axes[1].set_ylabel("Normalized amplitude")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Phase [°]")
    axes[2].set_xlabel("Frequency [Hz]")

    fig.tight_layout()
    return fig, axes


def plot_prestack_wavelet(
    wavelet: grid.PreStackWavelet,
    figsize: Tuple[int, int] = (9, 6),
    three_angles: Tuple[int] = None,  # type: ignore
    title: str = "Predicted wavelet",
    plot_params: dict = None,  # type: ignore
    fmax: float = None,  # type: ignore
    phi_max: float = 100,
) -> tuple:
    """绘制叠前子波在三个角度下的时域与频域特征。

    Parameters
    ----------
    wavelet : grid.PreStackWavelet
        叠前子波集合，``values`` shape 为 ``(n_traces, n_samples)``。
    figsize : Tuple[int, int], default=(9, 6)
        图幅尺寸（英寸）。
    three_angles : Tuple[int], optional
        指定绘制的 3 个角度（度）。为空时自动取首、中、末角度。
    title : str, default="Predicted wavelet"
        待确认：当前实现未使用该参数设置标题。
    plot_params : dict, optional
        待确认：当前实现未使用该参数。
    fmax : float, optional
        频率坐标上限（Hz）。
    phi_max : float, default=100
        相位图 y 轴范围 ``[-phi_max, phi_max]``（度）。

    Returns
    -------
    tuple
        ``(fig, axes)``，共 9 个子图：3 个时域、3 个振幅谱、3 个相位谱。

    Raises
    ------
    AssertionError
        当不确定性频率轴与计算频率轴不一致时触发。

    Examples
    --------
    >>> fig, axes = plot_prestack_wavelet(prestack_wavelet)
    >>> plt.show()
    """
    if plot_params is None:
        plot_params = {}

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 3)

    axes = [
        fig.add_subplot(gs[:2, 0]),
        fig.add_subplot(gs[:2, 1]),
        fig.add_subplot(gs[:2, 2]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
        fig.add_subplot(gs[3, 0]),
        fig.add_subplot(gs[3, 1]),
        fig.add_subplot(gs[3, 2]),
    ]

    if three_angles is None:
        three_angles = [wavelet.angles[0], wavelet.angles[wavelet.angles.size // 2], wavelet.angles[-1]]

    # Wavelets
    for i, ax in enumerate([axes[0], axes[1], axes[2]]):
        theta = three_angles[i]
        _values = wavelet[theta].values
        ax.plot(wavelet.basis, _values, color="k", alpha=0.5, lw=0.5)
        ax.fill_between(wavelet.basis, _values, where=(_values >= 0.0), color="b", alpha=0.8, interpolate=True)  # type: ignore
        ax.fill_between(wavelet.basis, _values, where=(_values < 0.0), color="r", alpha=0.8, interpolate=True)  # type: ignore

        ax.plot(wavelet.basis, np.zeros_like(wavelet.basis), color="k", lw=0.5)
        ax.set_xlim((wavelet.basis[0], wavelet.basis[-1]))
        # ax.set_ylim((2.0*_values.min(), 1.15*_values.max()))
        ax.set_ylim((-0.5, 1.15))
        ax.set_title(("Wavelet at %d °" % theta))

    axes[0].set_xlabel("Time [s]")

    # Spectrum
    for i, idx in enumerate(range(3, 6, 1)):
        theta = three_angles[i]
        _values = wavelet[theta].values
        ff, ampl, _, phase = compute_spectrum(_values, wavelet.sampling_rate, to_degree=True)
        ampl /= ampl.max()

        axes[idx].plot(ff, ampl, "-")
        axes[idx].set_ylim((0.0, 1.05 * ampl.max()))
        axes[idx].set_xlim((ff[0], fmax))

        axes[idx + 3].plot(ff, phase, "+")
        axes[idx + 3].plot(ff, np.zeros_like(phase), color="k", lw=0.5, alpha=0.5, linestyle="--")
        axes[idx + 3].set_xlim((ff[0], fmax))
        axes[idx + 3].set_ylim((-phi_max, phi_max))

        # uncertainities
        wavelet_theta = wavelet[theta]
        assert isinstance(wavelet_theta, grid.Wavelet)

        if wavelet_theta.uncertainties is not None:
            assert np.allclose(ff, wavelet_theta.uncertainties.ff)

            # amplitude
            ampl_mean = wavelet_theta.uncertainties.ampl_mean
            ampl_std = wavelet_theta.uncertainties.ampl_std
            max_ = ampl_mean.max()
            ampl_mean /= max_
            ampl_std /= max_

            axes[idx].fill_between(ff, ampl_mean - ampl_std, ampl_mean + ampl_std, color="gray", alpha=0.7)

            # phase
            axes[idx + 3].plot(ff, wavelet_theta.uncertainties.phase_mean, color="gray", alpha=0.9, lw=0.9)
            axes[idx + 3].fill_between(
                ff,
                wavelet_theta.uncertainties.phase_mean - wavelet_theta.uncertainties.phase_std,
                wavelet_theta.uncertainties.phase_mean + wavelet_theta.uncertainties.phase_std,
                color="gray",
                alpha=0.6,
            )

    axes[3].set_ylabel("Normalized amplitude")
    axes[3].set_xlabel("Frequency [Hz]")
    axes[6].set_ylabel("Phase [°]")
    axes[6].set_xlabel("Frequency [Hz]")

    fig.tight_layout()
    return fig, axes


def plot_logsets_overlay(
    logset1: grid.LogSet,
    logset2: grid.LogSet,
    figsize: Tuple[int, int] = (7, 6),
    title: str = "Well Logs",
    fig_axes: tuple = None,  # type: ignore
) -> tuple:
    """将两套测井曲线叠加在同一组坐标轴上比较。

    Parameters
    ----------
    logset1 : grid.LogSet
        第一套测井数据。
    logset2 : grid.LogSet
        第二套测井数据，要求与 ``logset1`` 具有相同 ``basis_type``。
    figsize : Tuple[int, int], default=(7, 6)
        新建图窗尺寸（英寸）。
    title : str, default="Well Logs"
        待确认：当前实现未直接使用该参数。
    fig_axes : tuple, optional
        传入已有 ``(fig, axes)``。

    Returns
    -------
    tuple
        ``(fig, axes)``，当两套数据均包含 ``vs`` 时为 3 个子图，否则为 2 个子图。

    Raises
    ------
    AssertionError
        当两套测井 ``basis_type`` 不一致时触发。

    Examples
    --------
    >>> fig, axes = plot_logsets_overlay(logset_a, logset_b)
    >>> plt.show()
    """
    assert logset1.basis_type == logset2.basis_type
    is_vs = (logset1.vs is not None) and (logset2.vs is not None)

    if fig_axes is None:
        if is_vs:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, axes = fig_axes

    plot_logset(logset1, fig_axes=(fig, axes))
    plot_logset(logset2, fig_axes=(fig, axes), plot_params=dict(linewidth=1.0))

    return fig, axes


def plot_logset(
    logset: grid.LogSet,
    figsize: Tuple[int, int] = (7, 6),
    title: str = "Well Logs",
    plot_params: dict = None,  # type: ignore
    fig_axes: tuple = None,  # type: ignore
) -> tuple:
    """绘制单套测井曲线（Vp/Vs/Rho 或 Vp/Rho）。

    Parameters
    ----------
    logset : grid.LogSet
        测井集合。``vp`` 与 ``rho`` 必须存在；``vs`` 可选。
    figsize : Tuple[int, int], default=(7, 6)
        新建图窗尺寸（英寸）。
    title : str, default="Well Logs"
        图标题。
    plot_params : dict, optional
        透传给 ``matplotlib.axes.Axes.plot`` 的参数。
    fig_axes : tuple, optional
        传入已有 ``(fig, axes)``。

    Returns
    -------
    tuple
        ``(fig, axes)``，横轴单位分别为 km/s（Vp、Vs）与 g/cc（Rho）。

    Examples
    --------
    >>> fig, axes = plot_logset(logset)
    >>> plt.show()
    """
    is_vs = logset.vs is not None

    if fig_axes is None:
        if is_vs:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, axes = fig_axes

    if plot_params is None:
        plot_params = {}

    axes[0].plot(logset.vp / 1000, logset.basis, **plot_params)

    if is_vs:
        axes[1].plot(logset.vs / 1000, logset.basis, **plot_params)  # type: ignore
        axes[2].plot(logset.rho, logset.basis, **plot_params)
    else:
        axes[1].plot(logset.rho, logset.basis, **plot_params)

    axes[0].set_ylabel(logset.basis_type)
    axes[0].set_xlabel("Vp [km/s]")

    if is_vs:
        axes[1].set_xlabel("Vs [km/s]")
        axes[2].set_xlabel("Rho [g/cm³]")
    else:
        axes[1].set_xlabel("Rho [g/cm³]")

    for ax in axes:
        ax.set_ylim((logset.basis[0], logset.basis[-1]))
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(FormatStrFormatter("%0.1f"))

    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))

    for ax in axes[1:]:
        ax.set_yticklabels("")

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def plot_wellpath(wellpath: grid.WellPath, figsize: Tuple[int, int] = (5, 5), fig_axes: tuple = None) -> tuple:  # type: ignore
    """绘制井斜轨迹（MD 与 TVDKB 对比）。

    Parameters
    ----------
    wellpath : grid.WellPath
        井轨迹对象，使用 ``md`` 与 ``tvdkb``，shape 均为 ``(n_samples,)``。
    figsize : Tuple[int, int], default=(5, 5)
        新建图窗尺寸（英寸）。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。

    Returns
    -------
    tuple
        ``(fig, ax)``。

    Examples
    --------
    >>> fig, ax = plot_wellpath(wellpath)
    >>> plt.show()
    """
    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    ax.plot(wellpath.md, wellpath.tvdkb, lw=1.5)
    ax.plot(wellpath.md, wellpath.md, color="k", lw=0.5)
    ax.set_xlabel(grid.MD_NAME)
    ax.set_ylabel(grid.TVDKB_NAME)
    ax.set_title("Well curvature")

    ax.set_xlim((wellpath.md[0], wellpath.md[-1]))
    ax.set_ylim((wellpath.tvdkb[0], wellpath.md[-1]))
    ax.xaxis.set_ticks_position("top")
    ax.invert_yaxis()

    fig.tight_layout()

    return fig, ax


def plot_td_table(
    table: grid.TimeDepthTable,
    figsize: Tuple[int, int] = (4, 4),
    plot_params: dict = None,  # type: ignore
    fig_axes: tuple = None,  # type: ignore
) -> tuple:
    """绘制时深关系表（Time-Depth table）。

    Parameters
    ----------
    table : grid.TimeDepthTable
        时深表对象。若 ``table.is_md_domain`` 为真，横轴使用 ``md``；否则使用 ``tvdss``。
        纵轴始终为 ``twt``（双程时，常用单位 s）。
    figsize : Tuple[int, int], default=(4, 4)
        新建图窗尺寸（英寸）。
    plot_params : dict, optional
        透传给 ``Axes.plot`` 的参数。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。

    Returns
    -------
    tuple
        ``(fig, ax)``。

    Examples
    --------
    >>> fig, ax = plot_td_table(td_table)
    >>> plt.show()
    """
    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    if plot_params is None:
        plot_params = {}

    # 根据 TimeDepthTable 的模式选择正确的深度数据和标签
    if table.is_md_domain:
        depth_data = table.md
        depth_label = grid.MD_NAME
    else:
        depth_data = table.tvdss
        depth_label = grid.TVDSS_NAME

    ax.plot(depth_data, table.twt, **plot_params)
    ax.set_xlabel(depth_label)
    ax.set_ylabel(grid.TWT_NAME)
    ax.set_title("Time-Depth table.")

    ax.set_xlim((depth_data[0], depth_data[-1]))
    ax.set_ylim((table.twt[0], table.twt[-1]))
    # ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position("top")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))
    ax.invert_yaxis()

    fig.tight_layout()

    return fig, ax


def plot_reflectivity(reflectivity: grid.Reflectivity, figsize: tuple = (3, 5), fig_axes: tuple = None) -> tuple:  # type: ignore
    """绘制叠后反射系数脉冲图。

    Parameters
    ----------
    reflectivity : grid.Reflectivity
        反射系数序列，``values`` shape 为 ``(n_samples,)``，零值位置不绘制脉冲。
    figsize : tuple, default=(3, 5)
        新建图窗尺寸（英寸）。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。

    Returns
    -------
    tuple
        ``(fig, ax)``。

    Examples
    --------
    >>> fig, ax = plot_reflectivity(reflectivity)
    >>> plt.show()
    """
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


def plot_prestack_reflectivity(
    reflectivity: grid.PreStackReflectivity,
    scaling: float = 10.0,
    decimate_every_n: int = 1,
    figsize: tuple = (7, 6),
    fig_axes: tuple = None,  # type: ignore
    hline_params: dict = None,  # type: ignore
) -> tuple:
    """按角度绘制叠前反射系数脉冲图。

    Parameters
    ----------
    reflectivity : grid.PreStackReflectivity
        叠前反射系数，``values`` shape 为 ``(n_traces, n_samples)``。
    scaling : float, default=10.0
        振幅缩放系数（无量纲），用于增强横向可见性。
    decimate_every_n : int, default=1
        角度抽样步长；``n`` 表示每隔 ``n`` 条角度轨迹绘制一次。
    figsize : tuple, default=(7, 6)
        新建图窗尺寸（英寸）。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。
    hline_params : dict, optional
        透传给 ``Axes.hlines`` 的样式参数。

    Returns
    -------
    tuple
        ``(fig, ax)``，横轴为入射角（度）。

    Examples
    --------
    >>> fig, ax = plot_prestack_reflectivity(prestack_reflectivity, decimate_every_n=2)
    >>> plt.show()
    """
    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    if hline_params is None:
        hline_params = dict(color="r", lw=1.0)

    basis = reflectivity.basis

    for j, angle in enumerate(reflectivity.angles[::decimate_every_n]):
        values = scaling * reflectivity[angle].values
        x_picks = np.where(values != 0)[0]
        y_picks = values[values != 0.0]

        ax.plot(np.zeros_like(basis) + angle, basis, color="k", alpha=0.5, lw=0.5)
        for i in range(len(x_picks)):
            ax.hlines(basis[x_picks[i]], xmin=angle, xmax=y_picks[i] + angle, **hline_params)

    fig.suptitle(reflectivity.name)
    ax.set_ylabel(reflectivity.basis_type)

    ax.set_ylim((basis[0], basis[-1]))
    ax.invert_yaxis()

    ax.set_xlabel(grid.ANGLE_NAME)

    ax.yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax


def plot_trace(
    trace: grid.BaseTrace,
    figsize: Tuple[int, int] = (3, 5),
    fig_axes: tuple = None,  # type: ignore
    plot_params: dict = None,  # type: ignore
) -> tuple:
    """绘制单道曲线（通用一维 trace）。

    Parameters
    ----------
    trace : grid.BaseTrace
        一维道对象，``values`` 与 ``basis`` shape 均为 ``(n_samples,)``。
    figsize : Tuple[int, int], default=(3, 5)
        新建图窗尺寸（英寸）。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。
    plot_params : dict, optional
        透传给 ``Axes.plot`` 的样式参数。

    Returns
    -------
    tuple
        ``(fig, ax)``。

    Examples
    --------
    >>> fig, ax = plot_trace(trace)
    >>> plt.show()
    """
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


def plot_prestack_trace(
    trace: grid.BasePrestackTrace,
    scaling: float = 1.0,
    figsize: Tuple[int, int] = (3, 5),
    fig_axes: tuple = None,  # type: ignore
    plot_params: dict = None,  # type: ignore
) -> tuple:
    """按角度绘制叠前道集曲线。

    Parameters
    ----------
    trace : grid.BasePrestackTrace
        叠前道集，``values`` shape 为 ``(n_traces, n_samples)``。
    scaling : float, default=1.0
        振幅缩放系数（无量纲）。
    figsize : Tuple[int, int], default=(3, 5)
        新建图窗尺寸（英寸）。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。
    plot_params : dict, optional
        透传给 ``Axes.plot`` 的样式参数；为空时默认蓝色曲线。

    Returns
    -------
    tuple
        ``(fig, ax)``，横轴为角度（度）。

    Examples
    --------
    >>> fig, ax = plot_prestack_trace(prestack_trace, scaling=8.0)
    >>> plt.show()
    """
    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    if plot_params is None:
        plot_params = {"color": "b"}

    for theta in trace.angles:
        ax.plot(scaling * trace[theta].values + theta, trace.basis, **plot_params)

    fig.suptitle(trace.name)

    ax.set_xlabel(grid.ANGLE_NAME)
    # ax.xaxis.set_label_position("top")
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


def plot_prestack_wiggle_trace(
    trace: grid.BasePrestackTrace,
    figsize: Tuple[int, int] = (6, 6),
    scaling: float = 10.0,
    decimate_every_n: int = 1,
    fig_axes: tuple = None,  # type: ignore
) -> tuple:
    """绘制叠前道集的 wiggle 填充图。

    Parameters
    ----------
    trace : grid.BasePrestackTrace
        叠前道集，``values`` shape 为 ``(n_traces, n_samples)``。
    figsize : Tuple[int, int], default=(6, 6)
        新建图窗尺寸（英寸）。
    scaling : float, default=10.0
        振幅横向偏移缩放（无量纲）。
    decimate_every_n : int, default=1
        角度抽样步长。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。

    Returns
    -------
    tuple
        ``(fig, ax)``。

    Examples
    --------
    >>> fig, ax = plot_prestack_wiggle_trace(prestack_trace, decimate_every_n=2)
    >>> plt.show()
    """
    if fig_axes is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig, ax = fig_axes

    for theta in trace.angles[::decimate_every_n]:
        _x = scaling * trace[theta].values + theta
        ax.plot(_x, trace.basis, color="k", lw=0.5)
        ax.plot(theta * np.ones_like(_x), trace.basis, color="k", lw=0.2)

        ax.fill_betweenx(trace.basis, theta, _x, where=(_x >= theta), color="b", alpha=0.6, interpolate=True)
        ax.fill_betweenx(trace.basis, theta, _x, where=(_x < theta), color="r", alpha=0.6, interpolate=True)

    fig.suptitle(trace.name)
    ax.set_xlabel(grid.ANGLE_NAME)
    ax.set_ylim((trace.basis[0], trace.basis[-1]))
    ax.invert_yaxis()
    # ax.xaxis.set_label_position("top")
    ax.set_ylabel(trace.basis_type)

    # _space = 0.05*np.abs(trace.values.max())
    # ax.set_xlim((trace.values.min() - _space, (trace.values.max() + scaling + _space)))
    # ax.set_xticks([])

    if trace.is_twt:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax


def plot_wiggle_trace(
    trace: grid.BaseTrace,
    figsize: Tuple[int, int] = (4, 4),
    repeat_n_times: int = 7,
    scaling: float = 1.0,
    fig_axes: tuple = None,  # type: ignore
) -> tuple:
    """绘制单道 wiggle（振幅填充）图。

    Parameters
    ----------
    trace : grid.BaseTrace
        单道对象，``values`` shape 为 ``(n_samples,)``。
    figsize : Tuple[int, int], default=(4, 4)
        新建图窗尺寸（英寸）。
    repeat_n_times : int, default=7
        重复绘制道数，用于形成视觉道集；当为 1 时道线放置在 ``scaling / 2``。
    scaling : float, default=1.0
        道间水平偏移尺度（无量纲）。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。

    Returns
    -------
    tuple
        ``(fig, ax)``。

    Examples
    --------
    >>> fig, ax = plot_wiggle_trace(seismic_trace, repeat_n_times=5, scaling=1.2)
    >>> plt.show()
    """
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


def plot_trace_as_pixels(
    trace: grid.BaseTrace,
    repeat_n_times: int = 16,
    figsize: Tuple[int, int] = (3, 4),
    wiggle_scale: float = 2.0,
    fig_axes: tuple = None,  # type: ignore
    im_params: dict = None,  # type: ignore
) -> tuple:
    """将单道重复为像素图并叠加中心 wiggle 线。

    Parameters
    ----------
    trace : grid.BaseTrace
        单道对象，``values`` shape 为 ``(n_samples,)``。
    repeat_n_times : int, default=16
        横向重复次数，对应像素图 ``n_traces``。
    figsize : Tuple[int, int], default=(3, 4)
        新建图窗尺寸（英寸）。
    wiggle_scale : float, default=2.0
        中心 wiggle 叠加缩放系数（无量纲）。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。
    im_params : dict, optional
        透传给 ``Axes.imshow`` 的参数；默认色标范围 ``[-0.85, 0.85]``。

    Returns
    -------
    tuple
        ``(fig, ax, cbar)``，其中 ``cbar`` 为 Matplotlib colorbar 对象。

    Examples
    --------
    >>> fig, ax, cbar = plot_trace_as_pixels(trace)
    >>> plt.show()
    """
    absmax = np.abs(trace.values).max()

    pixels = np.empty((repeat_n_times, trace.size))
    for i in range(repeat_n_times):
        pixels[i, :] = trace.values

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    extent = [0, repeat_n_times, trace.basis[0], trace.basis[-1]]

    if im_params is None:
        im_params = dict(cmap="RdYlBu_r", vmin=-0.85, vmax=0.85)

    im = ax.imshow(pixels.T, interpolation="bilinear", aspect="auto", extent=extent, **im_params)  # type: ignore

    ax.plot(
        (repeat_n_times // 2) * np.ones_like(trace.values) + wiggle_scale * (trace.values / absmax),
        trace.basis,
        color="k",
    )

    ax.plot((repeat_n_times // 2) * np.ones_like(trace.values), trace.basis, color="k", lw=0.2, alpha=0.8)

    ax.set_xlabel(trace.name)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(trace.basis_type)
    ax.set_xticks([])
    if trace.is_twt or trace.is_tlag:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))

    cbar = fig.colorbar(im, ax=ax, shrink=1.0, orientation="horizontal", pad=0.02, aspect=5)
    cbar.set_label("", fontsize=11, rotation=90, y=0.5)

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax, cbar


def plot_prestack_trace_as_pixels(
    trace: grid.BasePrestackTrace,
    figsize: Tuple[int, int] = (4, 5),
    wiggle_scale: float = 2.0,
    fig_axes: tuple = None,  # type: ignore
    im_params: dict = None,  # type: ignore
    decimate_wiggles: int = 1,
) -> tuple:
    """将叠前道集绘制为像素图并叠加抽样 wiggle。

    Parameters
    ----------
    trace : grid.BasePrestackTrace
        叠前道集，``values`` shape 为 ``(n_traces, n_samples)``。
    figsize : Tuple[int, int], default=(4, 5)
        新建图窗尺寸（英寸）。
    wiggle_scale : float, default=2.0
        叠加 wiggle 缩放系数（无量纲）。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。
    im_params : dict, optional
        透传给 ``Axes.imshow`` 的参数。
    decimate_wiggles : int, default=1
        叠加 wiggle 的角度抽样步长。

    Returns
    -------
    tuple
        ``(fig, ax, cbar)``。

    Examples
    --------
    >>> fig, ax, cbar = plot_prestack_trace_as_pixels(prestack_trace, decimate_wiggles=2)
    >>> plt.show()
    """
    absmax = np.abs(trace.values).max()

    pixels = trace.values

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    extent = [trace.angles[0], trace.angles[-1], trace.basis[0], trace.basis[-1]]

    if im_params is None:
        im_params = dict(cmap="RdYlBu_r")

    im = ax.imshow(pixels.T, extent=extent, interpolation="bilinear", aspect="auto", **im_params)  # type: ignore

    for theta in trace.angles[::decimate_wiggles]:
        ax.plot(
            theta * np.ones_like(trace[theta].values) + wiggle_scale * (trace[theta].values / absmax),
            trace.basis,
            color="k",
            lw=0.5,
        )

    fig.suptitle(trace.name)
    ax.set_xlabel(grid.ANGLE_NAME)
    # ax.xaxis.set_label_position("top")
    ax.set_ylabel(trace.basis_type)
    # ax.set_xticks([])
    if trace.is_twt or trace.is_tlag:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, orientation="horizontal", pad=0.15, aspect=20)
    cbar.set_label("", fontsize=11, rotation=90, y=0.5)

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax, cbar


def plot_prestack_residual_as_pixels(
    trace1: grid.BasePrestackTrace,
    trace2: grid.BasePrestackTrace,
    figsize: Tuple[int, int] = (6, 6),
    im_params: dict = None,  # type: ignore
) -> tuple:
    """并排绘制两组叠前道集及其残差像素图。

    Parameters
    ----------
    trace1 : grid.BasePrestackTrace
        第一组叠前道集，shape 为 ``(n_traces, n_samples)``。
    trace2 : grid.BasePrestackTrace
        第二组叠前道集，shape 为 ``(n_traces, n_samples)``。
    figsize : Tuple[int, int], default=(6, 6)
        图幅尺寸（英寸）。
    im_params : dict, optional
        透传给 ``Axes.imshow`` 的参数。

    Returns
    -------
    tuple
        ``(fig, axes, cbar)``，``axes`` 长度为 3（trace1、trace2、residual）。

    Examples
    --------
    >>> fig, axes, cbar = plot_prestack_residual_as_pixels(real_gather, syn_gather)
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    extent = [trace1.angles[0], trace1.angles[-1], trace1.basis[0], trace1.basis[-1]]

    if im_params is None:
        im_params = dict(cmap="RdYlBu_r")

    im = axes[0].imshow(trace1.values.T, extent=extent, interpolation="bilinear", aspect="auto", **im_params)
    axes[1].imshow(trace2.values.T, extent=extent, interpolation="bilinear", aspect="auto", **im_params)

    # residual
    residual = []
    for theta in trace1.angles:
        residual.append(
            grid.Seismic(trace1[theta].values - trace2[theta].values, trace1.basis, "twt", name="Residual", theta=theta)
        )
    residual = grid.PreStackSeismic(residual, name="Residual")  # type: ignore

    axes[2].imshow(residual.values.T, extent=extent, interpolation="bilinear", aspect="auto", **im_params)

    axes[0].set_title(trace1.name)
    axes[1].set_title(trace2.name)
    axes[2].set_title(residual.name)

    axes[0].set_xlabel(grid.ANGLE_NAME)
    # ax.xaxis.set_label_position("top")
    axes[0].set_ylabel(trace1.basis_type)
    # ax.set_xticks([])
    if trace1.is_twt or trace1.is_tlag:
        axes[0].yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))
    for ax in axes[1:]:
        ax.set_yticklabels("")
        ax.set_xticklabels("")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, orientation="vertical")

    cbar.set_label("", fontsize=11, rotation=90, y=0.5)

    # fig.tight_layout()

    return fig, axes, cbar


def plot_dynamic_xcorr(
    dxcorr: grid.DynamicXCorr,
    figsize: Tuple[int, int] = (4, 5),
    fig_axes: tuple = None,  # type: ignore
    im_params: dict = None,  # type: ignore
) -> tuple:
    """绘制动态互相关矩阵（lag-time 热力图）。

    Parameters
    ----------
    dxcorr : grid.DynamicXCorr
        动态互相关对象。``values`` 为二维矩阵，x 轴使用 ``lags_basis``（转为 ms），
        y 轴使用 ``basis``（常见为 twt，单位 s）。
    figsize : Tuple[int, int], default=(4, 5)
        新建图窗尺寸（英寸）。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。
    im_params : dict, optional
        透传给 ``Axes.imshow`` 的参数；默认色标范围 ``[-0.85, 0.85]``。

    Returns
    -------
    tuple
        ``(fig, ax, cbar)``。

    Examples
    --------
    >>> fig, ax, cbar = plot_dynamic_xcorr(dxcorr)
    >>> plt.show()
    """
    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    # basis in seconds, lags in ms
    extent = [1000 * dxcorr.lags_basis[0], 1000 * dxcorr.lags_basis[-1], dxcorr.basis[-1], dxcorr.basis[0]]

    if im_params is None:
        im_params = dict(cmap="RdYlBu_r", vmin=-0.85, vmax=0.85)

    im = ax.imshow(dxcorr.values, extent=extent, interpolation="bilinear", aspect="auto", **im_params)  # type: ignore

    # fig.suptitle(dxcorr.name)
    ax.set_xlabel(dxcorr.name)  # type: ignore
    ax.xaxis.set_label_position("top")
    # ax.set_xlabel(dxcorr.lag_type)
    # ax.xaxis.set_label_position("top")
    ax.set_ylabel(dxcorr.basis_type)
    # ax.set_xticks([])
    if dxcorr.is_twt or dxcorr.is_tlag:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%0.3f"))

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, orientation="vertical", pad=0.05, aspect=30)
    cbar.set_label("", fontsize=11, rotation=90, y=0.5)

    if fig_axes is None:
        fig.tight_layout()

    return fig, ax, cbar


def plot_optimization_objective(ax_client, fig_axes: tuple = None, figsize=(6, 3)):  # type: ignore
    """绘制优化迭代过程中目标函数变化曲线。

    Parameters
    ----------
    ax_client : object
        需实现 ``get_trials_data_frame()`` 方法，且返回结果包含
        ``trial_index`` 与 ``goodness_of_match`` 两列。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。
    figsize : tuple, default=(6, 3)
        新建图窗尺寸（英寸）。

    Returns
    -------
    tuple
        ``(fig, ax)``。

    Examples
    --------
    >>> fig, ax = plot_optimization_objective(ax_client)
    >>> plt.show()
    """
    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    df = ax_client.get_trials_data_frame().sort_values("trial_index")
    tidx = df["trial_index"]
    ax.plot(tidx, df["goodness_of_match"])

    ax.set_title("Optimization objective")
    ax.set_xlabel("Iteration #")
    ax.set_ylabel("Central correlation")
    ax.set_xlim([tidx.values[0], tidx.values[-1]])  # type: ignore

    fig.tight_layout()

    return fig, ax


#######################
# warping
#######################


def plot_warping(
    ref_trace: grid.BaseTrace,
    other_trace: grid.BaseTrace,
    lags: grid.DynamicLag,
    scale: float = 1.0,
    figsize: Tuple[int, int] = (6.5, 4.5),  # type: ignore
    fig_axes: tuple = None,  # type: ignore
) -> tuple:
    """绘制两条道之间的 warping 对齐路径。

    该图将参考道与待对齐道分别归一化后上下错开显示，并使用连线展示每个采样点的
    对齐映射关系（由动态时移 ``lags`` 给出）。

    Parameters
    ----------
    ref_trace : grid.BaseTrace
        参考道（通常为真实地震），``values`` shape 为 ``(n_samples,)``。
    other_trace : grid.BaseTrace
        待对齐道（通常为合成地震），``values`` shape 为 ``(n_samples,)``。
    lags : grid.DynamicLag
        动态时移序列，``values`` shape 为 ``(n_samples,)``。其单位与 ``sampling_rate``
        一致，索引偏移由 ``round(values / sampling_rate)`` 计算。
    scale : float, default=1.0
        两条曲线的垂向分离尺度（无量纲）。
    figsize : Tuple[int, int], default=(6.5, 4.5)
        新建图窗尺寸（英寸）。
    fig_axes : tuple, optional
        传入已有 ``(fig, ax)``。

    Returns
    -------
    tuple
        ``(fig, ax)``。

    Raises
    ------
    AssertionError
        当 ``ref_trace``、``other_trace`` 与 ``lags`` 的 ``basis_type`` 不一致时触发。

    Notes
    -----
    可视化思路改写自 ``dtaidistance`` 项目中的 DTW 可视化实现。

    Examples
    --------
    >>> fig, ax = plot_warping(real_seismic, synthetic_seismic, dynamic_lag)
    >>> plt.show()
    """
    assert ref_trace.basis_type == other_trace.basis_type
    assert ref_trace.basis_type == lags.basis_type

    s1 = np.copy(ref_trace.values)
    s2 = np.copy(other_trace.values)

    s1 /= np.abs(s1).max()
    s1 += scale

    s2 /= np.abs(s2).max()
    s2 -= scale

    if fig_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_axes

    ax.plot(ref_trace.basis, s1, color="royalblue")
    ax.plot(other_trace.basis, s2, color="royalblue")
    ax.set_yticks([])
    ax.set_xlabel(ref_trace.basis_type)

    lines = []
    line_options = {"linewidth": 0.5, "color": "orange", "alpha": 0.8}

    lags_idx = np.round(lags.values / lags.sampling_rate).astype(int)
    path = [(i, i + lag) for i, lag in enumerate(lags_idx)]

    for r_c, c_c in path:
        if r_c < 0 or c_c < 0:
            continue
        if r_c >= s1.size or c_c >= s1.size:
            continue

        r_c_value = ref_trace.basis[r_c]
        c_c_value = ref_trace.basis[c_c]

        con = ConnectionPatch(
            xyA=[r_c_value, s1[r_c]],  # type: ignore
            coordsA=ax.transData,
            xyB=[c_c_value, s2[c_c]],  # type: ignore
            coordsB=ax.transData,
            **line_options,
        )
        lines.append(con)

    for line in lines:
        fig.add_artist(line)

    fig.suptitle("Warping Path")
    fig.tight_layout()

    return fig, ax
