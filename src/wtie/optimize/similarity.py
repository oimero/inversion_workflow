"""wtie.optimize.similarity: 相似度与互相关计算工具。

该模块用于计算地震道/合成道的整体相似度、归一化互相关及其动态版本，
并提供叠前数据的角度维聚合接口。

边界说明
--------
- 仅负责数值相似度计算与结果封装，不负责数据读取、预处理或可视化。
- 不在本模块内做重采样或单位换算；输入道对象需满足可比性约束。

核心公开对象
------------
1. central_xcorr_coeff: 统一入口，按是否叠前自动选择中心互相关系数计算方式。
2. traces_normalized_xcorr: 单道归一化互相关，返回 `grid.XCorr`。
3. dynamic_normalized_xcorr: 滑窗动态互相关，返回 `grid.DynamicXCorr`。
4. pep: 以能量比定义的相似度指标（PEP）。

Examples
--------
>>> import numpy as np
>>> from wtie.optimize.similarity import normalized_xcorr_maximum
>>> normalized_xcorr_maximum(np.array([1., 2., 3.]), np.array([1., 2., 3.]))
1.0
"""

import numpy as np

from wtie.processing import grid


def pep(seismic: grid.Seismic, synthetic: grid.Seismic, normalize: bool = False) -> float:
    """计算 PEP（基于能量比的相似度指标）。

    该实现按公式 ``1 - residual_energy / trace_energy`` 计算，其中残差为
    ``seismic.values - synthetic.values``。当 ``normalize=True`` 时，会先将两条
    道分别按各自绝对值最大值归一化。

    Parameters
    ----------
    seismic : grid.Seismic
        参考地震道，参与能量基准计算。
    synthetic : grid.Seismic
        合成地震道，需与 `seismic` 在采样点数与物理域上可对齐。
    normalize : bool, default=False
        是否在计算前分别做幅值归一化。

    Returns
    -------
    float
        PEP 值（无量纲）。理论上当残差能量不超过参考道能量时位于 [0, 1]，
        但该实现未做截断，实际可小于 0 或大于 1。

    Notes
    -----
    当参考道或归一化分母为 0 时，结果可能出现 `nan`/`inf`。

    Examples
    --------
    >>> # 待确认：以下示例依赖项目中的 grid.Seismic 构造方式
    >>> # p = pep(seismic_trace, synthetic_trace, normalize=False)
    >>> # float(p)
    """

    if normalize:
        seismic_values = seismic.values / np.abs(seismic.values).max()
        synth_values = synthetic.values / np.abs(synthetic.values).max()
    else:
        seismic_values = seismic.values
        synth_values = synthetic.values

    trace_energy = energy(seismic_values)

    residual = seismic_values - synth_values
    residual_energy = energy(residual)

    p = 1.0 - residual_energy / trace_energy

    return p


def normalized_xcorr_maximum(a: np.ndarray, b: np.ndarray) -> float:
    """返回归一化互相关序列的最大值。

    Parameters
    ----------
    a : numpy.ndarray
        一维序列，shape 为 `(n_samples,)`。
    b : numpy.ndarray
        一维序列，shape 为 `(n_samples,)`。

    Returns
    -------
    float
        归一化互相关序列中的最大系数（无量纲）。

    Examples
    --------
    >>> import numpy as np
    >>> normalized_xcorr_maximum(np.array([1., 0., -1.]), np.array([1., 0., -1.]))
    1.0
    """
    return normalized_xcorr(a, b).max()


def normalized_xcorr_central_coeff(a: np.ndarray, b: np.ndarray) -> float:
    """返回归一化互相关在零时移（中心点）处的系数。

    Parameters
    ----------
    a : numpy.ndarray
        一维序列，shape 为 `(n_samples,)`。
    b : numpy.ndarray
        一维序列，shape 为 `(n_samples,)`。

    Returns
    -------
    float
        互相关中心系数（无量纲），用于衡量零时移对齐下的相似程度。

    Examples
    --------
    >>> import numpy as np
    >>> normalized_xcorr_central_coeff(np.array([1., 2., 3.]), np.array([1., 2., 3.]))
    1.0
    """
    xcorr = normalized_xcorr(a, b)
    return xcorr[xcorr.size // 2]


def normalized_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算两个一维序列的归一化互相关（full 模式）。

    先对输入做去均值与标准差归一化，再调用 ``numpy.correlate(..., 'full')``，
    最后按 ``max(len(a), len(b))`` 缩放。

    Parameters
    ----------
    a : numpy.ndarray
        输入序列，shape 为 `(n_samples,)`。
    b : numpy.ndarray
        输入序列，shape 为 `(n_samples,)`。

    Returns
    -------
    numpy.ndarray
        归一化互相关序列，shape 为 `(2 * n_samples - 1,)`（当两序列等长时）。
        返回值无量纲。

    Notes
    -----
    若输入标准差为 0，会出现除以 0，结果可能包含 `nan`/`inf`。

    Examples
    --------
    >>> import numpy as np
    >>> normalized_xcorr(np.array([1., 2., 3.]), np.array([1., 2., 3.])).shape
    (5,)
    """
    a = (a - np.mean(a)) / (np.std(a))
    b = (b - np.mean(b)) / (np.std(b))
    xcorr = np.correlate(a, b, "full") / max(len(a), len(b))
    return xcorr


def energy(x: np.ndarray) -> float:
    """计算一维序列的能量（平方和）。

    Parameters
    ----------
    x : numpy.ndarray
        输入序列，shape 为 `(n_samples,)`。

    Returns
    -------
    float
        序列能量，即 ``sum(x**2)``。单位为输入幅值单位的平方。

    Examples
    --------
    >>> import numpy as np
    >>> energy(np.array([1., 2., 3.]))
    14.0
    """
    return np.sum(np.square(x))


def traces_normalized_xcorr(trace1: grid.BaseTrace, trace2: grid.BaseTrace) -> grid.XCorr:
    """计算两条单道的归一化互相关并封装为 `grid.XCorr`。

    该函数要求两条道的物理域类型一致（如时间域/深度域）且采样间隔 `dt`
    相同；返回结果中的 lag 轴由输入采样间隔推导。

    Parameters
    ----------
    trace1 : grid.BaseTrace
        输入道 1，需提供 `values`、`sampling_rate`、`basis_type` 与 `is_twt`。
    trace2 : grid.BaseTrace
        输入道 2，约束同 `trace1`。

    Returns
    -------
    grid.XCorr
        互相关对象。其 `values` 为互相关系数序列（无量纲），
        `basis` 为 lag 轴（单位与输入 basis 一致；时间域常见为 s，待确认）。

    Raises
    ------
    AssertionError
        当两条道的 `basis_type` 不一致，或 `sampling_rate` 不一致时触发。

    Examples
    --------
    >>> # 待确认：trace1/trace2 需为已对齐采样的 grid.BaseTrace 对象
    >>> # xc = traces_normalized_xcorr(trace1, trace2)
    >>> # xc.values.shape
    """

    # verify basis
    assert trace1.basis_type == trace2.basis_type
    assert np.allclose(trace1.sampling_rate, trace2.sampling_rate)

    # xcorr
    xcorr = normalized_xcorr(trace1.values, trace2.values)

    # lag
    sr = trace1.sampling_rate
    duration = xcorr.size * sr
    lag = np.arange(-duration / 2, (duration - sr) / 2, sr)

    if trace1.is_twt:
        btype = "tlag"
    else:
        btype = "zlag"

    return grid.XCorr(xcorr, lag, btype, name="XCorr")


def prestack_traces_normalized_xcorr(
    trace1: grid.BasePrestackTrace, trace2: grid.BasePrestackTrace
) -> grid.PreStackXCorr:
    """按角度逐道计算叠前互相关并聚合。

    对 `trace1.angles` 中每个角度 `theta`，分别调用
    :func:`traces_normalized_xcorr` 计算互相关，并返回 `grid.PreStackXCorr`。

    Parameters
    ----------
    trace1 : grid.BasePrestackTrace
        叠前道集 1，包含多个角度子道。
    trace2 : grid.BasePrestackTrace
        叠前道集 2，角度集合需与 `trace1` 完全一致。

    Returns
    -------
    grid.PreStackXCorr
        叠前互相关结果，内部包含各角度对应的 `grid.XCorr`。

    Raises
    ------
    AssertionError
        当两者角度集合不一致时触发。

    Examples
    --------
    >>> # 待确认：pst1/pst2 需为角度集合一致的叠前道集
    >>> # pxc = prestack_traces_normalized_xcorr(pst1, pst2)
    >>> # len(pxc.xcorr)
    """

    assert (trace1.angles == trace2.angles).all()

    xcorr = []
    for theta in trace1.angles:
        xc = traces_normalized_xcorr(trace1[theta], trace2[theta])
        xc.theta = theta
        xcorr.append(xc)

    return grid.PreStackXCorr(xcorr)  # type: ignore


def prestack_mean_central_xcorr_coeff(trace1: grid.BasePrestackTrace, trace2: grid.BasePrestackTrace) -> float:
    """计算叠前多角度中心互相关系数的均值。

    Parameters
    ----------
    trace1 : grid.BasePrestackTrace
        叠前道集 1。
    trace2 : grid.BasePrestackTrace
        叠前道集 2。

    Returns
    -------
    float
        各角度中心互相关系数均值（无量纲）。

    Examples
    --------
    >>> # 待确认：pst1/pst2 为可比较的 grid.BasePrestackTrace
    >>> # rc = prestack_mean_central_xcorr_coeff(pst1, pst2)
    >>> # float(rc)
    """
    xcorr = prestack_traces_normalized_xcorr(trace1, trace2)
    return np.mean(xcorr.Rc)  # type: ignore


def central_xcorr_coeff(trace1: grid.trace_t, trace2: grid.trace_t) -> float:
    """统一计算中心互相关系数（支持叠后与叠前）。

    当 `trace1.is_prestack` 为 True 时，调用叠前均值中心互相关；
    否则对单道 `values` 计算零时移中心系数。

    Parameters
    ----------
    trace1 : grid.trace_t
        输入道 1，可为叠后单道或叠前道集。
    trace2 : grid.trace_t
        输入道 2，类型应与 `trace1` 匹配。

    Returns
    -------
    float
        中心互相关系数（无量纲）。

    Examples
    --------
    >>> # 待确认：对叠后道或叠前道集均可调用
    >>> # rc = central_xcorr_coeff(trace1, trace2)
    >>> # float(rc)
    """
    if trace1.is_prestack:
        return prestack_mean_central_xcorr_coeff(trace1, trace2)  # type: ignore
    else:
        return normalized_xcorr_central_coeff(trace1.values, trace2.values)


def dynamic_normalized_xcorr(trace1: np.ndarray, trace2: np.ndarray, window_lenght: float = 0.070) -> grid.DynamicXCorr:
    """计算滑动窗口动态归一化互相关。

    对每个采样点，以长度 `window_lenght` 的窗口在两条道上截取片段，
    计算局部归一化互相关，并沿样点堆叠为二维结果。

    Parameters
    ----------
    trace1 : numpy.ndarray
        道对象（非纯 ndarray），需具备 `basis`、`values`、`sampling_rate`、`size`
        与 `basis_type` 属性。
    trace2 : numpy.ndarray
        道对象，约束同 `trace1`。
    window_lenght : float, default=0.070
        滑窗长度。单位与输入 `basis` 一致；时间域常见为 s（待确认）。

    Returns
    -------
    grid.DynamicXCorr
        动态互相关对象。
        - `values` shape 为 `(n_samples, n_lags)`
        - `basis` 为原始采样轴（shape 为 `(n_samples,)`）
        - lag 轴由 `grid.DynamicXCorr` 内部管理（待确认）

    Raises
    ------
    AssertionError
        当 `trace1.basis` 与 `trace2.basis` 不一致（容差 `1e-3`）时触发。

    Notes
    -----
    窗口越界部分通过随机噪声填充，噪声标准差取两条道标准差较小值的 10%。

    Examples
    --------
    >>> # 待确认：trace1/trace2 需提供 basis、values、sampling_rate 等属性
    >>> # dxc = dynamic_normalized_xcorr(trace1, trace2, window_lenght=0.07)
    >>> # dxc.values.shape
    """

    assert np.allclose(trace1.basis, trace2.basis, atol=1e-3)

    half_index = int(round(window_lenght / trace1.sampling_rate)) // 2

    # boundary
    _std = min(np.std(trace1.values), np.std(trace2.values))
    pady = lambda: np.random.normal(scale=0.1 * _std, size=(half_index,))
    s1 = np.concatenate((pady(), trace1.values, pady()))
    s2 = np.concatenate((pady(), trace2.values, pady()))

    # prealloc
    s1_w_ = s1[: 2 * half_index]
    s2_w_ = s2[: 2 * half_index]
    xcorr_ = normalized_xcorr(s1_w_, s2_w_)
    dxcorr = np.zeros((trace1.basis.size, xcorr_.size))

    # sliding window
    for i in range(trace1.size):
        # window
        j = i + half_index
        s1_w = s1[j - half_index : j + half_index]
        s2_w = s2[j - half_index : j + half_index]

        # correlation
        dxcorr[i, :] = normalized_xcorr(s1_w, s2_w)

    return grid.DynamicXCorr(dxcorr, trace1.basis, grid._inverted_name(trace1.basis_type), name="Dynamic X-Correlation")
