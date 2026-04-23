"""wtie.optimize.logs: 井震标定中测井处理与合成计算工具。

本模块提供测井去噪与块化、时深关系扰动与重建、反射系数计算、
以及叠后/叠前合成地震生成等函数，用于支撑井震对齐参数搜索流程。

边界说明
--------
- 本模块不负责优化器策略、目标函数设计与训练流程调度。
- 本模块仅实现单井曲线与时深表的基础数值处理，不承担上游数据质检。

核心公开对象
------------
1. filter_log / filter_logs: 单条或批量测井去尖峰、补空与平滑。
2. block_logs: 基于 Vp、AI 或自身分段规则的测井块化。
3. get_tdt_from_vp: 由 Vp 与参考检查点关系构建时深表。
4. compute_prestack_reflectivity / compute_acoustic_relfectiviy: 叠前/零偏移反射系数计算。
5. compute_synthetic_seismic / compute_synthetic_prestack_seismic: 叠后与叠前合成地震生成。

Examples
--------
>>> from wtie.optimize import logs as opt_logs
>>> filt = opt_logs.filter_log(vp_log, median_size=31, threshold=4.0, std=2.0)
>>> r0 = opt_logs.compute_acoustic_relfectiviy(logs_twt)
"""

import random

import numpy as np

from wtie.modeling.modeling import convolution_modeling
from wtie.processing import grid

# from wtie.processing.grid import convert_log_from_md_to_twt as _md_to_twt
# from wtie.processing.logs import despike, interpolate_nans, smooth, blocking
# from wtie.processing.logs import _compute_block_segments, _block_from_segments
from wtie.processing import logs as _logs

# from wtie.processing.reflection import vertical_acoustic_reflectivity, prestack_rpp
from wtie.processing import reflection as _reflection
from wtie.utils.types_ import List


##################
# LOGS
##################
def update_log_values(new_values: np.ndarray, current_log: grid.Log) -> grid.Log:
    """用新数值更新测井对象并保持原坐标轴与元信息。

    Parameters
    ----------
    new_values : numpy.ndarray
        新的曲线采样值，shape 为 ``(n_samples,)``。
    current_log : wtie.processing.grid.Log
        被更新的原始测井对象。其 ``basis``、basis 类型与名称将被保留。

    Returns
    -------
    wtie.processing.grid.Log
        更新后的测井对象，采样轴与 ``current_log`` 一致。
    """
    return grid.update_trace_values(new_values, current_log)  # type: ignore
    # return grid.Log(new_values,
    #                current_log.basis,
    #                grid._inverted_name(current_log.basis_type),
    #                name=current_log.name)


def old_temporal_strech_squeeze(
    table: grid.TimeDepthTable,
    central_idx: int,
    delta_idx: int,
    pert_ratio: float,
    dt: float = 0.012,
    mode: str = "slinear",
) -> grid.TimeDepthTable:
    """对时深表做局部拉伸/压缩扰动（旧实现，保留兼容）。

    该函数先将输入时深表按 ``dt`` 重采样，然后在 ``central_idx`` 附近以
    ``delta_idx`` 为窗口删除一段样点，并在中心位置插入扰动后的单点时间，
    最终再次插值为均匀时间采样。

    Parameters
    ----------
    table : wtie.processing.grid.TimeDepthTable
        输入时深表。
    central_idx : int
        中心样点索引。
    delta_idx : int
        扰动窗口半宽（样点数）。
    pert_ratio : float
        中心时间扰动比例。中心时间将变为原值的 ``(1 + pert_ratio)`` 倍。
    dt : float, default=0.012
        时间采样间隔。单位与 ``table.twt`` 一致，按项目约定通常为 s。
    mode : str, default='slinear'
        时间插值模式，直接传递给 ``temporal_interpolation``。

    Returns
    -------
    wtie.processing.grid.TimeDepthTable
        扰动后的时深表，时间轴为均匀采样。

    Raises
    ------
    AssertionError
        当 ``central_idx - delta_idx <= 0`` 或
        ``central_idx + delta_idx >= table.size`` 时触发。
    """

    table = table.temporal_interpolation(dt=dt)
    assert central_idx - delta_idx > 0
    assert central_idx + delta_idx < table.size

    twt_ = np.concatenate(
        (
            table.twt[: central_idx - delta_idx],
            table.twt[central_idx] * (1 + pert_ratio) * np.ones((1,)),
            table.twt[central_idx + delta_idx :],
        )
    )
    tvd_ = np.concatenate(
        (
            table.tvdss[: central_idx - delta_idx],
            table.tvdss[central_idx] * np.ones((1,)),
            table.tvdss[central_idx + delta_idx :],
        )
    )
    tdt_ = grid.TimeDepthTable(twt_, tvd_)
    return tdt_.temporal_interpolation(dt=dt, mode=mode)


def filter_log(
    log: grid.Log,
    median_size: int = 31,
    threshold: float = 4.0,
    std: float = 2.0,
    std2: float = None,  # type: ignore
) -> grid.Log:
    """对单条测井曲线执行去尖峰、插值补空与高斯平滑。

    Parameters
    ----------
    log : wtie.processing.grid.Log
        输入测井曲线。
    median_size : int, default=31
        中值滤波窗口长度（样点数），用于去尖峰。
    threshold : float, default=4.0
        去尖峰阈值，直接传递给 ``wtie.processing.logs.despike``。
    std : float, default=2.0
        第一次高斯平滑标准差（样点尺度）。
    std2 : float or None, default=None
        若提供且为真值，则执行第二次高斯平滑。

    Returns
    -------
    wtie.processing.grid.Log
        处理后的测井曲线，``basis`` 与输入一致，``values`` shape 为
        ``(n_samples,)``。
    """
    f_log = _logs.despike(log.values, median_size=median_size, threshold=threshold)
    f_log = _logs.interpolate_nans(f_log)
    f_log = _logs.smooth(f_log, std=std)
    if std2:
        f_log = _logs.smooth(f_log, std=std2)

    return update_log_values(f_log, log)


def filter_logs(
    logset: grid.LogSet,
    median_size: int = 31,
    threshold: float = 4.0,
    std: float = 2.0,
    std2: float = None,  # type: ignore
    log_keys: List[str] = None,  # type: ignore
) -> grid.LogSet:
    """批量过滤 ``LogSet`` 中的测井曲线。

    默认处理 ``logset`` 的全部曲线；若指定 ``log_keys``，仅处理给定键。

    Parameters
    ----------
    logset : wtie.processing.grid.LogSet
        输入测井集合。
    median_size : int, default=31
        中值滤波窗口长度（样点数）。
    threshold : float, default=4.0
        去尖峰阈值。
    std : float, default=2.0
        第一次高斯平滑标准差（样点尺度）。
    std2 : float or None, default=None
        第二次高斯平滑标准差；为 ``None`` 时不执行第二次平滑。
    log_keys : list of str or None, default=None
        需要处理的曲线键列表。为 ``None`` 时处理全部曲线。

    Returns
    -------
    wtie.processing.grid.LogSet
        仅包含被处理曲线的新 ``LogSet``。

    Notes
    -----
    返回对象中的键集合遵循当前实现：当 ``log_keys`` 非空时，仅返回
    ``log_keys`` 对应曲线，而不是回填未处理曲线。
    """

    keys_ = logset.Logs.keys() if log_keys is None else log_keys

    new_logs = {}
    for key_ in keys_:
        new_logs[key_] = filter_log(logset[key_], median_size, threshold, std, std2)

    return grid.LogSet(new_logs)


def block_logs(
    logset: grid.LogSet,
    threshold_perc: float,
    maximum_length: int = None,  # type: ignore
    baseline: str = "AI",
    log_keys: List[str] = None,  # type: ignore
) -> grid.LogSet:
    """按分段策略对测井曲线进行块化（blocking）。

    分段可基于 ``Vp``、``AI`` 或每条曲线自身计算。对于 ``Vp``/``Vs``，
    块内采用 harmonic 平均；其余曲线采用 arithmetic 平均。

    Parameters
    ----------
    logset : wtie.processing.grid.LogSet
        输入测井集合。
    threshold_perc : float
        分段阈值百分比，例如 5 表示内部阈值 0.05。推荐范围待确认。
    maximum_length : int or None, default=None
        单段最大长度（样点数）。为 ``None`` 时取 ``round(n_samples / 4)``。
    baseline : {'Vp', 'AI', 'itself'}, default='AI'
        分段基准：
        ``'Vp'`` 使用 ``logset.vp``，``'AI'`` 使用 ``logset.ai``，
        ``'itself'`` 对每条曲线独立分段。
    log_keys : list of str or None, default=None
        需要块化的曲线键列表。为 ``None`` 时处理全部曲线。

    Returns
    -------
    wtie.processing.grid.LogSet
        块化后的测井集合。

    Raises
    ------
    ValueError
        当 ``baseline`` 不是 ``'Vp'``、``'AI'`` 或 ``'itself'`` 时触发。
    """
    # Filter log keys
    keys_ = logset.Logs.keys() if log_keys is None else log_keys

    # Parameters
    threshold = threshold_perc / 100

    if maximum_length is None:
        maximum_length = int(round(logset.basis.size // 4))

    # Compute segments
    if baseline == "Vp":
        segments = _logs._compute_block_segments(logset.vp, threshold, maximum_length)
        segments = len(keys_) * [segments]
    elif baseline == "AI":
        segments = _logs._compute_block_segments(logset.ai, threshold, maximum_length)
        segments = len(keys_) * [segments]
    elif baseline == "itself":
        segments = [
            _logs._compute_block_segments(logset.Logs[key_].values, threshold, maximum_length) for key_ in keys_
        ]
    else:
        raise ValueError
    segments = [tuple(s) for s in segments]

    # Block logs
    new_logs = {}
    for i, key_ in enumerate(keys_):
        log = logset[key_]
        if key_ in ["Vp", "Vs"]:
            log_b = _logs._block_from_segments(log.values, segments[i], "harmonic")
        else:
            log_b = _logs._block_from_segments(log.values, segments[i], "arithmetic")

        new_logs[key_] = update_log_values(log_b, log)

    return grid.LogSet(new_logs)


###################
# TD TABLES
###################

_apply_poly = lambda x, p: np.poly1d(p)(x)


def _perturbe_poly(p, delta_abs):
    """对多项式系数做相对幅度随机扰动。"""
    return [p_i + p_i * (random.uniform(-delta_abs, delta_abs)) for i, p_i in enumerate(p[::-1])][::-1]


def OLDget_perturbed_time_depth_tables(
    tdt: grid.TimeDepthTable, n: int = 100, delta_abs: float = 0.03, order: int = 6
) -> List[grid.TimeDepthTable]:
    """基于多项式拟合生成扰动时深表样本（旧实现，保留兼容）。

    Parameters
    ----------
    tdt : wtie.processing.grid.TimeDepthTable
        基准时深表。
    n : int, default=100
        目标生成样本数。
    delta_abs : float, default=0.03
        多项式系数相对扰动幅度。
    order : int, default=5
        拟合多项式阶数。

    Returns
    -------
    list of wtie.processing.grid.TimeDepthTable
        扰动后的时深表列表，长度最多为 ``n``。

    Raises
    ------
    ValueError
        当连续尝试超过 ``4 * n`` 次仍无法构造有效时深表时触发。
    """

    tables = []
    poly = np.polyfit(tdt.tvdss[1:], tdt.twt[1:], order)

    i = 0
    _i = 0
    while i < n:
        _i += 1
        poly_pert = _perturbe_poly(poly, delta_abs)
        p_twt_ = _apply_poly(tdt.tvdss[1:], poly_pert)
        p_twt_ = np.concatenate(
            (
                np.zeros(
                    1,
                ),
                p_twt_,
            )
        )
        try:
            tables.append(grid.TimeDepthTable(twt=p_twt_, tvdss=tdt.tvdss))
            i += 1
        except:
            # unrealsitic perurbations
            if _i > 4 * n:
                raise ValueError
            continue

    return tables


def get_tdt_from_vp(Vp: grid.Log, tdt: grid.TimeDepthTable, wp: grid.WellPath = None) -> grid.TimeDepthTable:  # type: ignore
    """根据 Vp 曲线与检查点关系构建 TVDSS-TWT 时深表。

    Parameters
    ----------
    Vp : wtie.processing.grid.Log
        纵波速度曲线，单位通常为 m/s。支持 MD 或 TWT 作为 ``basis``。
    tdt : wtie.processing.grid.TimeDepthTable
        参考时深表（通常来自 checkshot）。
    wp : wtie.processing.grid.WellPath or None, default=None
        井斜轨迹。当 ``Vp`` 以 MD 为坐标时参与深度转换。

    Returns
    -------
    wtie.processing.grid.TimeDepthTable
        由 ``Vp`` 推导的时深关系表。

    Raises
    ------
    NotImplementedError
        当 ``Vp`` 既不是 MD 也不是 TWT 坐标时触发。

    Notes
    -----
    代码中会计算 ``z_error`` 与 ``t_error``，但当前实现未使用这些误差量。
    """

    if Vp.is_md:
        t_start, z_error = grid.TimeDepthTable.get_twt_start_from_checkshots(Vp, wp, tdt)
        sonic_tdt_pert = grid.TimeDepthTable.get_tvdss_twt_relation_from_vp(Vp, wp=wp, origin=t_start)
    elif Vp.is_twt:
        z_start, t_error = grid.TimeDepthTable.get_tvdss_start_from_checkshots(Vp, tdt)
        sonic_tdt_pert = grid.TimeDepthTable.get_tvdss_twt_relation_from_vp(Vp, origin=z_start)

    else:
        raise NotImplementedError()

    return sonic_tdt_pert


def OLD_get_pertubed_tdt_from_vp(
    Vp: grid.Log,
    wp: grid.WellPath,
    tdt: grid.TimeDepthTable,
    p_pert_ratio: float = 0.02,
    t_pert_ratio: float = 0.01,
    max_degree: int = 5,
    N: int = 50,
) -> List[grid.TimeDepthTable]:
    """通过扰动 Vp 多项式与起始时间生成多组时深表（旧实现）。

    Parameters
    ----------
    Vp : wtie.processing.grid.Log
        MD 坐标下的纵波速度曲线，``values`` shape 为 ``(n_samples,)``。
    wp : wtie.processing.grid.WellPath
        井斜轨迹。
    tdt : wtie.processing.grid.TimeDepthTable
        参考时深表。
    p_pert_ratio : float, default=0.02
        速度多项式系数扰动比例。
    t_pert_ratio : float, default=0.01
        起始时间扰动比例。
    max_degree : int, default=5
        多项式最高阶数，实际阶数在 ``[1, max_degree]`` 内随机采样。
    N : int, default=50
        目标生成样本数。

    Returns
    -------
    list of wtie.processing.grid.TimeDepthTable
        扰动得到的时深表列表。

    Raises
    ------
    AssertionError
        当 ``Vp`` 不是 MD 坐标时触发。
    ValueError
        当连续尝试超过 ``4 * N`` 次仍无法生成足够有效样本时触发。
    """
    assert Vp.is_md

    tables = []

    i = 0
    i_ = 0
    while i < N:
        i_ += 1
        deg = np.random.randint(1, max_degree + 1)
        poly = np.polyfit(Vp.basis, Vp.values, deg)
        poly_pert = _perturbe_poly(poly, p_pert_ratio**deg)

        vl = _apply_poly(Vp.basis, poly)
        vl_p = _apply_poly(Vp.basis, poly_pert)
        Vp_p = grid.Log(Vp.values - vl + vl_p, Vp.basis, "md")

        t_start, z_error = grid.TimeDepthTable.get_tvdss_start_from_checkshots(Vp_p, wp, tdt)  # type: ignore
        t_start += random.uniform(-t_pert_ratio, t_pert_ratio) * t_start

        try:
            sonic_tdt_pert = grid.TimeDepthTable.get_tvdss_twt_relation_from_vp(Vp_p, wp, t_start=t_start)
            tables.append(sonic_tdt_pert)
            i += 1
        except:
            # unrealsitic perurbations
            if i_ > 4 * N:
                raise ValueError
            continue

    return tables


#####################
# OTHER
#####################


def compute_prestack_reflectivity(
    logs: grid.LogSet, theta_start: int, theta_end: int, delta_theta: int = 2
) -> grid.PreStackReflectivity:
    """由 Vp/Vs/rho 计算叠前角度反射系数。

    Parameters
    ----------
    logs : wtie.processing.grid.LogSet
        输入测井集合，需至少包含 ``Vp``、``Vs``、``rho``，且 ``basis`` 为 TWT。
        时间单位与项目约定一致，通常为 s。
    theta_start : int
        起始入射角（度）。
    theta_end : int
        终止入射角（度，包含端点）。
    delta_theta : int, default=2
        角度步长（度）。

    Returns
    -------
    wtie.processing.grid.PreStackReflectivity
        叠前反射系数对象，内部每个角度对应一条 1D 曲线，shape 为
        ``(n_samples - 1,)``。

    Raises
    ------
    AssertionError
        当 ``logs`` 不是 TWT 坐标时触发。
    """

    # verify basis
    assert logs.is_twt

    values = _reflection.prestack_rpp(logs.vp, logs.vs, logs.rho, theta_start, theta_end, delta_theta)  # type: ignore

    thetas = range(theta_start, theta_end + delta_theta, delta_theta)

    reflectivities = [grid.Reflectivity(values[i, :], logs.basis[1:], theta=thetas[i]) for i in range(values.shape[0])]

    return grid.PreStackReflectivity(reflectivities)  # type: ignore


def compute_acoustic_relfectiviy(logs: grid.LogSet) -> grid.Reflectivity:
    """由 Vp 与 rho 计算垂直入射（零偏移）声学反射系数。

    Parameters
    ----------
    logs : wtie.processing.grid.LogSet
        输入测井集合，需包含 ``Vp`` 与 ``rho``，且 ``basis`` 为 TWT。

    Returns
    -------
    wtie.processing.grid.Reflectivity
        声学反射系数曲线，``values`` shape 为 ``(n_samples - 1,)``。

    Raises
    ------
    AssertionError
        当 ``logs`` 不是 TWT 坐标时触发。
    """

    # verify basis
    assert logs.is_twt

    # R0
    reflectivity = _reflection.vertical_acoustic_reflectivity(logs.vp, logs.rho)

    return grid.Reflectivity(reflectivity, logs.basis[1:])


def convert_logs_from_md_to_tvdss(
    logset: grid.LogSet,
    trajectory: grid.WellPath,
    dz: float = None,  # type: ignore
    interpolation="linear",  # type: ignore
) -> grid.LogSet:
    """将 MD 坐标测井集合转换到 TVDSS 坐标。

    Parameters
    ----------
    logset : wtie.processing.grid.LogSet
        输入测井集合，``basis`` 必须为 MD。
    trajectory : wtie.processing.grid.WellPath
        井斜轨迹，用于 MD 到 TVDSS 的映射。
    dz : float or None, default=None
        目标深度采样间隔。单位与深度轴一致，按项目约定通常为 m。
        为 ``None`` 时使用下游默认行为。
    interpolation : str, default='linear'
        插值方法，传递至底层转换函数。

    Returns
    -------
    wtie.processing.grid.LogSet
        TVDSS 坐标下的测井集合。

    Raises
    ------
    AssertionError
        当 ``logset`` 不是 MD 坐标时触发。
    """
    # verify basis
    assert logset.is_md

    logs_tvd = grid._apply_trace_process_to_logset(
        grid._convert_log_from_md_to_tvdss, logset, trajectory, dz=dz, interpolation=interpolation
    )
    return logs_tvd


def convert_logs_from_md_to_twt(
    logset: grid.LogSet, table: grid.TimeDepthTable, trajectory: grid.WellPath, dt: float, interpolation="linear"
) -> grid.LogSet:
    """将 MD 坐标测井集合转换到 TWT 坐标。

    Parameters
    ----------
    logset : wtie.processing.grid.LogSet
        输入测井集合，``basis`` 必须为 MD。
    table : wtie.processing.grid.TimeDepthTable
        时深关系表（TVDSS-TWT）。
    trajectory : wtie.processing.grid.WellPath
        井斜轨迹。
    dt : float
        目标时间采样间隔。单位与 TWT 一致，按项目约定通常为 s。
    interpolation : str, default='linear'
        插值方法，传递至底层转换函数。

    Returns
    -------
    wtie.processing.grid.LogSet
        TWT 坐标下的测井集合。

    Raises
    ------
    AssertionError
        当 ``logset`` 不是 MD 坐标时触发。
    """
    # verify basis
    assert logset.is_md

    logs_twt = grid._apply_trace_process_to_logset(
        grid.convert_log_from_md_to_twt, logset, table, trajectory, dt, interpolation=interpolation
    )
    return logs_twt


def compute_synthetic_seismic(wavelet: grid.Wavelet, reflectivity: grid.Reflectivity) -> grid.Seismic:
    """基于子波与反射系数生成叠后合成地震道。

    Parameters
    ----------
    wavelet : wtie.processing.grid.Wavelet
        输入子波，``values`` shape 为 ``(n_samples,)``。
    reflectivity : wtie.processing.grid.Reflectivity
        输入反射系数，``basis`` 必须为 TWT，``values`` shape 为
        ``(n_samples,)``。

    Returns
    -------
    wtie.processing.grid.Seismic
        合成地震道，时间坐标继承自 ``reflectivity.basis``。

    Raises
    ------
    AssertionError
        当 ``reflectivity`` 不是 TWT 坐标，或
        ``reflectivity.sampling_rate`` 与 ``wavelet.sampling_rate``
        不一致（容差 ``1e-5``）时触发。
    """
    # verify basis
    assert reflectivity.is_twt
    assert np.allclose(reflectivity.sampling_rate, wavelet.sampling_rate, atol=1e-5)  # tolerance at 0.1 millisecond

    seismic = convolution_modeling(wavelet.values, reflectivity.values, noise=None, mode="same")  # type: ignore
    return grid.Seismic(seismic, reflectivity.basis, "twt")


def compute_synthetic_prestack_seismic(
    wavelet: grid.PreStackWavelet, reflectivity: grid.PreStackReflectivity
) -> grid.PreStackSeismic:
    """基于多角度子波与反射系数生成叠前合成地震道集。

    Parameters
    ----------
    wavelet : wtie.processing.grid.PreStackWavelet
        叠前子波集合，按角度索引。
    reflectivity : wtie.processing.grid.PreStackReflectivity
        叠前反射系数集合，按角度索引，时间坐标必须为 TWT。

    Returns
    -------
    wtie.processing.grid.PreStackSeismic
        叠前合成地震对象。每个角度对应一条 1D 地震道，shape 为
        ``(n_samples,)``。

    Raises
    ------
    AssertionError
        当 ``reflectivity`` 不是 TWT 坐标、采样率不一致（容差 ``1e-5``），
        或 ``wavelet.angles`` 与 ``reflectivity.angles`` 不一致时触发。
    """
    # verify basis
    assert reflectivity.is_twt
    assert np.allclose(reflectivity.sampling_rate, wavelet.sampling_rate, atol=1e-5)  # tolerance at 0.1 millisecond

    assert wavelet.angles == reflectivity.angles

    seismic = []
    for theta in wavelet.angles:
        values = convolution_modeling(wavelet[theta].values, reflectivity[theta].values, noise=None, mode="same")  # type: ignore

        seismic.append(grid.Seismic(values, reflectivity.basis, "twt", theta=theta))

    return grid.PreStackSeismic(seismic)  # type: ignore
