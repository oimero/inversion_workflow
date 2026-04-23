"""wtie.optimize.tie: 井震标定流程的高层编排入口。

本模块组织测井预处理、MD 到 TWT 转换、反射系数计算、子波估计与
合成地震生成等步骤，面向井震对齐任务提供可复用函数。

边界说明
--------
- 本模块不实现底层地球物理算子，具体计算由 logs、wavelet、similarity 等子模块负责。
- 本模块不负责训练流程管理、优化器策略配置或可视化渲染细节。

核心公开对象
------------
1. resample_seismic: 对叠后或叠前地震道做重采样。
2. filter_md_logs: 在测深（MD）域过滤测井曲线。
3. convert_logs_from_md_to_twt: 将 MD 测井校深后转换到双程时（TWT）域。
4. compute_reflectivity: 从 TWT 测井计算叠后或叠前反射系数。
5. compute_wavelet: 估计并可选缩放子波。
6. compute_synthetic_seismic: 由子波与反射系数生成合成地震。

Examples
--------
>>> # 假设 seismic/logset_md/wellpath/td_table/modeler/evaluator 已准备
>>> logset_twt = convert_logs_from_md_to_twt(logset_md, wellpath, td_table, dt=0.001)
>>> reflectivity = compute_reflectivity(logset_twt)
>>> wavelet = compute_wavelet(seismic, reflectivity, modeler, evaluator)
"""

from time import sleep

import numpy as np

from wtie import grid
from wtie.learning.model import BaseEvaluator
from wtie.modeling.modeling import ModelingCallable
from wtie.optimize import logs as _logs
from wtie.optimize import similarity as _similarity
from wtie.optimize import wavelet as _wavelet

# For accessibility from tie
from wtie.utils.types_ import FunctionType

VERY_FINE_DT: float = 0.0005
FINE_DT: float = 0.001


def resample_seismic(seismic: grid.seismic_t, dt: float):
    """对地震数据做 sinc 重采样。

    Parameters
    ----------
    seismic : grid.seismic_t
        输入地震数据（叠后或叠前）。shape 约定遵循 `grid`：
        常见为 `(n_samples,)` 或 `(n_traces, n_samples)`。
    dt : float
        目标采样间隔 ``dt``。单位与输入地震时间轴一致（通常为 s）。

    Returns
    -------
    grid.seismic_t
        重采样后的地震对象，语义与输入一致。

    Examples
    --------
    >>> seismic_rs = resample_seismic(seismic, dt=0.001)
    """
    return grid.resample_trace(seismic, dt)


def filter_md_logs(logset: grid.LogSet, **kwargs) -> grid.LogSet:
    """在测深（MD）域过滤测井曲线。

    Parameters
    ----------
    logset : grid.LogSet
        输入测井集合，必须位于 MD 域（``logset.is_md`` 为 ``True``）。
    **kwargs
        透传给 :func:`wtie.optimize.logs.filter_logs` 的过滤参数。

    Returns
    -------
    grid.LogSet
        过滤后的 MD 域测井集合。

    Raises
    ------
    AssertionError
        当 ``logset`` 不在 MD 域时触发。

    Examples
    --------
    >>> filtered = filter_md_logs(logset_md, median_size=7)
    """
    assert logset.is_md

    # log filtering
    filtered_logset = _logs.filter_logs(logset, **kwargs)

    return filtered_logset


def convert_logs_from_md_to_twt(
    logset: grid.LogSet, wellpath: grid.WellPath, table: grid.TimeDepthTable, dt: float
) -> grid.LogSet:
    """将 MD 域测井校深后转换到双程时（TWT）域并重采样。

    处理流程为：先用井轨迹 ``wellpath`` 与时深关系 ``table`` 完成 MD->TWT
    转换（内部先插值到更细采样），再按目标 ``dt`` 下采样。

    Parameters
    ----------
    logset : grid.LogSet
        输入测井集合，必须在 MD 域。
    wellpath : grid.WellPath
        井轨迹信息，用于深度校正。
    table : grid.TimeDepthTable
        时深关系表，用于 MD 到 TWT 的映射。
    dt : float
        输出 TWT 域采样间隔 ``dt``，单位通常为 s。

    Returns
    -------
    grid.LogSet
        TWT 域测井集合，采样间隔为 ``dt``。

    Raises
    ------
    AssertionError
        当 ``logset`` 不在 MD 域时触发。

    Examples
    --------
    >>> logset_twt = convert_logs_from_md_to_twt(logset_md, wellpath, td_table, dt=0.001)
    """
    assert logset.is_md, "Input logs must be in measured depth."

    logset_twt = _logs.convert_logs_from_md_to_twt(logset, table, wellpath, VERY_FINE_DT)
    logset_twt = grid.downsample_logset(logset_twt, new_sampling=dt)
    return logset_twt


def compute_reflectivity(logset: grid.LogSet, angle_range: tuple = None) -> grid.ref_t:  # type: ignore
    """根据测井计算反射系数（叠后或叠前）。

    当 ``angle_range`` 为空时，计算零偏移（垂直入射）反射系数；
    当 ``angle_range`` 提供时，计算角度相关的叠前反射系数。

    Parameters
    ----------
    logset : grid.LogSet
        输入测井集合，必须位于 TWT 域。
    angle_range : tuple, optional
        角度范围 ``(theta_start, theta_end, delta_theta)``。
        为 ``None`` 时执行叠后反射系数计算；否则执行叠前反射系数计算。

    Returns
    -------
    grid.ref_t
        反射系数对象。叠后结果名称被设为 ``"R0"``，叠前结果名称被设为 ``"Rpp"``。

    Raises
    ------
    AssertionError
        当 ``logset`` 不在 TWT 域时触发。

    Examples
    --------
    >>> r0 = compute_reflectivity(logset_twt)
    >>> rpp = compute_reflectivity(logset_twt, angle_range=(0, 30, 5))
    """
    assert logset.is_twt, "Input logs must be in two-way time."

    if angle_range is None:
        r = _logs.compute_acoustic_relfectiviy(logset)
        r.name = "R0"
    else:
        theta_start, theta_end, delta_theta = angle_range
        r = _logs.compute_prestack_reflectivity(logset, theta_start, theta_end, delta_theta=delta_theta)
        r.name = "Rpp"
    return r


def _matching_seismic_mask(seismic_basis: np.ndarray, reflectivity_basis: np.ndarray) -> np.ndarray:
    """Return seismic samples that are inside the reflectivity time support."""
    if reflectivity_basis.size < 2:
        raise ValueError("Reflectivity must contain at least two samples for interpolation.")

    eps = max(
        float(abs(seismic_basis[1] - seismic_basis[0])), float(abs(reflectivity_basis[1] - reflectivity_basis[0]))
    )
    eps *= 1e-6
    mask = (seismic_basis >= reflectivity_basis[0] - eps) & (seismic_basis <= reflectivity_basis[-1] + eps)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Seismic and reflectivity do not share enough TWT samples.")
    return mask


def _crop_seismic_to_mask(seismic: grid.Seismic, mask: np.ndarray) -> grid.Seismic:
    return grid.Seismic(
        seismic.values[mask],
        seismic.basis[mask],
        "twt",
        theta=seismic.theta,
        name=seismic.name,
    )


def _interpolate_reflectivity_to_basis(
    reflectivity: grid.Reflectivity,
    target_basis: np.ndarray,
) -> grid.Reflectivity:
    interp_basis = np.clip(target_basis, reflectivity.basis[0], reflectivity.basis[-1])
    values = np.interp(interp_basis, reflectivity.basis, reflectivity.values)
    return grid.Reflectivity(
        values,
        target_basis,
        theta=reflectivity.theta,
        name=reflectivity.name,
    )


def match_seismic_and_reflectivity(seismic: grid.seismic_t, reflectivity: grid.ref_t):
    """裁剪并对齐地震与反射系数的共同 TWT 区间。

    Parameters
    ----------
    seismic : grid.seismic_t
        输入地震数据。
    reflectivity : grid.ref_t
        输入反射系数数据。

    Returns
    -------
    tuple
        ``(seismic_matched, reflectivity_matched)``，两者位于共同定义的 TWT 区间，
        便于后续子波估计与正演。

    Examples
    --------
    >>> seis_m, ref_m = match_seismic_and_reflectivity(seismic, reflectivity)
    """
    assert seismic.basis_type == reflectivity.basis_type
    assert seismic.is_twt and reflectivity.is_twt
    assert np.allclose(seismic.sampling_rate, reflectivity.sampling_rate, rtol=1e-4)

    if seismic.is_prestack:
        assert reflectivity.is_prestack
        assert (seismic.angles == reflectivity.angles).all()  # type: ignore

        mask = _matching_seismic_mask(seismic.basis, reflectivity.basis)
        target_basis = seismic.basis[mask]

        seismics = []
        reflectivities = []
        for theta in seismic.angles:  # type: ignore
            seismics.append(_crop_seismic_to_mask(seismic[theta], mask))  # type: ignore
            reflectivities.append(_interpolate_reflectivity_to_basis(reflectivity[theta], target_basis))  # type: ignore

        return grid.PreStackSeismic(seismics, name=seismic.name), grid.PreStackReflectivity(  # type: ignore
            reflectivities,
            name=reflectivity.name,  # type: ignore
        )

    assert not reflectivity.is_prestack
    mask = _matching_seismic_mask(seismic.basis, reflectivity.basis)
    seismic_match = _crop_seismic_to_mask(seismic, mask)  # type: ignore
    reflectivity_match = _interpolate_reflectivity_to_basis(reflectivity, seismic_match.basis)  # type: ignore
    return seismic_match, reflectivity_match


def compute_wavelet(
    seismic: grid.seismic_t,
    reflectivity: grid.ref_t,
    modeler: ModelingCallable,
    wavelet_extractor: BaseEvaluator,
    similarity_measure: FunctionType = None,  # type: ignore
    zero_phasing: bool = False,
    scaling: bool = True,
    scaling_params: dict = None,  # type: ignore
    expected_value: bool = False,
    n_sampling: int = 60,
) -> grid.wlt_t:
    """估计子波并按需执行绝对子波幅值缩放。

    当 ``expected_value=True`` 时，调用评估器的期望子波接口；
    否则通过网格搜索在候选子波中选择最优子波。对叠前/叠后数据会自动选择对应实现。

    Parameters
    ----------
    seismic : grid.seismic_t
        输入地震数据，支持叠前与叠后。
    reflectivity : grid.ref_t
        与 ``seismic`` 对齐后的反射系数。
    modeler : ModelingCallable
        正演算子，用于由反射系数与子波生成合成记录。
    wavelet_extractor : BaseEvaluator
        子波提取/评估器。
    similarity_measure : FunctionType, optional
        相似度函数。默认使用
        :func:`wtie.optimize.similarity.normalized_xcorr_central_coeff`。
        相似度量纲通常为无量纲，范围由具体函数定义。
    zero_phasing : bool, default=False
        是否对估计子波执行零相位处理。
    scaling : bool, default=True
        是否执行绝对幅值缩放。
    scaling_params : dict, optional
        缩放参数字典。若 ``scaling=True``，至少应包含
        ``"wavelet_min_scale"`` 与 ``"wavelet_max_scale"``，可选 ``"num_iters"``。
    expected_value : bool, default=False
        是否使用期望子波路径（非网格搜索路径）。
    n_sampling : int, default=60
        网格搜索候选子波数量，仅在 ``expected_value=False`` 时使用。

    Returns
    -------
    grid.wlt_t
        估计后的子波对象（叠前返回叠前子波，叠后返回叠后子波）。

    Raises
    ------
    TypeError
        当 ``scaling=True`` 且 ``scaling_params`` 为 ``None`` 时，索引参数会触发类型错误。
    KeyError
        当 ``scaling=True`` 但 ``scaling_params`` 缺少必要键
        ``"wavelet_min_scale"`` 或 ``"wavelet_max_scale"`` 时触发。

    Examples
    --------
    >>> wavelet = compute_wavelet(
    ...     seismic, reflectivity, modeler, evaluator,
    ...     scaling=True,
    ...     scaling_params={"wavelet_min_scale": 0.5, "wavelet_max_scale": 1.5},
    ... )
    """
    if similarity_measure is None:
        similarity_measure = _similarity.normalized_xcorr_central_coeff

    # Compute wavelet
    if expected_value:
        if seismic.is_prestack:
            wavelet = _wavelet.compute_expected_prestack_wavelet(
                wavelet_extractor,
                seismic,  # type: ignore
                reflectivity,  # type: ignore
                zero_phasing=zero_phasing,  # type: ignore
            )
        else:
            wavelet = _wavelet.compute_expected_wavelet(
                wavelet_extractor,
                seismic,  # type: ignore
                reflectivity,  # type: ignore
                zero_phasing=zero_phasing,  # type: ignore
            )
    else:
        if seismic.is_prestack:
            _search_wlt = _wavelet.grid_search_best_prestack_wavelet
        else:
            _search_wlt = _wavelet.grid_search_best_wavelet

        wavelet = _search_wlt(
            wavelet_extractor,
            seismic,  # type: ignore
            reflectivity,  # type: ignore
            modeler,
            similarity_measure,
            zero_phasing=zero_phasing,
            num_wavelets=n_sampling,
        )

    # Find absolute scaling
    if scaling:
        _scale_wlt = _wavelet.scale_prestack_wavelet if seismic.is_prestack else _wavelet.scale_wavelet

        print("Find wavelet absolute scale")
        sleep(1.0)
        wavelet, _ = _scale_wlt(
            wavelet,  # type: ignore
            seismic,  # type: ignore
            reflectivity,  # type: ignore
            modeler,
            min_scale=scaling_params["wavelet_min_scale"],
            max_scale=scaling_params["wavelet_max_scale"],
            num_iters=scaling_params.get("num_iters", 70),
        )
    return wavelet


def compute_synthetic_seismic(
    modeler: ModelingCallable, wavelet: grid.wlt_t, reflectivity: grid.ref_t
) -> grid.seismic_t:
    """根据子波与反射系数生成合成地震。

    若输入为叠前子波，则调用叠前正演路径；若输入为叠后子波，
    则调用叠后正演路径。

    Parameters
    ----------
    modeler : ModelingCallable
        正演算子。
    wavelet : grid.wlt_t
        子波对象。仅支持 ``grid.PreStackWavelet`` 或 ``grid.Wavelet``。
    reflectivity : grid.ref_t
        反射系数对象。

    Returns
    -------
    grid.seismic_t
        合成地震对象，叠前/叠后形态与 ``wavelet`` 类型一致。

    Raises
    ------
    TypeError
        当 ``wavelet`` 不是 ``grid.PreStackWavelet`` 或 ``grid.Wavelet`` 时触发。

    Examples
    --------
    >>> synthetic = compute_synthetic_seismic(modeler, wavelet, reflectivity)
    """
    if type(wavelet) is grid.PreStackWavelet:
        return _wavelet.compute_synthetic_prestack_seismic(modeler, wavelet, reflectivity)  # type: ignore
    elif type(wavelet) is grid.Wavelet:
        return _wavelet.compute_synthetic_seismic(modeler, wavelet, reflectivity)  # type: ignore
    else:
        raise TypeError
