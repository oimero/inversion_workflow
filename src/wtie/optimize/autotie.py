"""wtie.optimize.autotie: 自动井震标定（auto tie）流程配方。

本模块在 :mod:`wtie.optimize.tie` 的基础算子之上，提供 version 1
自动井震标定流程：贝叶斯参数搜索、中间标定、可选拉伸压缩（stretch and squeeze）
与最终子波缩放。

边界说明
--------
- 本模块不实现底层反射系数计算、子波提取或正演算子，这些能力由 tie/wavelet/logs 子模块提供。
- 本模块不负责输入数据质控与可视化展示，仅负责流程编排与参数搜索。

核心公开对象
------------
1. tie_v1: 自动井震标定主流程（v1）。
2. stretch_and_squeeze: 基于动态时移更新时深表并重新标定。
3. get_default_search_space_v1: v1 默认参数搜索空间。

Examples
--------
>>> search_space = get_default_search_space_v1()
>>> outputs = tie_v1(
...     inputs, wavelet_extractor, modeler, wavelet_scaling_params,
...     search_space=search_space,
... )
"""

from time import sleep

from tqdm import tqdm

from wtie.learning.model import BaseEvaluator
from wtie.modeling.modeling import ModelingCallable
from wtie.optimize import optimizer as _optimizer
from wtie.optimize import similarity as _similarity
from wtie.optimize import tie as _tie
from wtie.optimize import warping as _warping
from wtie.processing import grid
from wtie.utils.datasets.utils import InputSet, OutputSet

# Some constants affecting the workflow
# when both values are True, results are a bit less good but final wavelets
# have smaller absolute phase.
# when both values are False, results are better, but wavelets can have strong
# positive or negative phase.
# when INTERMEDIATE_EXPECTED_VALUE is False, prestack auto-tie can take twice as long.
INTERMEDIATE_ZERO_PHASING: bool = True
INTERMEDIATE_EXPECTED_VALUE: bool = True


def stretch_and_squeeze(
    inputs: InputSet,
    current_outputs: OutputSet,
    wavelet_extractor: BaseEvaluator,
    modeler: ModelingCallable,
    wavelet_scaling_params: dict,
    best_params: dict,
    stretch_and_squeeze_params: dict,
):
    """执行一次拉伸压缩校正并更新标定结果。

    该函数先基于真实地震与当前合成地震计算动态时移（lag），再用时移修正
    时深关系表并重新执行中间标定，最后重新估计最终子波与合成地震。

    Parameters
    ----------
    inputs : InputSet
        自动井震标定输入集合，需包含 ``seismic``、``logset_md``、``wellpath`` 等字段。
    current_outputs : OutputSet
        当前标定结果，需包含 ``seismic``、``synth_seismic`` 与 ``table``。
    wavelet_extractor : BaseEvaluator
        子波提取/评估器。
    modeler : ModelingCallable
        正演算子。
    wavelet_scaling_params : dict
        最终子波绝对幅值缩放参数，需与 :func:`wtie.optimize.tie.compute_wavelet`
        的 ``scaling_params`` 约定一致。
    best_params : dict
        中间标定阶段使用的最优参数集合。
    stretch_and_squeeze_params : dict
        动态时移参数，透传给 :func:`wtie.optimize.warping.compute_dynamic_lag`。
        可选键 ``reference_angle``（叠前时使用）会被弹出（pop）。

    Returns
    -------
    OutputSet
        更新后的标定结果，额外写入 ``dlags``、``xcorr`` 与 ``dxcorr``。
        ``xcorr`` 为统一到 0.001 s 采样的相关曲线；叠前场景下 ``dxcorr`` 为 ``None``。

    Raises
    ------
    KeyError
        当 ``best_params`` 或 ``wavelet_scaling_params`` 缺少下游流程所需键时触发。

    Notes
    -----
    该函数会原位修改 ``stretch_and_squeeze_params``：若存在 ``reference_angle`` 键，
    会在函数内部被删除。
    """

    from_seismic = current_outputs.seismic
    to_seismic = current_outputs.synth_seismic

    if inputs.seismic.is_prestack:
        first_angle = from_seismic.angles[0]  # type: ignore
        ref_angle = stretch_and_squeeze_params.get("reference_angle", first_angle)
        stretch_and_squeeze_params.pop("reference_angle", None)
        from_seismic = from_seismic[ref_angle]  # type: ignore
        to_seismic = to_seismic[ref_angle]  # type: ignore

    dlags = _warping.compute_dynamic_lag(from_seismic, to_seismic, **stretch_and_squeeze_params)  # type: ignore

    warped_table = _warping.apply_lags_to_table(current_outputs.table, dlags)

    outputs = _intermediate_tie_v1(
        inputs.logset_md,
        inputs.wellpath,
        warped_table,
        inputs.seismic,  # type: ignore
        wavelet_extractor,
        modeler,
        best_params,  # type: ignore
    )

    outputs.dlags = dlags

    # Final wavelet
    wavelet = _tie.compute_wavelet(
        outputs.seismic,
        outputs.r,
        modeler,
        wavelet_extractor,
        zero_phasing=False,
        scaling=True,
        expected_value=False,
        scaling_params=wavelet_scaling_params,
    )
    # final synth
    synth_seismic = _tie.compute_synthetic_seismic(modeler, wavelet, outputs.r)

    # overwrite w/ new data
    outputs.wavelet = wavelet
    outputs.synth_seismic = synth_seismic

    # similarity
    if not inputs.seismic.is_prestack:
        xcorr = _similarity.traces_normalized_xcorr(outputs.seismic, outputs.synth_seismic)  # type: ignore
        xcorr = grid.resample_trace(xcorr, 0.001)
        dxcorr = _similarity.dynamic_normalized_xcorr(outputs.seismic, outputs.synth_seismic)  # type: ignore
    else:
        xcorr = _similarity.prestack_traces_normalized_xcorr(outputs.seismic, outputs.synth_seismic)  # type: ignore
        xcorr = grid.resample_trace(xcorr, 0.001)
        dxcorr = None

    outputs.xcorr = xcorr  # type: ignore
    outputs.dxcorr = dxcorr  # type: ignore

    return outputs


def tie_v1(
    inputs: InputSet,
    wavelet_extractor: BaseEvaluator,
    modeler: ModelingCallable,
    wavelet_scaling_params: dict,
    search_params: dict = None,  # type: ignore
    search_space: dict = None,  # type: ignore
    stretch_and_squeeze_params: dict = None,  # type: ignore
) -> OutputSet:
    """执行自动井震标定主流程（v1）。

    v1 流程包括：
    1) 贝叶斯搜索日志滤波与时深表平移参数；
    2) 生成中间标定结果；
    3) 可选 stretch and squeeze 动态时移校正；
    4) 重新估计并缩放最终子波，输出合成地震与相似度指标。

    Parameters
    ----------
    inputs : InputSet
        井震标定输入集合。
    wavelet_extractor : BaseEvaluator
        子波提取/评估器。
    modeler : ModelingCallable
        正演算子。
    wavelet_scaling_params : dict
        最终子波绝对幅值缩放参数。常用键包括
        ``"wavelet_min_scale"``、``"wavelet_max_scale"``，可选 ``"num_iters"``。
    search_params : dict, optional
        贝叶斯搜索控制参数。可选键：
        ``"num_iters"``（默认 80）、``"similarity_std"``（默认 0.01）、
        ``"random_ratio"``（默认 0.6，范围建议 [0, 1]）。
    search_space : dict, optional
        搜索空间定义。为 ``None`` 时使用 :func:`get_default_search_space_v1`。
    stretch_and_squeeze_params : dict, optional
        若提供，则启用动态时移校正。常用键包括 ``"window_length"``、``"max_lag"``
        （单位通常为 s）；叠前可额外提供 ``"reference_angle"``。

    Returns
    -------
    OutputSet
        自动井震标定输出，包含最终子波、合成地震、相关指标，以及搜索得到的
        ``ax_client`` 与可选 ``dlags``。

    Raises
    ------
    KeyError
        当输入参数字典缺少下游流程必要键时触发。

    Examples
    --------
    >>> outputs = tie_v1(
    ...     inputs,
    ...     wavelet_extractor,
    ...     modeler,
    ...     wavelet_scaling_params={"wavelet_min_scale": 0.5, "wavelet_max_scale": 1.5},
    ... )
    """

    if search_space is None:
        search_space = get_default_search_space_v1()

    # Search
    # Optimize for best parameters
    # at this point, the wavelet is zero-phased
    if search_params is None:
        search_params = {}
    num_iters = search_params.get("num_iters", 80)
    similarity_std = search_params.get("similarity_std", 0.01)
    random_ratio = search_params.get("random_ratio", 0.6)

    ax_client = _search_best_params_v1(
        inputs, wavelet_extractor, modeler, search_space, num_iters, random_ratio, similarity_std
    )
    best_params = ax_client.get_best_parameters()[0]  # type: ignore

    # Intermediate tie
    shifted_table = grid.TimeDepthTable.t_bulk_shift(inputs.table, best_params["table_t_shift"])  # type: ignore

    outputs_tmp1 = _intermediate_tie_v1(
        inputs.logset_md,
        inputs.wellpath,
        shifted_table,
        inputs.seismic,  # type: ignore
        wavelet_extractor,
        modeler,
        best_params,  # type: ignore
    )

    # Optional stretch and squeeze
    if stretch_and_squeeze_params is not None:
        from_seismic = outputs_tmp1.seismic
        to_seismic = outputs_tmp1.synth_seismic
        if outputs_tmp1.seismic.is_prestack:
            first_angle = from_seismic.angles[0]  # type: ignore
            ref_angle = stretch_and_squeeze_params.get("reference_angle", first_angle)
            stretch_and_squeeze_params.pop("reference_angle", None)
            from_seismic = from_seismic[ref_angle]  # type: ignore
            to_seismic = to_seismic[ref_angle]  # type: ignore

        dlags = _warping.compute_dynamic_lag(from_seismic, to_seismic, **stretch_and_squeeze_params)  # type: ignore

        warped_table = _warping.apply_lags_to_table(outputs_tmp1.table, dlags)

        outputs_tmp2 = _intermediate_tie_v1(
            inputs.logset_md,
            inputs.wellpath,
            warped_table,
            inputs.seismic,  # type: ignore
            wavelet_extractor,
            modeler,
            best_params,  # type: ignore
        )

        outputs_tmp2.dlags = dlags

    else:
        outputs_tmp2 = outputs_tmp1

    # Final wavelet
    wavelet = _tie.compute_wavelet(
        outputs_tmp2.seismic,
        outputs_tmp2.r,
        modeler,
        wavelet_extractor,
        zero_phasing=False,
        scaling=True,
        expected_value=False,
        scaling_params=wavelet_scaling_params,
    )

    # Final synthetic
    synth_seismic = _tie.compute_synthetic_seismic(modeler, wavelet, outputs_tmp2.r)

    # overwrite w/ new data
    outputs_tmp2.ax_client = ax_client
    outputs_tmp2.wavelet = wavelet
    outputs_tmp2.synth_seismic = synth_seismic

    # Similarity between synthetic and real seismic
    if not inputs.seismic.is_prestack:
        xcorr = _similarity.traces_normalized_xcorr(outputs_tmp2.seismic, outputs_tmp2.synth_seismic)  # type: ignore
        xcorr = grid.resample_trace(xcorr, 0.001)
        dxcorr = _similarity.dynamic_normalized_xcorr(outputs_tmp2.seismic, outputs_tmp2.synth_seismic)  # type: ignore
    else:
        xcorr = _similarity.prestack_traces_normalized_xcorr(outputs_tmp2.seismic, outputs_tmp2.synth_seismic)  # type: ignore
        xcorr = grid.resample_trace(xcorr, 0.001)
        dxcorr = None

    outputs_tmp2.xcorr = xcorr  # type: ignore
    outputs_tmp2.dxcorr = dxcorr  # type: ignore

    return outputs_tmp2


def _intermediate_tie_v1(
    logset_md: grid.LogSet,
    wellpath: grid.WellPath,
    table: grid.TimeDepthTable,
    seismic: grid.Seismic,
    wavelet_extractor: BaseEvaluator,
    modeler: ModelingCallable,
    parameters: dict,
) -> OutputSet:
    """执行 v1 的中间标定步骤并返回中间结果。

    中间步骤包括：地震重采样、公共预处理（日志过滤/时深转换/反射系数计算/匹配）、
    估计中间子波（可设零相位）并生成中间合成地震。

    Parameters
    ----------
    logset_md : grid.LogSet
        MD 域测井集合。
    wellpath : grid.WellPath
        井轨迹。
    table : grid.TimeDepthTable
        当前使用的时深关系表。
    seismic : grid.Seismic
        输入地震（叠后）。
    wavelet_extractor : BaseEvaluator
        子波提取/评估器。
    modeler : ModelingCallable
        正演算子。
    parameters : dict
        公共预处理参数，至少包含 ``logs_median_size``、``logs_median_threshold``、``logs_std``。

    Returns
    -------
    OutputSet
        中间输出集合，包含中间子波、匹配地震、合成地震、时深表和反射系数等。
    """
    # Resampling
    seismic = _tie.resample_seismic(seismic, wavelet_extractor.expected_sampling)  # type: ignore

    # Common steps
    logset_twt, seis_match, r_match = _common_steps_tie_v1(
        logset_md, wellpath, table, seismic, wavelet_extractor, modeler, parameters
    )

    # (Zero-phased) unscaled wavelet
    wlt = _tie.compute_wavelet(
        seis_match,  # type: ignore
        r_match,
        modeler,
        wavelet_extractor,
        zero_phasing=INTERMEDIATE_ZERO_PHASING,
        scaling=False,
        expected_value=False,
    )

    synth_seismic = _tie.compute_synthetic_seismic(modeler, wlt, r_match)

    return OutputSet(wlt, logset_twt, seis_match, synth_seismic, table, r_match, wellpath)  # type: ignore


def _search_best_params_v1(
    inputs: InputSet,
    wavelet_extractor: BaseEvaluator,
    modeler: ModelingCallable,
    search_space: dict,
    num_iters: int,
    random_ratio: float,
    similarity_std: float,
) -> _optimizer.AxClient:
    """通过 Ax 贝叶斯优化搜索 v1 的最优参数。

    优化目标为 ``goodness_of_match`` 最大化。每次试验会执行公共预处理、
    中间子波估计与合成，并以中心互相关系数作为评分。

    Parameters
    ----------
    inputs : InputSet
        自动井震标定输入集合。
    wavelet_extractor : BaseEvaluator
        子波提取/评估器。
    modeler : ModelingCallable
        正演算子。
    search_space : dict
        Ax 参数空间定义列表。
    num_iters : int
        搜索迭代次数 ``n``。
    random_ratio : float
        随机探索比例，通常取值范围 [0, 1]。
    similarity_std : float
        评分不确定度估计（无量纲）。

    Returns
    -------
    _optimizer.AxClient
        已完成试验记录的 AxClient，可用于获取最优参数。

    Notes
    -----
    若 ``ax_client.get_next_trial`` 在迭代中抛出 ``RuntimeError``，
    函数会提前停止并返回当前结果。
    """
    # Resampling
    seismic = _tie.resample_seismic(inputs.seismic, wavelet_extractor.expected_sampling)

    # Optim client
    ax_client = _optimizer.create_ax_client(num_iters, random_ratio=random_ratio)

    from ax.service.ax_client import ObjectiveProperties

    ax_client.create_experiment(
        name="auto well tie",
        parameters=search_space,  # type: ignore
        objectives={"goodness_of_match": ObjectiveProperties(minimize=False)},
        choose_generation_strategy_kwargs=None,
    )

    # Optimization
    print("Search for optimal parameters")
    sleep(1.0)
    for i in tqdm(range(num_iters)):
        try:
            h_params, trial_index = ax_client.get_next_trial()
        except RuntimeError:  # NotPSDError:
            print("Early stopping after %d/%d iterations." % (i + 1, num_iters))
            break

        # table
        current_table = grid.TimeDepthTable.t_bulk_shift(inputs.table, h_params["table_t_shift"])
        # common steps
        logset_twt, seis_match, r_match = _common_steps_tie_v1(
            inputs.logset_md,
            inputs.wellpath,
            current_table,
            seismic,  # type: ignore
            wavelet_extractor,
            modeler,
            h_params,  # type: ignore
        )

        # (zero-phased) unscaled wavelet
        current_wlt = _tie.compute_wavelet(
            seis_match,  # type: ignore
            r_match,
            modeler,
            wavelet_extractor,
            zero_phasing=INTERMEDIATE_ZERO_PHASING,
            scaling=False,
            expected_value=INTERMEDIATE_EXPECTED_VALUE,
        )

        # synthetic seismic
        synth_seismic = _tie.compute_synthetic_seismic(modeler, current_wlt, r_match)

        # similarity
        current_score = _similarity.central_xcorr_coeff(
            _tie.resample_seismic(seis_match, _tie.FINE_DT),  # type: ignore
            _tie.resample_seismic(synth_seismic, _tie.FINE_DT),  # type: ignore
        )

        ax_client.complete_trial(trial_index=trial_index, raw_data=(current_score, similarity_std))

    return ax_client


def _common_steps_tie_v1(
    logset_md: grid.LogSet,
    wellpath: grid.WellPath,
    table: grid.TimeDepthTable,
    seismic: grid.seismic_t,
    wavelet_extractor: BaseEvaluator,
    modeler: ModelingCallable,
    params: dict,
):
    """执行 v1 在搜索与中间标定中复用的公共步骤。

    公共步骤顺序为：
    1) MD 域日志过滤；
    2) MD 到 TWT 转换并按提取器采样间隔重采样；
    3) 计算反射系数（叠后或叠前取决于 ``seismic.angle_range``）；
    4) 地震与反射系数时窗匹配。

    Parameters
    ----------
    logset_md : grid.LogSet
        MD 域测井集合。
    wellpath : grid.WellPath
        井轨迹。
    table : grid.TimeDepthTable
        时深关系表。
    seismic : grid.seismic_t
        输入地震，可为叠后或叠前。
    wavelet_extractor : BaseEvaluator
        子波提取/评估器，其 ``expected_sampling`` 用于目标采样间隔 ``dt``。
    modeler : ModelingCallable
        正演算子（当前函数不直接调用，保留用于统一调用接口）。
    params : dict
        过滤参数字典，需包含 ``logs_median_size``、``logs_median_threshold``、``logs_std``。

    Returns
    -------
    tuple
        ``(logset_twt, seis_match, r0_match)``：
        TWT 域测井、匹配后的地震、匹配后的反射系数。

    Raises
    ------
    KeyError
        当 ``params`` 缺少必要键时触发。
    """

    # log filtering
    logset_md = _tie.filter_md_logs(
        logset_md,
        median_size=params["logs_median_size"],
        threshold=params["logs_median_threshold"],
        std=params["logs_std"],
        std2=0.8 * params["logs_std"],
    )

    # convertion
    logset_twt = _tie.convert_logs_from_md_to_twt(logset_md, wellpath, table, wavelet_extractor.expected_sampling)

    # reflectivity
    r0 = _tie.compute_reflectivity(logset_twt, angle_range=seismic.angle_range)  # type: ignore

    # matching
    seis_match, r0_match = _tie.match_seismic_and_reflectivity(seismic, r0)  # type: ignore

    return logset_twt, seis_match, r0_match


#####################
# Default params
#####################
def get_default_search_space_v1():
    """返回 v1 默认贝叶斯搜索空间定义。

    搜索空间包含 4 个参数：
    - ``logs_median_size``: 中值滤波窗口长度（采样点数 ``n``）。
    - ``logs_median_threshold``: 中值滤波阈值（相对日志标准差，无量纲）。
    - ``logs_std``: 高斯平滑标准差（采样点域）。
    - ``table_t_shift``: 时深关系表整体平移量（s）。

    Returns
    -------
    list of dict
        兼容 Ax 的参数定义列表，可直接用于 ``AxClient.create_experiment``。

    Examples
    --------
    >>> search_space = get_default_search_space_v1()
    >>> isinstance(search_space, list)
    True
    """
    median_length_choice = dict(
        name="logs_median_size", type="choice", values=[i for i in range(11, 73, 2)], value_type="int"
    )

    median_th_choice = dict(name="logs_median_threshold", type="range", bounds=[0.1, 5.5], value_type="float")

    std_choice = dict(name="logs_std", type="range", bounds=[0.5, 6.5], value_type="float")

    table_t_shift_choice = dict(name="table_t_shift", type="range", bounds=[-0.012, 0.012], value_type="float")

    search_space = [
        median_length_choice,
        median_th_choice,
        std_choice,
        table_t_shift_choice,
    ]
    return search_space
