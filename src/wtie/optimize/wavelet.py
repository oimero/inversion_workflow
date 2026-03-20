"""wtie.optimize.wavelet: 子波估计、筛选与幅值缩放工具集。

本模块提供叠后/叠前子波的期望估计、候选网格搜索、合成地震生成与
基于黑盒优化（Ax）的绝对幅值缩放流程。

边界说明
--------
- 本模块不负责井震标定流程编排，流程级调用由 wtie.optimize.tie/autotie 负责。
- 本模块不负责神经网络训练，仅调用外部 evaluator 的推理接口。

核心公开对象
------------
1. compute_expected_wavelet: 估计叠后期望子波并可计算谱不确定性。
2. grid_search_best_wavelet: 在采样子波集合中搜索最优叠后子波。
3. scale_wavelet: 通过 Ax 估计叠后子波绝对幅值缩放系数。
4. compute_expected_prestack_wavelet: 分角度估计叠前期望子波。
5. grid_search_best_prestack_wavelet: 分角度搜索叠前最优子波。
6. scale_prestack_wavelet: 分角度缩放叠前子波。

Examples
--------
>>> wlt = compute_expected_wavelet(evaluator, seismic, reflectivity)
>>> scaled_wlt, ax_client = scale_wavelet(wlt, seismic, reflectivity, modeler)
>>> synth = compute_synthetic_seismic(modeler, scaled_wlt, reflectivity)
"""

import numpy as np
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models as ax_Models
from ax.service.ax_client import AxClient, ObjectiveProperties
from tqdm import tqdm

from wtie.learning.model import BaseEvaluator
from wtie.modeling.modeling import ModelingCallable
from wtie.optimize import similarity as _similarity
from wtie.processing import grid
from wtie.processing.spectral import compute_spectrum as _compute_spectrum
from wtie.processing.spectral import zero_phasing as _zero_phasing
from wtie.utils.types_ import FunctionType, List


def _preprocess_real_seismic(seismic: grid.Seismic, inverse_polarity: bool = False) -> np.ndarray:
    """将叠后真实地震归一化并整理为网络输入张量。

    Parameters
    ----------
    seismic : grid.Seismic
        叠后地震对象，原始振幅 shape 为 ``(n_samples,)``。
    inverse_polarity : bool, default=False
        是否执行极性反转。为 ``True`` 时输出整体乘以 ``-1``。

    Returns
    -------
    numpy.ndarray
        网络输入张量，shape 为 ``(1, 1, n_samples)``，dtype 为 ``float32``。
    """
    # norm
    abs_max = max(seismic.values.max(), -seismic.values.min())
    new_seismic = seismic.values / abs_max
    if inverse_polarity:
        new_seismic *= -1.0
    return _prepare_for_input_to_network(new_seismic)


def _preprocess_reflectivity(reflectivity: grid.Reflectivity) -> np.ndarray:
    """将叠后反射系数归一化并整理为网络输入张量。

    Parameters
    ----------
    reflectivity : grid.Reflectivity
        叠后反射系数对象，shape 为 ``(n_samples,)``。

    Returns
    -------
    numpy.ndarray
        网络输入张量，shape 为 ``(1, 1, n_samples)``，dtype 为 ``float32``。
    """
    # norm
    abs_max = max(reflectivity.values.max(), -reflectivity.values.min())
    new_ref = reflectivity.values / abs_max
    return _prepare_for_input_to_network(new_ref)


def _prepare_for_input_to_network(x: np.ndarray) -> np.ndarray:
    """将一维序列升维为网络输入格式。

    Parameters
    ----------
    x : numpy.ndarray
        输入一维数组，shape 为 ``(n_samples,)``。

    Returns
    -------
    numpy.ndarray
        输出数组，shape 为 ``(1, 1, n_samples)``，dtype 为 ``float32``。
    """
    # tensor shape
    x = x[np.newaxis, :]  # batch
    x = x[np.newaxis, :]  # channels
    return x.astype(np.float32)


def _get_wavelet_object(wavelet: np.ndarray, dt: float, name: str = None) -> grid.Wavelet:  # type: ignore
    """把子波样值数组封装为 ``grid.Wavelet`` 对象。

    Parameters
    ----------
    wavelet : numpy.ndarray
        子波振幅序列，shape 为 ``(n_samples,)``。
    dt : float
        采样间隔 ``dt``，单位通常为 s。
    name : str, optional
        子波名称。

    Returns
    -------
    grid.Wavelet
        子波对象，时间基准由 ``[-duration/2, duration/2)`` 等间隔构造。
    """
    duration = wavelet.size * dt
    # t = np.arange(-duration/2, (duration-dt)/2, dt)
    t = np.arange(-duration / 2, duration / 2, dt)
    return grid.Wavelet(wavelet, t, name=name)


def zero_phasing_wavelet(wavelet: grid.Wavelet) -> grid.Wavelet:
    """对输入子波执行零相位处理。

    Parameters
    ----------
    wavelet : grid.Wavelet
        输入叠后子波。

    Returns
    -------
    grid.Wavelet
        零相位子波，保留原 ``basis`` 与 ``name``。
    """
    wlt_0 = _zero_phasing(wavelet.values)
    return grid.Wavelet(wlt_0, wavelet.basis, name=wavelet.name)


def get_phase(wavelet: grid.Wavelet, to_degree: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """计算子波相位谱。

    Parameters
    ----------
    wavelet : grid.Wavelet
        输入叠后子波。
    to_degree : bool, default=True
        是否将相位单位转换为角度（degree）。

    Returns
    -------
    tuple of numpy.ndarray
        ``(ff, phase)``：频率轴（Hz）与相位谱。
    """
    ff, ampl, power, phase = _compute_spectrum(wavelet.values, wavelet.sampling_rate, to_degree=to_degree)
    return ff, phase


def get_spectrum(
    wavelet: grid.Wavelet, to_degree: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算子波频谱特征。

    Parameters
    ----------
    wavelet : grid.Wavelet
        输入叠后子波。
    to_degree : bool, default=True
        是否将相位单位转换为角度（degree）。

    Returns
    -------
    tuple of numpy.ndarray
        ``(ff, ampl, power, phase)``：频率轴（Hz）、振幅谱、功率谱与相位谱。
    """
    ff, ampl, power, phase = _compute_spectrum(wavelet.values, wavelet.sampling_rate, to_degree=to_degree)
    return ff, ampl, power, phase


def compute_expected_wavelet(
    evaluator: BaseEvaluator,
    seismic: grid.Seismic,
    reflectivity: grid.Reflectivity,
    n_sampling: int = 50,
    inverse_polarity: bool = False,
    zero_phasing: bool = False,
) -> grid.Wavelet:
    """估计叠后期望子波，并可附带频谱不确定性统计。

    当 ``zero_phasing=False`` 时，会额外采样 ``n_sampling`` 个子波，计算
    振幅谱与相位谱均值/标准差并写入 ``Wavelet.uncertainties``。

    Parameters
    ----------
    evaluator : BaseEvaluator
        子波评估器，需支持 ``expected_wavelet`` 与 ``sample_n_times``。
    seismic : grid.Seismic
        叠后地震，shape 为 ``(n_samples,)``。
    reflectivity : grid.Reflectivity
        叠后反射系数，shape 为 ``(n_samples,)``。
    n_sampling : int, default=50
        估计不确定性时的采样次数 ``n``。
    inverse_polarity : bool, default=False
        是否对输入/输出执行极性反转。
    zero_phasing : bool, default=False
        是否强制输出零相位子波。

    Returns
    -------
    grid.Wavelet
        期望子波对象；当 ``zero_phasing=False`` 时包含谱不确定性。

    Raises
    ------
    AssertionError
        当 ``seismic`` 与 ``reflectivity`` 时间基准不一致，或
        ``seismic.sampling_rate`` 与 ``evaluator.expected_sampling`` 不一致时触发。

    Examples
    --------
    >>> wlt = compute_expected_wavelet(evaluator, seismic, reflectivity, n_sampling=80)
    """
    assert np.allclose(seismic.basis, reflectivity.basis, rtol=1e-3)
    assert np.allclose(seismic.sampling_rate, evaluator.expected_sampling, rtol=1e-3)

    seismic_ = _preprocess_real_seismic(seismic, inverse_polarity=inverse_polarity)
    ref_ = _preprocess_reflectivity(reflectivity)

    # Expected wavelet
    expected_wlt = evaluator.expected_wavelet(seismic=seismic_, reflectivity=ref_, squeeze=True)  # type: ignore
    if inverse_polarity:
        expected_wlt *= -1.0

    expected_wlt = _get_wavelet_object(expected_wlt, seismic.sampling_rate)

    if zero_phasing:
        expected_wlt = zero_phasing_wavelet(expected_wlt)
    else:
        # Uncertainties
        wavelets = evaluator.sample_n_times(seismic=seismic_, reflectivity=ref_, n=n_sampling)  # type: ignore
        amp_spectrums = []
        phase_spectrums = []
        for wlt in wavelets:
            if inverse_polarity:
                wlt *= -1.0
            wlt = _get_wavelet_object(wlt, seismic.sampling_rate)
            ff, ampl, power, phase = get_spectrum(wlt, to_degree=True)
            amp_spectrums.append(ampl)
            phase_spectrums.append(phase)
        amp_spectrums = np.stack(amp_spectrums, axis=0)
        phase_spectrums = np.stack(phase_spectrums, axis=0)
        uncertainties = grid.WaveletUncertainties(
            ff,
            np.mean(amp_spectrums, axis=0),
            np.std(amp_spectrums, axis=0),
            np.mean(phase_spectrums, axis=0),
            np.std(phase_spectrums, axis=0),
        )
        expected_wlt.uncertainties = uncertainties
    return expected_wlt


def compute_expected_prestack_wavelet(
    evaluator: BaseEvaluator,
    seismic: grid.PreStackSeismic,
    reflectivity: grid.PreStackReflectivity,
    zero_phasing: bool = False,
    inverse_polarity: bool = False,
) -> grid.PreStackWavelet:
    """分角度估计叠前期望子波。

    Parameters
    ----------
    evaluator : BaseEvaluator
        子波评估器。
    seismic : grid.PreStackSeismic
        叠前地震，常见 shape 为 ``(n_traces, n_samples)``。
    reflectivity : grid.PreStackReflectivity
        叠前反射系数，shape 与角度集合需与 ``seismic`` 一致。
    zero_phasing : bool, default=False
        是否对每个角度子波执行零相位处理。
    inverse_polarity : bool, default=False
        是否执行极性反转。

    Returns
    -------
    grid.PreStackWavelet
        叠前子波集合，每个子波 ``theta`` 与输入角度对应。

    Raises
    ------
    AssertionError
        当 ``seismic.angles`` 与 ``reflectivity.angles`` 不一致时触发。
    """
    assert (seismic.angles == reflectivity.angles).all()

    wavelets = []
    for theta in seismic.angles:
        wlt_ = compute_expected_wavelet(
            evaluator,
            seismic[theta],  # type: ignore
            reflectivity[theta],  # type: ignore
            zero_phasing=zero_phasing,
            inverse_polarity=inverse_polarity,
        )
        wlt_.theta = theta
        wavelets.append(wlt_)
    return grid.PreStackWavelet(wavelets)  # type: ignore


def grid_search_best_wavelet(
    evaluator: BaseEvaluator,
    seismic: grid.Seismic,
    reflectivity: grid.Reflectivity,
    modeler: ModelingCallable,
    similarity: FunctionType,
    num_wavelets: int = 60,
    inverse_polarity: bool = False,
    zero_phasing: bool = False,
) -> grid.Wavelet:
    """在候选子波样本中网格搜索叠后最优子波。

    评分策略为：
    1) 先按相似度得分降序；
    2) 取前 10% 候选；
    3) 若 ``zero_phasing=False``，再按 0-60 Hz 区间平均绝对相位最小化筛选。

    Parameters
    ----------
    evaluator : BaseEvaluator
        子波评估器，需支持 ``sample_n_times``。
    seismic : grid.Seismic
        叠后真实地震。
    reflectivity : grid.Reflectivity
        叠后反射系数。
    modeler : ModelingCallable
        正演算子。
    similarity : FunctionType
        相似度函数，输入为两个一维数组，返回无量纲分数（范围待确认）。
    num_wavelets : int, default=60
        候选子波数量 ``n``。
    inverse_polarity : bool, default=False
        是否反转极性。
    zero_phasing : bool, default=False
        是否对候选子波先做零相位处理。

    Returns
    -------
    grid.Wavelet
        搜索得到的最优子波，且写入候选集合统计得到的不确定性。

    Raises
    ------
    AssertionError
        当输入采样率或时间基准不满足一致性约束时触发。
    IndexError
        当 ``num_wavelets`` 过小导致前 10% 候选为空时可能触发。

    Examples
    --------
    >>> best = grid_search_best_wavelet(
    ...     evaluator, seismic, reflectivity, modeler, _similarity.normalized_xcorr_maximum
    ... )
    """
    assert np.allclose(seismic.basis, reflectivity.basis, rtol=1e-3)
    assert np.allclose(seismic.sampling_rate, evaluator.expected_sampling, rtol=1e-3)

    seismic_ = _preprocess_real_seismic(seismic, inverse_polarity=inverse_polarity)
    ref_ = _preprocess_reflectivity(reflectivity)

    wavelets = evaluator.sample_n_times(seismic=seismic_, reflectivity=ref_, n=num_wavelets)  # type: ignore
    # uncertainties
    amp_spectrums = []
    phase_spectrums = []
    for wlt in wavelets:
        wlt = _get_wavelet_object(wlt, seismic.sampling_rate)
        ff, ampl, power, phase = get_spectrum(wlt, to_degree=True)
        amp_spectrums.append(ampl)
        phase_spectrums.append(phase)
    amp_spectrums = np.stack(amp_spectrums, axis=0)
    phase_spectrums = np.stack(phase_spectrums, axis=0)
    uncertainties = grid.WaveletUncertainties(
        ff,
        np.mean(amp_spectrums, axis=0),
        np.std(amp_spectrums, axis=0),
        np.mean(phase_spectrums, axis=0),
        np.std(phase_spectrums, axis=0),
    )

    pack = []

    for i in range(len(wavelets)):
        wlt = _get_wavelet_object(wavelets[i], seismic.sampling_rate)
        if zero_phasing:
            wlt = zero_phasing_wavelet(wlt)
        synth_seismic = compute_synthetic_seismic(modeler, wlt, reflectivity)
        current_score = similarity(seismic.values, synth_seismic.values)

        # only look at phase before 60° for stability reasons
        ff, current_phase = get_phase(wlt, to_degree=True)
        valid_idx = np.argmin(np.abs(ff - 60))
        current_mean_abs_phase = np.mean(np.abs(current_phase[:valid_idx]))
        pack.append((wlt, current_score, current_mean_abs_phase))

    # sort list based on similarity score
    pack.sort(key=lambda a: a[1], reverse=True)
    # print("FIRST")
    # for p in pack:
    #    print(p[1],p[2])

    # take 10% best
    ntop = int(0.1 * num_wavelets)
    pack = pack[:ntop]

    # take smallest absolute phase amoung top scores
    if not zero_phasing:
        pack.sort(key=lambda a: a[2], reverse=False)
        # print("SECOND")
        # for p in pack:
        #    print(p[1],p[2])

    best_wavelet = pack[0][0]
    # print("FINAL")
    # print(pack[0][1],pack[0][2])

    if inverse_polarity:
        best_wavelet *= -1.0

    # set uncertainites
    best_wavelet.uncertainties = uncertainties

    return best_wavelet


def grid_search_best_prestack_wavelet(
    evaluator: BaseEvaluator,
    seismic: grid.PreStackSeismic,
    reflectivity: grid.PreStackReflectivity,
    modeler: ModelingCallable,
    similarity: FunctionType,
    num_wavelets: int = 60,
    zero_phasing: bool = False,
) -> grid.Wavelet:
    """分角度执行叠前最优子波搜索。

    Parameters
    ----------
    evaluator : BaseEvaluator
        子波评估器。
    seismic : grid.PreStackSeismic
        叠前真实地震。
    reflectivity : grid.PreStackReflectivity
        叠前反射系数。
    modeler : ModelingCallable
        正演算子。
    similarity : FunctionType
        相似度函数。
    num_wavelets : int, default=60
        每个角度的候选子波数量 ``n``。
    zero_phasing : bool, default=False
        是否对子波执行零相位处理。

    Returns
    -------
    grid.PreStackWavelet
        叠前最优子波集合。

    Raises
    ------
    AssertionError
        当输入角度集合不一致时触发。
    """
    assert (seismic.angles == reflectivity.angles).all()

    best_wavelet = []
    for i in range(seismic.num_traces):
        wlt = grid_search_best_wavelet(
            evaluator,
            seismic.traces[i],  # type: ignore
            reflectivity.traces[i],  # type: ignore
            modeler,
            similarity,
            num_wavelets=num_wavelets,
            zero_phasing=zero_phasing,
        )
        wlt.theta = seismic.angles[i]
        best_wavelet.append(wlt)

    return grid.PreStackWavelet(best_wavelet)  # type: ignore


def compute_synthetic_seismic(
    modeler: ModelingCallable,
    wavelet: grid.Wavelet,
    reflectivity: grid.Reflectivity,
) -> grid.Seismic:
    """由叠后子波与反射系数生成叠后合成地震。

    Parameters
    ----------
    modeler : ModelingCallable
        正演算子，输入为 ``(wavelet.values, reflectivity.values)``。
    wavelet : grid.Wavelet
        叠后子波，shape 为 ``(n_samples,)``。
    reflectivity : grid.Reflectivity
        叠后反射系数，shape 为 ``(n_samples,)``，域需为 TWT。

    Returns
    -------
    grid.Seismic
        合成叠后地震对象，名称固定为 ``"Synthetic seismic"``。

    Raises
    ------
    AssertionError
        当 ``wavelet`` 与 ``reflectivity`` 采样率不一致或
        ``reflectivity`` 不在 TWT 域时触发。
    """
    assert np.allclose(wavelet.sampling_rate, reflectivity.sampling_rate)
    assert reflectivity.is_twt

    seismic = modeler(wavelet.values, reflectivity.values)  # type: ignore
    return grid.Seismic(seismic, reflectivity.basis, "twt", name="Synthetic seismic")


def compute_synthetic_prestack_seismic(
    modeler: ModelingCallable,
    wavelet: grid.PreStackWavelet,
    reflectivity: grid.PreStackReflectivity,
) -> grid.PreStackSeismic:
    """由叠前子波与叠前反射系数生成叠前合成地震。

    Parameters
    ----------
    modeler : ModelingCallable
        正演算子。
    wavelet : grid.PreStackWavelet
        叠前子波集合。
    reflectivity : grid.PreStackReflectivity
        叠前反射系数集合。

    Returns
    -------
    grid.PreStackSeismic
        叠前合成地震集合，角度标签与输入一致。

    Raises
    ------
    AssertionError
        当 ``wavelet.angles`` 与 ``reflectivity.angles`` 不一致时触发。
    """
    assert (wavelet.angles == reflectivity.angles).all()

    seismics = []
    for theta in wavelet.angles:
        seis_ = compute_synthetic_seismic(modeler, wavelet[theta], reflectivity[theta])  # type: ignore
        seis_.theta = theta
        seismics.append(seis_)
    return grid.PreStackSeismic(seismics)  # type: ignore


############################################################
############################################################


def select_best_wavelet(
    wavelets: List[np.ndarray],
    seismic: np.ndarray,
    reflectivity: np.ndarray,
    modeler: ModelingCallable,
    num_iters: int = None,  # type: ignore
    noise_perc: float = 10,
):
    """从候选子波集合中选择最优子波（当前未实现）。

    Parameters
    ----------
    wavelets : List[numpy.ndarray]
        候选子波列表，每个元素 shape 为 ``(n_samples,)``。
    seismic : numpy.ndarray
        目标地震道，shape 为 ``(n_samples,)``。
    reflectivity : numpy.ndarray
        反射系数道，shape 为 ``(n_samples,)``。
    modeler : ModelingCallable
        正演算子。
    num_iters : int, optional
        计划迭代次数。当前实现未使用。
    noise_perc : float, default=10
        噪声百分比。当前实现未使用。

    Raises
    ------
    NotImplementedError
        函数入口立即抛出，后续代码当前不可达。
    """
    raise NotImplementedError()

    if num_iters is None:
        # overkill but its cheap
        num_iters = int(len(wavelets) * 1.2)

    n_sobol = int(0.7 * num_iters)  # 70%
    n_bayes = num_iters - n_sobol  # 30%

    ax_gen_startegy = GenerationStrategy(
        [
            GenerationStep(ax_Models.SOBOL, num_trials=n_sobol),
            GenerationStep(ax_Models.LEGACY_BOTORCH, num_trials=n_bayes),
        ]
    )

    ax_client = AxClient(generation_strategy=ax_gen_startegy, verbose_logging=False)

    choice = dict(
        name="wavelet_choice",
        type="range",
        bounds=[0, len(wavelets) - 1],
        value_type="int",
    )

    search_space = [choice]

    # Maximization
    ax_client.create_experiment(
        name="wavelet_distribution_choice",
        parameters=search_space,
        objectives={"choice_performance": ObjectiveProperties(minimize=False)},
        choose_generation_strategy_kwargs=None,
    )

    noise_level = (noise_perc / 100) * np.std(seismic)
    noise1_ = np.random.normal(scale=noise_level, size=seismic.shape)
    noise2_ = np.random.normal(scale=noise_level, size=seismic.shape)
    xcorr_coeff_error = 1.0 - _similarity.normalized_xcorr_maximum(seismic + noise1_, seismic + noise2_)

    for i in tqdm(range(num_iters)):
        h_params, trial_index = ax_client.get_next_trial()
        current_choice = h_params["wavelet_choice"]
        current_wavelet = wavelets[current_choice]

        current_seismic = modeler(wavelet=current_wavelet, reflectivity=reflectivity, noise=None)

        current_coeff = _similarity.normalized_xcorr_maximum(current_seismic, seismic)

        ax_client.complete_trial(trial_index=trial_index, raw_data=(current_coeff, xcorr_coeff_error))

    best_choice = ax_client.get_best_parameters()[0]["wavelet_choice"]

    return wavelets[best_choice], ax_client


def scale_wavelet(
    wavelet: grid.Wavelet,
    seismic: grid.Seismic,
    reflectivity: grid.Reflectivity,
    modeler: ModelingCallable,
    min_scale: float = 0.01,
    max_scale: float = 100,
    num_iters: int = 80,
    noise_perc: float = 5,
    is_tqdm: bool = True,
):
    """估计叠后子波的绝对幅值缩放系数。

    该函数使用 Ax（Sobol + BoTorch）最小化能量比误差：
    ``abs(E(synth)/E(real) - 1)``。

    Parameters
    ----------
    wavelet : grid.Wavelet
        待缩放子波。
    seismic : grid.Seismic
        目标真实地震。
    reflectivity : grid.Reflectivity
        反射系数。
    modeler : ModelingCallable
        正演算子。
    min_scale : float, default=0.01
        缩放系数下界。
    max_scale : float, default=100
        缩放系数上界。
    num_iters : int, default=80
        优化迭代次数 ``n``。
    noise_perc : float, default=5
        噪声比例（百分数），用于误差不确定度估计。
    is_tqdm : bool, default=True
        是否显示进度条。

    Returns
    -------
    tuple
        ``(scaled_wlt, ax_client)``：
        - ``scaled_wlt`` 为缩放后的 ``grid.Wavelet``；
        - ``ax_client`` 为保存试验历史的 AxClient。

    Examples
    --------
    >>> scaled_wlt, ax_client = scale_wavelet(wlt, seismic, reflectivity, modeler)
    """

    n_sobol = int(0.65 * num_iters)  # 65%
    n_bayes = num_iters - n_sobol  # 35%

    ax_gen_startegy = GenerationStrategy(
        [
            GenerationStep(ax_Models.SOBOL, num_trials=n_sobol),
            GenerationStep(ax_Models.LEGACY_BOTORCH, num_trials=n_bayes),
        ]
    )

    ax_client = AxClient(generation_strategy=ax_gen_startegy, verbose_logging=False)

    scaler = dict(name="scaler", type="range", bounds=[min_scale, max_scale], value_type="float")

    search_space = [scaler]

    ax_client.create_experiment(
        name="wavelet_absolute_scale_estimation",
        parameters=search_space,  # type: ignore
        objectives={"scaling_loss": ObjectiveProperties(minimize=True)},
        choose_generation_strategy_kwargs=None,
    )

    noise_level = (noise_perc / 100) * np.std(seismic.values)
    seismic_energy = _similarity.energy(seismic.values)

    for i in tqdm(range(num_iters), disable=(not is_tqdm)):
        h_params, trial_index = ax_client.get_next_trial()
        current_scaler = h_params["scaler"]
        current_wavelet_ = current_scaler * wavelet.values

        current_seismic = modeler(wavelet=current_wavelet_, reflectivity=reflectivity.values, noise=None)

        current_energy = _similarity.energy(current_seismic)

        # error = np.abs(current_energy - seismic_energy)
        error = np.abs(current_energy / seismic_energy - 1.0)

        # TODO: IS THIS CORRECT?
        error_std = _similarity.energy(np.random.normal(scale=noise_level, size=seismic.shape))

        ax_client.complete_trial(trial_index=trial_index, raw_data=(error, error_std))

    best_scaler = ax_client.get_best_parameters()[0]["scaler"]  # type: ignore

    scaled_wlt = grid.Wavelet(
        best_scaler * wavelet.values,
        wavelet.basis,
        name=wavelet.name,
        uncertainties=wavelet.uncertainties,
        theta=wavelet.theta,
    )

    return scaled_wlt, ax_client


def scale_prestack_wavelet(
    wavelet: grid.PreStackWavelet,
    seismic: grid.PreStackSeismic,
    reflectivity: grid.PreStackReflectivity,
    modeler: ModelingCallable,
    min_scale: float = 0.01,
    max_scale: float = 100,
    num_iters: int = 50,
    noise_perc: float = 5,
):
    """按角度缩放叠前子波集合的绝对幅值。

    Parameters
    ----------
    wavelet : grid.PreStackWavelet
        叠前子波集合。
    seismic : grid.PreStackSeismic
        叠前真实地震集合。
    reflectivity : grid.PreStackReflectivity
        叠前反射系数集合。
    modeler : ModelingCallable
        正演算子。
    min_scale : float, default=0.01
        每个角度缩放系数下界。
    max_scale : float, default=100
        每个角度缩放系数上界。
    num_iters : int, default=50
        每个角度优化迭代次数 ``n``。
    noise_perc : float, default=5
        噪声比例（百分数）。

    Returns
    -------
    tuple
        ``(scaled_wavelet, ax_clients)``：
        - ``scaled_wavelet`` 为缩放后的 ``grid.PreStackWavelet``；
        - ``ax_clients`` 为各角度对应的 AxClient 列表。

    Raises
    ------
    AssertionError
        当 ``wavelet``、``seismic``、``reflectivity`` 的角度集合不一致时触发。
    """
    assert (wavelet.angles == seismic.angles).all()
    assert (wavelet.angles == reflectivity.angles).all()

    scaled_wavelet = []
    ax_clients = []

    for i in tqdm(range(wavelet.num_traces)):
        wlt, ax_client = scale_wavelet(
            wavelet.traces[i],  # type: ignore
            seismic.traces[i],  # type: ignore
            reflectivity.traces[i],  # type: ignore
            modeler,
            min_scale=min_scale,
            max_scale=max_scale,
            num_iters=num_iters,
            noise_perc=noise_perc,
            is_tqdm=False,
        )

        wlt.theta = wavelet.angles[i]

        scaled_wavelet.append(wlt)
        ax_clients.append(ax_client)

    return grid.PreStackWavelet(scaled_wavelet, name=wavelet.name), ax_clients  # type: ignore
