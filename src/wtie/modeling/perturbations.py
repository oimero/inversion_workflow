"""wtie.modeling.perturbations: 一维信号随机扰动变换组件。

本模块提供面向井震对齐/波子增强的一维随机变换，
包括相位旋转、时移、噪声注入、陷波滤波与振幅缩放，
并支持通过 ``Compose`` 按顺序组合执行。

边界说明
--------
- 本模块不负责数据读取写入、地质解释或训练流程编排。
- 本模块仅提供扰动算子与组合逻辑，参数有效性与输入质量控制由上游负责。

核心公开对象
------------
1. BaseTransform: 扰动变换统一接口与公共状态。
2. Compose: 顺序组合多个扰动并按需随机跳过。
3. RandomConstantPhaseRotation / RandomIndependentPhaseRotation: 相位扰动入口。
4. RandomTimeShift / RandomAmplitudeScaling: 时域平移与振幅缩放。
5. RandomSimplexNoise / RandomWhitexNoise / RandomNotchFilter: 噪声与频带扰动。

Examples
--------
>>> import numpy as np
>>> from wtie.modeling.perturbations import Compose, RandomTimeShift
>>> x = np.ones((128,), dtype=float)
>>> y = Compose([RandomTimeShift(max_samples=4)])(x)
>>> x.shape == y.shape
True
"""

import numpy as np

from wtie.modeling import noise
from wtie.processing import spectral
from wtie.utils.types_ import List, Tuple


class BaseTransform:
    """扰动变换基类。

    该类定义统一的调用接口与公共开关，子类应在 ``__call__`` 中实现具体变换逻辑。

    Attributes
    ----------
    apply : bool
        当前变换是否允许被执行。上层 ``Compose`` 可结合该标志做条件应用。
    dt : float or None
        采样间隔（s）。仅在需要频率合法性检查的子类中使用；为空时表示未提供。
    """

    def __init__(self, apply: bool = True, dt: float = None):  # type: ignore
        """初始化基类状态。

        Parameters
        ----------
        apply : bool, default=True
            是否启用该变换。
        dt : float or None, default=None
            采样间隔（s）。
        """
        self.apply = apply
        self.dt = dt

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """对输入信号执行变换。

        Parameters
        ----------
        signal : numpy.ndarray
            输入一维信号，shape 为 ``(n_samples,)``。

        Returns
        -------
        numpy.ndarray
            变换后的一维信号，shape 为 ``(n_samples,)``。

        Notes
        -----
        基类仅定义接口，不提供实现；子类应覆盖该方法。
        """
        pass

    def _verify_frequency(self, f: float):
        """校验频率是否落在有效范围内。

        Parameters
        ----------
        f : float
            待校验频率（Hz）。

        Raises
        ------
        AssertionError
            当 ``f <= 0`` 或 ``f > 1 / (2 * dt)``（Nyquist）时抛出。
        """
        assert f > 0.0
        assert f <= 1 / (2 * self.dt)  # Nyquist


class Compose:
    """按顺序组合并执行多个扰动变换。

    支持两级开关：
    1) 变换实例自身的 ``apply`` 标志；
    2) 组合器级别的随机关闭机制（``random_switch`` + ``p``）。

    Attributes
    ----------
    transformations : List[BaseTransform]
        需依次执行的变换列表。
    random_switch : bool
        是否启用“对每个可应用变换随机跳过”的机制。
    p : float
        跳过单个变换的概率，范围通常为 ``[0, 1]``。
    """

    def __init__(self, transformations: List[BaseTransform], random_switch: bool = False, p: float = 0.5):
        """初始化变换组合器。

        Parameters
        ----------
        transformations : List[BaseTransform]
            由 ``BaseTransform`` 子类实例组成的有序列表。
        random_switch : bool, default=False
            若为 ``True``，则对每个 ``apply=True`` 的变换按概率随机跳过。
        p : float, default=0.5
            随机跳过概率，建议范围 ``[0, 1]``。
        """
        self.transformations = transformations
        self.random_switch = random_switch
        self.p = p

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """依次执行组合中的变换。

        Parameters
        ----------
        signal : numpy.ndarray
            输入一维信号，shape 为 ``(n_samples,)``。

        Returns
        -------
        numpy.ndarray
            变换后信号，shape 为 ``(n_samples,)``。

        Notes
        -----
        - 会先复制输入数组，再在副本上顺序应用变换。
        - 若变换对象包含 ``apply`` 属性且其值为 ``False``，该变换始终跳过。
        - 当 ``random_switch=True`` 时，``apply=True`` 的变换会以概率 ``p`` 被跳过。
        """
        signal = np.copy(signal)

        for f in self.transformations:
            # check if there is an internal switch
            if hasattr(f, "apply"):
                apply = f.apply

                # randomly switch off if random_swith is set to True by user
                # never apply if internal flag apply exists and is False
                if self.random_switch and apply:
                    apply = np.random.rand() > self.p

            else:
                # always apply if no internal switch
                apply = True

            if apply:
                signal = f(signal)

        return signal

    def __str__(self):
        """返回当前启用变换的名称列表字符串。

        Returns
        -------
        str
            每行一个类名，仅包含静态 ``apply=True`` 的变换。
        """
        s = ""
        for f in self.transformations:
            if "apply" in vars(f).keys():
                applied = vars(f)["apply"]
            else:
                applied = True

            if applied:
                s += f.__class__.__name__ + "\n"
        return s


class RandomConstantPhaseRotation(BaseTransform):
    """对输入信号施加随机常相位旋转。"""

    def __init__(self, angle_range: Tuple[float, float], **kwargs):
        """初始化常相位旋转扰动。

        Parameters
        ----------
        angle_range : tuple(float, float)
            相位角随机采样区间（度），格式为 ``(min_deg, max_deg)``。
        **kwargs
            透传给 ``BaseTransform`` 的参数（如 ``apply``、``dt``）。
        """
        super().__init__(**kwargs)

        self.min_angle = np.deg2rad(angle_range[0])
        self.max_angle = np.deg2rad(angle_range[1])

    def __call__(self, signal):
        """执行一次常相位旋转。

        Parameters
        ----------
        signal : numpy.ndarray
            输入一维信号，shape 为 ``(n_samples,)``。

        Returns
        -------
        numpy.ndarray
            旋转后的信号，shape 为 ``(n_samples,)``。
        """
        angle = np.random.uniform(self.min_angle, self.max_angle)
        return spectral.constant_phase_rotation(signal, angle)


class RandomTimeShift(BaseTransform):
    """对输入信号做随机整数采样点时移。

    Attributes
    ----------
    max_samples : int
        最大平移采样点数 ``n``，实际平移量在 ``[0, n]`` 内随机采样。
    """

    def __init__(self, max_samples: int, **kwargs):
        """初始化时移扰动。

        Parameters
        ----------
        max_samples : int
            最大平移采样点数 ``n``。
        **kwargs
            透传给 ``BaseTransform`` 的参数（如 ``apply``、``dt``）。
        """
        super().__init__(**kwargs)

        self.max_samples = max_samples

    def __call__(self, signal):
        """执行随机时移。

        Parameters
        ----------
        signal : numpy.ndarray
            输入一维信号，shape 为 ``(n_samples,)``。

        Returns
        -------
        numpy.ndarray
            时移后的信号，shape 为 ``(n_samples,)``。

        Notes
        -----
        当平移点数大于 0 时：
        - 随机选择左移或右移；
        - 空缺位置以 0 填充。
        """

        n_samples = np.random.randint(0, self.max_samples + 1)

        if n_samples != 0:
            zeros = np.zeros((n_samples,))

            # left
            if np.random.randint(2) == 0:
                signal = np.concatenate((zeros, signal[:-n_samples]))
            # right
            else:
                signal = np.concatenate((signal[n_samples:], zeros))

        return signal


class TMPRandomLinearPhaseRotation(BaseTransform):
    """对输入信号施加随机线性相位旋转（临时实现）。

    Notes
    -----
    类名中的 ``TMP`` 表明该实现处于临时/实验状态。
    """

    def __init__(self, angle_range: Tuple[float, float], **kwargs):
        """初始化线性相位旋转扰动。

        Parameters
        ----------
        angle_range : tuple(float, float)
            相位角随机采样区间（度），格式为 ``(min_deg, max_deg)``。
            内部会转换为弧度后采样起止相位。
        **kwargs
            透传给 ``BaseTransform`` 的参数（如 ``apply``、``dt``）。
        """
        super().__init__(**kwargs)

        self.min_angle = np.deg2rad(angle_range[0])
        self.max_angle = np.deg2rad(angle_range[1])

    def __call__(self, signal):
        """执行一次线性相位旋转。

        Parameters
        ----------
        signal : numpy.ndarray
            输入一维信号，shape 为 ``(n_samples,)``。

        Returns
        -------
        numpy.ndarray
            旋转后的信号，shape 为 ``(n_samples,)``。
        """

        start_angle = np.random.uniform(self.min_angle, self.max_angle)
        end_angle = np.random.uniform(start_angle, self.max_angle)

        return spectral.linear_phase_rotation(signal, start_angle, end_angle)


class RandomIndependentPhaseRotation(BaseTransform):
    """对各频率分量施加独立随机相位旋转。"""

    def __init__(self, angle_range: Tuple[float, float], **kwargs):
        """初始化独立频率相位旋转扰动。

        Parameters
        ----------
        angle_range : tuple(float, float)
            相位角随机采样区间（度），格式为 ``(min_deg, max_deg)``。
        **kwargs
            透传给 ``BaseTransform`` 的参数（如 ``apply``、``dt``）。
        """
        super().__init__(**kwargs)

        self.min_angle = np.deg2rad(angle_range[0])
        self.max_angle = np.deg2rad(angle_range[1])

    def __call__(self, signal):
        """执行独立频率相位旋转。

        Parameters
        ----------
        signal : numpy.ndarray
            输入一维信号，shape 为 ``(n_samples,)``。

        Returns
        -------
        numpy.ndarray
            旋转后的信号，shape 为 ``(n_samples,)``。
        """
        return spectral.random_phase_rotation(signal, self.min_angle, self.max_angle)


class RandomSimplextPhaseRotation(BaseTransform):
    """基于 simplex 随机场对相位进行频率相关旋转。

    Attributes
    ----------
    max_abs_angle : int
        最大绝对相位角上限（度）。每次调用会在 ``[5, max_abs_angle]`` 采样。
    scale_percentage_factor : float
        传递给底层频率扰动生成器的尺度因子（无量纲）。
    """

    def __init__(self, max_abs_angle: int = 60, scale_percentage_factor: float = 2.5, **kwargs):
        """初始化 simplex 相位旋转扰动。

        Parameters
        ----------
        max_abs_angle : int, default=60
            最大绝对相位角上限（度）。
        scale_percentage_factor : float, default=2.5
            控制相位扰动尺度的无量纲因子。
        **kwargs
            透传给 ``BaseTransform`` 的参数（如 ``apply``、``dt``）。
        """
        super().__init__(**kwargs)

        self.max_abs_angle = max_abs_angle
        self.scale_percentage_factor = scale_percentage_factor

    def __call__(self, signal):
        """执行 simplex 相位旋转并对长度进行对齐。

        Parameters
        ----------
        signal : numpy.ndarray
            输入一维信号，shape 为 ``(n_samples,)``。

        Returns
        -------
        numpy.ndarray
            处理后的信号，shape 为 ``(n_samples,)``。

        Notes
        -----
        若底层函数返回长度小于输入长度，会在尾部以 0 补齐。
        """

        max_abs_angle_ = np.random.randint(5, self.max_abs_angle + 1)

        modified = spectral.random_simplex_phase_rotation(
            signal, max_abs_angle=max_abs_angle_, scale_percentage_factor=self.scale_percentage_factor
        )

        pad = signal.size - modified.size
        if pad > 0:
            modified = np.concatenate((modified, np.zeros((pad,), dtype=modified.dtype)))

        return modified


class RandomSimplexNoise(BaseTransform):
    """向输入信号叠加随机 simplex 噪声。

    Attributes
    ----------
    scale_range : Tuple[float, float]
        噪声主尺度随机范围。
    variation_scale_range : Tuple[float, float]
        噪声变化尺度随机范围。
    octaves : int
        叠加 octave 数量。
    """

    def __init__(
        self, scale_range: Tuple[float, float], variation_scale_range: Tuple[float, float], octaves: int = 3, **kwargs
    ):
        """初始化 simplex 噪声扰动。

        Parameters
        ----------
        scale_range : Tuple[float, float]
            噪声主尺度采样范围。
        variation_scale_range : Tuple[float, float]
            噪声变化尺度采样范围。
        octaves : int, default=3
            octave 数量。
        **kwargs
            透传给 ``BaseTransform`` 的参数（如 ``apply``、``dt``）。
        """
        super().__init__(**kwargs)

        self.scale_range = scale_range
        self.variation_scale_range = variation_scale_range
        self.octaves = octaves

    def __call__(self, signal):
        """生成并叠加一次随机 simplex 噪声。

        Parameters
        ----------
        signal : numpy.ndarray
            输入一维信号，shape 为 ``(n_samples,)``。

        Returns
        -------
        numpy.ndarray
            叠加噪声后的信号，shape 为 ``(n_samples,)``。

        Notes
        -----
        噪声会随机执行反转与符号翻转，并在叠加前做去均值处理。
        """

        # octaves = np.random.randint(self.octave_range[0], self.octave_range[1] + 1)
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

        var_scale = np.random.uniform(self.variation_scale_range[0], self.variation_scale_range[1])

        noise_ = noise.open_simplex_noise(signal.size, scale, octaves=self.octaves, variation_scale=var_scale)

        if np.random.randint(2) == 1:
            noise_ = noise_[::-1]

        if np.random.randint(2) == 1:
            noise_ *= -1.0

        noise_ -= noise_.mean()

        return signal + noise_


class RandomWhitexNoise(BaseTransform):
    """向输入信号叠加白噪声。

    Attributes
    ----------
    scale : float
        白噪声幅度尺度参数。
    """

    def __init__(self, scale: float, **kwargs):
        """初始化白噪声扰动。

        Parameters
        ----------
        scale : float
            白噪声幅度尺度参数。
        **kwargs
            透传给 ``BaseTransform`` 的参数（如 ``apply``、``dt``）。
        """
        super().__init__(**kwargs)

        self.scale = scale

    def __call__(self, signal):
        """生成并叠加白噪声。

        Parameters
        ----------
        signal : numpy.ndarray
            输入一维信号，shape 为 ``(n_samples,)``。

        Returns
        -------
        numpy.ndarray
            叠加白噪声后的信号，shape 为 ``(n_samples,)``。
        """

        noise_ = noise.white_noise(signal.size, self.scale)

        return signal + noise_


class RandomNotchFilter(BaseTransform):
    """对输入信号施加随机陷波（band-cut）滤波。

    Attributes
    ----------
    dt : float
        采样间隔 ``dt``（s）。
    freq_range : Tuple[float, float]
        陷波中心频率随机范围（Hz）。
    band_range : Tuple[float, float]
        陷波带宽随机范围（Hz，按整数采样）。
    """

    def __init__(self, dt: float, freq_range: Tuple[float, float], band_range: Tuple[float, float], **kwargs):
        """初始化随机陷波滤波扰动。

        Parameters
        ----------
        dt : float
            采样间隔 ``dt``（s）。
        freq_range : Tuple[float, float]
            陷波中心频率随机范围（Hz）。
        band_range : Tuple[float, float]
            陷波带宽随机范围（Hz）。实现中按整数采样。
        **kwargs
            透传给 ``BaseTransform`` 的参数（如 ``apply``）。
        """
        super().__init__(**kwargs)

        self.dt = dt
        self.freq_range = freq_range
        self.band_range = band_range

    def __call__(self, signal):
        """执行一次随机陷波滤波。

        Parameters
        ----------
        signal : numpy.ndarray
            输入一维信号，shape 为 ``(n_samples,)``。

        Returns
        -------
        numpy.ndarray
            滤波后的信号，shape 为 ``(n_samples,)``。
        """
        freq = np.random.uniform(self.freq_range[0], self.freq_range[1])
        band = np.random.randint(self.band_range[0], self.band_range[1] + 1)  # type: ignore
        return spectral.apply_notch_filter(signal, self.dt, freq, band, order=8)


class RandomAmplitudeScaling(BaseTransform):
    """对输入信号施加随机振幅缩放。

    Attributes
    ----------
    amplitude_range : Tuple[float, float]
        振幅缩放因子随机范围（无量纲）。
    """

    def __init__(self, amplitude_range: Tuple[float, float], **kwargs):
        """初始化振幅缩放扰动。

        Parameters
        ----------
        amplitude_range : Tuple[float, float]
            振幅缩放因子随机范围。
        **kwargs
            透传给 ``BaseTransform`` 的参数（如 ``apply``、``dt``）。
        """
        super().__init__(**kwargs)

        self.amplitude_range = amplitude_range

    def __call__(self, signal):
        """执行一次随机振幅缩放。

        Parameters
        ----------
        signal : numpy.ndarray
            输入一维信号，shape 为 ``(n_samples,)``。

        Returns
        -------
        numpy.ndarray
            缩放后的信号，shape 为 ``(n_samples,)``。
        """

        a = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
        signal *= a

        return signal
