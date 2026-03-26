"""wtie.modeling.wavelet: 地震子波生成与随机化工具。

本模块提供 Ricker、Ormby、Butterworth 子波生成函数，
以及将随机基子波封装为 Wavelet 对象的工厂工具。

边界说明
--------
- 本模块不负责地震道卷积、井震对齐策略或训练流程控制。
- 本模块仅处理子波构造与基础扰动/加窗封装，不执行文件 I/O。

核心公开对象
------------
1. Wavelet: 子波数据容器（原始/扰动振幅与时间轴）。
2. RandomWaveletCallable: 随机选择基子波生成器并返回 Wavelet。
3. ricker / ormby / butterworth: 三类常用子波生成入口函数。
4. RandomRickerTools / RandomOrmbyTools / RandomButterworthTools: 对应参数随机采样器。

Examples
--------
>>> gen = RandomRickerTools(f_range=(20.0, 35.0), dt=0.002, n_samples=128)
>>> rw = RandomWaveletCallable([gen])
>>> w = rw()
>>> w.y.shape, w.t.shape
((128,), (128,))
"""

import random
import warnings

import numpy as np

from wtie.processing.sampling import Resampler
from wtie.processing.spectral import apply_butter_bandpass_filter
from wtie.processing.taper import _Taper
from wtie.utils.types_ import Callable, FunctionType, List, Tuple


class RandomWaveletCallable:
    """随机子波工厂，按给定策略采样基子波并构造 ``Wavelet``。

    Attributes
    ----------
    random_base_wavelet_gens : list of callable
        候选基子波生成器列表；每个生成器调用后应返回
        ``(func, func_args, func_kwargs)``。
    perturbations : callable or None
        对基子波振幅进行扰动的可调用对象；输入/输出 shape 均为 ``(n_samples,)``。
    resampler : Resampler or None
        预留参数；当前传入非 ``None`` 会在 ``Wavelet`` 中触发 ``DeprecationWarning``。
    taper : _Taper or None
        对扰动后子波施加窗函数的可调用对象。
    """

    def __init__(
        self,
        random_base_wavelet_gens: List[FunctionType],
        perturbations: Callable[[np.ndarray], np.ndarray] = None,  # type: ignore
        resampler: Resampler = None,  # type: ignore
        taper: _Taper = None,  # type: ignore
    ):
        """初始化随机子波工厂。

        Parameters
        ----------
        random_base_wavelet_gens : List[FunctionType]
            基子波生成器列表。每个元素在调用后应返回
            ``(func, func_args, func_kwargs)``，其中 ``func`` 需返回 ``(t, y)``。
        perturbations : callable, optional
            子波扰动算子。输入为一维振幅数组 ``(n_samples,)``，返回同 shape 数组。
            默认 ``None``，表示不做扰动。
        resampler : Resampler, optional
            预留重采样器。默认 ``None``。
            若传入非 ``None``，后续构造 ``Wavelet`` 时会抛出 ``DeprecationWarning``。
        taper : _Taper, optional
            子波加窗算子。默认 ``None``。
        """

        self.random_base_wavelet_gens = random_base_wavelet_gens
        self.perturbations = perturbations
        self.resampler = resampler
        self.taper = taper

    def __call__(self):
        """随机采样一个基子波配置并返回 ``Wavelet`` 对象。

        Returns
        -------
        Wavelet
            包含时间轴与振幅数据的子波对象。
        """
        # random select the wavelet base
        func, func_args, func_kwargs = random.choice(self.random_base_wavelet_gens)()

        # apply perturbations
        return Wavelet(
            func=func,
            func_args=func_args,
            func_kwargs=func_kwargs,
            perturbations=self.perturbations,
            resampler=self.resampler,
            taper=self.taper,
        )


class Wavelet:
    """地震子波数据容器，封装基子波及其扰动结果。

    约束：时间轴与振幅均按一维数组组织，shape 为 ``(n_samples,)``；
    采样间隔 ``dt`` 的单位为秒（s）。

    Attributes
    ----------
    t_original : np.ndarray
        基子波时间轴，shape ``(n_samples,)``，单位 s。
    y_original : np.ndarray
        基子波振幅，shape ``(n_samples,)``。
    perturbations : callable or None
        扰动算子；``None`` 表示不做扰动。
    y_perturbed : np.ndarray
        扰动后的子波振幅，shape ``(n_samples,)``。
    t : np.ndarray
        当前子波时间轴，shape ``(n_samples,)``，单位 s。
    dt_original : float
        基子波采样间隔，单位 s。
    dt : float
        当前子波采样间隔，单位 s。
    y : np.ndarray
        ``y_perturbed`` 的别名，shape ``(n_samples,)``。
    """

    def __init__(
        self,
        func: FunctionType,
        func_args: list = None,  # type: ignore
        func_kwargs: dict = None,  # type: ignore
        perturbations: Callable[[np.ndarray], np.ndarray] = None,  # type: ignore
        resampler: Resampler = None,  # type: ignore
        taper: _Taper = None,  # type: ignore
    ):
        """构造子波对象并按需执行扰动/加窗。

        Parameters
        ----------
        func : FunctionType
            基子波生成函数，调用后应返回 ``(t, y)``，两者 shape 均为 ``(n_samples,)``。
        func_args : list, optional
            传给 ``func`` 的位置参数。
        func_kwargs : dict, optional
            传给 ``func`` 的关键字参数。默认 ``None``，内部按空字典处理。
        perturbations : callable, optional
            子波扰动算子。输入/输出 shape 均为 ``(n_samples,)``。
        resampler : Resampler, optional
            已弃用参数。当前若不为 ``None``，会立即抛出 ``DeprecationWarning``。
        taper : _Taper, optional
            子波加窗算子。若提供，则在扰动后执行。

        Raises
        ------
        DeprecationWarning
            当 ``resampler`` 非 ``None`` 时触发。
        """

        if func_kwargs is None:
            func_kwargs = {}

        # base wavelet
        self.t_original, self.y_original = func(*func_args, **func_kwargs)

        self._func_name = func.__name__
        # self._args = func_args
        self._args = {key: value for (key, value) in zip(func.__code__.co_varnames[: len(func_args)], func_args)}

        self._kwargs = func_kwargs

        # random perturbations
        self.perturbations = perturbations
        if perturbations is None:
            self.y_perturbed = np.copy(self.y_original)
        else:
            self.y_perturbed = perturbations(self.y_original)

        # resampling
        if resampler is not None:
            raise DeprecationWarning("Don't use resampler anymore")
            # self.y_perturbed, self.t = resampler(self.y_perturbed, self.t_original)
        # else:
        # self.t = np.copy(self.t_original)
        self.t = np.copy(self.t_original)

        # tapering
        if taper is not None:
            self.y_perturbed = taper(self.y_perturbed)

        # sampling rate
        self.dt_original = self.t_original[1] - self.t_original[0]
        self.dt = self.t[1] - self.t[0]

        # alias
        self.y = self.y_perturbed

    def __str__(self):
        """返回子波关键信息的可读字符串表示。"""
        s = "base wavelet: " + str(self._func_name) + "\n"
        s += "num samples: " + str(len(self.y)) + "\n"
        s += "sampling rate: " + str(self.dt) + "\n"
        s += "duration (seconds): " + str(self.dt * len(self.y)) + "\n"
        s += "base args: " + str(self._args) + "\n"
        s += "base kwargs: " + str(self._kwargs) + "\n"
        s += "transformations:\n"
        for line in str(self.perturbations).split("\n")[:-1]:
            s += "\t- " + line + "\n"
        # s +=  str(self.perturbations)
        return s


##################################################
# utils
##################################################


class RandomRickerTools:
    """Ricker 基子波参数的随机采样器。

    Attributes
    ----------
    f_range : tuple of float
        主频采样范围，单位 Hz。
    dt : float
        采样间隔，单位 s。
    n_samples : int
        采样点数 ``n``，建议为偶数。
    """

    def __init__(self, f_range: Tuple[float, float], dt: float, n_samples: int):
        """初始化 Ricker 参数采样器。

        Parameters
        ----------
        f_range : tuple of float
            主频采样范围 ``(f_min, f_max)``，单位 Hz。
        dt : float
            采样间隔，单位 s。
        n_samples : int
            采样点数 ``n``。当前实现要求为偶数。

        Raises
        ------
        AssertionError
            当 ``n_samples`` 为奇数时触发。
        """

        assert n_samples % 2 == 0, "Best specify an even number of samples..."

        self.f_range = f_range
        self.dt = dt
        self.n_samples = n_samples

    def __call__(self):
        """随机生成一组 Ricker 子波构造参数。

        Returns
        -------
        tuple
            三元组 ``(func, func_args, func_kwargs)``，其中
            ``func is ricker``。
        """
        # returns function, args, kwargs
        f = np.random.uniform(self.f_range[0], self.f_range[1])
        return ricker, (f, self.dt, self.n_samples), None


class RandomOrmbyTools:
    """Ormby 基子波参数的随机采样器。

    该采样器会强制频率满足严格递增关系
    ``f0 < f1 < f2 < f3``，并在相邻频点间保留至少 9 Hz 间隔。

    Attributes
    ----------
    f0_range, f1_range, f2_range, f3_range : tuple of float
        四个特征频率各自的采样范围，单位 Hz。
    dt : float
        采样间隔，单位 s。
    n_samples : int
        采样点数 ``n``，当前实现要求为偶数。
    """

    def __init__(
        self,
        f0_range: Tuple[float, float],
        f1_range: Tuple[float, float],
        f2_range: Tuple[float, float],
        f3_range: Tuple[float, float],
        dt: float,
        n_samples: int,
    ):
        """初始化 Ormby 参数采样器。

        Parameters
        ----------
        f0_range, f1_range, f2_range, f3_range : tuple of float
            四个特征频率的采样范围，单位 Hz。
        dt : float
            采样间隔，单位 s。
        n_samples : int
            采样点数 ``n``。当前实现要求为偶数。

        Raises
        ------
        AssertionError
            当 ``n_samples`` 为奇数时触发。

        Warns
        -----
        UserWarning
            当 ``f3_range`` 上限超过 Nyquist 频率时触发告警。
        """

        assert n_samples % 2 == 0, "Best specify an even number of samples..."

        self.f0_range = f0_range
        self.f1_range = f1_range
        self.f2_range = f2_range
        self.f3_range = f3_range
        self.dt = dt
        self.n_samples = n_samples

        _warn_nyquist(f3_range[-1], dt)

    def __call__(self):
        """随机生成一组满足约束的 Ormby 子波参数。

        Returns
        -------
        tuple
            三元组 ``(func, func_args, func_kwargs)``，其中
            ``func is ormby``。
        """
        # returns function, args, kwargs
        Df = 9

        f0 = np.random.uniform(self.f0_range[0], self.f0_range[1])

        f1_min = max(self.f1_range[0], f0 + Df)
        f1 = np.random.uniform(f1_min, max(self.f1_range[1], f1_min))

        f2_min = max(self.f2_range[0], f1 + Df)
        f2 = np.random.uniform(f2_min, max(self.f2_range[1], f2_min))

        f3_min = max(self.f3_range[0], f2 + Df)
        f3 = np.random.uniform(f3_min, max(self.f3_range[1], f3_min))
        return ormby, ((f0, f1, f2, f3), self.dt, self.n_samples), None


class RandomButterworthTools:
    """Butterworth 子波参数的随机采样器。

    Attributes
    ----------
    lowcut_range : tuple of float
        低截止频率采样范围，单位 Hz。
    highcut_range : tuple of float
        高截止频率采样范围，单位 Hz。
    dt : float
        采样间隔，单位 s。
    n_samples : int
        采样点数 ``n``。
    order_range : tuple of float
        滤波器阶数采样范围，取整后用于随机采样。
    """

    def __init__(
        self,
        lowcut_range: Tuple[float, float],
        highcut_range: Tuple[float, float],
        dt: float,
        n_samples: int,
        order_range: Tuple[float, float] = (6, 6),
    ):
        """初始化 Butterworth 参数采样器。

        Parameters
        ----------
        lowcut_range : tuple of float
            低截止频率采样范围，单位 Hz。
        highcut_range : tuple of float
            高截止频率采样范围，单位 Hz。
        dt : float
            采样间隔，单位 s。
        n_samples : int
            采样点数 ``n``。
        order_range : tuple of float, optional
            滤波器阶数采样范围。默认 ``(6, 6)``，即固定 6 阶。
        """

        self.lowcut_range = lowcut_range
        self.highcut_range = highcut_range
        self.dt = dt
        self.n_samples = n_samples
        self.order_range = order_range

    def __call__(self):
        """随机生成一组 Butterworth 子波构造参数。

        Returns
        -------
        tuple
            三元组 ``(func, func_args, func_kwargs)``，其中
            ``func is butterworth``。
        """
        # retunrs function, args, kwargs
        lowcut = np.random.uniform(self.lowcut_range[0], self.lowcut_range[1])
        highcut = np.random.uniform(self.highcut_range[0], self.highcut_range[1])
        order = np.random.randint(self.order_range[0], self.order_range[1] + 1)  # type: ignore
        return butterworth, (lowcut, highcut, self.dt, self.n_samples), dict(order=order)


def ricker(f: float, dt: float, n_samples: int):
    """生成 Ricker 子波。

    Parameters
    ----------
    f : float
        子波主频，单位 Hz。
    dt : float
        采样间隔 ``dt``，单位 s。
    n_samples : int
        采样点数 ``n``。

    Returns
    -------
    t : np.ndarray
        时间轴，shape 为 ``(n_samples,)``，单位 s。
    y : np.ndarray
        子波振幅，shape 为 ``(n_samples,)``。
    """
    # n_samples = int(round(duration / dt))
    duration = n_samples * dt
    t = np.arange(-duration / 2, (duration - dt) / 2, dt)
    # t = np.arange(-duration/2, duration/2, dt)

    y = (1.0 - 2.0 * (np.pi**2) * (f**2) * (t**2)) * np.exp(-(np.pi**2) * (f**2) * (t**2))
    return t, y


def ormby(f: Tuple[float], dt: float, n_samples: int):
    """生成 Ormby（梯形频谱）子波。

    Parameters
    ----------
    f : tuple of float
        四个特征频率 ``(f0, f1, f2, f3)``，单位 Hz，且需满足
        ``0 < f0 < f1 < f2 < f3``。
    dt : float
        采样间隔 ``dt``，单位 s。
    n_samples : int
        采样点数 ``n``。

    Returns
    -------
    t : np.ndarray
        时间轴，shape 为 ``(n_samples,)``，单位 s。
    w : np.ndarray
        归一化子波振幅，shape 为 ``(n_samples,)``，最大值为 1。

    Raises
    ------
    AssertionError
        当 ``f`` 不是长度为 4 的频率序列，或频率不严格递增时触发。
    """

    assert len(f) == 4, "You need to specify 4 frequencies for the trapezoid definition"
    assert 0 < f[0] < f[1] < f[2] < f[3], "Frequencies must be strictly increasing"
    f0, f1, f2, f3 = f

    duration = n_samples * dt
    # t = np.arange(-duration/2, (duration-dt)/2, dt)
    t = np.arange(-duration / 2, duration / 2, dt)

    def g(f, t):
        return (np.sinc(f * t) ** 2) * ((np.pi * f) ** 2)

    pf32 = (np.pi * f3) - (np.pi * f2)
    pf10 = (np.pi * f1) - (np.pi * f0)

    w = (g(f3, t) / pf32) - (g(f2, t) / pf32) - (g(f1, t) / pf10) + (g(f0, t) / pf10)

    w /= np.max(w)
    return t, w


def butterworth(
    lowcut: float,
    highcut: float,
    dt: float,
    n_samples: int,
    order: int = 8,
    rescale: bool = True,
    zero_phase=True,
) -> Tuple[np.ndarray, np.ndarray]:
    """通过带通 Butterworth 滤波器脉冲响应构造零相位/非零相位子波。

    Parameters
    ----------
    lowcut : float
        低截止频率，单位 Hz。
    highcut : float
        高截止频率，单位 Hz。
    dt : float
        采样间隔 ``dt``，单位 s。
    n_samples : int
        输出采样点数 ``n``。
    order : int, optional
        滤波器阶数。默认 8。
    rescale : bool, optional
        是否将输出振幅按最大绝对值归一化到 ``[-1, 1]``。默认 ``True``。
    zero_phase : bool, optional
        是否采用零相位滤波。默认 ``True``。

    Returns
    -------
    t : np.ndarray
        时间轴，shape 为 ``(n_samples,)``，单位 s。
    y : np.ndarray
        子波振幅，shape 为 ``(n_samples,)``。

    Raises
    ------
    AssertionError
        当内部裁剪后的振幅长度与 ``n_samples`` 不一致时触发。

    Notes
    -----
    当前函数签名的返回类型标注为 ``np.ndarray``，但代码实际返回 ``(t, y)``。
    该注解与实现不一致，待确认是否需要在后续版本统一。
    """

    # for edge effects
    n_samples_tmp = 3 * n_samples

    spike = np.zeros((n_samples_tmp,), dtype=float)
    spike[n_samples_tmp // 2] = 1.0

    fs = 1 / dt

    duration = n_samples * dt
    t = np.arange(-duration / 2, (duration - dt) / 2, dt)

    y = apply_butter_bandpass_filter(spike, lowcut, highcut, fs, order=order, zero_phase=zero_phase)

    y = y[n_samples:-n_samples]

    assert t.size == n_samples == y.size

    if rescale:
        y /= max(y.max(), -y.min())

    return t, y


#########################
# utils
#########################
def _warn_nyquist(f: float, dt: float):
    """当频率超过 Nyquist 频率时发出告警。

    Parameters
    ----------
    f : float
        待检查频率，单位 Hz。
    dt : float
        采样间隔，单位 s。

    Warns
    -----
    UserWarning
        当 ``f > 1 / (2 * dt)`` 时触发。
    """
    # assumes Hertz and seconds
    fN = 1 / (2 * dt)
    if f > fN:
        warnings.warn("%.1f Hz greater than Nyquist frequency (%.1f Hz)" % (f, fN))
