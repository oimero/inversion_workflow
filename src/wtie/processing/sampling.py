"""采样与重采样工具。

该模块提供 1D 信号重采样相关接口，包括通用重采样类入口与降采样函数。

边界说明
--------
- 仅处理数组级采样变换，不负责道对象封装、I/O 或可视化。
- 不在本模块内统一管理物理单位换算，时间单位按调用方约定（常见为 s）。

核心公开对象
------------
1. downsample: 基于低通滤波 + decimate 的降采样实现。
2. Resampler: 线性重采样类接口（当前构造函数未实现）。

Examples
--------
>>> import numpy as np
>>> from wtie.processing.sampling import downsample
>>> x = np.sin(np.linspace(0, 4 * np.pi, 64))
>>> y = downsample(x, div_factor=2)
>>> y.shape
(32,)
"""

import numpy as np
from scipy.signal import decimate as _decimate
from scipy.signal import resample

from wtie.utils.types_ import Tuple


class Resampler:
    """一维线性重采样接口。

    该类设计用于将当前采样间隔 `current_dt` 的信号重采样到 `resampling_dt`。
    注意：当前 `__init__` 直接抛出 `NotImplementedError`，实例默认不可用。

    Attributes
    ----------
    div_factor : float
        采样间隔比值，定义为 ``resampling_dt / current_dt``。
        代码中该属性赋值位于异常之后，默认实例化流程下不会被设置（待确认是否为
        预留实现）。
    """

    def __init__(self, current_dt: float, resampling_dt: float):
        """初始化重采样器参数。

        Parameters
        ----------
        current_dt : float
            当前采样间隔 `dt`，单位通常为 s。
        resampling_dt : float
            目标采样间隔 `dt`，单位通常为 s。

        Raises
        ------
        NotImplementedError
            当前实现始终抛出该异常，实例化不会继续执行。
        """

        raise NotImplementedError()

        # division factor: n_samples_new = n_samples_old // div_factor
        self.div_factor = resampling_dt / current_dt

    def __call__(self, signal: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """执行一维信号重采样并返回新信号与新采样轴。

        Parameters
        ----------
        signal : numpy.ndarray
            输入一维信号，shape 为 `(n_samples,)`。
        t : numpy.ndarray
            输入采样轴，shape 为 `(n_samples,)`，应与 `signal` 对齐。

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            `(signal_resamp, t_resamp)`：
            - `signal_resamp` 为重采样后信号，shape 为 `(n_samples_new,)`
            - `t_resamp` 为重采样后采样轴，shape 为 `(n_samples_new,)`

        Notes
        -----
        其中 ``n_samples_new = int(len(signal) // self.div_factor)``。
        """
        n_samples_new = int(len(signal) // self.div_factor)
        signal_resamp, t_resamp = resample(signal, num=n_samples_new, t=t)
        return signal_resamp, t_resamp


def downsample(s: np.ndarray, div_factor: int) -> np.ndarray:
    """对一维信号执行降采样并做直流偏置校正。

    先调用 `scipy.signal.decimate` 完成低通滤波与抽取，再将输出均值平移到
    与输入信号均值一致。

    Parameters
    ----------
    s : numpy.ndarray
        输入一维信号，shape 为 `(n_samples,)`。
    div_factor : int
        降采样因子，要求 `div_factor > 1`。

    Returns
    -------
    numpy.ndarray
        降采样后信号，shape 约为 `(n_samples / div_factor,)`（由 decimate 决定）。

    Raises
    ------
    AssertionError
        当 `div_factor <= 1` 时触发。

    Examples
    --------
    >>> import numpy as np
    >>> s = np.linspace(0.0, 1.0, 20)
    >>> downsample(s, 2).shape
    (10,)
    """
    assert div_factor > 1
    # lowpass and decimate
    signal_resamp = _decimate(s, div_factor)
    # correct for DC bias
    signal_resamp = signal_resamp - signal_resamp.mean() + s.mean()
    return signal_resamp
