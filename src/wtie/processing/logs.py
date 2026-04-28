"""wtie.processing.logs: 测井曲线基础预处理函数。

本模块提供 1D 测井序列的去尖峰、缺失值插值、高斯平滑与分段阻塞工具，
用于在井震对齐与反演前完成基础数值清洗与形态整饰。

边界说明
--------
- 本模块不负责 LAS/井数据文件读取、井名管理或单位体系转换。
- 本模块不包含反演优化策略、训练流程或可视化渲染逻辑。

核心公开对象
------------
1. despike: 基于中值滤波残差阈值的去尖峰入口函数。
2. interpolate_nans: 双向缺失值插值函数，尽量填补边界缺失。
3. smoothly_interpolate_nans: 去尖峰 + 平滑 + 插值的一体化缺失值回填函数。
4. smooth: 自动区分含 NaN/不含 NaN 场景的高斯平滑函数。
5. blocking: 基于相邻变化阈值与最大段长的分段均值化函数。

Examples
--------
>>> import numpy as np
>>> from wtie.processing.logs import smoothly_interpolate_nans
>>> x = np.array([1.0, np.nan, 3.0, 20.0, 5.0])
>>> y = smoothly_interpolate_nans(x, {'median_size': 3}, {'std': 1.0})
>>> y.shape
(5,)
"""

import numpy as np
import pandas as pd
from numba import njit

# from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

# from scipy.stats import hmean


def despike(
    data: np.ndarray,
    median_size: int = 31,
    threshold: float = 1.0,
    xmin_clip: float = None,  # type: ignore
    xmax_clip: float = None,  # type: ignore
) -> np.ndarray:
    """对一维曲线做去尖峰与可选阈值裁剪。

    先以中值滤波结果为基线计算残差噪声，再将超过阈值的样点置为 NaN。
    之后可按上下界做额外裁剪。

    Parameters
    ----------
    data : np.ndarray
        输入一维数组，shape 为 (n_samples,)。
    median_size : int, default=31
        中值滤波窗口长度，传递给 scipy.signal.medfilt。
    threshold : float, default=1.0
        噪声阈值系数，实际阈值为 ``threshold * nanstd(noise)``。
    xmin_clip : float or None, default=None
        下界裁剪值。仅当该参数在布尔意义上为 True 时生效。
    xmax_clip : float or None, default=None
        上界裁剪值。仅当该参数在布尔意义上为 True 时生效。

    Returns
    -------
    np.ndarray
        去尖峰后的数组，shape 为 (n_samples,)。被判定为异常点或越界点的位置为 NaN。

    Notes
    -----
    该函数不修改原始输入数组。
    """

    data = np.copy(data)
    med = medfilt(np.copy(data), median_size)
    noise = np.abs(data - med)
    threshold = threshold * np.nanstd(noise)  # type: ignore
    mask = np.abs(noise) > threshold
    data[mask] = np.nan
    if xmin_clip:
        data[data < xmin_clip] = np.nan
    if xmax_clip:
        data[data > xmax_clip] = np.nan
    return data


def interpolate_nans(x: np.ndarray, method: str = "linear", **kwargs) -> np.ndarray:
    """对 NaN/inf 执行双向插值，尽量填补序列两端空值。

    先按原方向插值，再对反转后的序列插值并反转回来，以减少单向插值
    对边界缺失值的残留。输入中的 ``inf`` / ``-inf`` 会先被视为 NaN。

    Parameters
    ----------
    x : np.ndarray
        输入一维数组，shape 为 (n_samples,)。
    method : str, default='linear'
        pandas.Series.interpolate 的插值方法名称，例如 'linear'、'slinear'。
    **kwargs
        透传给 pandas.Series.interpolate 的其他参数。

    Returns
    -------
    np.ndarray
        插值后的数组，shape 为 (n_samples,)。

    Raises
    ------
    ValueError
        当输入没有任何有限样点时抛出。
    """
    values = np.asarray(x, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    if np.isnan(values).all():
        raise ValueError("Cannot interpolate an all-NaN array.")

    interp_r = np.array(pd.Series(values).interpolate(method=method, **kwargs))  # type: ignore
    interp_l = np.array(pd.Series(interp_r[::-1]).interpolate(method=method, **kwargs))  # type: ignore
    return interp_l[::-1]
    # return np.array(pd.Series(x).interpolate(method=method,**kwargs))


def smoothly_interpolate_nans(
    x: np.ndarray, despike_params: dict, smooth_params: dict, method: str = "slinear"
) -> np.ndarray:
    """先去尖峰和平滑，再仅回填原始 NaN/inf 位置。

    处理流程为：despike -> smooth -> interpolate_nans。
    输入中的 ``inf`` / ``-inf`` 会先被视为 NaN。最终结果只会替换输入中
    原本为非有限值的样点，有限样点保持原值。

    Parameters
    ----------
    x : np.ndarray
        输入一维数组，shape 为 (n_samples,)。
    despike_params : dict
        传递给 despike 的参数字典。
    smooth_params : dict
        传递给 smooth 的参数字典。
    method : str, default='slinear'
        传递给 interpolate_nans 的插值方法名称。

    Returns
    -------
    np.ndarray
        与输入同 shape 的数组 (n_samples,)。
        仅输入中的 NaN/inf 位置会被填充，其他位置保持与输入一致。

    Raises
    ------
    ValueError
        当输入没有任何有限样点时抛出。
    """
    values = np.asarray(x, dtype=float).copy()
    missing_mask = ~np.isfinite(values)
    values[missing_mask] = np.nan
    if missing_mask.all():
        raise ValueError("Cannot interpolate an all-NaN array.")

    # interpolate on despiked
    x2 = despike(values, **despike_params)
    x2 = smooth(x2, **smooth_params)
    x2 = interpolate_nans(x2, method=method)

    # fill on original
    x3 = values.copy()
    x3[missing_mask] = x2[missing_mask]
    return x3


def smooth(x: np.ndarray, std: float = 1.0, mode="reflect", **kwargs) -> np.ndarray:
    """对一维序列执行高斯平滑，自动分流 NaN 与非 NaN 场景。

    Parameters
    ----------
    x : np.ndarray
        输入一维数组，shape 为 (n_samples,)。
    std : float, default=1.0
        高斯核标准差（以样点数为单位）。
    mode : str, default='reflect'
        边界处理模式，传递给 scipy.ndimage.gaussian_filter1d。
    **kwargs
        透传给 scipy.ndimage.gaussian_filter1d 的其他参数。

    Returns
    -------
    np.ndarray
        平滑后的数组，shape 为 (n_samples,)。
    """
    if np.isnan(x).any():
        return _nan_smooth(x, std, mode=mode, **kwargs)
    else:
        return _smooth(x, std, mode=mode, **kwargs)


def _smooth(x: np.ndarray, std: float = 1.0, mode="reflect", **kwargs) -> np.ndarray:
    """高斯平滑的无 NaN 快速路径。

    Parameters
    ----------
    x : np.ndarray
        输入一维数组，shape 为 (n_samples,)。
    std : float, default=1.0
        高斯核标准差（以样点数为单位）。
    mode : str, default='reflect'
        边界处理模式，传递给 scipy.ndimage.gaussian_filter1d。
    **kwargs
        透传给 scipy.ndimage.gaussian_filter1d 的其他参数。

    Returns
    -------
    np.ndarray
        平滑后的数组，shape 为 (n_samples,)。
    """
    return gaussian_filter1d(x, std, mode=mode, **kwargs)


def _nan_smooth(x: np.ndarray, std: float = 1.0, mode="reflect", **kwargs) -> np.ndarray:
    """对含 NaN 的序列做归一化加权高斯平滑。

    该实现先对值数组与权重数组分别做同样的高斯滤波，再用二者比值恢复
    有效样点的平滑结果，最后将原始 NaN 位置重新设为 NaN。

    Parameters
    ----------
    x : np.ndarray
        输入一维数组，shape 为 (n_samples,)。
    std : float, default=1.0
        高斯核标准差（以样点数为单位）。
    mode : str, default='reflect'
        边界处理模式，传递给 scipy.ndimage.gaussian_filter1d。
    **kwargs
        透传给 scipy.ndimage.gaussian_filter1d 的其他参数。

    Returns
    -------
    np.ndarray
        平滑后的数组，shape 为 (n_samples,)。原始 NaN 位置保持为 NaN。

    Notes
    -----
    参考思路见 https://stackoverflow.com/questions/18697532。
    """
    V = x.copy()
    V[np.isnan(x)] = 0
    VV = _smooth(V, std, mode=mode, **kwargs)

    W = 0.0 * x.copy() + 1.0
    W[np.isnan(x)] = 0.0
    WW = _smooth(W, std, mode=mode, **kwargs)

    Z = VV / (WW + 1e-10)
    Z[np.isnan(x)] = np.nan
    return Z


# @njit()
def blocking(x: np.ndarray, threshold: float, maximum_length: int, mean_type: str = "arithmetic") -> np.ndarray:
    """按变化阈值与最大段长将曲线分段并做段内均值化。

    Parameters
    ----------
    x : np.ndarray
        输入一维数组，shape 为 (n_samples,)。
    threshold : float
        相邻样点变化的相对阈值系数。当
        ``abs(x[i] - x[i-1]) > threshold * abs(x[i-1])`` 时触发新分段。
    maximum_length : int
        单段允许的最大长度上限（样点数）。超过后强制开启新分段。
    mean_type : {'arithmetic', 'harmonic'}, default='arithmetic'
        段内均值类型。'arithmetic' 为算术平均，'harmonic' 为调和平均。

    Returns
    -------
    np.ndarray
        阻塞后的数组，shape 为 (n_samples,)。每段内取同一均值。

    Raises
    ------
    ValueError
        当 mean_type 不是 'arithmetic' 或 'harmonic' 时抛出。
    """
    segments = _compute_block_segments(x, threshold, maximum_length)
    return _block_from_segments(x, tuple(segments), mean_type)


@njit()
def _compute_block_segments(x: np.ndarray, threshold: float, maximum_length: int) -> list:
    """计算 blocking 使用的分段起止索引序列。

    Parameters
    ----------
    x : np.ndarray
        输入一维数组，shape 为 (n_samples,)。
    threshold : float
        相邻变化相对阈值系数。
    maximum_length : int
        单段最大长度上限（样点数）。

    Returns
    -------
    list
        分段边界索引列表，包含首样点索引 0 和末样点索引 ``n_samples - 1``。
    """

    # find segements
    segments = [0]
    s_current = 0
    for i in range(1, x.size):
        cond1 = abs(x[i] - x[i - 1]) > threshold * abs(x[i - 1])
        cond2 = (i - s_current) > maximum_length
        if cond1 or cond2:
            segments.append(i)
            s_current = i

    if s_current != x.size - 1:
        segments.append(x.size - 1)

    return segments


@njit()
def _block_from_segments(x: np.ndarray, segments: tuple, mean_type: str = "arithmetic") -> np.ndarray:
    """根据给定分段边界执行段内均值替换。

    Parameters
    ----------
    x : np.ndarray
        输入一维数组，shape 为 (n_samples,)。
    segments : tuple
        分段边界索引元组，通常来自 _compute_block_segments。
    mean_type : {'arithmetic', 'harmonic'}, default='arithmetic'
        段内均值类型。'arithmetic' 为算术平均，'harmonic' 为调和平均。

    Returns
    -------
    np.ndarray
        阻塞后的数组，shape 为 (n_samples,)。

    Raises
    ------
    ValueError
        当 mean_type 不是 'arithmetic' 或 'harmonic' 时抛出。
    """
    x_blocked = np.copy(x)
    s = segments

    for i in range(len(s) - 1):
        x_seg = x[s[i] : s[i + 1]]
        if mean_type == "arithmetic":
            mean_ = np.mean(x_seg)
        elif mean_type == "harmonic":
            mean_ = (s[i + 1] - s[i]) / np.sum(1.0 / x_seg)
        else:
            raise ValueError

        x_blocked[s[i] : s[i + 1]] = mean_

    return x_blocked
