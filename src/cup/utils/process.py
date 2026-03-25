"""通用曲线处理工具。"""

from __future__ import annotations

from copy import deepcopy
from numbers import Real
from typing import cast

import numpy as np

from wtie.processing.grid import BaseTrace, Log, LogSet


def recompute_trace_basis(
    trace: BaseTrace,
    basis_start: float,
    sampling_rate: float | None = None,
) -> BaseTrace:
    """根据起点与采样率重建曲线 basis，并返回新对象。

    Parameters
    ----------
    trace : BaseTrace
            输入曲线对象，支持 BaseTrace 及其子类。
    basis_start : float
            新 basis 起点，对应 i=0 的采样坐标。
    sampling_rate : float or None, default=None
            新采样率。为 None 时沿用 ``trace.sampling_rate``。
            该值必须为大于 0 的数值。

    Returns
    -------
    BaseTrace
            与输入同类型的新曲线对象。values 及其它元信息保持不变，
            basis 按 ``b_i = b_0 + i * sampling_rate`` 重新计算。

    Raises
    ------
    TypeError
            当 ``trace`` 不是 BaseTrace（或其子类）实例，或输入参数不是数值时触发。
    ValueError
            当 ``sampling_rate`` 小于等于 0 时触发。

    Notes
    -----
    - 采样点数保持不变，仍为 ``len(trace)``。
    - 该函数返回新对象，不会原地修改输入对象。
    """
    if not isinstance(trace, BaseTrace):
        raise TypeError("trace must be an instance of BaseTrace or its subclass.")
    if not isinstance(basis_start, Real):
        raise TypeError("basis_start must be a numeric value.")

    sr = trace.sampling_rate if sampling_rate is None else sampling_rate
    if not isinstance(sr, Real):
        raise TypeError("sampling_rate must be a numeric value.")
    if sr <= 0:
        raise ValueError("sampling_rate must be greater than 0.")

    new_basis = float(basis_start) + np.arange(trace.size, dtype=float) * float(sr)

    new_trace = deepcopy(trace)
    new_trace.basis = new_basis

    # Keep geometric descriptors consistent with the updated basis.
    new_trace.sampling_rate = float(sr)
    new_trace.size = int(new_basis.size)
    new_trace.shape = new_trace.values.shape
    new_trace.duration = float(new_basis[-1] - new_basis[0])

    return new_trace


def recompute_logset_basis(
    logset: LogSet,
    basis_start: float,
    sampling_rate: float | None = None,
) -> LogSet:
    """根据起点与采样率重建 LogSet 内所有 Log 的 basis，并返回新对象。

    Parameters
    ----------
    logset : LogSet
            输入 LogSet 对象。内部每条 Log 均会按统一规则重建 basis。
    basis_start : float
            新 basis 起点，对应 i=0 的采样坐标。
    sampling_rate : float or None, default=None
            新采样率。为 None 时沿用 ``logset.sampling_rate``。
            该值必须为大于 0 的数值。

    Returns
    -------
    LogSet
            新的 LogSet 对象。每条 Log 的 values 与元信息保持不变，
            basis 按 ``b_i = b_0 + i * sampling_rate`` 统一重建。

    Raises
    ------
    TypeError
            当 ``logset`` 不是 LogSet 实例时触发。
    ValueError
            当 ``sampling_rate`` 小于等于 0 时触发。

    Notes
    -----
    - 采样点数保持不变。
    - 该函数返回新对象，不会原地修改输入对象。
    - 返回对象通过新 Log 集合重新构造，以保证 ``df``、快捷字段与几何属性一致。
    """
    if not isinstance(logset, LogSet):
        raise TypeError("logset must be an instance of LogSet.")

    new_logs = {
        key: cast(Log, recompute_trace_basis(log, basis_start=basis_start, sampling_rate=sampling_rate))
        for key, log in logset.Logs.items()
    }

    return LogSet(new_logs)
