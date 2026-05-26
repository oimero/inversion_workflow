"""cup.utils.coerce: 类型强制转换辅助工具。

本模块提供跨脚本复用的布尔/数值强制转换函数。

边界说明
--------
- 不依赖任何地球物理库。
- 输入安全性由调用方保证，本模块仅做类型转换。

核心公开对象
------------
1. as_bool: 将任意值强制转换为 bool。
2. optional_float: 将有限数值转换为 float，否则返回 None。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def as_bool(value: Any) -> bool:
    """Coerce a value to a strict boolean.

    Accepts Python ``bool`` directly. Strings ``"true"``, ``"1"``, ``"yes"``,
    ``"y"`` (case-insensitive) are ``True``; all other strings are ``False``.
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    return text in {"true", "1", "yes", "y"}


def optional_float(value: Any) -> float | None:
    """若输入可转换为有限浮点数则返回该数值，否则返回 ``None``。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
