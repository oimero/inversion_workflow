"""cup.utils.config: 配置合并辅助工具。

本模块提供脚本配置段的默认值合并逻辑。

边界说明
--------
- 不依赖任何地球物理库。
- ``merge_dict_defaults`` 是就地修改，调用方如需保留原始配置应自行深拷贝。
- ``deep_merge_dict`` 返回新 dict，不修改输入。

核心公开对象
------------
1. merge_dict_defaults: 将默认值字典合并到配置 key 下（就地修改）。
2. deep_merge_dict: 递归合并两个 dict，返回新的合并结果。
3. require_latest_mode: 校验 ``source_runs.mode == "latest"``。
"""

from __future__ import annotations

from typing import Any, Mapping


def merge_dict_defaults(config: dict[str, Any], key: str, defaults: dict[str, Any]) -> None:
    """Merge *defaults* into ``config[key]`` in place.

    If ``config[key]`` is ``None`` or missing it is set to a copy of
    *defaults*.  If it is already a mapping, *defaults* are applied
    underneath it (existing keys kept).
    """
    value = config.get(key)
    if value is None:
        config[key] = dict(defaults)
        return
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping, got {type(value).__name__}.")
    merged = dict(defaults)
    merged.update(value)
    config[key] = merged


def deep_merge_dict(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict that recursively merges *updates* into *base*.

    Neither input is modified.  When a key exists in both dicts and both
    values are dicts, the merge is deep; otherwise the update value wins.
    """
    out = {key: (dict(value) if isinstance(value, dict) else value) for key, value in base.items()}
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def require_latest_mode(source_cfg: Mapping[str, Any], *, section: str) -> None:
    """Raise ``ValueError`` if ``source_runs.mode`` is not ``"latest"``.

    *section* is used in the error message to identify which config block is
    being checked (e.g. ``"well_constraints"``).
    """
    mode = str(source_cfg.get("mode", "latest")).strip().casefold()
    if mode != "latest":
        raise ValueError(f"{section}.source_runs.mode only supports 'latest' for now, got {mode!r}.")
