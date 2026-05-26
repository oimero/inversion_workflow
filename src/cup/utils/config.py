"""cup.utils.config: Config merging helpers shared across workflow scripts."""

from __future__ import annotations

from typing import Any


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
