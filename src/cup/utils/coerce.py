"""cup.utils.coerce: Type coercion helpers shared across workflow scripts."""

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
    """Return ``float(value)`` if finite, otherwise ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
