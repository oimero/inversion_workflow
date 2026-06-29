"""Explicit adapters between ``wtie.processing.grid`` traces and NumPy kernels."""

from __future__ import annotations

from typing import Any

import numpy as np

from cup.physics.numpy_backend import forward_depth, forward_time


def _require_trace(trace: Any, *, name: str) -> None:
    required = ("values", "basis", "basis_type")
    missing = [attribute for attribute in required if not hasattr(trace, attribute)]
    if missing:
        raise TypeError(f"{name} must be a grid trace; missing attributes {missing}.")
    values = np.asarray(trace.values)
    basis = np.asarray(trace.basis)
    if values.ndim != 1 or basis.ndim != 1 or values.shape != basis.shape:
        raise ValueError(f"{name} must contain matching one-dimensional values and basis.")


def _require_basis(trace: Any, *, name: str, expected: str) -> None:
    flag_name = f"is_{expected}"
    if not bool(getattr(trace, flag_name, False)):
        raise ValueError(
            f"{name} must use {expected} basis, got {getattr(trace, 'basis_type', None)!r}."
        )


def _require_unit(trace: Any, *, name: str, expected: str) -> None:
    actual = getattr(trace, "unit", None)
    if actual != expected:
        raise ValueError(f"{name}.unit must be {expected!r}, got {actual!r}.")


def forward_time_log(
    log_ai: Any,
    wavelet_time_s: Any,
    wavelet_amp: Any,
    *,
    name: str = "Synthetic",
):
    """Forward one TWT ``grid.Log`` and return a lower-sample ``grid.Seismic``."""
    from wtie.processing import grid

    _require_trace(log_ai, name="log_ai")
    _require_basis(log_ai, name="log_ai", expected="twt")
    values = forward_time(log_ai.values, wavelet_time_s, wavelet_amp)
    basis = np.asarray(log_ai.basis)[1:]
    if basis.size < 2:
        raise ValueError("grid time adapter requires at least three log_ai samples.")
    return grid.Seismic(
        values,
        basis,
        "twt",
        name=name,
        allow_nan=False,
    )


def forward_depth_log(
    log_ai: Any,
    velocity_mps: Any,
    wavelet_time_s: Any,
    wavelet_amp: Any,
    *,
    output_chunk_size: int = 64,
    name: str = "Synthetic",
):
    """Forward aligned TVDSS log/velocity traces and return ``grid.Seismic``."""
    from wtie.processing import grid

    _require_trace(log_ai, name="log_ai")
    _require_trace(velocity_mps, name="velocity_mps")
    _require_basis(log_ai, name="log_ai", expected="tvdss")
    _require_basis(velocity_mps, name="velocity_mps", expected="tvdss")
    _require_unit(velocity_mps, name="velocity_mps", expected="m/s")
    log_basis = np.asarray(log_ai.basis)
    velocity_basis = np.asarray(velocity_mps.basis)
    if not np.array_equal(log_basis, velocity_basis):
        raise ValueError("log_ai and velocity_mps must have exactly matching TVDSS axes.")
    values = forward_depth(
        log_ai.values,
        velocity_mps.values,
        log_basis,
        wavelet_time_s,
        wavelet_amp,
        output_chunk_size=output_chunk_size,
    )
    return grid.Seismic(
        values,
        log_basis,
        "tvdss",
        name=name,
        allow_nan=False,
    )


__all__ = ["forward_depth_log", "forward_time_log"]
