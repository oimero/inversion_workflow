"""Generic statistics and fitting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def pearson_r(
    x: np.ndarray, y: np.ndarray, *, mask: np.ndarray | None = None,
) -> float:
    """Pearson correlation coefficient, ignoring non-finite entries.

    Parameters
    ----------
    x, y : array-like
    mask : bool array, optional
        Extra boolean mask applied on top of the finite check.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    n = int(valid.sum())
    if n < 5:
        return np.nan
    xs = x[valid]
    ys = y[valid]
    if np.std(xs) <= 0.0 or np.std(ys) <= 0.0:
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient, ignoring non-finite entries."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 5:
        return np.nan
    rx = pd.Series(x[valid]).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y[valid]).rank(method="average").to_numpy(dtype=float)
    return pearson_r(rx, ry)


def ols_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Ordinary least-squares fit y = a + b*x.

    Returns a dict with keys *intercept*, *slope*, *n_samples*, *pearson_r*,
    *r2*, *mae*, *rmse*.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 5:
        raise ValueError("Need at least five finite samples for OLS fitting.")
    xv, yv = x[valid], y[valid]
    design = np.column_stack([np.ones_like(xv), xv])
    intercept, slope = np.linalg.lstsq(design, yv, rcond=None)[0]
    pred = intercept + slope * xv
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = np.nan if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "n_samples": int(valid.sum()),
        "pearson_r": pearson_r(x, y),
        "r2": float(r2),
        "mae": float(np.mean(np.abs(yv - pred))),
        "rmse": float(np.sqrt(np.mean((yv - pred) ** 2))),
    }


def normalized_mae(
    reference: np.ndarray, estimate: np.ndarray, *, mask: np.ndarray | None = None,
) -> float:
    """Normalised mean absolute error: |ref - est| / |ref| (L1)."""
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    valid = np.isfinite(ref) & np.isfinite(est)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if int(valid.sum()) < 5:
        return np.nan
    denom = float(np.sum(np.abs(ref[valid])))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(np.abs(ref[valid] - est[valid])) / denom)
