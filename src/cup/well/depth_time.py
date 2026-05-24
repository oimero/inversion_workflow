"""Depth/time helpers for time-domain well tying."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import lasio
import numpy as np
import pandas as pd

from wtie.processing import grid
from wtie.processing.logs import interpolate_nans


def _finite_positive(values: np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= 0.0] = np.nan
    if np.all(np.isnan(arr)):
        raise ValueError(f"{label} contains no positive finite samples.")
    return arr


def _read_petrel_checkshot_dataframe(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    in_data = False
    with Path(path).open("r", encoding="utf-8", errors="ignore") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            if line == "END HEADER":
                in_data = True
                continue
            if not in_data or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                rows.append(
                    {
                        "x_m": float(parts[0]),
                        "y_m": float(parts[1]),
                        "z_m": float(parts[2]),
                        "twt_ms": float(parts[3]),
                        "md_m": float(parts[4]),
                        "well_name": parts[5].strip('"'),
                    }
                )
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric Petrel checkshot rows found in {path}.")
    return pd.DataFrame.from_records(rows)


def read_time_depth_table(path: str | Path, *, domain: Literal["md", "tvdss"] = "md") -> grid.TimeDepthTable:
    """Read a Petrel checkshot/time-depth table into a wtie table.

    Petrel exports in this project store picked TWT as negative milliseconds.
    The time-domain workflow uses positive seconds, so this reader applies
    ``abs(twt_ms) / 1000`` and sorts by the requested depth domain.
    """
    path = Path(path)
    df = _read_petrel_checkshot_dataframe(path)
    domain = str(domain).strip().lower()  # type: ignore[assignment]
    if domain not in {"md", "tvdss"}:
        raise ValueError(f"Unsupported TDT domain: {domain}.")

    depth_col = "md_m" if domain == "md" else "z_m"
    depth = df[depth_col].to_numpy(dtype=np.float64)
    if domain == "tvdss":
        depth = np.abs(depth)
    twt = np.abs(df["twt_ms"].to_numpy(dtype=np.float64)) / 1000.0

    finite = np.isfinite(depth) & np.isfinite(twt)
    depth = depth[finite]
    twt = twt[finite]
    if depth.size < 2:
        raise ValueError(f"Time-depth table has fewer than 2 valid samples: {path}")

    order = np.argsort(depth)
    depth = depth[order]
    twt = twt[order]

    # Keep the first sample for duplicate depths. Petrel exports are dense and
    # occasionally contain repeated values after rounding.
    unique_depth, unique_indices = np.unique(depth, return_index=True)
    depth = unique_depth
    twt = twt[unique_indices]
    if depth.size < 2:
        raise ValueError(f"Time-depth table has fewer than 2 unique depth samples: {path}")

    if np.any(np.diff(depth) < 0.0):
        raise ValueError(f"Time-depth table depth is not monotonic after sorting: {path}")

    # Some Petrel exports contain a short shallow turnaround or millisecond
    # rounding plateaus.  wtie requires strictly increasing TWT, so preserve the
    # physically usable monotonic segment instead of rejecting the whole well.
    start = int(np.nanargmin(twt))
    depth = depth[start:]
    twt = twt[start:]
    keep = np.zeros(twt.shape, dtype=bool)
    last_twt = -np.inf
    for index, value in enumerate(twt):
        if value > last_twt + 1e-9:
            keep[index] = True
            last_twt = float(value)
    depth = depth[keep]
    twt = twt[keep]
    if depth.size < 2:
        raise ValueError(f"Time-depth table has fewer than 2 strictly increasing TWT samples: {path}")
    if np.any(np.diff(twt) <= 0.0):
        raise ValueError(f"Time-depth table TWT is not strictly increasing after monotonic filtering: {path}")

    if domain == "md":
        return grid.TimeDepthTable(twt=twt, md=depth)
    return grid.TimeDepthTable(twt=twt, tvdss=depth)


def build_vp_rho_logset_from_preprocessed_las(path: str | Path) -> grid.LogSet:
    """Build MD-domain Vp/Rho LogSet from third-step standard LAS."""
    las = lasio.read(str(path))
    df = las.df()
    missing = [name for name in ("DT_USM", "RHO_GCC") if name not in df.columns]
    if missing:
        raise ValueError(f"Preprocessed LAS is missing required curves {missing}: {path}")

    md = np.asarray(df.index.to_numpy(dtype=np.float64), dtype=np.float64)
    dt_usm = _finite_positive(df["DT_USM"].to_numpy(dtype=np.float64), label="DT_USM")
    rho = _finite_positive(df["RHO_GCC"].to_numpy(dtype=np.float64), label="RHO_GCC")
    vp = 1_000_000.0 / dt_usm

    vp = interpolate_nans(vp, method="linear")
    rho = interpolate_nans(rho, method="linear")
    if np.any(~np.isfinite(vp)) or np.any(~np.isfinite(rho)):
        raise ValueError(f"Vp/Rho still contain non-finite samples after interpolation: {path}")

    return grid.LogSet(
        {
            "Vp": grid.Log(vp, md, "md", name="Vp", unit="m/s", allow_nan=False),
            "Rho": grid.Log(rho, md, "md", name="Rho", unit="g/cm3", allow_nan=False),
        }
    )


def validate_time_depth_table(
    table: grid.TimeDepthTable,
    log_basis_md: np.ndarray,
    *,
    min_overlap_samples: int = 64,
) -> dict[str, float | int]:
    """Validate MD overlap between a table and log basis."""
    if not table.is_md_domain:
        raise ValueError("vertical_with_tdt expects an MD-domain TimeDepthTable.")
    log_md = np.asarray(log_basis_md, dtype=np.float64)
    table_md = np.asarray(table.md, dtype=np.float64)
    overlap_min = max(float(log_md[0]), float(table_md[0]))
    overlap_max = min(float(log_md[-1]), float(table_md[-1]))
    if overlap_max <= overlap_min:
        raise ValueError(
            f"Log MD range [{log_md[0]}, {log_md[-1]}] does not overlap TDT MD range [{table_md[0]}, {table_md[-1]}]."
        )
    count = int(np.count_nonzero((log_md >= overlap_min) & (log_md <= overlap_max)))
    if count < int(min_overlap_samples):
        raise ValueError(f"Too few log samples overlap TDT: {count} < {min_overlap_samples}.")
    return {
        "overlap_md_min_m": overlap_min,
        "overlap_md_max_m": overlap_max,
        "overlap_log_sample_count": count,
    }


def write_time_depth_table_csv(table: grid.TimeDepthTable, path: str | Path) -> None:
    """Write a wtie TimeDepthTable to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_name = "md_m" if table.is_md_domain else "tvdss_m"
    pd.DataFrame({"twt_s": table.twt, depth_name: table.depth}).to_csv(path, index=False)
