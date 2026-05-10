"""Build a depth-domain well-resolution prior for stage-2 enhancement.

The script samples shifted LAS AI logs on the depth GINN/LFM axis and stores the
log-AI residual against the AI LFM as a ``ginn_well_resolution_prior_v1`` NPZ.

Usage::

    python scripts/well_resolution_prior_depth.py
    python scripts/well_resolution_prior_depth.py --config experiments/common_depth.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lasio
import numpy as np
import pandas as pd

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.petrel.load import import_well_heads_petrel  # noqa: E402
from cup.seismic.survey import open_survey  # noqa: E402
from cup.utils.io import load_yaml_config, resolve_relative_path  # noqa: E402
from enhance.prior import (  # noqa: E402
    WellResolutionPriorBundle,
    save_well_resolution_prior_npz,
    summarize_well_resolution_prior,
)
from ginn_depth.data import load_lfm_depth_npz  # noqa: E402

# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/common_depth.yaml"),
        help="Depth-domain common config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <output_root>/well_resolution_prior_depth_<timestamp>.",
    )
    return parser.parse_args()


# LAS helpers


def _normalize_name(name: object) -> str:
    return str(name).strip()


def _find_curve(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    normalized = {str(col).strip().upper(): str(col) for col in df.columns}
    for name in names:
        hit = normalized.get(name.upper())
        if hit is not None:
            return hit
    return None


def _read_ai_from_shifted_las(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    las = lasio.read(path)
    df = las.df()
    md = df.index.to_numpy(dtype=float)
    ai_col = _find_curve(df, ("AI",))
    if ai_col is not None:
        ai = df[ai_col].to_numpy(dtype=float)
        return md, ai, "AI"

    vp_col = _find_curve(df, ("VP_MPS", "VP"))
    rho_col = _find_curve(df, ("RHO_GCC", "RHO"))
    if vp_col is None or rho_col is None:
        raise ValueError("LAS must contain AI or both VP_MPS/VP and RHO_GCC/RHO curves.")
    ai = df[vp_col].to_numpy(dtype=float) * df[rho_col].to_numpy(dtype=float)
    return md, ai, f"{vp_col}*{rho_col}"


def _confidence_from_corr(corr: object, *, floor: float, span: float) -> float:
    corr_value = _finite_float_or_nan(corr)
    if not np.isfinite(corr_value):
        return 0.0
    if span <= 0.0:
        raise ValueError(f"confidence_corr_span must be positive, got {span}.")
    return float(np.clip((corr_value - floor) / span, 0.0, 1.0))


def _finite_float_or_nan(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _interpolate_ai_to_depth_axis(
    md: np.ndarray,
    ai: np.ndarray,
    kb_m: float,
    depth_axis_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tvdss = np.asarray(md, dtype=float) - float(kb_m)
    ai = np.asarray(ai, dtype=float)
    valid = np.isfinite(tvdss) & np.isfinite(ai) & (ai > 0.0)
    if int(valid.sum()) < 2:
        raise ValueError("Too few finite positive AI samples after MD->TVDSS conversion.")

    tvdss = tvdss[valid]
    ai = ai[valid]
    order = np.argsort(tvdss)
    tvdss = tvdss[order]
    ai = ai[order]
    unique_tvdss, unique_idx = np.unique(tvdss, return_index=True)
    unique_ai = ai[unique_idx]
    if unique_tvdss.size < 2:
        raise ValueError("Too few unique TVDSS samples.")

    ai_on_axis = np.interp(depth_axis_m, unique_tvdss, unique_ai, left=np.nan, right=np.nan)
    mask = np.isfinite(ai_on_axis) & (ai_on_axis > 0.0)
    if int(mask.sum()) < 2:
        raise ValueError("Too few AI samples overlap the depth LFM axis.")
    return ai_on_axis.astype(np.float32), mask


def _nearest_flat_index(ai_lfm: Any, iline: float, xline: float) -> tuple[int, int, int, float, float]:
    il_idx = int(np.argmin(np.abs(ai_lfm.ilines - float(iline))))
    xl_idx = int(np.argmin(np.abs(ai_lfm.xlines - float(xline))))
    flat_idx = il_idx * int(ai_lfm.xlines.size) + xl_idx
    return flat_idx, il_idx, xl_idx, float(ai_lfm.ilines[il_idx]), float(ai_lfm.xlines[xl_idx])


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = cfg.get("well_resolution_prior_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'well_resolution_prior_depth' section in config.")

    data_root = resolve_relative_path(str(cfg.get("data_root", "data")), root=REPO_ROOT)
    output_root = resolve_relative_path(str(cfg.get("output_root", "scripts/output")), root=REPO_ROOT)

    source_batch_dir = resolve_relative_path(str(script_cfg["source_batch_dir"]), root=REPO_ROOT)
    shifted_las_dir = resolve_relative_path(
        str(script_cfg.get("shifted_las_dir", source_batch_dir / "shifted_las")),
        root=source_batch_dir,
    )
    metrics_path = resolve_relative_path(
        str(script_cfg.get("metrics_file", source_batch_dir / "wavelet_batch_metrics.csv")),
        root=source_batch_dir,
    )
    ai_lfm_file = resolve_relative_path(str(script_cfg["ai_lfm_file"]), root=REPO_ROOT)
    seismic_file = resolve_relative_path(str(cfg["segy"]["file"]), root=data_root)
    well_heads_file = resolve_relative_path(str(cfg["well"]["well_heads_file"]), root=data_root)

    for path in [seismic_file, well_heads_file, shifted_las_dir, metrics_path, ai_lfm_file]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"well_resolution_prior_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prior_path = output_dir / "well_resolution_prior_depth.npz"
    qc_path = output_dir / "well_resolution_prior_depth_qc.csv"
    run_summary_path = output_dir / "run_summary.json"

    confidence_corr_floor = float(script_cfg.get("confidence_corr_floor", 0.3))
    confidence_corr_span = float(script_cfg.get("confidence_corr_span", 0.4))

    segy_cfg = cfg["segy"]
    segy_options = {
        "iline": segy_cfg["iline_byte"],
        "xline": segy_cfg["xline_byte"],
        "istep": segy_cfg["istep"],
        "xstep": segy_cfg["xstep"],
    }

    print("=== Well Resolution Prior (Depth) ===")
    print(f"Shifted LAS dir: {shifted_las_dir}")
    print(f"Batch metrics: {metrics_path}")
    print(f"AI LFM: {ai_lfm_file}")
    print(f"Output dir: {output_dir}")

    ai_lfm = load_lfm_depth_npz(ai_lfm_file)
    depth_axis_m = ai_lfm.samples.astype(np.float32)
    well_heads_df = import_well_heads_petrel(well_heads_file)
    metrics_df = pd.read_csv(metrics_path)
    survey = open_survey(seismic_file, seismic_type="segy", segy_options=segy_options)

    heads_by_name = {_normalize_name(row["Name"]): row for _, row in well_heads_df.iterrows()}
    heads_by_upper = {_normalize_name(row["Name"]).upper(): row for _, row in well_heads_df.iterrows()}
    metrics_by_name = {_normalize_name(row["well_name"]): row for _, row in metrics_df.iterrows()}
    metrics_by_upper = {_normalize_name(row["well_name"]).upper(): row for _, row in metrics_df.iterrows()}

    las_paths = sorted(shifted_las_dir.glob("*.las"))
    if not las_paths:
        raise ValueError(f"No shifted LAS files found in {shifted_las_dir}")

    records: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for las_path in las_paths:
        well_name = las_path.stem
        well_name_upper = well_name.upper()
        qc: dict[str, Any] = {"well_name": well_name, "las_path": str(las_path), "status": "skipped"}
        try:
            head = heads_by_name.get(well_name)
            if head is None:
                head = heads_by_upper.get(well_name_upper)
            if head is None:
                raise ValueError("Missing well head row.")
            kb_m = float(head["Well datum value"])
            well_x = float(head["Surface X"])
            well_y = float(head["Surface Y"])
            if not np.isfinite(kb_m) or not np.isfinite(well_x) or not np.isfinite(well_y):
                raise ValueError("Non-finite KB or surface coordinates.")

            iline, xline = survey.coord_to_line(well_x, well_y)
            flat_idx, il_idx, xl_idx, nearest_inline, nearest_xline = _nearest_flat_index(ai_lfm, iline, xline)
            md, ai, ai_source = _read_ai_from_shifted_las(las_path)
            ai_on_axis, mask = _interpolate_ai_to_depth_axis(md, ai, kb_m, depth_axis_m)
            lfm_ai = np.asarray(ai_lfm.trace_by_index(il_idx, xl_idx), dtype=np.float32)
            lfm_valid = np.isfinite(lfm_ai) & (lfm_ai > 0.0)
            mask = mask & lfm_valid
            if int(mask.sum()) < 2:
                raise ValueError("Too few AI samples overlap finite positive LFM samples.")

            log_ai = np.zeros_like(ai_on_axis, dtype=np.float32)
            lfm_log_ai = np.zeros_like(lfm_ai, dtype=np.float32)
            residual_log_ai = np.zeros_like(ai_on_axis, dtype=np.float32)
            log_ai[mask] = np.log(np.clip(ai_on_axis[mask], 1e-6, None)).astype(np.float32)
            lfm_log_ai[mask] = np.log(np.clip(lfm_ai[mask], 1e-6, None)).astype(np.float32)
            residual_log_ai[mask] = (log_ai[mask] - lfm_log_ai[mask]).astype(np.float32)

            metric = metrics_by_name.get(well_name)
            if metric is None:
                metric = metrics_by_upper.get(well_name_upper)
            corr = np.nan if metric is None else _finite_float_or_nan(metric.get("corr", np.nan))
            confidence = _confidence_from_corr(
                corr,
                floor=confidence_corr_floor,
                span=confidence_corr_span,
            )
            weight = np.zeros_like(ai_on_axis, dtype=np.float32)
            weight[mask] = confidence

            records.append(
                {
                    "well_name": well_name,
                    "flat_idx": int(flat_idx),
                    "inline": float(nearest_inline),
                    "xline": float(nearest_xline),
                    "log_ai": log_ai,
                    "lfm_log_ai": lfm_log_ai,
                    "residual_log_ai": residual_log_ai,
                    "ai": np.nan_to_num(ai_on_axis, nan=0.0).astype(np.float32),
                    "lfm_ai": np.where(lfm_valid, lfm_ai, 0.0).astype(np.float32),
                    "mask": mask.astype(bool),
                    "weight": weight,
                }
            )
            qc.update(
                {
                    "status": "ok",
                    "ai_source": ai_source,
                    "kb_m": kb_m,
                    "well_x": well_x,
                    "well_y": well_y,
                    "resolved_inline": float(iline),
                    "resolved_xline": float(xline),
                    "nearest_inline": float(nearest_inline),
                    "nearest_xline": float(nearest_xline),
                    "flat_idx": int(flat_idx),
                    "corr": corr,
                    "confidence": confidence,
                    "valid_samples": int(mask.sum()),
                }
            )
        except Exception as exc:
            qc.update({"status": "failed", "error": str(exc)})
        qc_rows.append(qc)

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(qc_path, index=False)
    print(qc_df.to_string(index=False))

    if not records:
        raise ValueError(f"No successful wells; inspect {qc_path} before building the prior.")

    duplicates = [flat for flat, count in Counter(row["flat_idx"] for row in records).items() if count > 1]
    if duplicates:
        duplicate_wells = [row["well_name"] for row in records if row["flat_idx"] in duplicates]
        raise ValueError(f"Duplicate nearest traces are not supported in v1: {duplicates}, wells={duplicate_wells}")

    sample_step = float(np.median(np.diff(depth_axis_m))) if depth_axis_m.size > 1 else None
    well_mask = np.stack([row["mask"] for row in records]).astype(bool)
    well_weight = np.stack([row["weight"] for row in records]).astype(np.float32)
    residual_log_ai = np.stack([row["residual_log_ai"] for row in records]).astype(np.float32)
    summary = summarize_well_resolution_prior(
        residual_log_ai,
        well_mask,
        sample_step=sample_step,
        well_weight=well_weight,
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": Path(__file__).name,
        "shifted_las_dir": str(shifted_las_dir),
        "metrics_path": str(metrics_path),
        "well_heads_file": str(well_heads_file),
        "ai_lfm_file": str(ai_lfm_file),
        "confidence_formula": f"clip((corr - {confidence_corr_floor}) / {confidence_corr_span}, 0, 1)",
        "qc_path": str(qc_path),
        "n_input_las": int(len(las_paths)),
        "n_successful_wells": int(len(records)),
        "status_counts": qc_df["status"].value_counts(dropna=False).to_dict(),
    }

    bundle = WellResolutionPriorBundle(
        sample_domain="depth",
        sample_unit="m",
        samples=depth_axis_m.astype(np.float32),
        flat_indices=np.array([row["flat_idx"] for row in records], dtype=np.int64),
        well_log_ai=np.stack([row["log_ai"] for row in records]).astype(np.float32),
        lfm_log_ai=np.stack([row["lfm_log_ai"] for row in records]).astype(np.float32),
        residual_log_ai=residual_log_ai,
        well_ai=np.stack([row["ai"] for row in records]).astype(np.float32),
        lfm_ai=np.stack([row["lfm_ai"] for row in records]).astype(np.float32),
        well_mask=well_mask,
        well_weight=well_weight,
        well_names=np.array([row["well_name"] for row in records]),
        inline=np.array([row["inline"] for row in records], dtype=np.float32),
        xline=np.array([row["xline"] for row in records], dtype=np.float32),
        summary=summary,
        metadata=metadata,
    )

    save_well_resolution_prior_npz(prior_path, bundle)
    with run_summary_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                **metadata,
                "prior_path": str(prior_path),
                "summary": summary,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved prior: {prior_path}")
    print(f"Saved QC: {qc_path}")
    print(f"Saved run summary: {run_summary_path}")


if __name__ == "__main__":
    main()
