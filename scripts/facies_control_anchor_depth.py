"""Build stage-1 log-AI anchors from depth-domain facies control points."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.seismic.facies_control_depth import build_target_layer_from_lfm_metadata, load_depth_facies_control_points_csv  # noqa: E402
from cup.seismic.survey import open_survey  # noqa: E402
from cup.utils.io import load_yaml_config, resolve_relative_path, to_json_compatible, write_json  # noqa: E402
from ginn.facies_anchor_depth import build_facies_control_anchor_bundle  # noqa: E402
from ginn.log_ai_anchor import save_log_ai_anchor_npz  # noqa: E402
from ginn_depth.data import load_lfm_depth_npz  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common_depth.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _jsonify_horizons(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "horizon_values" in out.columns:
        out["horizon_values"] = out["horizon_values"].map(lambda value: json.dumps(to_json_compatible(value), ensure_ascii=False))
    return out


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = cfg.get("facies_control_anchor_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'facies_control_anchor_depth' section in config.")

    data_root = resolve_relative_path(str(cfg.get("data_root", "data")), root=REPO_ROOT)
    output_root = resolve_relative_path(str(cfg.get("output_root", "scripts/output")), root=REPO_ROOT)
    ai_lfm_file = resolve_relative_path(str(script_cfg["reference_ai_lfm_file"]), root=REPO_ROOT)
    control_points_file = resolve_relative_path(str(script_cfg["control_points_file"]), root=REPO_ROOT)
    seismic_file = resolve_relative_path(str(cfg["segy"]["file"]), root=data_root)
    for path in (ai_lfm_file, control_points_file, seismic_file):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    if args.output_dir is None:
        output_dir = output_root / f"facies_control_anchor_depth_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    anchor_path = output_dir / "facies_control_log_ai_anchor_depth.npz"
    qc_path = output_dir / "facies_control_anchor_depth_qc.csv"
    summary_path = output_dir / "run_summary.json"
    segy_cfg = cfg["segy"]
    survey = open_survey(
        seismic_file,
        seismic_type="segy",
        segy_options={
            "iline": int(segy_cfg["iline_byte"]),
            "xline": int(segy_cfg["xline_byte"]),
            "istep": int(segy_cfg["istep"]),
            "xstep": int(segy_cfg["xstep"]),
        },
    )

    print("=== Facies Control Anchor (Depth) ===")
    print(f"Reference AI LFM: {ai_lfm_file}")
    print(f"Control points: {control_points_file}")
    print(f"Seismic: {seismic_file}")
    print(f"Output dir: {output_dir}")

    ai_lfm = load_lfm_depth_npz(ai_lfm_file)
    target_layer = build_target_layer_from_lfm_metadata(ai_lfm.metadata, ai_lfm.geometry, qc_output_dir=output_dir / "target_layer_qc")
    control_points = load_depth_facies_control_points_csv(control_points_file)
    result = build_facies_control_anchor_bundle(
        control_points=control_points,
        ilines=ai_lfm.ilines,
        xlines=ai_lfm.xlines,
        samples=ai_lfm.samples,
        target_layer=target_layer,
        survey=survey,
        metadata={
            "source_script": Path(__file__).name,
            "reference_ai_lfm_file": str(ai_lfm_file),
            "control_points_file": str(control_points_file),
            "seismic_file": str(seismic_file),
            "anchor_target_mode": "fixed_target_ai",
            "anchor_weight_mode": "constant_strength_inside_mask",
            "qc_path": str(qc_path),
        },
    )
    save_log_ai_anchor_npz(anchor_path, result.bundle)
    qc_for_csv = _jsonify_horizons(result.qc)
    qc_for_csv.to_csv(qc_path, index=False, encoding="utf-8-sig")
    write_json(
        summary_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_script": Path(__file__).name,
            "artifact": {
                "path": str(anchor_path),
                "schema": result.bundle.schema_version,
                "summary": result.bundle.summary,
            },
            "inputs": {
                "reference_ai_lfm_file": str(ai_lfm_file),
                "control_points_file": str(control_points_file),
                "seismic_file": str(seismic_file),
            },
            "qc_path": str(qc_path),
            "n_controls": int(len(control_points)),
        },
    )
    print(qc_for_csv.to_string(index=False))
    print(f"Saved facies-control anchor: {anchor_path}")
    print(f"Saved QC: {qc_path}")
    print(f"Saved run summary: {summary_path}")


if __name__ == "__main__":
    main()
