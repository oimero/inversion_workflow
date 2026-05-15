"""Merge depth-domain well and facies-control log-AI anchor bundles."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.seismic.survey import open_survey  # noqa: E402
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, write_json  # noqa: E402
from ginn_depth.facies import merge_well_and_facies_anchor_bundles  # noqa: E402
from ginn.anchor import load_log_ai_anchor_npz, save_log_ai_anchor_npz  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common_depth.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = cfg.get("merge_log_ai_anchor_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'merge_log_ai_anchor_depth' section in config.")

    data_root = resolve_relative_path(str(cfg.get("data_root", "data")), root=REPO_ROOT)
    output_root = resolve_relative_path(str(cfg.get("output_root", "scripts/output")), root=REPO_ROOT)
    well_anchor_file = resolve_relative_path(str(script_cfg["well_anchor_file"]), root=REPO_ROOT)
    facies_anchor_file = resolve_relative_path(str(script_cfg["facies_anchor_file"]), root=REPO_ROOT)
    seismic_file = resolve_relative_path(str(cfg["segy"]["file"]), root=data_root)
    min_sep = float(script_cfg.get("min_well_facies_separation_m", 0.0))
    for path in (well_anchor_file, facies_anchor_file, seismic_file):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    if args.output_dir is None:
        output_dir = output_root / f"merge_log_ai_anchor_depth_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "combined_log_ai_anchor_depth.npz"
    qc_path = output_dir / "combined_log_ai_anchor_depth_qc.csv"
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

    print("=== Merge Log-AI Anchor (Depth) ===")
    print(f"Well anchor: {well_anchor_file}")
    print(f"Facies anchor: {facies_anchor_file}")
    print(f"Min well/facies XY separation: {min_sep:.3f} m")
    print(f"Output dir: {output_dir}")

    well_bundle = load_log_ai_anchor_npz(well_anchor_file)
    facies_bundle = load_log_ai_anchor_npz(facies_anchor_file)
    result = merge_well_and_facies_anchor_bundles(
        well_bundle=well_bundle,
        facies_bundle=facies_bundle,
        survey=survey,
        min_well_facies_separation_m=min_sep,
        metadata={
            "source_script": Path(__file__).name,
            "path_style": "repo_relative",
            "well_anchor_file": repo_relative_path(well_anchor_file, root=REPO_ROOT),
            "facies_anchor_file": repo_relative_path(facies_anchor_file, root=REPO_ROOT),
            "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT),
            "qc_path": repo_relative_path(qc_path, root=REPO_ROOT),
        },
    )
    save_log_ai_anchor_npz(combined_path, result.bundle)
    result.qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    write_json(
        summary_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_script": Path(__file__).name,
            "artifact": {
                "path": repo_relative_path(combined_path, root=REPO_ROOT),
                "schema": result.bundle.schema_version,
                "summary": result.bundle.summary,
            },
            "inputs": {
                "path_style": "repo_relative",
                "well_anchor_file": repo_relative_path(well_anchor_file, root=REPO_ROOT),
                "facies_anchor_file": repo_relative_path(facies_anchor_file, root=REPO_ROOT),
                "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT),
                "min_well_facies_separation_m": min_sep,
            },
            "qc_path": repo_relative_path(qc_path, root=REPO_ROOT),
            "n_facies_kept": int((result.qc["status"] == "kept").sum()) if not result.qc.empty else 0,
            "n_facies_skipped": int((result.qc["status"] != "kept").sum()) if not result.qc.empty else 0,
        },
    )
    print(result.qc.to_string(index=False))
    print(f"Saved combined anchor: {combined_path}")
    print(f"Saved QC: {qc_path}")
    print(f"Saved run summary: {summary_path}")


if __name__ == "__main__":
    main()
