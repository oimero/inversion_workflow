"""Source-run and calibration artifact resolution for synthoseis-lite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from cup.synthetic.calibration import ImpedanceCalibration
from cup.synthetic.hashing import sha256_file
from cup.utils.io import resolve_artifact_path, resolve_relative_path


def resolve_sources(script_cfg: Mapping[str, Any], *, repo_root: Path) -> dict[str, Path]:
    sources = {
        key: resolve_relative_path(value, root=repo_root)
        for key, value in script_cfg["source_runs"].items()
    }
    required = {
        "forward_observability_dir": [
            "run_summary.json",
            "frequency_evidence_bands.csv",
            "well_frequency_sensitivity.csv",
        ],
        "well_preprocess_dir": ["well_preprocess_status.csv"],
        "well_auto_tie_dir": ["well_tie_metrics.csv"],
        "wavelet_generation_dir": [
            "selected_wavelet.csv",
            "selected_wavelet_summary.json",
            "evaluation_well_spatial_clusters.csv",
        ],
    }
    for key, names in required.items():
        directory = sources[key]
        if not directory.is_dir():
            raise FileNotFoundError(f"Source run does not exist: {directory}")
        missing = [name for name in names if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{key} is missing {missing}: {directory}")
    with (sources["forward_observability_dir"] / "run_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        summary = json.load(handle)
    recorded = summary.get("source_runs") or {}
    for key in ("well_preprocess_dir", "well_auto_tie_dir", "wavelet_generation_dir"):
        if key not in recorded:
            raise ValueError(f"source_run_mismatch: observability summary lacks {key}")
        if resolve_relative_path(recorded[key], root=repo_root).resolve() != sources[key].resolve():
            raise ValueError(f"source_run_mismatch:{key}")
    return sources

def _artifact(value: Any, *, run_dir: Path, repo_root: Path, label: str) -> Path:
    path = resolve_artifact_path(value, root=repo_root, run_dir=run_dir)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path

def load_calibration(path: Path) -> ImpedanceCalibration:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ImpedanceCalibration.from_dict(json.load(handle))
