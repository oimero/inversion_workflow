"""Load the wavelet and physical relation used by seismic forward workflows."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np

from cup.physics.relations import AIVelocityRelation
from cup.seismic.wavelet import load_wavelet_csv, validate_wavelet_normalization
from cup.utils.io import resolve_relative_path


def load_forward_inputs(
    run_dir: Path,
    *,
    repo_root: Path,
    domain: str,
    depth_basis: str | None,
) -> tuple[np.ndarray, np.ndarray, AIVelocityRelation | None, dict[str, Any]]:
    """Read ``forward_model_inputs.json`` from a forward-input run directory."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    path = run_dir / "forward_model_inputs.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "forward_model_inputs_v3":
        raise ValueError("Forward inputs require forward_model_inputs_v3.")
    if payload.get("sample_domain") != domain or payload.get("depth_basis") != depth_basis:
        raise ValueError("Frozen forward inputs do not match the seismic SampleAxis domain.")
    wavelet_info = payload.get("wavelet")
    if not isinstance(wavelet_info, Mapping):
        raise ValueError("forward_model_inputs.wavelet must be a mapping.")
    wavelet_path = resolve_relative_path(
        str(wavelet_info.get("path") or ""),
        root=Path(repo_root),
    )
    time_s, amplitude = load_wavelet_csv(wavelet_path)
    amplitude, qc = validate_wavelet_normalization(
        time_s,
        amplitude,
        allow_small_renormalization=False,
    )
    if qc.status != "ok":
        raise ValueError(f"Frozen wavelet failed normalization QC: {qc.reasons}")
    relation_info = payload.get("ai_velocity_relation")
    if domain == "depth" and not isinstance(relation_info, Mapping):
        raise ValueError("Depth forward inputs must contain ai_velocity_relation.")
    relation = AIVelocityRelation.from_mapping(relation_info) if relation_info is not None else None
    return time_s, amplitude, relation, payload
