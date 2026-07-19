"""Freeze the depth-domain forward model from rock physics and a Step-4 wavelet."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.config.sources import resolve_source_run
from cup.config.workflow import WorkflowConfig
from cup.physics.numpy_backend import forward_time
from cup.synthetic.schemas import (
    DEPTH_FORWARD_MODEL_INPUTS_RUN_SCHEMA_VERSION,
    FORWARD_MODEL_INPUTS_SCHEMA_VERSION,
    ROCK_PHYSICS_ANALYSIS_SCHEMA_VERSION,
)
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    is_consumable_contract_status,
    load_yaml_config,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_artifact_path,
    resolve_relative_path,
    resolve_timestamped_output_dir,
    write_json,
)
from cup.well.contracts import DEPTH_VERTICAL_AUTO_TIE_SCHEMA_VERSION


DEFAULT_COMMON_CONFIG = Path("experiments/common/common.yaml")
RELATION_SCHEMA = "rock_physics_relation_v1"
RUN_SCHEMA = DEPTH_FORWARD_MODEL_INPUTS_RUN_SCHEMA_VERSION
FORWARD_INPUTS_SCHEMA = FORWARD_MODEL_INPUTS_SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_COMMON_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], *, path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown keys: {unknown}.")


def _parse_config(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(
        config.get("depth_forward_model_inputs"),
        path="depth_forward_model_inputs",
    )
    _reject_unknown(
        root,
        {"source_runs", "source_well_name"},
        path="depth_forward_model_inputs",
    )
    source_runs = _mapping(
        root.get("source_runs") or {},
        path="depth_forward_model_inputs.source_runs",
    )
    _reject_unknown(
        source_runs,
        {"rock_physics_analysis_dir", "vertical_well_auto_tie_depth_dir"},
        path="depth_forward_model_inputs.source_runs",
    )
    source_well_name = str(root.get("source_well_name") or "").strip()
    if not source_well_name:
        raise ValueError(
            "depth_forward_model_inputs.source_well_name must be non-empty."
        )
    return {
        "source_well_name": source_well_name,
        "source_runs": {
            key: str(source_runs.get(key) or "").strip()
            for key in (
                "rock_physics_analysis_dir",
                "vertical_well_auto_tie_depth_dir",
            )
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_wavelet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["time_s", "amplitude"]:
        raise ValueError(
            "Wavelet CSV columns must be exactly ['time_s', 'amplitude']."
        )
    time_s = frame["time_s"].to_numpy(dtype=np.float64)
    amplitude = frame["amplitude"].to_numpy(dtype=np.float64)
    forward_time(np.array([0.0, 0.0], dtype=np.float64), time_s, amplitude)
    if amplitude.size < 3 or amplitude.size % 2 == 0:
        raise ValueError("Depth forward wavelet must have odd sample count >= 3.")
    zero_index = int(np.argmin(np.abs(time_s)))
    if zero_index != amplitude.size // 2 or not np.isclose(
        time_s[zero_index], 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Depth forward wavelet must be centered at zero time.")
    return time_s, amplitude


def _validate_relation(payload: Mapping[str, Any]) -> tuple[float, float, list[str]]:
    if payload.get("schema") != RELATION_SCHEMA:
        raise ValueError(f"Rock-physics relation must use {RELATION_SCHEMA}.")
    if payload.get("formula") != "AI = a * Vp + b":
        raise ValueError("Rock-physics relation formula is unsupported.")
    units = dict(payload.get("units") or {})
    expected_units = {
        "ai": "m/s*g/cm3",
        "vp": "m/s",
        "a": "g/cm3",
        "b": "m/s*g/cm3",
    }
    if units != expected_units:
        raise ValueError("Rock-physics relation units are invalid.")
    coefficients = dict(payload.get("coefficients") or {})
    a = float(coefficients.get("a"))
    b = float(coefficients.get("b"))
    if not np.isfinite(a) or a <= 0.0 or not np.isfinite(b):
        raise ValueError("Rock-physics relation coefficients are invalid.")
    accepted_wells = [str(value) for value in list(payload.get("accepted_wells") or [])]
    if not accepted_wells or len(accepted_wells) != len(set(accepted_wells)):
        raise ValueError("Rock-physics relation accepted_wells must be non-empty and unique.")
    return a, b, accepted_wells


def _resolve_sources(
    *,
    workflow: WorkflowConfig,
    script_config: Mapping[str, Any],
) -> dict[str, Path]:
    output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    configured = dict(script_config["source_runs"])
    source_well_name = str(script_config["source_well_name"])
    return {
        "rock_physics_analysis_dir": resolve_source_run(
            configured.get("rock_physics_analysis_dir") or None,
            output_root=output_root,
            prefix="rock_physics_analysis",
            required_files=[
                "run_summary.json",
                "modules/ai_vp_linear/rock_physics_relation.json",
            ],
            root=REPO_ROOT,
            label="rock_physics_analysis",
        ),
        "vertical_well_auto_tie_depth_dir": resolve_source_run(
            configured.get("vertical_well_auto_tie_depth_dir") or None,
            output_root=output_root,
            prefix="vertical_well_auto_tie_depth",
            required_files=[
                f"run_summary_{source_well_name}.json",
                f"wavelet_201ms_{source_well_name}.csv",
            ],
            root=REPO_ROOT,
            label="vertical_well_auto_tie_depth",
        ),
    }


def run(config_path: Path, output_dir_arg: Path | None = None) -> Path:
    config_file = resolve_relative_path(config_path, root=REPO_ROOT)
    raw_config = load_yaml_config(config_file)
    workflow = WorkflowConfig.from_mapping(raw_config)
    if workflow.seismic.domain != "depth" or workflow.seismic.depth_basis != "tvdss":
        raise ValueError(
            "depth_forward_model_inputs requires seismic.domain='depth' and "
            "seismic.depth_basis='tvdss'."
        )
    script_config = _parse_config(raw_config)
    source_well_name = str(script_config["source_well_name"])
    sources = _resolve_sources(workflow=workflow, script_config=script_config)

    rock_dir = sources["rock_physics_analysis_dir"]
    rock_summary_path = rock_dir / "run_summary.json"
    rock_summary = _load_json(rock_summary_path)
    if (
        rock_summary.get("schema") != ROCK_PHYSICS_ANALYSIS_SCHEMA_VERSION
        or not is_consumable_contract_status(rock_summary.get("status"))
        or rock_summary.get("sample_domain") != "depth"
        or rock_summary.get("depth_basis") != "tvdss"
    ):
        raise ValueError("Rock-physics source is not a successful depth/TVDSS run.")
    module_summary = dict(dict(rock_summary.get("modules") or {}).get("ai_vp_linear") or {})
    if not is_consumable_contract_status(module_summary.get("status")):
        raise ValueError("Rock-physics AI–Vp module did not succeed.")
    relation_path = rock_dir / "modules" / "ai_vp_linear" / "rock_physics_relation.json"
    recorded_relation_path = resolve_artifact_path(
        dict(dict(rock_summary.get("artifacts") or {}).get("rock_physics_relation") or {}).get("path"),
        root=REPO_ROOT,
        run_dir=rock_dir,
    )
    if recorded_relation_path is None or recorded_relation_path.resolve() != relation_path.resolve():
        raise ValueError("Rock-physics summary relation path does not match its artifact.")
    relation_payload = _load_json(relation_path)
    a, b, accepted_wells = _validate_relation(relation_payload)

    auto_tie_dir = sources["vertical_well_auto_tie_depth_dir"]
    auto_tie_summary_path = auto_tie_dir / f"run_summary_{source_well_name}.json"
    auto_tie_summary = _load_json(auto_tie_summary_path)
    if (
        auto_tie_summary.get("schema_version") != DEPTH_VERTICAL_AUTO_TIE_SCHEMA_VERSION
        or not is_consumable_contract_status(auto_tie_summary.get("status"))
        or str(auto_tie_summary.get("well_name") or "") != source_well_name
    ):
        raise ValueError("Depth auto-tie source does not match the configured source well.")
    wavelet_path = auto_tie_dir / f"wavelet_201ms_{source_well_name}.csv"
    wavelet_time_s, wavelet_amplitude = _load_wavelet(wavelet_path)

    input_contracts = {
        "rock_physics_analysis": {
            "path": repo_relative_path(rock_summary_path, root=REPO_ROOT),
            "contract_fingerprint_sha256": require_contract_fingerprint(
                rock_summary, label=f"rock physics run {rock_dir}"
            ),
        },
        "vertical_well_auto_tie_depth": {
            "path": repo_relative_path(auto_tie_summary_path, root=REPO_ROOT),
            "contract_fingerprint_sha256": require_contract_fingerprint(
                auto_tie_summary, label=f"depth auto-tie run {auto_tie_dir}"
            ),
        },
    }
    source_runs = {
        key: repo_relative_path(value, root=REPO_ROOT)
        for key, value in sources.items()
    }
    forward_inputs = {
        "schema": FORWARD_INPUTS_SCHEMA,
        "sample_domain": "depth",
        "depth_basis": "tvdss",
        "wavelet": {
            "source_well": source_well_name,
            "path": repo_relative_path(wavelet_path, root=REPO_ROOT),
            "time_unit": "s",
            "sample_count": int(wavelet_time_s.size),
            "dt_s": float(wavelet_time_s[1] - wavelet_time_s[0]),
            "l2_energy": float(np.sum(np.square(wavelet_amplitude))),
        },
        "ai_velocity_relation": {
            "path": repo_relative_path(relation_path, root=REPO_ROOT),
            "formula": "AI = a * Vp + b",
            "a": a,
            "b": b,
            "ai_unit": "m/s*g/cm3",
            "vp_unit": "m/s",
            "a_unit": "g/cm3",
            "b_unit": "m/s*g/cm3",
            "accepted_wells": accepted_wells,
        },
        "source_runs": source_runs,
        "input_contracts": input_contracts,
    }

    output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    output_dir = (
        resolve_relative_path(output_dir_arg, root=REPO_ROOT)
        if output_dir_arg is not None
        else resolve_timestamped_output_dir(output_root, "depth_forward_model_inputs")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    forward_inputs_path = output_dir / "forward_model_inputs.json"
    write_json(forward_inputs_path, forward_inputs)
    contract_fingerprint = contract_fingerprint_sha256(
        contract_schema_version=RUN_SCHEMA,
        semantics={
            "sample_domain": "depth",
            "sample_unit": "m",
            "depth_basis": "tvdss",
            "forward_inputs_schema": FORWARD_INPUTS_SCHEMA,
            "relation_schema": RELATION_SCHEMA,
        },
        business_config={"source_well_name": source_well_name},
        input_contracts=input_contracts,
        primary_artifacts={"forward_model_inputs": forward_inputs_path},
    )
    summary = {
        "schema": RUN_SCHEMA,
        "script": "depth_forward_model_inputs.py",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "success",
        "sample_domain": "depth",
        "sample_unit": "m",
        "depth_basis": "tvdss",
        "source_well_name": source_well_name,
        "source_runs": source_runs,
        "input_contracts": input_contracts,
        "artifacts": {
            "forward_model_inputs": {
                "path": repo_relative_path(forward_inputs_path, root=REPO_ROOT),
            },
        },
        "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
        "contract_fingerprint_sha256": contract_fingerprint,
    }
    write_json(output_dir / "run_summary.json", summary)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = run(args.config, args.output_dir)
    print(
        "Wrote depth forward-model inputs to "
        f"{repo_relative_path(output_dir, root=REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
