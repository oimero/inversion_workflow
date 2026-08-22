"""Sweep the GINN V2 body/residual Gaussian FWHM boundary on trusted wells.

Usage::

    python scripts/ginn_v2_body_fwhm_sweep.py
    python scripts/ginn_v2_body_fwhm_sweep.py --smoke
    python scripts/ginn_v2_body_fwhm_sweep.py --output-dir experiments/ginn_v2/results/body_fwhm_sweep_manual
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.config.workflow import WorkflowConfig, deep_merge_dict
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.seismic.target_zone_io import build_workflow_target_zone
from cup.utils.io import (
    load_yaml_config,
    repo_relative_path,
    resolve_relative_path,
    resolve_timestamped_output_dir,
)
from cup.utils.logging import configure_run_logger
from cup.well.body_fwhm_sweep import BodyFwhmSweepPolicy, run_body_fwhm_sweep
from cup.well.body_fwhm_sweep_artifacts import write_body_fwhm_sweep_artifacts
from cup.well.real_field_control_qc import load_depth_forward_inputs
from cup.well.real_field_controls import load_well_control_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/ginn_v2/ginn_v2.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run only the first configured trusted well.")
    return parser.parse_args()


def _load_composed_config(path: Path) -> dict[str, Any]:
    experiment = load_yaml_config(path)
    workflow_config = str(experiment.get("workflow_config") or "").strip()
    if not workflow_config:
        return experiment
    common = load_yaml_config(resolve_relative_path(workflow_config, root=REPO_ROOT))
    overlay = {key: value for key, value in experiment.items() if key != "workflow_config"}
    return deep_merge_dict(common, overlay)


def _resolve_experiment(raw: dict[str, Any]) -> tuple[dict[str, Any], Path, Path, Path]:
    config = dict(raw.get("ginn_v2_body_fwhm_sweep") or {})
    expected = {"inputs", "trusted_well_names", "sweep", "output_root"}
    if set(config) != expected:
        raise ValueError(f"ginn_v2_body_fwhm_sweep must contain exactly {sorted(expected)}.")
    inputs = dict(config["inputs"])
    expected_inputs = {"well_control_run_dir", "forward_model_inputs_run_dir"}
    if set(inputs) != expected_inputs:
        raise ValueError(f"body FWHM sweep inputs must contain exactly {sorted(expected_inputs)}.")
    well_control_run = resolve_relative_path(inputs["well_control_run_dir"], root=REPO_ROOT)
    forward_inputs_run = resolve_relative_path(inputs["forward_model_inputs_run_dir"], root=REPO_ROOT)
    output_root = resolve_relative_path(config["output_root"], root=REPO_ROOT)
    if not well_control_run.is_dir():
        raise FileNotFoundError(well_control_run)
    if not forward_inputs_run.is_dir():
        raise FileNotFoundError(forward_inputs_run)
    BodyFwhmSweepPolicy.from_mapping(dict(config["sweep"]))
    trusted = tuple(str(name).strip() for name in config["trusted_well_names"])
    if not trusted or any(not name for name in trusted):
        raise ValueError("trusted_well_names must contain non-empty names.")
    return config, well_control_run, forward_inputs_run, output_root


def _resolve_output_dir(args: argparse.Namespace, output_root: Path) -> Path:
    if args.output_dir is not None:
        return resolve_relative_path(args.output_dir, root=REPO_ROOT)
    return resolve_timestamped_output_dir(output_root, "body_fwhm_sweep")


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    raw = _load_composed_config(config_path)
    workflow = WorkflowConfig.from_mapping(raw)
    if workflow.seismic.domain != "depth" or workflow.seismic.depth_basis != "tvdss":
        raise ValueError("The configured sweep requires depth/TVDSS seismic.")
    experiment, well_control_run, forward_inputs_run, output_root = _resolve_experiment(raw)
    output_dir = _resolve_output_dir(args, output_root)
    if output_dir.exists():
        raise FileExistsError(f"Body FWHM sweep output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    log = configure_run_logger(
        output_dir,
        logger_name="ginn_v2_body_fwhm_sweep",
        file_name="sweep.log",
    )

    controls = load_well_control_set(well_control_run, repo_root=REPO_ROOT)
    trusted_well_names = tuple(str(name).strip() for name in experiment["trusted_well_names"])
    if args.smoke:
        trusted_well_names = trusted_well_names[:1]
    policy = BodyFwhmSweepPolicy.from_mapping(dict(experiment["sweep"]))
    wavelet_time_s, wavelet_amplitude, relation_a, relation_b, forward_inputs_path = (
        load_depth_forward_inputs(forward_inputs_run, repo_root=REPO_ROOT)
    )

    data_root = resolve_relative_path(workflow.data_root, root=REPO_ROOT)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    survey = open_survey(
        seismic_path,
        workflow.seismic.type,
        segy_options=segy_options_from_config(workflow.seismic.as_dict()),
    )
    target_zone, horizon_sources = build_workflow_target_zone(
        raw_config=raw,
        survey=survey,
        data_root=data_root,
        repo_root=REPO_ROOT,
    )
    log.info(
        "body FWHM sweep start | wells=%d | candidates=%s | output=%s",
        len(trusted_well_names),
        ",".join(f"{value:g}" for value in policy.fwhm_values_m),
        output_dir,
    )
    result = run_body_fwhm_sweep(
        controls,
        trusted_well_names=trusted_well_names,
        survey=survey,
        target_zone=target_zone,
        wavelet_time_s=wavelet_time_s,
        wavelet_amplitude=wavelet_amplitude,
        relation_a=relation_a,
        relation_b=relation_b,
        policy=policy,
        logger=log,
    )
    resolved_config = {
        "inputs": {
            "well_control_run_dir": repo_relative_path(well_control_run, root=REPO_ROOT),
            "forward_model_inputs_run_dir": repo_relative_path(forward_inputs_run, root=REPO_ROOT),
        },
        "trusted_well_names": list(trusted_well_names),
        "sweep": dict(experiment["sweep"]),
        "output_root": repo_relative_path(output_root, root=REPO_ROOT),
        "smoke": bool(args.smoke),
    }
    manifest = write_body_fwhm_sweep_artifacts(
        result,
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        resolved_config=resolved_config,
        inputs={
            "config": repo_relative_path(config_path, root=REPO_ROOT),
            "well_control_run": repo_relative_path(well_control_run, root=REPO_ROOT),
            "forward_model_inputs": repo_relative_path(forward_inputs_path, root=REPO_ROOT),
            "seismic": repo_relative_path(seismic_path, root=REPO_ROOT),
        },
        horizon_sources=horizon_sources,
    )
    log.info("body FWHM sweep complete | manifest=%s", output_dir / "manifest.json")
    print("=== GINN V2 body FWHM sweep ===")
    print(f"Output: {output_dir}")
    print(f"Wells: {len(result.wells)}")
    print(f"Candidates: {len(policy.fwhm_values_m)}")
    print(f"Summary: {REPO_ROOT / manifest['tables']['fwhm_summary']}")


if __name__ == "__main__":
    main()
