"""Prepare canonical real-field WellControlSet artifacts for every LFM variant.

Usage::

    python scripts/real_field_well_controls.py
    python scripts/real_field_well_controls.py --config experiments/my_project.yaml
    python scripts/real_field_well_controls.py --output-dir scripts/output/well_controls_test
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping



SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.config.workflow import WorkflowConfig, deep_merge_dict
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.seismic.target_zone_io import build_workflow_target_zone
from cup.seismic.wavelet_consensus import WAVELET_GENERATION_SCHEMA_VERSION
from cup.synthetic.schemas import FORWARD_MODEL_INPUTS_SCHEMA_VERSION
from cup.config.artifacts import is_consumable_contract_status, latest_checked_run
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, write_json
from cup.well.controls import (
    WELL_TIE_SOURCE_SCHEMA,
    build_well_control_set,
    write_well_control_qc,
    write_well_control_set,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_composed_config(path: Path) -> dict[str, Any]:
    experiment = load_yaml_config(path)
    workflow_config = str(experiment.get("workflow_config") or "").strip()
    if not workflow_config:
        return experiment
    common = load_yaml_config(resolve_relative_path(workflow_config, root=REPO_ROOT))
    overlay = {key: value for key, value in experiment.items() if key != "workflow_config"}
    return deep_merge_dict(common, overlay)


def _source_validator(*, domain: str, depth_basis: str | None):
    def validate(path: Path) -> None:
        with (path / "run_summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("schema_version") != WELL_TIE_SOURCE_SCHEMA:
            raise ValueError(f"schema_version is not {WELL_TIE_SOURCE_SCHEMA}")
        if summary.get("sample_domain") != domain or summary.get("depth_basis") != depth_basis:
            raise ValueError("source domain/depth_basis does not match workflow seismic")
        if not is_consumable_contract_status(summary.get("status")):
            raise ValueError("source run is not successful")

    return validate


def _resolve_source_run(config: dict[str, Any], workflow: WorkflowConfig) -> Path:
    source_runs = config.get("source_runs")
    if not isinstance(source_runs, Mapping):
        raise ValueError("real_field_well_controls.source_runs must be a mapping.")
    explicit = str(source_runs.get("well_auto_tie_dir") or "").strip()
    if explicit:
        return resolve_relative_path(explicit, root=REPO_ROOT)
    return latest_checked_run(
        resolve_relative_path(workflow.output_root, root=REPO_ROOT),
        "well_auto_tie",
        required_files=["run_summary.json", "well_tie_metrics.csv"],
        validator=lambda path: _source_validator(
            domain=workflow.seismic.domain, depth_basis=workflow.seismic.depth_basis
        )(path),
    )


def _resolved_config(raw: dict[str, Any], workflow: WorkflowConfig) -> dict[str, Any]:
    config = dict(raw.get("real_field_well_controls") or {})
    if not config:
        raise ValueError("Config lacks real_field_well_controls section.")
    config.pop("qc", None)
    source_run_dir = _resolve_source_run(config, workflow)
    with (source_run_dir / "run_summary.json").open("r", encoding="utf-8") as handle:
        source_summary = json.load(handle)
    source_runs = dict(config.get("source_runs") or {})

    def resolve_input_dir(explicit: Any, keys: tuple[str, ...], label: str) -> Path:
        text = str(explicit or "").strip()
        if text:
            path = resolve_relative_path(text, root=REPO_ROOT)
            if not path.is_dir():
                raise FileNotFoundError(f"{label}: {path}")
            return path
        inputs = dict(source_summary.get("inputs") or {})
        for key in keys:
            value = str(inputs.get(key) or "").strip()
            if not value:
                continue
            path = resolve_relative_path(value, root=REPO_ROOT)
            if path.is_file():
                path = path.parent
            if path.is_dir():
                return path
        raise FileNotFoundError(f"Cannot derive {label} from Step-4 run_summary.inputs.")

    source_runs["well_auto_tie_dir"] = str(source_run_dir)
    source_runs["well_preprocess_dir"] = str(resolve_input_dir(
        source_runs.get("well_preprocess_dir"),
        ("well_preprocess_dir",),
        "well_preprocess_dir",
    ))
    source_runs["well_inventory_dir"] = str(resolve_input_dir(
        source_runs.get("well_inventory_dir"),
        ("well_inventory_dir",),
        "well_inventory_dir",
    ))
    config["source_runs"] = source_runs
    config["well_trace_dir"] = workflow.assets.well_trace_dir
    config["time_depth_dir"] = workflow.assets.time_depth_dir
    inventory_path = Path(source_runs["well_inventory_dir"]) / "well_inventory.csv"
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    import pandas as pd
    inventory = pd.read_csv(inventory_path)
    if "well_name" not in inventory.columns:
        raise ValueError(f"well inventory is missing well_name: {inventory_path}")
    workflow.well_usage.validate_names(inventory["well_name"])
    metrics = pd.read_csv(source_run_dir / "well_tie_metrics.csv")
    if "well_name" not in metrics.columns or "tie_status" not in metrics.columns:
        raise ValueError("Step 4 well_tie_metrics.csv must contain well_name and tie_status.")
    metric_by_name = {
        str(row["well_name"]).strip().casefold(): row
        for _, row in metrics.iterrows()
    }
    plan_path = source_run_dir / "well_tie_plan.csv"
    plan = pd.read_csv(plan_path) if plan_path.is_file() else pd.DataFrame()
    plan_by_name = {
        str(row["well_name"]).strip().casefold(): row
        for _, row in plan.iterrows()
    } if "well_name" in plan.columns else {}
    explicit_trusted = workflow.well_usage.trusted_well_names is not None
    roles: dict[str, str] = {}
    role_reasons: dict[str, str] = {}
    for name in inventory["well_name"]:
        display = str(name).strip()
        key = display.casefold()
        central_role = workflow.well_usage.role_for(display)
        if central_role == "background_only":
            role = "background_only"
        elif explicit_trusted:
            role = central_role
        else:
            metric = metric_by_name.get(key)
            success = metric is not None and str(metric.get("tie_status") or "").casefold() == "success"
            role = "trusted" if success else "excluded"
        roles[display] = role
        if role == "excluded":
            metric = metric_by_name.get(key)
            plan_row = plan_by_name.get(key)
            reason = str(metric.get("reasons") or "").strip() if metric is not None else ""
            if reason.casefold() in {"nan", "none", "null"}:
                reason = ""
            if not reason and plan_row is not None:
                reason = str(plan_row.get("reasons") or "").strip()
                if reason.casefold() in {"nan", "none", "null"}:
                    reason = ""
            role_reasons[display] = reason or "step4_well_not_successful"
    config["well_roles"] = roles
    config["well_role_reasons"] = role_reasons
    config["central_trusted_well_names"] = (
        list(workflow.well_usage.trusted_well_names)
        if workflow.well_usage.trusted_well_names is not None else None
    )
    return config


def _resolve_qc_config(raw: dict[str, Any], workflow: WorkflowConfig) -> dict[str, Any]:
    section = raw.get("real_field_well_controls")
    qc = section.get("qc") if isinstance(section, Mapping) else None
    if not isinstance(qc, Mapping) or not qc:
        raise ValueError("Config requires a real_field_well_controls.qc mapping.")
    config = dict(qc)
    body = section.get("body_smoothing")
    if body is not None:
        if not isinstance(body, Mapping) or set(body) != {"fwhm_axis_units"}:
            raise ValueError("real_field_well_controls.body_smoothing must contain exactly fwhm_axis_units.")
        if "body_smoothing_fwhm_m" in config or "body_smoothing_fwhm_s" in config:
            raise ValueError("Configure body smoothing FWHM once, under real_field_well_controls.body_smoothing.")
        width_key = "body_smoothing_fwhm_m" if workflow.seismic.domain == "depth" else "body_smoothing_fwhm_s"
        config[width_key] = body["fwhm_axis_units"]
    output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    if workflow.seismic.domain == "depth":
        key, prefix = "forward_model_inputs_run_dir", "depth_forward_model_inputs"
        required = ["run_summary.json", "forward_model_inputs.json"]

        def validate(path: Path) -> None:
            with (path / "forward_model_inputs.json").open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if (payload.get("schema") != FORWARD_MODEL_INPUTS_SCHEMA_VERSION or payload.get("sample_domain") != "depth"
                    or payload.get("depth_basis") != "tvdss"):
                raise ValueError("QC forward inputs do not match the depth/TVDSS workflow.")
    else:
        key, prefix = "wavelet_generation_run_dir", "wavelet_generation"
        required = ["run_summary.json", "selected_wavelet.csv"]

        def validate(path: Path) -> None:
            with (path / "run_summary.json").open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if (payload.get("schema_version") != WAVELET_GENERATION_SCHEMA_VERSION
                    or not is_consumable_contract_status(payload.get("status"))
                    or payload.get("sample_domain") != "time" or payload.get("depth_basis") is not None):
                raise ValueError("QC wavelet run does not match the time-domain workflow.")
    explicit = str(config.get(key) or "").strip()
    if explicit:
        path = resolve_relative_path(explicit, root=REPO_ROOT)
        validate(path)
    else:
        path = latest_checked_run(
            output_root=output_root,
            prefix=prefix,
            required_files=required,
            validator=validate,
        )
    config[key] = str(path)
    return config


def _resolve_output_dir(args: argparse.Namespace, workflow: WorkflowConfig) -> Path:
    if args.output_dir is not None:
        return resolve_relative_path(args.output_dir, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    return output_root / f"real_field_well_controls_{timestamp}"


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    raw = _load_composed_config(config_path)
    workflow = WorkflowConfig.from_mapping(raw)
    config = _resolved_config(raw, workflow)
    data_root = resolve_relative_path(workflow.data_root, root=REPO_ROOT)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    survey = open_survey(
        seismic_path,
        workflow.seismic.type,
        segy_options=segy_options_from_config(workflow.seismic.as_dict()),
    )
    sample_axis = survey.sample_axis(workflow.seismic.domain)
    controls, manifest = build_well_control_set(
        config=config,
        sample_axis=sample_axis,
        line_geometry=survey.line_geometry,
        domain=workflow.seismic.domain,
        depth_basis=workflow.seismic.depth_basis,
        repo_root=REPO_ROOT,
        data_root=data_root,
        seismic_path=seismic_path,
    )
    output_dir = _resolve_output_dir(args, workflow)
    summary = write_well_control_set(
        controls,
        manifest,
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        resolved_config=config,
    )
    target_zone, _horizon_sources = build_workflow_target_zone(
        raw_config=raw,
        survey=survey,
        data_root=data_root,
        repo_root=REPO_ROOT,
    )
    qc_dir = None
    qc_config = _resolve_qc_config(raw, workflow)
    qc_dir = output_dir / "qc"
    qc_manifest = write_well_control_qc(
        controls,
        survey=survey,
        target_zone=target_zone,
        output_dir=qc_dir,
        repo_root=REPO_ROOT,
        config=qc_config,
    )
    summary["outputs"]["qc_manifest"] = repo_relative_path(qc_dir / "manifest.json", root=REPO_ROOT)
    summary["outputs"]["evaluation_support"] = qc_manifest["evaluation_support"]
    summary["qc"] = qc_manifest
    write_json(output_dir / "run_summary.json", summary)
    print("=== Real-field Well Controls ===")
    print(f"Output: {output_dir}")
    print(f"Successful wells: {summary['counts']['successful_wells']}")
    if qc_dir is not None:
        print(f"QC figures: {qc_dir / 'figures'}")


if __name__ == "__main__":
    main()
