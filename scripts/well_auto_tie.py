"""Run Step 4 well auto-tie for time-domain or depth-domain seismic.

Usage::

    python scripts/well_auto_tie.py
    python scripts/well_auto_tie.py --config experiments/common/common.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.config.sources import resolve_source_run
from cup.config.workflow import WorkflowConfig, load_workflow_config, merge_dict_defaults
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.utils.io import repo_relative_path, write_json
from cup.well.pretrained import resolve_wtie_asset
from cup.well.petrel import import_well_tops_petrel
from cup.well.tie import WELL_AUTO_TIE_SCHEMA_VERSION, TieRoute, WellTiePlan, WellTieResult, plans_dataframe, results_dataframe
from cup.well.auto_tie import (
    resolve_repo_path, resolve_data_path, load_anchor_config, load_manual_shift_config,
    ensure_output_dirs, prepare_time_route, prepare_depth_route, execute_well,
    build_plans, select_plans, RerouteToAnchor, load_wavelet_extractor, write_wavelet_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--well", type=str, default=None, help="Optional single well execution filter.")
    return parser.parse_args()


# Configuration and upstream run selection

@dataclass(frozen=True)
class AutoTieInputs:
    inventory_dir: Path
    preprocess_dir: Path
    trajectory_dir: Path | None
    inventory_file: Path
    preprocess_status_file: Path
    preprocessed_las_dir: Path
    trajectory_file: Path | None

@dataclass(frozen=True)
class AutoTieSettings:
    raw: dict[str, Any]
    inputs: AutoTieInputs

    @property
    def search_space(self) -> dict[str, Any]:
        return dict(self.raw["search_space"])

    @property
    def search_params(self) -> dict[str, Any]:
        return dict(self.raw["search_params"])

    @property
    def wavelet_scaling(self) -> dict[str, Any]:
        return dict(self.raw["wavelet_scaling"])

    @property
    def reject(self) -> dict[str, Any]:
        return dict(self.raw["reject"])

def normalize_script_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and fill the local Step4 configuration defaults."""
    script_cfg = dict(cfg.get("well_auto_tie") or {})
    merge_dict_defaults(
        script_cfg,
        "source_runs",
        {
            "well_inventory_dir": None,
            "well_preprocess_dir": None,
            "well_trajectory_dir": None,
        },
    )
    merge_dict_defaults(
        script_cfg,
        "target_interval",
        {
            "top_horizon": "",
            "bottom_horizon": "",
            "margin_top_ms": 100.0,
            "margin_bottom_ms": 100.0,
            "margin_top_m": 100.0,
            "margin_bottom_m": 100.0,
            "twt_unit": "auto",
        },
    )
    script_cfg.setdefault("enabled_routes", None)
    script_cfg.setdefault("tutorial_model", None)
    script_cfg.setdefault("tutorial_params", None)
    script_cfg.setdefault("target_crop_ms", 201.0)
    merge_dict_defaults(
        script_cfg,
        "search_space",
        {
            "logs_median_size_values": [51, 71, 91, 111],
            "logs_median_threshold_bounds": [0.5, 3.0],
            "logs_std_bounds": [20, 50],
            "table_t_shift_bounds": [-0.030, 0.030],
        },
    )
    merge_dict_defaults(script_cfg, "search_params", {"num_iters": 60, "similarity_std": 0.02})
    merge_dict_defaults(
        script_cfg,
        "wavelet_scaling",
        {"min_scale": 50000, "max_scale": 500000, "num_iters": 60},
    )
    merge_dict_defaults(
        script_cfg,
        "reject",
        {
            "allow_near_outside": False,
            "min_tie_samples": 64,
            "max_trajectory_outside_fraction": 0.05,
            "max_short_log_gap_s": 0.010,
        },
    )
    merge_dict_defaults(
        script_cfg,
        "coarse_correction",
        {
            "anchor": {
                "enabled": True,
                "apply_to_routes": ["vertical_anchor_from_tops"],
                "config_file": "",
            },
            "manual_shift": {
                "default_ms": 0.0,
                "by_route_ms": {},
                "config_file": "",
            },
        },
    )
    if script_cfg["enabled_routes"] is None:
        script_cfg["enabled_routes"] = [
            "vertical_with_tdt",
            "vertical_anchor_from_tops",
            "deviated_with_tdt",
        ]
    script_cfg["enabled_routes"] = [str(route) for route in script_cfg["enabled_routes"]]
    return script_cfg

def resolve_inputs(
    cfg: dict[str, Any],
    script_cfg: dict[str, Any],
    *,
    repo_root: Path,
) -> AutoTieInputs:
    source_runs = dict(script_cfg.get("source_runs") or {})
    output_root = resolve_repo_path(str(cfg.get("output_root", "scripts/output")), repo_root=repo_root)
    inventory_dir = resolve_source_run(
        source_runs.get("well_inventory_dir"),
        output_root=output_root,
        prefix="well_inventory",
        required_files=["well_inventory.csv"],
        root=repo_root,
        label="well_inventory",
    )
    preprocess_dir = resolve_source_run(
        source_runs.get("well_preprocess_dir"),
        output_root=output_root,
        prefix="well_preprocess",
        required_files=["well_preprocess_status.csv"],
        root=repo_root,
        label="well_preprocess",
    )
    trajectory_value = source_runs.get("well_trajectory_dir")
    if trajectory_value is None:
        trajectory_dir = None
        try:
            trajectory_dir = resolve_source_run(
                None,
                output_root=output_root,
                prefix="well_trajectory",
                required_files=["well_trajectory.csv"],
                root=repo_root,
                label="well_trajectory",
            )
        except FileNotFoundError:
            trajectory_dir = None
    else:
        trajectory_dir = resolve_source_run(
            trajectory_value,
            output_root=output_root,
            prefix="well_trajectory",
            required_files=["well_trajectory.csv"],
            root=repo_root,
            label="well_trajectory",
        )
    trajectory_file = None if trajectory_dir is None else trajectory_dir / "well_trajectory.csv"
    if trajectory_file is not None and not trajectory_file.exists():
        raise FileNotFoundError(trajectory_file)
    return AutoTieInputs(
        inventory_dir=inventory_dir,
        preprocess_dir=preprocess_dir,
        trajectory_dir=trajectory_dir,
        inventory_file=inventory_dir / "well_inventory.csv",
        preprocess_status_file=preprocess_dir / "well_preprocess_status.csv",
        preprocessed_las_dir=preprocess_dir / "preprocessed_las",
        trajectory_file=trajectory_file,
    )

def load_settings(cfg: dict[str, Any], workflow: WorkflowConfig, *, repo_root: Path) -> AutoTieSettings:
    script_cfg = normalize_script_config(cfg)
    inputs = resolve_inputs(cfg, script_cfg, repo_root=repo_root)
    return AutoTieSettings(raw=script_cfg, inputs=inputs)

def read_input_tables(inputs: AutoTieInputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    inventory = pd.read_csv(inputs.inventory_file)
    preprocess = pd.read_csv(inputs.preprocess_status_file)
    trajectory = pd.read_csv(inputs.trajectory_file) if inputs.trajectory_file is not None else None
    return inventory, preprocess, trajectory


# Batch orchestration and workflow reports

def _output_dir(args_output_dir: Path | None, cfg: dict[str, Any], *, repo_root: Path) -> Path:
    if args_output_dir is not None:
        return resolve_repo_path(args_output_dir, repo_root=repo_root)
    output_root = resolve_repo_path(str(cfg.get("output_root", "scripts/output")), repo_root=repo_root)
    return output_root / f"well_auto_tie_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def _series_counts(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}

def _failed_result(plan: WellTiePlan, exc: Exception) -> WellTieResult:
    return WellTieResult(well_name=plan.well_name, route=plan.route, tie_status="failed", reasons=f"{type(exc).__name__}: {exc}")

def _metrics_with_extras(
    results: list[WellTieResult],
    extras: dict[str, dict[str, Any]],
    plans: list[WellTiePlan],
    *,
    domain: str,
    max_short_log_gap_s: float,
) -> pd.DataFrame:
    base = results_dataframe(results)
    plan_by_name = {plan.well_name.casefold(): plan for plan in plans}
    rows: list[dict[str, Any]] = []
    for result in results:
        extra = dict(extras.get(result.well_name) or {})
        report = dict(extra.get("tie_window") or {})
        plan = plan_by_name.get(result.well_name.casefold())
        row: dict[str, Any] = {
            "well_name": result.well_name,
            "route": result.route,
            "tie_status": result.tie_status,
            "sample_domain": domain,
            "max_short_log_gap_s": float(max_short_log_gap_s),
            "well_role": plan.well_role if plan is not None else "trusted",
            "tie_window_start_s": report.get("tie_window_start_s"),
            "tie_window_end_s": report.get("tie_window_end_s"),
            "tie_window_md_min_m": report.get("tie_window_md_min_m"),
            "tie_window_md_max_m": report.get("tie_window_md_max_m"),
            "target_top_tvdss_m": report.get("target_top_tvdss_m"),
            "target_bottom_tvdss_m": report.get("target_bottom_tvdss_m"),
            "filtered_las_file": extra.get("filtered_las_file"),
            "optimized_tdt_file": extra.get("optimized_tdt_file") or result.optimized_tdt_file,
            "seismic_trace_file": extra.get("seismic_trace_file"),
            "petrel_checkshot_file": extra.get("petrel_checkshot_file"),
            "optimized_trace_sample_plan_file": extra.get("optimized_trace_sample_plan_file"),
            "aligned_depth_file": extra.get("aligned_depth_file"),
            "source_depth_mapping_file": extra.get("source_depth_mapping_file"),
            "depth_seismic_trace_file": extra.get("depth_seismic_trace_file"),
        }
        rows.append(row)
    extra_df = pd.DataFrame.from_records(rows)
    if base.empty:
        return extra_df
    return base.merge(extra_df.drop(columns=["route", "tie_status"], errors="ignore"), on="well_name", how="left")

def _run_workflow(
    *,
    config_path: Path,
    output_dir: Path | None = None,
    well_name: str | None = None,
    repo_root: Path,
) -> Path:
    cfg = load_workflow_config(config_path, repo_root=repo_root)
    workflow = WorkflowConfig.from_mapping(cfg)
    settings = load_settings(cfg, workflow, repo_root=repo_root)
    script_cfg = settings.raw
    domain = workflow.seismic.domain
    if domain == "depth":
        if cfg.get("well_auto_tie", {}).get("enabled_routes") is None:
            script_cfg["enabled_routes"] = ["vertical_local_depth", "deviated_local_depth"]
    output = _output_dir(output_dir, cfg, repo_root=repo_root)
    output.mkdir(parents=True, exist_ok=False)
    ensure_output_dirs(output, sample_domain=domain)
    inventory, preprocess, trajectory = read_input_tables(settings.inputs)
    data_root = resolve_repo_path(workflow.data_root, repo_root=repo_root)
    time_depth_dir = resolve_data_path(workflow.assets.time_depth_dir, data_root=data_root)
    well_trace_dir = resolve_data_path(workflow.assets.well_trace_dir, data_root=data_root)
    plans = build_plans(
        inventory_df=inventory,
        preprocess_df=preprocess,
        trajectory_df=trajectory,
        time_depth_dir=time_depth_dir,
        well_trace_dir=well_trace_dir,
        enabled_routes=list(script_cfg["enabled_routes"]),
        workflow=workflow,
        sample_domain=domain,
        allow_near_outside=bool(script_cfg["reject"]["allow_near_outside"]),
    )
    plans = select_plans(plans, well_name)
    plan_df = plans_dataframe(plans)
    plan_path = output / "well_tie_plan.csv"
    plan_df.to_csv(plan_path, index=False)
    seismic_path = resolve_data_path(workflow.seismic.file, data_root=data_root)
    survey = open_survey(
        seismic_path,
        seismic_type=workflow.seismic.type,
        segy_options=segy_options_from_config(workflow.seismic.as_dict()) or None,
    )
    model_path = resolve_wtie_asset(script_cfg["tutorial_model"], filename="trained_net_state_dict.pt", data_root=data_root, repo_root=repo_root)
    params_path = resolve_wtie_asset(script_cfg["tutorial_params"], filename="network_parameters.yaml", data_root=data_root, repo_root=repo_root)
    wavelet_extractor = load_wavelet_extractor(model_path, params_path)
    script_cfg["wavelet_sampling_s"] = float(wavelet_extractor.expected_sampling)
    from wtie.modeling.modeling import ConvModeler

    modeler = ConvModeler()
    well_tops = pd.DataFrame()
    anchor_cfg = dict(script_cfg["coarse_correction"].get("anchor") or {})
    manual_cfg = load_manual_shift_config(
        dict(script_cfg["coarse_correction"].get("manual_shift") or {}), repo_root=repo_root
    )
    needs_anchor = domain == "time" and any(
        plan.route in {"vertical_anchor_from_tops", "deviated_anchor_from_tops"} or plan.route in set(anchor_cfg.get("apply_to_routes") or [])
        for plan in plans
        if plan.route_status == "planned"
    )
    well_tops_file = resolve_data_path(workflow.assets.well_tops_file, data_root=data_root)
    if needs_anchor:
        well_tops = import_well_tops_petrel(well_tops_file)
        anchor_cfg = load_anchor_config(anchor_cfg, repo_root=repo_root)
    horizon_cache: dict[str, Any] = {}
    results: list[WellTieResult] = []
    extras: dict[str, dict[str, Any]] = {}
    planned = [plan for plan in plans if plan.route_status == "planned"]
    for plan in planned:
        try:
            depth_prepared = None
            seismic_override = None
            optimized_trace_trajectory = None
            if domain == "depth":
                depth_prepared = prepare_depth_route(
                    plan=plan,
                    survey=survey,
                    config=script_cfg,
                    output_dir=output,
                    data_root=data_root,
                    repo_root=repo_root,
                    horizon_cache=horizon_cache,
                )
                prepared = depth_prepared.prepared
                seismic_override = depth_prepared.seismic
                extra_seed: dict[str, Any] = {"initial_table_source": "depth_vp", "target_depth_basis": "tvdss"}
            else:
                prepared, extra_seed, seismic_override, optimized_trace_trajectory = prepare_time_route(
                    plan=plan,
                    survey=survey,
                    config=script_cfg,
                    output_dir=output,
                    data_root=data_root,
                    repo_root=repo_root,
                    well_tops_df=well_tops,
                    anchor_config=anchor_cfg,
                    manual_shift_config=manual_cfg,
                    horizon_cache=horizon_cache,
                )
            result, extra = execute_well(
                plan=plan,
                prepared=prepared,
                survey=survey,
                wavelet_extractor=wavelet_extractor,
                modeler=modeler,
                config=script_cfg,
                output_dir=output,
                repo_root=repo_root,
                sample_domain=domain,
                seismic_override=seismic_override,
                extra_seed=extra_seed,
                depth_prepared=depth_prepared,
                input_las=resolve_repo_path(plan.input_las, repo_root=repo_root),
                optimized_trace_trajectory=optimized_trace_trajectory,
            )
            results.append(result)
            extras[plan.well_name] = extra
        except RerouteToAnchor as reroute:
            try:
                if domain != "time":
                    raise
                if well_tops.empty:
                    well_tops = import_well_tops_petrel(well_tops_file)
                anchor_cfg = load_anchor_config(anchor_cfg, repo_root=repo_root)
                rerouted = WellTiePlan(**{**plan.to_row(), "route": TieRoute.VERTICAL_ANCHOR_FROM_TOPS.value})
                prepared, extra_seed, seismic_override, optimized_trace_trajectory = prepare_time_route(
                    plan=rerouted,
                    survey=survey,
                    config=script_cfg,
                    output_dir=output,
                    data_root=data_root,
                    repo_root=repo_root,
                    well_tops_df=well_tops,
                    anchor_config=anchor_cfg,
                    manual_shift_config=manual_cfg,
                    horizon_cache=horizon_cache,
                    rerouted_from=plan.route,
                    reroute_reason=reroute.reason,
                )
                result, extra = execute_well(
                    plan=rerouted,
                    prepared=prepared,
                    survey=survey,
                    wavelet_extractor=wavelet_extractor,
                    modeler=modeler,
                    config=script_cfg,
                    output_dir=output,
                    repo_root=repo_root,
                    sample_domain=domain,
                    seismic_override=seismic_override,
                    extra_seed=extra_seed,
                    input_las=resolve_repo_path(plan.input_las, repo_root=repo_root),
                    optimized_trace_trajectory=optimized_trace_trajectory,
                )
                results.append(result)
                extras[plan.well_name] = extra
            except Exception as exc:
                results.append(_failed_result(plan, exc))
        except Exception as exc:
            results.append(_failed_result(plan, exc))
    metrics_df = _metrics_with_extras(
        results,
        extras,
        plans,
        domain=domain,
        max_short_log_gap_s=float(script_cfg["reject"]["max_short_log_gap_s"]),
    )
    metrics_path = output / "well_tie_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    tie_window_rows = []
    for result in results:
        report = dict(extras.get(result.well_name, {}).get("tie_window") or {})
        if report:
            tie_window_rows.append({"well_name": result.well_name, "route": result.route, "tie_status": result.tie_status, **report})
    pd.DataFrame.from_records(tie_window_rows).to_csv(output / "tie_window_report.csv", index=False)
    plan_df.loc[plan_df["route_status"].isin(["rejected", "skipped_role", "skipped_disabled"])].to_csv(output / "rejected_wells.csv", index=False)
    wavelet_df = write_wavelet_inventory(results, extras, repo_root=repo_root, output_dir=output)
    anchor_columns = [
        "well_name",
        "well_top",
        "horizon",
        "anchor_md_m",
        "anchor_x_m",
        "anchor_y_m",
        "anchor_position_source",
        "horizon_twt_raw",
        "anchor_twt_s",
        "inline_float",
        "xline_float",
        "horizon_sample_method",
        "horizon_nearest_line_distance",
        "horizon_nearest_inline",
        "horizon_nearest_xline",
        "anchor_role",
        "tdt_twt_at_anchor_s",
        "anchor_shift_ms",
    ]
    anchor_rows = []
    for result in results:
        anchor = extras.get(result.well_name, {}).get("anchor")
        if anchor:
            anchor_rows.append({"well_name": result.well_name, **dict(anchor)})
    anchor_path = output / "anchor_report.csv"
    pd.DataFrame.from_records(anchor_rows, columns=anchor_columns).to_csv(anchor_path, index=False)
    successful = int(sum(result.tie_status == "success" for result in results))
    run_summary = {
        "schema_version": WELL_AUTO_TIE_SCHEMA_VERSION,
        "status": "success" if successful > 0 else "failed",
        "sample_domain": domain,
        "sample_unit": "s" if domain == "time" else "m",
        "depth_basis": "tvdss" if domain == "depth" else None,
        "script": "well_auto_tie.py",
        "config_file": repo_relative_path(config_path, root=repo_root),
        "inputs": {
            "well_inventory_dir": repo_relative_path(settings.inputs.inventory_dir, root=repo_root),
            "well_inventory_file": repo_relative_path(settings.inputs.inventory_file, root=repo_root),
            "well_preprocess_dir": repo_relative_path(settings.inputs.preprocess_dir, root=repo_root),
            "well_trajectory_dir": None if settings.inputs.trajectory_dir is None else repo_relative_path(settings.inputs.trajectory_dir, root=repo_root),
            "seismic_file": repo_relative_path(seismic_path, root=repo_root),
            "well_tops_file": repo_relative_path(well_tops_file, root=repo_root),
            "tutorial_model": repo_relative_path(model_path, root=repo_root),
            "tutorial_params": repo_relative_path(params_path, root=repo_root),
        },
        "well_roles": {plan.well_name: plan.well_role for plan in plans},
        "enabled_routes": list(script_cfg["enabled_routes"]),
        "input_row_counts": {"inventory": int(len(inventory)), "preprocess_status": int(len(preprocess)), "well_trajectory": int(len(trajectory)) if trajectory is not None else 0},
        "route_counts": _series_counts(plan_df["route"]),
        "route_status_counts": _series_counts(plan_df["route_status"]),
        "tie_status_counts": _series_counts(metrics_df["tie_status"]) if "tie_status" in metrics_df.columns else {},
        "planned_run_count": len(planned),
        "successful_tie_count": successful,
        "wavelet_count": int(len(wavelet_df)),
        "paths": {
            "well_tie_plan": repo_relative_path(plan_path, root=repo_root),
            "well_tie_metrics": repo_relative_path(metrics_path, root=repo_root),
            "wavelet_inventory": repo_relative_path(output / "wavelet_inventory.csv", root=repo_root),
            "tie_window_report": repo_relative_path(output / "tie_window_report.csv", root=repo_root),
            "rejected_wells": repo_relative_path(output / "rejected_wells.csv", root=repo_root),
            "anchor_report": repo_relative_path(anchor_path, root=repo_root),
        },
    }
    write_json(output / "run_summary.json", run_summary)
    return output

def main() -> None:
    args = parse_args()
    output = _run_workflow(config_path=args.config, output_dir=args.output_dir,
                           well_name=args.well, repo_root=REPO_ROOT)
    print(f"Wrote auto-tie artifacts to {output}")


if __name__ == "__main__":
    main()
