"""Create standard Step 4/5 artifacts for the Marmousi2 adapter.

Marmousi2 pseudo-wells have an exact model TWT table, so this adapter uses the
main Wtie evaluation primitives directly and publishes the same artifact
contracts consumed by later workflow stages.  It does not invoke the two CLI
scripts as subprocesses.
"""

from pathlib import Path
import json
import os
import shutil

import lasio
import numpy as np
import pandas as pd
import yaml

from cup.utils.io import repo_relative_path, write_json
from cup.well.pretrained import resolve_wtie_asset
from cup.well.tie import (
    TieEvaluationWell,
    WellTiePlan,
    prepare_continuous_tie_logs,
    prepare_well_for_evaluation,
    results_dataframe,
    WELL_AUTO_TIE_SCHEMA_VERSION,
)
from cup.well.las import load_standard_vp_rho_logs
from cup.well.trajectory import PreparedTieWindow
from cup.seismic.wavelet import wavelet_l2_normalize
from ginn_v3.workflow import load_config
from marmousi2.seismic import training_well_correlations
from wtie.processing import grid
from wtie.modeling.modeling import ConvModeler


def prepare_tie_inputs(prepared_dir: Path, *, repo_root: Path, pretrained_dir: Path | None = None) -> Path:
    """Export standard LAS and Petrel TDT without filtering known model curves."""
    prepared_dir, repo_root = Path(prepared_dir).resolve(), Path(repo_root).resolve()
    adapter = prepared_dir / "wavelet_inputs"
    relative = lambda p: repo_relative_path(p, root=repo_root)
    directories = {name: adapter / name for name in ("well_inventory", "well_screen", "well_preprocess", "time_depth", "interpre")}
    directories["las"] = directories["well_preprocess"] / "preprocessed_las"
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(prepared_dir / "evaluation" / "well_roles.csv")
    training = roles.loc[roles.role == "train"]
    inventory, preprocess = [], []
    for well in training.itertuples():
        source = lasio.read(prepared_dir / "assets" / "las" / f"{well.well_name}.las")
        standard = lasio.LASFile()
        standard.well.WELL = well.well_name
        standard.append_curve("DEPT", source["DEPT"], unit="m")
        standard.append_curve("DT_USM", 1e6 / source["VP"], unit="us/m")
        standard.append_curve("RHO_GCC", source["RHOB"], unit="g/cm3")
        standard.append_curve("AI", source["AI"], unit="m/s*g/cm3")
        las_path = directories["las"] / f"{well.well_name}.las"
        standard.write(str(las_path), version=2.0)
        tdt = pd.read_csv(prepared_dir / "assets" / "time_depth" / f"{well.well_name}.csv")
        lines = ["BEGIN HEADER", "X", "Y", "Z", "MD", "TWT", "Well", "END HEADER"]
        lines += [f"{well.x_m:.8g} 0 {-row.depth_m:.8g} {row.depth_m:.8g} {-row.twt_s * 1000:.10g} {well.well_name}"
                  for row in tdt.itertuples()]
        (directories["time_depth"] / f"{well.well_name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        inventory.append({"well_name": well.well_name, "survey_position": "inside", "wellbore_class": "vertical",
                          "surface_x": well.x_m, "surface_y": 0.0, "kb_m": 0.0,
                          "has_time_depth": True, "has_well_trace": False, "has_well_tops": False})
        preprocess.append({"well_name": well.well_name, "preprocess_status": "passed", "usable_p_sonic": True,
                           "usable_density": True, "preprocessed_las": relative(las_path)})
    pd.DataFrame(inventory).to_csv(directories["well_inventory"] / "well_inventory.csv", index=False)
    pd.DataFrame({"well_name": training.well_name}).to_csv(directories["well_screen"] / "well_screen.csv", index=False)
    pd.DataFrame(preprocess).to_csv(directories["well_preprocess"] / "well_preprocess_status.csv", index=False)
    with np.load(prepared_dir / "evaluation" / "truth.npz", allow_pickle=False) as saved:
        x, times = saved["x_m"], saved["twt_s"]
    preparation = json.loads((prepared_dir / "preparation_summary.json").read_text(encoding="utf-8"))
    target_top, target_bottom = preparation["target_interval_s"]
    for name, value in (("top", target_top), ("bottom", target_bottom)):
        lines = [f"INLINE : 0 XLINE : {i} {position:.8g} 0 {value:.8g}" for i, position in enumerate(x)]
        (directories["interpre"] / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = load_config(prepared_dir / "ginn_v3.yaml")
    config["assets"]["time_depth_dir"] = str(directories["time_depth"].relative_to(prepared_dir)).replace("\\", "/")
    public_wtie_root = Path(pretrained_dir).resolve() if pretrained_dir is not None else repo_root / "opendata" / "pretrained" / "wtie"
    config["well_auto_tie"] = {
        "source_runs": {f"{name}_dir": relative(directories[name]) for name in ("well_inventory", "well_screen", "well_preprocess")},
        "enabled_routes": ["vertical_with_tdt"],
        "target_interval": {"top_horizon": "wavelet_inputs/interpre/top.txt", "bottom_horizon": "wavelet_inputs/interpre/bottom.txt",
                            "margin_top_ms": 0, "margin_bottom_ms": 0, "twt_unit": "s"},
        # ``well_auto_tie.py`` resolves these against data_root.  Keeping the
        # paths relative to the prepared directory references the single
        # public copy under opendata/pretrained instead of duplicating it in
        # every prepared run.
        "tutorial_model": Path(os.path.relpath(public_wtie_root / "trained_net_state_dict.pt", prepared_dir)).as_posix(),
        "tutorial_params": Path(os.path.relpath(public_wtie_root / "network_parameters.yaml", prepared_dir)).as_posix(),
        "target_crop_ms": 201.0,
        "search_space": {"logs_median_size_values": [1], "logs_median_threshold_bounds": [100.0, 101.0],
                         "logs_std_bounds": [0.001, 0.002], "table_t_shift_bounds": [0.0, 0.0]},
        "search_params": {"num_iters": 24, "similarity_std": 0.02},
        "wavelet_scaling": {"min_scale": 0.1, "max_scale": 100.0, "num_iters": 16},
        "coarse_correction": {"anchor": {"enabled": False, "config_file": None},
                              "manual_shift": {"config_file": None, "default_ms": 0}},
    }
    names = training.well_name.tolist()
    config["wavelet_generation"] = {
        "source_runs": {"well_auto_tie_dir": relative(prepared_dir / "step4_well_auto_tie")},
        "candidate_filter": {"min_source_tie_corr": 0.0, "include_source_wells": names},
        "evaluation_wells": {"include_wells": names},
        "scoring": {"min_eval_well_count": 3, "on_insufficient_eval_wells": "select_best_source_tie"},
        "generation": {"optimizer": {"random_trials": 256, "max_refine_iters": 80, "seed": 20261004}},
    }
    path = prepared_dir / "ginn_v3_steps45.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _network_wtie_artifacts(
    prepared_dir: Path,
    *,
    repo_root: Path,
    pretrained_dir: Path | None,
) -> dict[str, object]:
    """Run the main Wtie neural extractor with fixed model TDTs.

    The exact Marmousi2 TDT is supplied as the initial table and the generic
    auto-tie search is constrained to zero table shift.  Wtie still performs
    neural wavelet extraction and its normal convolutional scoring; no ridge
    estimate is used on this default path.
    """
    import importlib.util

    module_spec = importlib.util.spec_from_file_location(
        "mero_main_well_auto_tie",
        repo_root / "scripts" / "well_auto_tie.py",
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError("Could not load the main well_auto_tie library adapter.")
    auto_tie = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(auto_tie)
    prepared_dir = Path(prepared_dir).resolve()
    repo_root = Path(repo_root).resolve()
    roles = pd.read_csv(prepared_dir / "evaluation" / "well_roles.csv")
    training = roles.loc[roles["role"].astype(str).str.casefold().eq("train")].copy()
    with np.load(prepared_dir / "evaluation" / "truth.npz", allow_pickle=False) as saved:
        times = np.asarray(saved["twt_s"], dtype=np.float64)
    with np.load(prepared_dir / "seismic.npz", allow_pickle=False) as saved:
        seismic = np.asarray(saved["seismic"], dtype=np.float64)
    if seismic.ndim == 3 and seismic.shape[0] == 1:
        seismic = seismic[0]
    model_path = resolve_wtie_asset(
        None if pretrained_dir is None else Path(pretrained_dir) / "trained_net_state_dict.pt",
        filename="trained_net_state_dict.pt",
        data_root=prepared_dir,
        repo_root=repo_root,
    )
    params_path = resolve_wtie_asset(
        None if pretrained_dir is None else Path(pretrained_dir) / "network_parameters.yaml",
        filename="network_parameters.yaml",
        data_root=prepared_dir,
        repo_root=repo_root,
    )
    with params_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    expected_dt = float(params["synthetic_dataset"]["dt"])
    preparation = json.loads((prepared_dir / "preparation_summary.json").read_text(encoding="utf-8"))
    target_top, target_bottom = preparation["target_interval_s"]
    wavelet_extractor = auto_tie._load_wavelet_extractor(model_path, params_path)
    modeler = ConvModeler()
    step4 = prepared_dir / "step4_well_auto_tie"
    step5 = prepared_dir / "step5_wavelet_generation"
    auto_tie._ensure_output_dirs(step4)
    for directory in (step5 / "synthetic_qc", step5 / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    results = []
    extras: dict[str, dict[str, object]] = {}
    plan_rows: list[dict[str, object]] = []
    for row in training.itertuples(index=False):
        name = str(row.well_name)
        safe = name.replace("/", "_").replace("\\", "_")
        filtered_las = prepared_dir / "wavelet_inputs" / "well_preprocess" / "preprocessed_las" / f"{name}.las"
        tdt_path = prepared_dir / "assets" / "time_depth" / f"{name}.csv"
        plan = WellTiePlan(
            well_name=name,
            route="vertical_with_tdt",
            route_status="planned",
            wellbore_class_initial="vertical",
            wellbore_class_qc="vertical",
            has_time_depth=True,
            has_well_trace=False,
            has_well_tops=False,
            usable_p_sonic=True,
            usable_density=True,
            input_las=repo_relative_path(filtered_las, root=repo_root),
            time_depth_file=repo_relative_path(tdt_path, root=repo_root),
            well_trace_file="",
            surface_x=float(row.x_m),
            surface_y=0.0,
            kb_m=0.0,
            reasons="exact_model_tdt",
        )
        standard_logs = load_standard_vp_rho_logs(filtered_las)
        source_tdt = pd.read_csv(prepared_dir / "assets" / "time_depth" / f"{name}.csv")
        table = grid.TimeDepthTable(
            twt=source_tdt["twt_s"].to_numpy(dtype=np.float64),
            md=source_tdt["depth_m"].to_numpy(dtype=np.float64),
        )
        continuous = prepare_continuous_tie_logs(
            standard_logs,
            table,
            window_start_s=float(target_top),
            window_end_s=float(target_bottom),
            max_short_gap_s=0.010,
            min_tie_samples=64,
            seismic_sample_interval_s=float(times[1] - times[0]),
        )
        table_rows = pd.DataFrame({
            "twt_s": table.twt,
            "md_m": table.md,
            "source": "original_tdt",
        })
        prepared = PreparedTieWindow(
            table=table,
            logset_md=continuous.logs,
            table_rows=table_rows,
            report={
                "target_top_name": "experiment_top",
                "target_bottom_name": "experiment_bottom",
                "target_top_twt_s": float(target_top),
                "target_bottom_twt_s": float(target_bottom),
                "target_window_start_s": float(target_top),
                "target_window_end_s": float(target_bottom),
                "tie_window_start_s": float(continuous.start_twt_s),
                "tie_window_end_s": float(continuous.end_twt_s),
            },
        )
        from wtie.optimize import autotie, tie as tie_ops

        paths = auto_tie._build_output_paths(step4, name)
        paths["figure_dir"].mkdir(parents=True, exist_ok=True)
        seismic_trace = grid.Seismic(seismic[int(row.profile_index)], times, "twt", name=name)
        params = {
            "logs_median_size": 1,
            "logs_median_threshold": 100.0,
            "logs_std": 0.001,
            "table_t_shift": 0.0,
        }
        outputs = autotie._intermediate_tie_v1(
            continuous.logs,
            None,
            table,
            seismic_trace,
            wavelet_extractor,
            modeler,
            params,
        )
        wavelet = tie_ops.compute_wavelet(
            outputs.seismic,
            outputs.r,
            modeler,
            wavelet_extractor,
            zero_phasing=False,
            scaling=True,
            expected_value=False,
            scaling_params={"wavelet_min_scale": 0.1, "wavelet_max_scale": 100.0, "num_iters": 16},
        )
        cropped_wavelet, _crop_info = auto_tie.crop_wavelet_center_energy_normalize(wavelet, 201.0)
        pd.DataFrame({"twt_s": outputs.seismic.basis, "seismic": outputs.seismic.values}).to_csv(paths["seismic_trace"], index=False)
        auto_tie.write_time_depth_table_csv(table, paths["initial_tdt"])
        auto_tie.write_time_depth_table_csv(table, paths["optimized_tdt"])
        pd.DataFrame({"time_s": cropped_wavelet.basis, "amplitude": cropped_wavelet.values}).to_csv(paths["wavelet"], index=False)
        filtered_las_path = paths["filtered_las"]
        shutil.copy2(filtered_las, filtered_las_path)
        seismic_norm, synthetic, optimized_corr, optimized_nmae, synthetic_scale = auto_tie.scaled_synthetic_metrics(
            modeler, cropped_wavelet, outputs.r, outputs.seismic,
        )
        pd.DataFrame({
            "twt_s": outputs.seismic.basis,
            "seismic_norm": seismic_norm,
            "reflectivity": outputs.r.values,
            "synthetic_cropped_scaled": synthetic,
            "residual": seismic_norm - synthetic,
        }).to_csv(paths["synthetic_qc"], index=False)
        result = auto_tie.WellTieResult(
            well_name=name,
            route="vertical_with_tdt",
            tie_status="success",
            initial_corr=float(optimized_corr),
            optimized_corr=float(optimized_corr),
            optimized_nmae=float(optimized_nmae),
            best_table_shift_ms=0.0,
            wavelet_file=repo_relative_path(paths["wavelet"], root=repo_root),
            optimized_tdt_file=repo_relative_path(paths["optimized_tdt"], root=repo_root),
            qc_figure_dir=repo_relative_path(paths["figure_dir"], root=repo_root),
            reasons="exact_model_tdt;wtie_neural_extractor;table_shift_fixed",
        )
        extra = {
            "initial_table_source": "exact_model_tdt",
            "table_t_shift_bounds": [0.0, 0.0],
            "synthetic_scale": float(synthetic_scale),
            "seismic_trace_file": repo_relative_path(paths["seismic_trace"], root=repo_root),
            "filtered_las_file": repo_relative_path(filtered_las_path, root=repo_root),
            "tie_window": prepared.report,
        }
        results.append(result)
        extras[name] = extra
        plan_rows.append(plan.to_row())
    plan_path = step4 / "well_tie_plan.csv"
    pd.DataFrame.from_records(plan_rows).to_csv(plan_path, index=False)
    metrics = results_dataframe(results)
    metrics.to_csv(step4 / "well_tie_metrics.csv", index=False)
    auto_tie._write_wavelet_inventory(results, step4)
    pd.DataFrame.from_records([
        {
            "well_name": result.well_name,
            "target_top_name": "experiment_top",
            "target_bottom_name": "experiment_bottom",
            "target_top_twt_s": float(target_top),
            "target_bottom_twt_s": float(target_bottom),
            "tie_window_start_s": float(extras[result.well_name]["tie_window"]["tie_window_start_s"]),
            "tie_window_end_s": float(extras[result.well_name]["tie_window"]["tie_window_end_s"]),
        }
        for result in results
    ]).to_csv(step4 / "tie_window_report.csv", index=False)
    pd.DataFrame(columns=["well_name", "reason"]).to_csv(step4 / "rejected_wells.csv", index=False)
    step4_summary = {
        "schema_version": WELL_AUTO_TIE_SCHEMA_VERSION,
        "status": "success",
        "route": "vertical_with_tdt",
        "calibration_method": "wtie_neural_wavelet_extractor_fixed_model_tdt",
        "wavelet_extractor": {
            "model": repo_relative_path(model_path, root=repo_root),
            "parameters": repo_relative_path(params_path, root=repo_root),
            "expected_sampling_s": expected_dt,
        },
        "table_t_shift_bounds_s": [0.0, 0.0],
        "well_count": len(results),
        "successful_tie_count": len(results),
        "outputs": {
            "well_tie_plan": repo_relative_path(plan_path, root=repo_root),
            "well_tie_metrics": repo_relative_path(step4 / "well_tie_metrics.csv", root=repo_root),
            "wavelet_inventory": repo_relative_path(step4 / "wavelet_inventory.csv", root=repo_root),
            "tie_window_report": repo_relative_path(step4 / "tie_window_report.csv", root=repo_root),
        },
    }
    write_json(step4 / "run_summary.json", step4_summary)
    _network_wavelet_consensus(step4, step5, repo_root=repo_root, seed=20261004)
    return {
        "step4": step4,
        "step5": step5,
        "results": results,
        "metrics": metrics,
        "times": times,
        "expected_dt_s": expected_dt,
        "model_path": model_path,
        "params_path": params_path,
    }


def _network_wavelet_consensus(
    step4: Path,
    step5: Path,
    *,
    repo_root: Path,
    seed: int,
) -> dict[str, object]:
    """Run the main consensus optimizer over Wtie candidate wavelets."""
    from cup.seismic.wavelet import load_wavelet_csv
    from cup.seismic.wavelet_consensus import (
        ConsensusSearchPolicy,
        build_wavelet_pca_basis,
        optimize_consensus_wavelet,
    )
    from cup.well.tie import (
        evaluate_wavelet_on_well,
        load_tie_artifacts,
        prepare_well_for_evaluation,
    )

    index = load_tie_artifacts(step4, repo_root=repo_root)
    candidates = index.candidate_wavelets()
    wells = index.evaluation_wells(status="success")
    if not candidates or not wells:
        raise ValueError("Wtie consensus requires at least one candidate and one successful evaluation well.")
    wavelet_time, first_wavelet = load_wavelet_csv(candidates[0].wavelet_file)
    wavelet_time = np.asarray(wavelet_time, dtype=np.float64)
    candidate_arrays = []
    candidate_names = []
    for candidate in candidates:
        candidate_time, values = load_wavelet_csv(candidate.wavelet_file)
        candidate_arrays.append(np.interp(wavelet_time, candidate_time, values))
        candidate_names.append(candidate.wavelet_file.stem)
    matrix = np.asarray(candidate_arrays, dtype=np.float64)
    dt_s = float(np.median(np.diff(wavelet_time)))
    well_cache = {
        well.well_name: prepare_well_for_evaluation(well, dt_s=dt_s)
        for well in wells
    }
    modeler = ConvModeler()

    def score_wavelet(values: np.ndarray, *, candidate_name: str) -> tuple[dict[str, object], list[dict[str, object]]]:
        metric_rows: list[dict[str, object]] = []
        for well in wells:
            seismic_match, reflectivity_match = well_cache[well.well_name]
            metric, _qc = evaluate_wavelet_on_well(
                wavelet_time_s=wavelet_time,
                wavelet_amplitude=values,
                well_artifact=well,
                candidate_wavelet=candidate_name,
                source_well=candidate_name,
                modeler=modeler,
                seismic_match=seismic_match,
                reflectivity_match=reflectivity_match,
            )
            metric_rows.append(metric.to_row())
        frame = pd.DataFrame.from_records(metric_rows)
        mean_corr = float(np.nanmean(frame["corr"].to_numpy(dtype=np.float64)))
        mean_nmae = float(np.nanmean(frame["nmae"].to_numpy(dtype=np.float64)))
        return {
            "candidate_wavelet": candidate_name,
            "mean_corr": mean_corr,
            "mean_nmae": mean_nmae,
            "score": mean_corr - 0.5 * mean_nmae,
            "evaluation_well_count": int(len(frame)),
        }, metric_rows

    aggregate_rows: list[dict[str, object]] = []
    all_metric_rows: list[dict[str, object]] = []
    for name, values in zip(candidate_names, matrix):
        aggregate, metric_rows = score_wavelet(values, candidate_name=name)
        aggregate_rows.append(aggregate)
        all_metric_rows.extend(metric_rows)
    aggregate_frame = pd.DataFrame.from_records(aggregate_rows)
    best_index = int(np.nanargmax(aggregate_frame["score"].to_numpy(dtype=np.float64)))
    best_values = matrix[best_index]
    best_name = candidate_names[best_index]
    basis = build_wavelet_pca_basis(
        matrix,
        n_components=min(4, matrix.shape[0]),
        coefficient_bounds="quantile",
        coefficient_quantiles=(0.05, 0.95),
    )

    def evaluator(values: np.ndarray) -> dict[str, float]:
        aggregate, _rows = score_wavelet(values, candidate_name="optimized_consensus")
        return {key: float(value) for key, value in aggregate.items() if isinstance(value, (float, int, np.floating, np.integer))}

    optimized = optimize_consensus_wavelet(
        basis,
        evaluator,
        policy=ConsensusSearchPolicy(
            random_trials=32,
            max_refine_iters=20,
            seed=int(seed),
            score_key="score",
        ),
    )
    selected_values = best_values
    selected_name = best_name
    selected_score = float(aggregate_rows[best_index]["score"])
    if np.isfinite(float(optimized.score)) and float(optimized.score) > selected_score:
        selected_values = np.asarray(optimized.wavelet, dtype=np.float64)
        selected_name = "optimized_consensus"
        selected_score = float(optimized.score)
        optimized_aggregate, optimized_rows = score_wavelet(selected_values, candidate_name=selected_name)
        aggregate_rows.append(optimized_aggregate)
        all_metric_rows.extend(optimized_rows)
    step5.mkdir(parents=True, exist_ok=True)
    selected_path = step5 / "selected_wavelet.csv"
    pd.DataFrame({"time_s": wavelet_time, "amplitude": selected_values}).to_csv(selected_path, index=False)
    pd.DataFrame.from_records(all_metric_rows).to_csv(step5 / "wavelet_candidate_metrics.csv", index=False)
    pd.DataFrame.from_records(aggregate_rows).to_csv(step5 / "wavelet_candidate_aggregate.csv", index=False)
    summary = {
        "schema_version": "wavelet_generation_v2",
        "status": "success",
        "selection_mode": "main_consensus_optimizer" if selected_name == "optimized_consensus" else "existing_candidate_wins",
        "selected_wavelet": selected_name,
        "selected_source_well": selected_name,
        "selected_score": selected_score,
        "candidate_count": len(candidates),
        "evaluation_well_count": len(wells),
        "selected_wavelet_file": repo_relative_path(selected_path, root=repo_root),
        "source_auto_tie_dir": repo_relative_path(step4, root=repo_root),
        "optimizer": {"random_trials": 32, "max_refine_iters": 20, "seed": int(seed)},
    }
    write_json(step5 / "selected_wavelet_summary.json", summary)
    write_json(step5 / "run_summary.json", summary)
    return {"selected_wavelet": selected_path, "summary": summary}


def run_wavelet_workflow(
    prepared_dir: Path,
    *,
    repo_root: Path,
    pretrained_dir: Path | None = None,
):
    """Build Wtie artifacts and connect the selected wavelet."""
    prepared_dir, repo_root = Path(prepared_dir).resolve(), Path(repo_root).resolve()
    prepare_tie_inputs(prepared_dir, repo_root=repo_root, pretrained_dir=pretrained_dir)
    _network_wtie_artifacts(
        prepared_dir,
        repo_root=repo_root,
        pretrained_dir=pretrained_dir,
    )
    return apply_selected_wavelet(prepared_dir, repo_root=repo_root)


def apply_selected_wavelet(prepared_dir: Path, *, repo_root: Path):
    """Connect a completed Step 5 result to this dataset and its LFM configs."""
    prepared_dir, repo_root = Path(prepared_dir).resolve(), Path(repo_root).resolve()
    selected_dir = prepared_dir / "step5_wavelet_generation"
    base_path = prepared_dir / "ginn_v3.yaml"
    config = load_config(base_path)
    wavelet = pd.read_csv(selected_dir / "selected_wavelet.csv")
    extractor_dt = float(np.median(np.diff(wavelet.time_s)))
    with np.load(prepared_dir / "evaluation" / "truth.npz", allow_pickle=False) as saved:
        truth, times = saved["log_ai"], saved["twt_s"]
    with np.load(prepared_dir / "seismic.npz", allow_pickle=False) as saved:
        seismic = saved["seismic"][0]
    dt = float(times[1] - times[0])
    half = int(round(max(abs(wavelet.time_s.iloc[0]), abs(wavelet.time_s.iloc[-1])) / dt))
    sampled_times = np.arange(-half, half + 1) * dt
    sampled_wavelet, _ = wavelet_l2_normalize(np.interp(sampled_times, wavelet.time_s, wavelet.amplitude))
    forward_dir = prepared_dir / "wavelet"
    forward_dir.mkdir(exist_ok=True)
    pd.DataFrame({"time_s": sampled_times, "amplitude": sampled_wavelet}).to_csv(forward_dir / "selected_wavelet.csv", index=False)
    config["ginn_v3_body_inversion"]["inputs"]["wavelet_generation_run_dir"] = repo_relative_path(forward_dir, root=repo_root)
    base_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    roles = pd.read_csv(prepared_dir / "evaluation" / "well_roles.csv")
    training = roles.loc[roles.role == "train"]
    metrics = pd.read_csv(prepared_dir / "step4_well_auto_tie" / "well_tie_metrics.csv")
    preparation = json.loads((prepared_dir / "preparation_summary.json").read_text(encoding="utf-8"))
    target_top, target_bottom = preparation["target_interval_s"]
    summary = {
        "method": "workflow_steps_4_5",
        "correlation_support_s": [target_top, target_bottom],
        "wavelet_method": "workflow",
        "training_well_names": training.well_name.tolist(),
        "step4": repo_relative_path(prepared_dir / "step4_well_auto_tie", root=repo_root),
        "step5": repo_relative_path(selected_dir, root=repo_root),
        "extractor_wavelet_dt_s": extractor_dt, "wavelet_dt_s": dt,
        "forward_wavelet": repo_relative_path(forward_dir / "selected_wavelet.csv", root=repo_root),
        "training_well_correlations_on_original_tdt": training_well_correlations(
            truth, seismic, times, sampled_times, sampled_wavelet, training.profile_index.to_numpy(), start_s=target_top, stop_s=target_bottom),
        "calibration_results": metrics[[c for c in ("well_name", "tie_status", "optimized_corr", "best_table_shift_ms") if c in metrics]].to_dict("records"),
        "labels": "Original model TWT/AI retained; fixed model TDT is used for Wtie calibration.",
    }
    write_json(prepared_dir / "wavelet" / "wavelet_estimation.json", summary)
    preparation_path = prepared_dir / "preparation_summary.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["wavelet_source"] = summary
    write_json(preparation_path, preparation)
    # Update variant configs too if the user recalibrates after creating LFMs.
    for path in (prepared_dir / "lfm_models").glob("*/configs/*.yaml"):
        variant = load_config(path)
        variant["ginn_v3_body_inversion"]["inputs"]["wavelet_generation_run_dir"] = repo_relative_path(forward_dir, root=repo_root)
        path.write_text(yaml.safe_dump(variant, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return summary
