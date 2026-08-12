"""Train the GINN V2 body inversion with a limited trial loop.

The command requires explicit Step-6, Step-7, and frozen-forward inputs.  A
configuration section named ``ginn_v2_body_inversion`` carries the settings;
the input identities can also be supplied as command-line overrides.

Example::

    python scripts/ginn_v2_body_inversion.py --config experiments/ginn_v2/ginn_v2.yaml
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.config.workflow import WorkflowConfig, deep_merge_dict
from cup.lfm.math import parse_lowpass_spec
from cup.physics.calibration import AIVelocityRelation
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.seismic.wavelet import load_wavelet_csv, validate_wavelet_normalization
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, write_json
from cup.utils.logging import configure_run_logger
from cup.well.real_field_controls import load_well_control_set
from ginn_v2.adapters import DepthDomainAdapter, TimeDomainAdapter
from ginn_v2.checkpoint import load_checkpoint
from ginn_v2.data import PatchReader, SurveyTraceSource, candidate_patch_keys, fit_lfm_normalization
from ginn_v2.evaluation import gate_improved
from ginn_v2.trainer import (
    BodyInversionConfig,
    BodyInversionTrainer,
    TrialResult,
    build_body_inversion_data,
    make_trial_adjustment,
)
from cup.lfm.artifacts import load_lfm_input
from ginn_v2.model import CenterTraceBodyNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/ginn_v2/ginn_v2.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lfm-run-dir", type=Path, default=None)
    parser.add_argument("--variant-id", type=str, default=None)
    parser.add_argument("--well-control-run-dir", type=Path, default=None)
    parser.add_argument("--forward-model-inputs", type=Path, default=None)
    return parser.parse_args()


def _load_composed_config(path: Path) -> dict[str, Any]:
    experiment = load_yaml_config(path)
    workflow_config = str(experiment.get("workflow_config") or "").strip()
    if not workflow_config:
        return experiment
    common = load_yaml_config(resolve_relative_path(workflow_config, root=REPO_ROOT))
    overlay = {key: value for key, value in experiment.items() if key != "workflow_config"}
    return deep_merge_dict(common, overlay)


def _required_input(stage_config: Mapping[str, Any], key: str, override: object) -> str:
    if override is not None:
        value = str(override).strip()
    else:
        inputs = stage_config.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("ginn_v2_body_inversion.inputs must explicitly contain all frozen input paths.")
        value = str(inputs.get(key) or "").strip()
    if not value:
        raise ValueError(f"ginn_v2_body_inversion input {key!r} must be explicit.")
    return value


def _load_forward_inputs(path: Path, *, domain: str, depth_basis: str | None) -> tuple[np.ndarray, np.ndarray, AIVelocityRelation | None, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "forward_model_inputs_v3":
        raise ValueError("Body inversion requires forward_model_inputs_v3.")
    if payload.get("sample_domain") != domain or payload.get("depth_basis") != depth_basis:
        raise ValueError("Frozen forward inputs do not match the seismic SampleAxis domain.")
    wavelet_info = payload.get("wavelet")
    if not isinstance(wavelet_info, Mapping):
        raise ValueError("forward_model_inputs.wavelet must be a mapping.")
    wavelet_path = resolve_relative_path(str(wavelet_info.get("path") or ""), root=REPO_ROOT)
    time_s, amplitude = load_wavelet_csv(wavelet_path)
    amplitude, qc = validate_wavelet_normalization(
        time_s,
        amplitude,
        allow_small_renormalization=False,
    )
    if qc.status != "ok":
        raise ValueError(f"Frozen wavelet failed normalization QC: {qc.reasons}")
    relation = None
    if domain == "depth":
        relation_info = payload.get("ai_velocity_relation")
        if not isinstance(relation_info, Mapping):
            raise ValueError("Depth forward inputs must contain ai_velocity_relation.")
        relation = AIVelocityRelation.from_mapping(relation_info)
    return time_s, amplitude, relation, payload


def _resolve_output_dir(value: Path | None, workflow: WorkflowConfig) -> Path:
    if value is not None:
        return resolve_relative_path(value, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return resolve_relative_path(workflow.output_root, root=REPO_ROOT) / f"ginn_v2_body_inversion_{timestamp}"


def _build_runtime(raw: Mapping[str, Any], args: argparse.Namespace) -> tuple[WorkflowConfig, BodyInversionConfig, Path, Path, Path, Path, str]:
    workflow = WorkflowConfig.from_mapping(raw)
    section = raw.get("ginn_v2_body_inversion")
    if not isinstance(section, Mapping):
        raise ValueError("Config lacks explicit ginn_v2_body_inversion section.")
    lfm_run_dir = resolve_relative_path(_required_input(section, "lfm_run_dir", args.lfm_run_dir), root=REPO_ROOT)
    variant_id = _required_input(section, "variant_id", args.variant_id)
    well_control_run_dir = resolve_relative_path(
        _required_input(section, "well_control_run_dir", args.well_control_run_dir),
        root=REPO_ROOT,
    )
    forward_inputs = resolve_relative_path(
        _required_input(section, "forward_model_inputs", args.forward_model_inputs),
        root=REPO_ROOT,
    )
    if section.get("training") is None:
        training_mapping = {
            key: value
            for key, value in section.items()
            if key not in {"inputs", "lfm_run_dir", "variant_id", "well_control_run_dir", "forward_model_inputs"}
        }
    else:
        training_mapping = dict(section.get("training") or {})
    if "trusted_well_names" not in training_mapping and "trusted_well_names" in section:
        training_mapping["trusted_well_names"] = section["trusted_well_names"]
    config = BodyInversionConfig.from_mapping(training_mapping)
    output_dir = _resolve_output_dir(args.output_dir, workflow)
    return workflow, config, lfm_run_dir, well_control_run_dir, forward_inputs, output_dir, variant_id


def _write_review_package(trainer: BodyInversionTrainer, model: CenterTraceBodyNet, output_dir: Path) -> dict[str, Any]:
    """Render deterministic well profiles and blind validation sections."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    review_dir = output_dir / "review_package"
    profiles_dir = review_dir / "fixed_well_profiles"
    blind_dir = review_dir / "blind_sections"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    blind_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    well_rows: dict[str, dict[int, list[float]]] = {}
    well_targets: dict[str, dict[int, float]] = {}
    well_patch_keys: dict[str, dict[int, Any]] = {}
    with torch.no_grad():
        items = trainer.data.validation_well_patches
        for start in range(0, len(items), trainer.config.batch_size):
            local = items[start : start + trainer.config.batch_size]
            batch = trainer.data.reader.batch(
                tuple(item.patch_key for item in local),
                center_visible=True,
                device=trainer.device,
            )
            body, _, _ = trainer._predict(model, batch)
            for row, item in enumerate(local):
                for sample_index in np.flatnonzero(item.target_mask):
                    index = int(sample_index)
                    well_rows.setdefault(item.well_name, {}).setdefault(index, []).append(
                        float(body[row, index].cpu())
                    )
                    well_targets.setdefault(item.well_name, {})[index] = float(item.target_values[index])
                    well_patch_keys.setdefault(item.well_name, {}).setdefault(index, item.patch_key)
    figure, axes = plt.subplots(max(1, len(well_rows)), 1, figsize=(8, max(3, 2.5 * len(well_rows))), squeeze=False)
    axis_values = trainer.data.reader.sample_axis.values
    for row, name in enumerate(sorted(well_rows)):
        indices = sorted(well_rows[name])
        predicted = [np.mean(well_rows[name][index]) for index in indices]
        target = [well_targets[name][index] for index in indices]
        lfm = [
            trainer.data.reader.lfm_log_ai[
                well_patch_keys[name][index].inline_index,
                well_patch_keys[name][index].xline_index,
                index,
            ]
            for index in indices
        ]
        current_axis = axes[row, 0]
        current_axis.plot(target, axis_values[indices], label="well body target")
        current_axis.plot(predicted, axis_values[indices], label="GINN body")
        current_axis.plot(lfm, axis_values[indices], label="LFM")
        current_axis.set_title(name)
        current_axis.set_xlabel("log-AI")
        current_axis.set_ylabel(trainer.data.reader.sample_axis.unit)
        current_axis.invert_yaxis()
        current_axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    well_figure = profiles_dir / "trusted_well_holdout_profiles.png"
    figure.savefig(well_figure, dpi=150)
    plt.close(figure)

    section_keys = {
        orientation: trainer.validation_section_keys(orientation)
        for orientation in trainer.config.orientations
    }
    review_keys = tuple(dict.fromkeys(key for keys in section_keys.values() for key in keys))
    trace_eval = trainer._trace_evaluation(model, review_keys, center_visible=True)
    section_files: list[str] = []
    for orientation in trainer.config.orientations:
        keys = section_keys[orientation]
        lateral_m = trainer.lateral_distance_m(keys)
        values = np.stack([trace_eval["bodies"][item] for item in keys])
        lfm_values = np.stack([trace_eval["lfm"][item] for item in keys])
        support = np.stack([trace_eval["supports"][item] for item in keys])
        values = np.where(support, values, np.nan)
        residual = np.where(support, values - lfm_values, np.nan)
        vertical_support = np.any(support, axis=0)
        sample_indices = np.flatnonzero(vertical_support)
        if sample_indices.size < 2:
            raise ValueError("Blind section has fewer than two target-zone samples.")
        sample_start, sample_stop = int(sample_indices[0]), int(sample_indices[-1])
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        extent = [lateral_m[0], lateral_m[-1], axis_values[sample_stop], axis_values[sample_start]]
        axes[0].imshow(values[:, sample_start : sample_stop + 1].T, aspect="auto", origin="upper", extent=extent)
        axes[0].set_title(f"Blind validation section — {orientation}")
        axes[0].set_ylabel(trainer.data.reader.sample_axis.unit)
        axes[1].imshow(residual[:, sample_start : sample_stop + 1].T, aspect="auto", origin="upper", extent=extent, cmap="RdBu_r")
        axes[1].set_title("GINN body minus LFM")
        axes[1].set_xlabel("lateral distance (m)")
        axes[1].set_ylabel(trainer.data.reader.sample_axis.unit)
        figure.tight_layout()
        path = blind_dir / f"blind_section_{orientation}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        section_files.append(repo_relative_path(path, root=REPO_ROOT))
    manifest = {
        "fixed_well_profile": repo_relative_path(well_figure, root=REPO_ROOT),
        "blind_sections": section_files,
        "validation_patch_key_count": len(trainer.data.spatial_split.review_keys),
        "validation_centers": [list(item) for item in trainer.data.spatial_split.validation_centers],
    }
    write_json(review_dir / "review_manifest.json", manifest)
    return manifest


def _write_trial_comparison(path: Path, results: list[TrialResult]) -> None:
    """Write the fixed limited-loop comparison as a human-readable table."""

    fieldnames = (
        "trial_id",
        "parent_trial_id",
        "action",
        "target_gate",
        "target_gate_improved",
        "selected_epoch",
        "stop_reason",
        "gate_passed",
        "first_failed_gate",
        "masked_corr_change_from_pretrain_median",
        "masked_shape_median",
        "visible_corr_median",
        "visible_shape_median",
        "well_pooled_rmse",
        "well_pooled_bias",
        "lfm_drift_rmse",
        "short_wave_energy_fraction",
        "roughness_ratio",
        "orientation_disagreement_rms_ratio",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            metrics = result.metrics
            details = result.gate.details
            writer.writerow(
                {
                    "trial_id": result.trial_id,
                    "parent_trial_id": "" if result.adjustment.parent_trial_id is None else result.adjustment.parent_trial_id,
                    "action": result.adjustment.action,
                    "target_gate": result.adjustment.target_gate or "",
                    "target_gate_improved": "" if result.target_gate_improved is None else result.target_gate_improved,
                    "selected_epoch": "" if result.selected_epoch is None else result.selected_epoch,
                    "stop_reason": result.stop_reason,
                    "gate_passed": result.gate.passed,
                    "first_failed_gate": result.gate.first_failed_gate or "",
                    "masked_corr_change_from_pretrain_median": details.get("masked_corr_change_from_pretrain_median", ""),
                    "masked_shape_median": float(np.median(metrics.masked_shape_loss)),
                    "visible_corr_median": float(np.median(metrics.visible_correlation)),
                    "visible_shape_median": float(np.median(metrics.visible_shape_loss)),
                    "well_pooled_rmse": metrics.well_pooled_rmse,
                    "well_pooled_bias": metrics.well_pooled_bias,
                    "lfm_drift_rmse": metrics.lfm_drift_rmse,
                    "short_wave_energy_fraction": metrics.short_wave_energy_fraction,
                    "roughness_ratio": metrics.roughness_ratio,
                    "orientation_disagreement_rms_ratio": metrics.orientation_disagreement_rms_ratio,
                }
            )


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    raw = _load_composed_config(config_path)
    workflow, config, lfm_run_dir, well_control_run_dir, forward_inputs_path, output_dir, variant_id = _build_runtime(raw, args)
    if output_dir.exists():
        raise FileExistsError(f"Body-inversion output directory already exists: {output_dir}; use a new output directory.")
    else:
        output_dir.mkdir(parents=True)
    logger = configure_run_logger(
        output_dir,
        logger_name="ginn_v2_body_inversion",
        file_name="training.log",
    )
    logger.info("body inversion start | config=%s | output=%s", config_path, output_dir)

    controls = load_well_control_set(well_control_run_dir, repo_root=REPO_ROOT)
    lfm = load_lfm_input(
        {
            "lfm_run_dir": str(lfm_run_dir),
            "variant_id": variant_id,
            "well_control_run_dir": str(well_control_run_dir),
        },
        repo_root=REPO_ROOT,
    )
    data_root = resolve_relative_path(workflow.data_root, root=REPO_ROOT)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    if not seismic_path.is_file():
        raise FileNotFoundError(seismic_path)
    survey_options = segy_options_from_config(workflow.seismic.as_dict()) if workflow.seismic.type == "segy" else {}
    survey = open_survey(seismic_path, workflow.seismic.type, segy_options=survey_options or None)
    sample_axis = survey.sample_axis(workflow.seismic.domain)
    if not np.array_equal(sample_axis.values, lfm.sample_axis.values) or not np.array_equal(sample_axis.values, controls.sample_axis.values):
        raise ValueError("Body-inversion seismic, LFM, and WellControlSet SampleAxis values differ.")
    if lfm.log_ai.ndim != 3:
        raise ValueError("Body inversion requires a volume LFM variant, not a section variant.")
    if lfm.log_ai.shape != (survey.line_geometry.inline_axis.count, survey.line_geometry.xline_axis.count, sample_axis.values.size):
        raise ValueError("LFM volume shape differs from current survey geometry.")
    target_zone_mask = np.asarray(lfm.valid_mask, dtype=bool)
    baseline_config = dict(lfm.variant.variant_metadata.get("resolved_baseline_config") or {})
    lfm_lowpass_spec = parse_lowpass_spec(
        dict(baseline_config.get("filter") or {}),
        sample_axis,
    )
    wavelet_time_s, wavelet_amplitude, relation, forward_payload = _load_forward_inputs(
        forward_inputs_path,
        domain=workflow.seismic.domain,
        depth_basis=workflow.seismic.depth_basis,
    )
    domain_extras: dict[str, np.ndarray] = {}
    if workflow.seismic.domain == "depth":
        if relation is None:
            raise ValueError("Depth body inversion requires a frozen AI--Vp relation.")
        valid = np.asarray(lfm.valid_mask, dtype=bool)
        velocity = np.full(lfm.log_ai.shape, np.nan, dtype=np.float64)
        velocity[valid] = relation.velocity_from_ai(np.exp(np.asarray(lfm.log_ai[valid], dtype=np.float64)))
        domain_extras["velocity_mps"] = velocity
        adapter = DepthDomainAdapter(
            torch.as_tensor(wavelet_time_s, dtype=torch.float32),
            torch.as_tensor(wavelet_amplitude, dtype=torch.float32),
        )
    else:
        adapter = TimeDomainAdapter(
            torch.as_tensor(wavelet_time_s, dtype=torch.float32),
            torch.as_tensor(wavelet_amplitude, dtype=torch.float32),
        )
    normalization = fit_lfm_normalization(lfm.log_ai, lfm.valid_mask, geometry=survey.line_geometry)
    source = SurveyTraceSource(survey=survey, sample_axis=sample_axis, geometry=survey.line_geometry)
    reader = PatchReader(
        source,
        lfm_log_ai=lfm.log_ai,
        lfm_valid_mask=lfm.valid_mask,
        ilines=lfm.ilines,
        xlines=lfm.xlines,
        sample_axis=sample_axis,
        normalization=normalization,
        patch_radius=config.patch_radius,
        domain_extras=domain_extras,
        cache_size=config.cache_size,
    )
    candidates = candidate_patch_keys(
        lfm.log_ai,
        lfm.valid_mask,
        patch_radius=config.patch_radius,
        orientations=config.orientations,
    )
    data = build_body_inversion_data(
        reader,
        controls,
        config=config,
        candidate_keys=candidates,
        target_zone_mask=target_zone_mask,
    )
    logger.info(
        "data ready | train_patches=%d | validation_patches=%d | train_well_samples=%d | validation_well_samples=%d",
        len(data.spatial_split.train_keys),
        len(data.spatial_split.validation_keys),
        sum(int(np.count_nonzero(item.target_mask)) for item in data.train_well_patches),
        sum(int(np.count_nonzero(item.target_mask)) for item in data.validation_well_patches),
    )
    trainer = BodyInversionTrainer(
        data,
        adapter=adapter,
        config=config,
        lfm_lowpass_spec=lfm_lowpass_spec,
        output_dir=output_dir,
        logger=logger,
    )
    logger.info("LFM-only baseline evaluation start")
    baseline = trainer.evaluate(None)
    logger.info(
        "LFM-only baseline ready | masked_corr=%.4f | visible_corr=%.4f | well_rmse=%.5f",
        float(np.median(baseline.masked_correlation)),
        float(np.median(baseline.visible_correlation)),
        baseline.well_pooled_rmse,
    )
    all_well_names = [control.well_name for control in controls.controls]
    trusted_well_names = list(config.trusted_well_names)
    trusted_lookup = {name.casefold() for name in trusted_well_names}
    diagnostic_well_names = [name for name in all_well_names if name.casefold() not in trusted_lookup]
    write_json(output_dir / "baseline_metrics.json", baseline.to_json_dict())
    write_json(output_dir / "split.json", data.split_description())
    write_json(
        output_dir / "well_roles.json",
        {
            "lfm_well_names": all_well_names,
            "body_well_names": trusted_well_names,
            "diagnostic_only_well_names": diagnostic_well_names,
        },
    )
    write_json(
        output_dir / "input_contract.json",
        {
            "sample_axis": sample_axis.describe(),
            "depth_basis": workflow.seismic.depth_basis,
            "body_smoothing_fwhm_m": config.body_smoothing_fwhm_m,
            "lfm_input_normalization": {
                "lfm_mean": normalization.lfm_mean,
                "lfm_scale": normalization.lfm_scale,
                "geometry_scale_m": normalization.geometry_scale_m,
            },
            "lfm_run_dir": repo_relative_path(lfm_run_dir, root=REPO_ROOT),
            "lfm_variant_id": variant_id,
            "lfm_artifact_lowpass": {
                "value": lfm.lowpass_value,
                "unit": lfm.lowpass_unit,
                "order": lfm_lowpass_spec.order,
                "buffer_mode": lfm_lowpass_spec.buffer_mode,
                "buffer_axis_units": lfm_lowpass_spec.buffer_axis_units,
            },
            "well_control_run_dir": repo_relative_path(well_control_run_dir, root=REPO_ROOT),
            "forward_model_inputs": repo_relative_path(forward_inputs_path, root=REPO_ROOT),
            "forward_adapter": getattr(adapter, "adapter_id", type(adapter).__name__),
            "wavelet": forward_payload.get("wavelet"),
            "well_roles": {
                "lfm_well_names": all_well_names,
                "body_well_names": trusted_well_names,
                "diagnostic_only_well_names": diagnostic_well_names,
            },
        },
    )

    logger.info("shared masked pretraining start")
    shared_pretrain_checkpoint, pretrain_metrics = trainer.run_pretraining()
    write_json(output_dir / "pretrain_metrics.json", pretrain_metrics.to_json_dict())
    logger.info(
        "shared masked pretraining ready | checkpoint=%s | masked_corr=%.4f | well_rmse=%.5f",
        shared_pretrain_checkpoint,
        float(np.median(pretrain_metrics.masked_correlation)),
        pretrain_metrics.well_pooled_rmse,
    )

    if (
        float(np.median(pretrain_metrics.masked_correlation))
        < float(np.median(baseline.masked_correlation)) + config.gates.pretrain_masked_corr_improvement
        or float(np.median(pretrain_metrics.masked_shape_loss))
        > config.gates.pretrain_masked_shape_ratio * float(np.median(baseline.masked_shape_loss))
    ):
        write_json(
            output_dir / "body_inversion_status.json",
            {"status": "pretraining_not_accepted", "checkpoint": repo_relative_path(shared_pretrain_checkpoint, root=REPO_ROOT)},
        )
        raise RuntimeError("Shared masked pretraining did not improve the LFM-only masked reconstruction.")

    results: list[TrialResult] = []
    failure_counts: dict[str, int] = {}
    previous: TrialResult | None = None
    selected: TrialResult | None = None
    for trial_id in range(1, config.max_trials + 1):
        adjustment = make_trial_adjustment(
            config,
            trial_id=trial_id,
            previous=previous,
            failure_counts=failure_counts,
        )
        logger.info(
            "trial %d/%d prepared | action=%s | target_gate=%s",
            trial_id,
            config.max_trials,
            adjustment.action,
            adjustment.target_gate or "none",
        )
        result = trainer.run_trial(
            adjustment,
            baseline=baseline,
            resume_checkpoint=shared_pretrain_checkpoint,
        )
        if previous is not None and adjustment.target_gate is not None and result.target_gate_improved is None:
            result = TrialResult(
                trial_id=result.trial_id,
                adjustment=result.adjustment,
                checkpoints=result.checkpoints,
                selected_checkpoint=result.selected_checkpoint,
                selected_epoch=result.selected_epoch,
                metrics=result.metrics,
                gate=result.gate,
                stop_reason=result.stop_reason,
                target_gate_improved=gate_improved(
                    result.metrics,
                    previous.metrics,
                    adjustment.target_gate,
                    threshold=0.0,
                ),
            )
        results.append(result)
        if result.gate.passed and result.selected_checkpoint is not None:
            selected = result
            break
        gate = result.gate.first_failed_gate
        if gate is None:
            break
        failure_counts[gate] = failure_counts.get(gate, 0) + 1
        if previous is not None and adjustment.target_gate is not None and result.target_gate_improved is False:
            break
        previous = result
    write_json(output_dir / "trial_results.json", {"trials": [item.to_json_dict() for item in results]})
    _write_trial_comparison(output_dir / "trial_comparison.csv", results)
    if selected is None:
        write_json(
            output_dir / "body_inversion_status.json",
            {
                "status": "not_accepted",
                "reason": "limited_trial_loop_exhausted_or_retracted",
                "trial_count": len(results),
            },
        )
        raise RuntimeError(
            f"GINN V2 body inversion did not produce an accepted checkpoint within {config.max_trials} trials."
        )
    selected_path = output_dir / "selected_checkpoint.pt"
    shutil.copyfile(selected.selected_checkpoint, selected_path)
    write_json(
        output_dir / "selected_checkpoint.json",
        {
            "status": "ok",
            "selected_checkpoint": repo_relative_path(selected_path, root=REPO_ROOT),
            "trial_id": selected.trial_id,
            "epoch": selected.selected_epoch,
            "selection_rule": "earliest_acceptable_checkpoint",
            "gate": selected.gate.to_json_dict(),
        },
    )
    selected_model = CenterTraceBodyNet(config.network).to(trainer.device)
    load_checkpoint(
        selected_path,
        model=selected_model,
        expected_network_config=config.network,
        map_location=trainer.device,
    )
    review = _write_review_package(trainer, selected_model, output_dir)
    write_json(
        output_dir / "body_inversion_status.json",
        {
            "status": "ok",
            "selected_checkpoint": repo_relative_path(selected_path, root=REPO_ROOT),
            "review_package": review,
            "trial_count": len(results),
        },
    )
    print("=== GINN V2 body inversion ===")
    print(f"Output: {output_dir}")
    print(f"Selected checkpoint: {selected_path}")
    print(f"Trial: {selected.trial_id}, epoch: {selected.selected_epoch}")


if __name__ == "__main__":
    main()
