"""Build and publish the Marmousi2 low-frequency models used by the benchmark.

The directory layout is defined by the construction recipe, not by the order in
which the variants happened to be requested:

    lfm_models/truth_huber_trend/    per-trace Huber line fitted on the full truth
    lfm_models/well_huber_trend/     the workflow ``trend`` builder on the training wells
    lfm_models/rgt_lowpass_slices/   low-passed well curves sliced along RGT
    lfm_models/rgt_structure/        the RGT field the slicing LFM consumes

Each LFM directory is a standard LFM run directory, so a training config can
point straight at it.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from cup.lfm.math import parse_lowpass_spec
from cup.lfm.pipeline import resolve_output_geometry, run_lfm_pipeline
from cup.lfm.types import LfmContext, LfmVariantResult
from cup.seismic.survey import open_survey
from cup.seismic.target_zone import TargetZone
from cup.utils.io import repo_relative_path, write_json
from cup.well.controls import load_well_control_set
from ginn_v2.workflow import load_config
from marmousi2.lfm_models import build_truth_huber_trend
from marmousi2.prepare import check_prepared_inputs
from marmousi2.rgt_inference import infer_rgt_section
from marmousi2.rgt_lfm import build_rgt_lowpass_slices


LFM_IDS = ("truth_huber_trend", "well_huber_trend", "rgt_lowpass_slices")
RGT_LFM_IDS = ("rgt_lowpass_slices",)
DEFAULT_LFM_IDS = ("well_huber_trend", "rgt_lowpass_slices")

# The shared 5 Hz operator: it filters training-well curves for the slicing LFM
# and simultaneously acts as the network's low-frequency projection and anchor.
DEFAULT_LOWPASS_CONFIG: dict[str, Any] = {
    "enabled": True, "cutoff_hz": 5.0, "order": 4,
    "buffer_mode": "reflect", "buffer_axis_units": 0.2,
}

TREND_FIT_CONFIG: dict[str, Any] = {
    "min_valid_samples_per_well": 8, "huber_f_scale_log_ai": 0.1,
}

DISPLAY_NAMES = {
    "truth_huber_trend": "Per-trace Huber log-AI trend | full truth",
    "well_huber_trend": "Training-well trend on TWT | workflow trend builder",
    "rgt_lowpass_slices": "RGT coordinate + 5 Hz low-pass well slices | training wells only",
}

def _training_wells(base: Mapping[str, Any], controls):
    """Return the trusted, complete, stationary training pseudo-wells in x order."""

    names = set(base["ginn_v2_body_inversion"]["training"]["trusted_well_names"])
    trained = [control for control in controls.controls if control.well_name in names]
    if not names or {control.well_name for control in trained} != names:
        raise ValueError("Trusted training pseudo-wells do not match the available controls.")
    trained.sort(key=lambda control: float(control.x_m_by_sample[0]))
    for control in trained:
        if not np.all(control.valid_mask) or np.ptp(control.x_m_by_sample) > 1e-8 or np.ptp(control.y_m_by_sample) > 1e-8:
            raise ValueError("These LFM builders require complete, stationary training pseudo-wells.")
    values = np.stack([control.model_grid_filtered_log_ai.values for control in trained])
    positions = np.asarray([control.x_m_by_sample[0] for control in trained], dtype=np.float64)
    return trained, values, positions


def _variant_inputs(prepared_dir: Path, repo_root: Path, lowpass_config: Mapping[str, Any]):
    base = load_config(prepared_dir / "ginn_v2.yaml")
    if (repo_root / base["data_root"]).resolve() != prepared_dir:
        raise ValueError("Prepared directory does not match the base configuration data_root.")
    inputs = base["ginn_v2_body_inversion"]["inputs"]
    controls_dir = repo_root / inputs["well_control_run_dir"]
    controls = load_well_control_set(controls_dir, repo_root=repo_root)
    survey_path = prepared_dir / base["seismic"]["file"]
    survey = open_survey(survey_path, base["seismic"]["type"])
    axis = survey.sample_axis("time")
    if survey.seismic.shape[0] != 1:
        raise ValueError("This adapter requires a singleton-inline 2D section.")
    if not np.array_equal(axis.values, controls.sample_axis.values):
        raise ValueError("Seismic and well-control sample axes differ.")
    geometry = resolve_output_geometry({"mode": "volume"}, survey=survey, sample_axis=axis)
    x_m = np.asarray(geometry.x_m[0], dtype=np.float64)
    if np.any(np.diff(x_m) <= 0.0) or np.ptp(geometry.y_m) > 1e-8:
        raise ValueError("This adapter requires a straight, increasing-x section.")
    truth_path = prepared_dir / "evaluation" / "truth.npz"
    with np.load(truth_path, allow_pickle=False) as saved:
        truth = np.asarray(saved["log_ai"], dtype=np.float64)
        truth_x = np.asarray(saved["x_m"], dtype=np.float64)
        twt_s = np.asarray(saved["twt_s"], dtype=np.float64)
    if truth.shape != (x_m.size, axis.values.size) or not np.array_equal(twt_s, axis.values):
        raise ValueError("Prepared truth shape/TWT differs from the seismic grid.")
    if not np.all(np.isfinite(truth)) or not np.allclose(truth_x, x_m, rtol=0.0, atol=1e-8):
        raise ValueError("Prepared truth must be finite and share the seismic lateral positions.")
    frames = {
        label: pd.DataFrame({"inline": np.zeros(x_m.size), "xline": geometry.xlines, "interpretation": value})
        for label, value in (("experiment_top", twt_s[0]), ("experiment_bottom", twt_s[-1]))
    }
    zone = TargetZone(frames, survey.describe_geometry("time"), list(frames))
    context = LfmContext(
        survey.line_geometry, axis, zone, geometry, None,
        {"seismic": {"path": repo_relative_path(survey_path, root=repo_root), "type": base["seismic"]["type"]}},
    )
    return {
        "base": base, "inputs": inputs, "controls": controls, "controls_dir": controls_dir,
        "survey": survey, "survey_path": survey_path, "axis": axis, "geometry": geometry,
        "x_m": x_m, "truth": truth, "truth_path": truth_path, "twt_s": twt_s, "context": context,
        "lowpass": parse_lowpass_spec(dict(lowpass_config), axis),
    }


def _training_config(runtime: Mapping[str, Any], variant_id: str, output_dir: Path, repo_root: Path) -> Path:
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    variant_config = deepcopy(runtime["base"])
    variant_config["ginn_v2_body_inversion"]["inputs"].update(
        lfm_run_dir=repo_relative_path(output_dir, root=repo_root), variant_id=variant_id,
    )
    path = configs_dir / f"{variant_id}.yaml"
    path.write_text(yaml.safe_dump(variant_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _load_published(output_dir: Path, variant_id: str) -> np.ndarray:
    with np.load(output_dir / "variants" / variant_id / "lfm.npz", allow_pickle=False) as data:
        return np.asarray(data["log_ai"], dtype=np.float64)[0]


def _row(
    *, variant_id: str, output_dir: Path, runtime: Mapping[str, Any], repo_root: Path,
    config_path: Path, check: Mapping[str, Any], uses_full_truth: bool,
    input_lowpass_applied: bool, training_well_names: Sequence[str], method: str,
) -> dict[str, Any]:
    ai = np.exp(_load_published(output_dir, variant_id))
    return {
        "variant_id": variant_id, "uses_full_truth": bool(uses_full_truth),
        "training_well_names": list(training_well_names),
        "input_lowpass_applied": bool(input_lowpass_applied), "output_lowpass_applied": False,
        "display_name": DISPLAY_NAMES[variant_id], "construction_method": method,
        "ai_min": float(ai.min()), "ai_mean": float(ai.mean()), "ai_max": float(ai.max()),
        "config": repo_relative_path(config_path, root=repo_root),
        "run": repo_relative_path(output_dir, root=repo_root),
        "adapter_check": check,
    }


def _publish_truth_huber_trend(
    *, output_dir: Path, runtime: Mapping[str, Any], repo_root: Path,
    lowpass_config: Mapping[str, Any], truth_f_scale: float, source: Mapping[str, Any],
) -> dict[str, Any]:
    variant_id = "truth_huber_trend"
    built = build_truth_huber_trend(runtime["truth"], runtime["twt_s"], f_scale=truth_f_scale)
    method = f"marmousi_{variant_id}"
    baseline_id = f"base_{variant_id}"
    baseline_config = {
        "method": method, "filter": dict(lowpass_config),
        "filter_role": "shared_network_low_frequency_projection_and_anchor",
        "uses_full_truth": True, "source": dict(source), "construction": dict(built.metadata),
    }
    truth = runtime["truth"]
    profile = np.broadcast_to(built.log_ai, truth.shape).copy()
    metadata = {
        **built.metadata, "variant_id": variant_id, "uses_full_truth": True,
        "source_scope": "full_profile_truth", "display_name": DISPLAY_NAMES[variant_id],
        "shared_low_frequency_filter": dict(lowpass_config),
        "input_lowpass_applied": False, "output_lowpass_applied": False,
        "training_well_names": [], "modifier_chain": [],
        "resolved_baseline_config": baseline_config, "resolved_modifier_configs": {},
    }
    result = LfmVariantResult(
        profile[None], np.ones((1, *truth.shape), dtype=bool), baseline_id, method,
        method_fields=dict(built.fields), metadata=metadata,
    )
    config = {
        "source_runs": {"well_control_run_dir": repo_relative_path(runtime["controls_dir"], root=repo_root)},
        "output_geometry": {"mode": "volume"}, "baselines": {baseline_id: baseline_config},
        "modifiers": {}, "variants": [{"variant_id": variant_id, "baseline_id": baseline_id, "modifier_ids": []}],
        "comparisons": [],
    }
    run_lfm_pipeline(
        config=config, controls=runtime["controls"], context=runtime["context"],
        controls_run=runtime["controls_dir"], horizon_sources=[],
        source_seismic_file=runtime["survey_path"], source_seismic_type=runtime["base"]["seismic"]["type"],
        seismic_options={}, output_dir=output_dir, repo_root=repo_root,
        prepared_results={variant_id: result},
    )
    config_path = _training_config(runtime, variant_id, output_dir, repo_root)
    check = check_prepared_inputs(
        config_path, repo_root=repo_root, report_dir=output_dir / "variants" / variant_id / "qc" / "adapter",
    )
    return _row(
        variant_id=variant_id, output_dir=output_dir, runtime=runtime, repo_root=repo_root,
        config_path=config_path, check=check, uses_full_truth=True, input_lowpass_applied=False,
        training_well_names=[], method=method,
    )


def _publish_well_huber_trend(
    *, output_dir: Path, runtime: Mapping[str, Any], repo_root: Path,
    lowpass_config: Mapping[str, Any], variogram: str, nugget: float,
    kriging_range_m: float | None, well_names: Sequence[str],
) -> dict[str, Any]:
    """Publish the workflow ``trend`` baseline: per-well robust line plus kriged coefficients."""

    variant_id = "well_huber_trend"
    baseline_id = "well_trend"
    if not np.all(np.isfinite(runtime["x_m"])):
        raise ValueError("Output lateral positions must be finite.")
    spatial: dict[str, Any] = {"variogram": variogram, "exact": True, "nugget": float(nugget)}
    if kriging_range_m is not None:
        spatial["range_m"] = float(kriging_range_m)
    baseline_config = {
        "method": "trend", "filter": dict(lowpass_config), "fit": dict(TREND_FIT_CONFIG), "spatial": spatial,
    }
    config = {
        "source_runs": {"well_control_run_dir": repo_relative_path(runtime["controls_dir"], root=repo_root)},
        "output_geometry": {"mode": "volume"}, "baselines": {baseline_id: baseline_config},
        "modifiers": {}, "variants": [{"variant_id": variant_id, "baseline_id": baseline_id, "modifier_ids": []}],
        "comparisons": [],
    }
    run_lfm_pipeline(
        config=config, controls=runtime["controls"], context=runtime["context"],
        controls_run=runtime["controls_dir"], horizon_sources=[],
        source_seismic_file=runtime["survey_path"], source_seismic_type=runtime["base"]["seismic"]["type"],
        seismic_options={}, output_dir=output_dir, repo_root=repo_root,
    )
    config_path = _training_config(runtime, variant_id, output_dir, repo_root)
    check = check_prepared_inputs(
        config_path, repo_root=repo_root, report_dir=output_dir / "variants" / variant_id / "qc" / "adapter",
    )
    return _row(
        variant_id=variant_id, output_dir=output_dir, runtime=runtime, repo_root=repo_root,
        config_path=config_path, check=check, uses_full_truth=False, input_lowpass_applied=False,
        training_well_names=well_names, method=f"marmousi_{variant_id}",
    )


def _publish_rgt_lowpass_slices(
    *, output_dir: Path, runtime: Mapping[str, Any], repo_root: Path,
    lowpass_config: Mapping[str, Any], well_values: np.ndarray, well_x: np.ndarray,
    raw_rgt: np.ndarray, source: Mapping[str, Any], well_names: Sequence[str],
    variogram: str, nugget: float, kriging_range_m: float,
) -> dict[str, Any]:
    variant_id = "rgt_lowpass_slices"
    built = build_rgt_lowpass_slices(
        well_values, well_x, runtime["x_m"], runtime["twt_s"], raw_rgt, lowpass_spec=runtime["lowpass"],
        variogram=variogram, nugget=nugget, kriging_range_m=kriging_range_m,
    )
    method = f"marmousi_{variant_id}"
    baseline_id = f"base_{variant_id}"
    baseline_config = {
        "method": method, "filter": dict(lowpass_config),
        "filter_role": "training_well_input_lowpass_and_shared_network_low_frequency_projection_and_anchor",
        "uses_full_truth": False, "source": dict(source), "construction": dict(built.metadata),
    }
    truth = runtime["truth"]
    profile = np.broadcast_to(built.log_ai, truth.shape).copy()
    metadata = {
        **built.metadata, "variant_id": variant_id, "uses_full_truth": False,
        "source_scope": "training_pseudo_wells_and_unlabeled_seismic",
        "display_name": DISPLAY_NAMES[variant_id], "shared_low_frequency_filter": dict(lowpass_config),
        "input_lowpass_applied": True, "output_lowpass_applied": False,
        "training_well_names": list(well_names), "modifier_chain": [],
        "resolved_baseline_config": baseline_config, "resolved_modifier_configs": {},
    }
    result = LfmVariantResult(
        profile[None], np.ones((1, *truth.shape), dtype=bool), baseline_id, method,
        method_fields=dict(built.fields), metadata=metadata,
    )
    config = {
        "source_runs": {"well_control_run_dir": repo_relative_path(runtime["controls_dir"], root=repo_root)},
        "output_geometry": {"mode": "volume"}, "baselines": {baseline_id: baseline_config},
        "modifiers": {}, "variants": [{"variant_id": variant_id, "baseline_id": baseline_id, "modifier_ids": []}],
        "comparisons": [],
    }
    run_lfm_pipeline(
        config=config, controls=runtime["controls"], context=runtime["context"],
        controls_run=runtime["controls_dir"], horizon_sources=[],
        source_seismic_file=runtime["survey_path"], source_seismic_type=runtime["base"]["seismic"]["type"],
        seismic_options={}, output_dir=output_dir, repo_root=repo_root,
        prepared_results={variant_id: result},
    )
    well_indices = [int(np.argmin(np.abs(runtime["x_m"] - position))) for position in well_x]
    reference = built.fields["filtered_training_well_log_ai"]
    well_error = float(np.max(np.abs(built.log_ai[well_indices] - reference)))
    if well_error > 1e-8:
        raise ValueError(f"RGT slice LFM does not preserve the low-pass training wells: {well_error:g}.")
    config_path = _training_config(runtime, variant_id, output_dir, repo_root)
    check = check_prepared_inputs(
        config_path, repo_root=repo_root, report_dir=output_dir / "variants" / variant_id / "qc" / "adapter",
    )
    row = _row(
        variant_id=variant_id, output_dir=output_dir, runtime=runtime, repo_root=repo_root,
        config_path=config_path, check=check, uses_full_truth=False, input_lowpass_applied=True,
        training_well_names=well_names, method=method,
    )
    row["training_well_lowpass_max_abs_error_log_ai"] = well_error
    return row


def prepare_lfm_models(
    prepared_dir: Path,
    *,
    repo_root: Path,
    checkpoint_path: Path | None = None,
    cig_bench_root: Path | None = None,
    infer_shape: tuple[int, int, int] = (384, 16, 512),
    device: str = "cuda",
    variants: Sequence[str] = DEFAULT_LFM_IDS,
    lowpass_config: Mapping[str, Any] | None = None,
    truth_f_scale: float = 0.1,
    variogram: str = "exponential",
    nugget: float = 0.0,
    kriging_range_m: float | None = None,
    kriging_range_scale: float = 4.0,
    default_variant: str | None = None,
) -> dict[str, Any]:
    """Build the requested LFMs plus the shared RGT field and numeric summaries.

    The training-well trend coefficients are interpolated with an exponential
    variogram whose length scale defaults to ``kriging_range_scale`` times the
    median nearest-neighbour well distance; ``kriging_range_m`` overrides it with
    an absolute value.
    """

    prepared_dir, repo_root = Path(prepared_dir).resolve(), Path(repo_root).resolve()
    selected = tuple(dict.fromkeys(str(name) for name in variants))
    unknown = sorted(set(selected) - set(LFM_IDS))
    if unknown:
        raise ValueError(f"Unknown LFM ids: {unknown}; choose from {LFM_IDS}.")
    if not selected:
        raise ValueError("At least one LFM variant must be requested.")
    if kriging_range_m is None and (not np.isfinite(kriging_range_scale) or kriging_range_scale <= 0.0):
        raise ValueError("kriging_range_scale must be finite and positive.")
    needs_rgt = any(name in RGT_LFM_IDS for name in selected)
    if needs_rgt and checkpoint_path is None:
        raise ValueError("The RGT slicing LFM needs checkpoint_path.")
    lowpass_config = dict(lowpass_config or DEFAULT_LOWPASS_CONFIG)
    parent = prepared_dir / "lfm_models"
    runtime = _variant_inputs(prepared_dir, repo_root, lowpass_config)
    trained, well_values, well_x = _training_wells(runtime["base"], runtime["controls"])
    well_names = [control.well_name for control in trained]
    well_spacing_m = float(np.median(np.diff(well_x))) if well_x.size > 1 else float("nan")
    resolved_range_m = (
        float(kriging_range_m) if kriging_range_m is not None else float(kriging_range_scale) * well_spacing_m
    )
    range_basis = "absolute_override" if kriging_range_m is not None else f"{kriging_range_scale:g}x_median_well_spacing"
    truth_source = {
        "path": repo_relative_path(runtime["truth_path"], root=repo_root),
    }
    well_source = {
        "well_controls": {
            "path": repo_relative_path(runtime["controls_dir"] / "run_summary.json", root=repo_root),
            "well_names": well_names,
        },
        "seismic": {"path": repo_relative_path(runtime["survey_path"], root=repo_root)},
    }

    published: dict[str, Any] = {}
    if "truth_huber_trend" in selected:
        published["truth_huber_trend"] = _publish_truth_huber_trend(
            output_dir=parent / "truth_huber_trend", runtime=runtime, repo_root=repo_root,
            lowpass_config=lowpass_config, truth_f_scale=truth_f_scale, source=truth_source,
        )
    if "well_huber_trend" in selected:
        published["well_huber_trend"] = _publish_well_huber_trend(
            output_dir=parent / "well_huber_trend", runtime=runtime, repo_root=repo_root,
            lowpass_config=lowpass_config, variogram=variogram, nugget=nugget,
            kriging_range_m=resolved_range_m, well_names=well_names,
        )

    rgt_source: dict[str, Any] | None = None
    raw_rgt: np.ndarray | None = None
    if needs_rgt:
        raw_rgt, inference = infer_rgt_section(
            runtime["survey"].seismic[0], checkpoint_path=Path(checkpoint_path).resolve(),
            cig_bench_root=None if cig_bench_root is None else Path(cig_bench_root).resolve(),
            infer_shape=tuple(infer_shape), device=device,
        )
        if raw_rgt.shape != runtime["truth"].shape:
            raise ValueError(f"RGT field shape {raw_rgt.shape} must equal the truth {runtime['truth'].shape}.")
        rgt_structure = parent / "rgt_structure"
        rgt_structure.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            rgt_structure / "rgt_fields.npz", rgt_raw=raw_rgt, x_m=runtime["x_m"], twt_s=runtime["twt_s"],
            metadata_json=np.asarray(json.dumps(inference, ensure_ascii=False)),
        )
        write_json(rgt_structure / "rgt_inference_summary.json", inference)
        rgt_source = {
            "rgt_inference": inference,
            "rgt_field": {
                "path": repo_relative_path(rgt_structure / "rgt_fields.npz", root=repo_root), "key": "rgt_raw",
                "array_shape": list(raw_rgt.shape),
            },
        }
        published["rgt_lowpass_slices"] = _publish_rgt_lowpass_slices(
            output_dir=parent / "rgt_lowpass_slices", runtime=runtime, repo_root=repo_root,
            lowpass_config=lowpass_config, well_values=well_values, well_x=well_x, raw_rgt=raw_rgt,
            source={**well_source, **rgt_source}, well_names=well_names,
            variogram=variogram, nugget=nugget, kriging_range_m=resolved_range_m,
        )

    arrays = {name: _load_published(parent / name, name) for name in selected}
    manifest_path = parent / "variant_manifest.csv"
    pd.DataFrame([
        {key: row[key] for key in (
            "variant_id", "display_name", "construction_method", "uses_full_truth",
            "input_lowpass_applied", "output_lowpass_applied", "ai_min", "ai_mean", "ai_max", "run", "config",
        )}
        for row in (published[name] for name in selected)
    ]).to_csv(manifest_path, index=False)

    resolved_default = default_variant or ("rgt_lowpass_slices" if "rgt_lowpass_slices" in selected else selected[0])
    if resolved_default not in selected:
        raise ValueError(f"default_variant {resolved_default!r} was not built in this run.")
    base_path = prepared_dir / "ginn_v2.yaml"
    base = runtime["base"]
    base["ginn_v2_body_inversion"]["inputs"].update(
        lfm_run_dir=repo_relative_path(parent / resolved_default, root=repo_root), variant_id=resolved_default,
    )
    base_path.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True), encoding="utf-8")

    summary = {
        "schema_version": "marmousi2_lfm_models_v1", "status": "ok", "prepared_dir": str(prepared_dir),
        "layout_basis": "construction_recipe", "variants": {name: published[name] for name in selected},
        "shared_low_frequency_filter": lowpass_config, "extra_gaussian_smoothing": False,
        "well_trend_fit": dict(TREND_FIT_CONFIG),
        "kriging_spatial": {
            "variogram": variogram, "exact": True, "nugget": float(nugget),
            "range_m": resolved_range_m, "range_basis": range_basis,
            "kriging_range_scale": None if kriging_range_m is not None else float(kriging_range_scale),
            "median_well_spacing_m": well_spacing_m,
            "applies_to": ["well_huber_trend coefficient fields", "rgt_lowpass_slices slice weights"],
        },
        "rgt_structure": None if rgt_source is None else rgt_source["rgt_field"],
        "default_variant": resolved_default,
        "default_config_updated": repo_relative_path(base_path, root=repo_root),
        "manifest": repo_relative_path(manifest_path, root=repo_root),
        "optimizer_updates": 0,
    }
    write_json(parent / "marmousi2_lfm_summary.json", summary)
    return summary
__all__ = ["DEFAULT_LOWPASS_CONFIG", "DEFAULT_LFM_IDS", "LFM_IDS", "RGT_LFM_IDS", "prepare_lfm_models"]
