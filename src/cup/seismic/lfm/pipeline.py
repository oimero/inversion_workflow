"""Variant graph orchestration, artifact publication, and comparison QC."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cup.config.workflow import WorkflowConfig
from cup.petrel.load import import_interpretation_petrel
from cup.seismic.horizon import normalize_interpretation_unit_for_geometry
from cup.seismic.lfm.builders import BUILDERS
from cup.seismic.lfm.framework import MODIFIERS
from cup.seismic.lfm.types import LfmContext, LfmVariantResult, OutputGeometry
from cup.seismic.target_zone import TargetZone
from cup.seismic.volume_export import export_volume_like_source, log_ai_to_ai_volume
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_relative_path,
    write_json,
)
from cup.well.anchor import sample_volume_trilinear
from cup.well.real_field_controls import WellControlSet


RUN_SCHEMA = "real_field_lfm_run_v3"
VARIANT_SCHEMA = "real_field_lfm_variant_v3"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_NUMBERED_MODEL_ID = re.compile(r"^m\d+(?:[_.-].*)?$", re.IGNORECASE)


def _validate_id(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not _SAFE_ID.fullmatch(text) or _NUMBERED_MODEL_ID.fullmatch(text):
        raise ValueError(f"{label} must be a descriptive filesystem-safe ID, got {text!r}.")
    return text


def _axis_subset(axis: np.ndarray, start: float, stop: float, *, name: str) -> np.ndarray:
    if not np.isfinite(start) or not np.isfinite(stop) or start > stop:
        raise ValueError(f"Invalid {name} window [{start}, {stop}].")
    selected = axis[(axis >= start - 1e-8) & (axis <= stop + 1e-8)]
    if selected.size == 0:
        raise ValueError(f"{name} window does not intersect the survey axis.")
    if not np.isclose(selected[0], start, rtol=0.0, atol=1e-8) or not np.isclose(selected[-1], stop, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} window endpoints must lie exactly on the explicit survey axis.")
    return selected


def _section_trace_lines(points: Sequence[Mapping[str, Any]], n_traces: int, context_geometry: Any) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2 or n_traces < 2:
        raise ValueError("section requires at least two points and n_traces >= 2.")
    line_points = np.asarray([[float(point["inline"]), float(point["xline"])] for point in points], dtype=np.float64)
    xy = np.asarray([context_geometry.line_to_coord(il, xl) for il, xl in line_points], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if np.any(segment_lengths <= 0.0):
        raise ValueError("section points must describe non-zero XY segments.")
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    targets = np.linspace(0.0, cumulative[-1], n_traces)
    segment_index = np.minimum(np.searchsorted(cumulative, targets, side="right") - 1, len(segment_lengths) - 1)
    segment_index = np.maximum(segment_index, 0)
    weight = (targets - cumulative[segment_index]) / segment_lengths[segment_index]
    lines = (1.0 - weight[:, None]) * line_points[segment_index] + weight[:, None] * line_points[segment_index + 1]
    return lines[:, 0], lines[:, 1]


def resolve_output_geometry(config: Mapping[str, Any], *, survey: Any, sample_axis: Any) -> OutputGeometry:
    raw = dict(config)
    mode = str(raw.get("mode") or "").casefold()
    full_il = survey.line_geometry.inline_axis.values()
    full_xl = survey.line_geometry.xline_axis.values()
    full_samples = np.asarray(sample_axis.values, dtype=np.float64)
    if mode == "volume":
        if set(raw) != {"mode"}:
            raise ValueError("volume output_geometry accepts only mode; it always uses full source axes.")
        x_m, y_m = survey.line_geometry.trace_xy_grids(full_il, full_xl)
        return OutputGeometry(mode=mode, ilines=full_il, xlines=full_xl, samples=full_samples, x_m=x_m, y_m=y_m)
    if mode == "window":
        required = {"mode", "inline_min", "inline_max", "xline_min", "xline_max", "sample_min", "sample_max"}
        if set(raw) != required:
            raise ValueError(f"window output_geometry must contain exactly {sorted(required)}.")
        ilines = _axis_subset(full_il, float(raw["inline_min"]), float(raw["inline_max"]), name="inline")
        xlines = _axis_subset(full_xl, float(raw["xline_min"]), float(raw["xline_max"]), name="xline")
        samples = _axis_subset(full_samples, float(raw["sample_min"]), float(raw["sample_max"]), name="sample")
        if ilines.size < 2 or xlines.size < 2 or samples.size < 2:
            raise ValueError("window output requires at least two inline, xline, and sample coordinates.")
        x_m, y_m = survey.line_geometry.trace_xy_grids(ilines, xlines)
        return OutputGeometry(mode=mode, ilines=ilines, xlines=xlines, samples=samples, x_m=x_m, y_m=y_m)
    if mode == "section":
        required = {"mode", "points", "n_traces", "sample_min", "sample_max"}
        if set(raw) != required or not isinstance(raw["points"], list):
            raise ValueError(f"section output_geometry must contain exactly {sorted(required)}.")
        ilines, xlines = _section_trace_lines(raw["points"], int(raw["n_traces"]), survey.line_geometry)
        samples = _axis_subset(full_samples, float(raw["sample_min"]), float(raw["sample_max"]), name="sample")
        xy = np.asarray([survey.line_geometry.line_to_coord(il, xl) for il, xl in zip(ilines, xlines)])
        return OutputGeometry(mode=mode, ilines=ilines, xlines=xlines, samples=samples, x_m=xy[:, 0], y_m=xy[:, 1])
    raise ValueError("output_geometry.mode must be volume, window, or section.")


def build_lfm_context(
    *,
    raw_config: Mapping[str, Any],
    workflow: WorkflowConfig,
    survey: Any,
    data_root: Path,
    repo_root: Path,
    common_sources: Mapping[str, Any],
) -> tuple[LfmContext, list[dict[str, str]]]:
    sample_axis = survey.sample_axis(workflow.seismic.domain)
    geometry = survey.describe_geometry(workflow.seismic.domain)
    target = raw_config.get("target_interval")
    if not isinstance(target, Mapping) or not isinstance(target.get("horizons"), list) or len(target["horizons"]) < 2:
        raise ValueError("target_interval.horizons must contain at least two ordered horizons.")
    raw_horizons: dict[str, pd.DataFrame] = {}
    horizon_sources: list[dict[str, str]] = []
    names: list[str] = []
    for index, item in enumerate(target["horizons"]):
        if not isinstance(item, Mapping):
            raise ValueError(f"target_interval.horizons[{index}] must be a mapping.")
        name = str(item.get("name") or "").strip()
        file_text = str(item.get("file") or "").strip()
        if not name or not file_text or name in raw_horizons:
            raise ValueError(f"Invalid/duplicate target horizon at index {index}.")
        path = resolve_relative_path(file_text, root=data_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = normalize_interpretation_unit_for_geometry(import_interpretation_petrel(path), geometry)
        values = pd.to_numeric(frame["interpretation"], errors="coerce").to_numpy(dtype=np.float64)
        frame = frame.copy()
        frame["interpretation"] = np.abs(values)
        raw_horizons[name] = frame
        names.append(name)
        horizon_sources.append({"name": name, "path": repo_relative_path(path, root=repo_root)})
    target_zone = TargetZone(raw_horizons, geometry, names, min_thickness=float(sample_axis.step))
    lfm_config = raw_config.get("real_field_lfm")
    if not isinstance(lfm_config, Mapping):
        raise ValueError("Config lacks real_field_lfm section.")
    output = resolve_output_geometry(dict(lfm_config.get("output_geometry") or {}), survey=survey, sample_axis=sample_axis)
    return (
        LfmContext(
            line_geometry=survey.line_geometry,
            sample_axis=sample_axis,
            target_zone=target_zone,
            output_geometry=output,
            depth_basis=workflow.seismic.depth_basis,
            common_sources=dict(common_sources),
        ),
        horizon_sources,
    )


def _variant_graph(config: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    allowed_top = {"source_runs", "output_geometry", "baselines", "modifiers", "variants", "comparisons"}
    if set(config) != allowed_top:
        raise ValueError(f"real_field_lfm must contain exactly {sorted(allowed_top)}.")
    baselines_raw = config.get("baselines")
    modifiers_raw = config.get("modifiers")
    variants_raw = config.get("variants")
    comparisons_raw = config.get("comparisons")
    if not isinstance(baselines_raw, Mapping) or not baselines_raw:
        raise ValueError("real_field_lfm.baselines must be a non-empty mapping.")
    if modifiers_raw is None:
        modifiers_raw = {}
    if not isinstance(modifiers_raw, Mapping):
        raise ValueError("real_field_lfm.modifiers must be a mapping.")
    if not isinstance(variants_raw, list) or not variants_raw:
        raise ValueError("real_field_lfm.variants must be a non-empty explicit list.")
    if comparisons_raw is None:
        comparisons_raw = []
    if not isinstance(comparisons_raw, list):
        raise ValueError("real_field_lfm.comparisons must be a list.")
    baselines = {_validate_id(key, label="baseline ID"): dict(value) for key, value in baselines_raw.items() if isinstance(value, Mapping)}
    modifiers = {_validate_id(key, label="modifier ID"): dict(value) for key, value in modifiers_raw.items() if isinstance(value, Mapping)}
    if len(baselines) != len(baselines_raw) or len(modifiers) != len(modifiers_raw):
        raise ValueError("Every baseline/modifier config must be a mapping.")
    for baseline_id, baseline in baselines.items():
        if str(baseline.get("method") or "") not in BUILDERS:
            raise ValueError(f"Unknown method for baseline {baseline_id!r}.")
    for modifier_id, modifier in modifiers.items():
        if str(modifier.get("method") or "") not in MODIFIERS:
            raise ValueError(f"Unknown method for modifier {modifier_id!r}.")
    variants: list[dict[str, Any]] = []
    seen_variants: set[str] = set()
    for index, raw in enumerate(variants_raw):
        if not isinstance(raw, Mapping) or set(raw) != {"variant_id", "baseline_id", "modifier_ids"}:
            raise ValueError(f"variants[{index}] must contain exactly variant_id/baseline_id/modifier_ids.")
        variant_id = _validate_id(raw["variant_id"], label="variant_id")
        if variant_id.casefold() in seen_variants:
            raise ValueError(f"Duplicate variant_id: {variant_id}")
        seen_variants.add(variant_id.casefold())
        baseline_id = str(raw["baseline_id"])
        modifier_ids = list(raw["modifier_ids"]) if isinstance(raw["modifier_ids"], list) else None
        if baseline_id not in baselines or modifier_ids is None or any(item not in modifiers for item in modifier_ids):
            raise ValueError(f"Variant {variant_id!r} references unknown baseline/modifier.")
        if len(modifier_ids) != len(set(modifier_ids)):
            raise ValueError(f"Variant {variant_id!r} repeats a modifier ID.")
        methods = [str(modifiers[item]["method"]) for item in modifier_ids]
        if len(methods) != len(set(methods)):
            raise ValueError(f"Variant {variant_id!r} repeats a modifier method and would overwrite sidecar fields.")
        variants.append({"variant_id": variant_id, "baseline_id": baseline_id, "modifier_ids": modifier_ids})
    comparisons: list[dict[str, Any]] = []
    seen_comparisons: set[str] = set()
    valid_variant_ids = {item["variant_id"] for item in variants}
    for index, raw in enumerate(comparisons_raw):
        if not isinstance(raw, Mapping):
            raise ValueError(f"comparisons[{index}] must be a mapping.")
        allowed = {"comparison_id", "left_variant_id", "right_variant_id", "sections"}
        if not set(raw).issubset(allowed) or not {"comparison_id", "left_variant_id", "right_variant_id"}.issubset(raw):
            raise ValueError(f"comparisons[{index}] has invalid keys.")
        comparison_id = _validate_id(raw["comparison_id"], label="comparison_id")
        if comparison_id.casefold() in seen_comparisons:
            raise ValueError(f"Duplicate comparison_id: {comparison_id}")
        seen_comparisons.add(comparison_id.casefold())
        left, right = str(raw["left_variant_id"]), str(raw["right_variant_id"])
        if left not in valid_variant_ids or right not in valid_variant_ids or left == right:
            raise ValueError(f"Comparison {comparison_id!r} references invalid/equal variants.")
        comparisons.append({"comparison_id": comparison_id, "left_variant_id": left, "right_variant_id": right, "sections": raw.get("sections", [])})
    all_ids = [*baselines, *modifiers, *(item["variant_id"] for item in variants), *(item["comparison_id"] for item in comparisons)]
    folded_ids = [item.casefold() for item in all_ids]
    duplicates = sorted({item for item in folded_ids if folded_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Baseline/modifier/variant/comparison IDs must be globally unique: {duplicates}")
    return baselines, modifiers, variants, comparisons


def _framework_well_effect_qc(
    *,
    baseline: LfmVariantResult,
    modified: LfmVariantResult,
    controls: WellControlSet,
    context: LfmContext,
) -> pd.DataFrame:
    class_fields = {
        key.split("__", 1)[1]: value
        for key, value in modified.modifier_fields.items()
        if key.startswith("class_probability__")
    }
    rows = []
    output = context.output_geometry
    for control in controls.controls:
        if output.is_section:
            section_xy = np.column_stack([output.x_m, output.y_m])
            control_xy = np.column_stack([control.x_m_by_sample, control.y_m_by_sample])
            trace_indices = np.argmin(np.linalg.norm(control_xy[:, None, :] - section_xy[None, :, :], axis=2), axis=1)
            sample_indices = np.searchsorted(output.samples, control.sample_axis.values)
            sample_indices = np.clip(sample_indices, 0, output.samples.size - 1)
            inside = np.isclose(output.samples[sample_indices], control.sample_axis.values, rtol=0.0, atol=1e-8)
            base_values = baseline.log_ai[trace_indices, sample_indices]
            modified_values = modified.log_ai[trace_indices, sample_indices]
            probabilities = {
                name: field[trace_indices, sample_indices] for name, field in class_fields.items()
            }
        else:
            kwargs = {
                "ilines": output.ilines,
                "xlines": output.xlines,
                "twt_s": output.samples,
                "inline_values": control.inline_by_sample,
                "xline_values": control.xline_by_sample,
                "sample_twt_s": control.sample_axis.values,
            }
            base_values, base_inside = sample_volume_trilinear(baseline.log_ai, **kwargs)
            modified_values, modified_inside = sample_volume_trilinear(modified.log_ai, **kwargs)
            inside = base_inside & modified_inside
            probabilities = {name: sample_volume_trilinear(field, **kwargs)[0] for name, field in class_fields.items()}
        valid = control.valid_mask & inside & np.isfinite(base_values) & np.isfinite(modified_values)
        delta = modified_values - base_values
        row: dict[str, Any] = {
            "well_name": control.well_name,
            "wellbore_class": control.wellbore_class,
            "sampling_mode": control.sampling_mode,
            "n_valid_samples": int(np.count_nonzero(valid)),
            "n_modified_samples": int(np.count_nonzero(valid & (np.abs(delta) > 0.0))),
            "delta_log_ai_mean": float(np.mean(delta[valid])) if np.any(valid) else np.nan,
            "delta_log_ai_max_abs": float(np.max(np.abs(delta[valid]))) if np.any(valid) else np.nan,
        }
        for class_name, probability in probabilities.items():
            row[f"{class_name}_probability_mean"] = float(np.mean(probability[valid])) if np.any(valid) else np.nan
            row[f"{class_name}_probability_max"] = float(np.max(probability[valid])) if np.any(valid) else np.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def build_lfm_variants(
    *, config: Mapping[str, Any], controls: WellControlSet, context: LfmContext, repo_root: Path
) -> tuple[dict[str, LfmVariantResult], list[dict[str, Any]]]:
    baselines, modifiers, variants, comparisons = _variant_graph(config)
    required_baselines = {item["baseline_id"] for item in variants}
    baseline_results: dict[str, LfmVariantResult] = {}
    for baseline_id in required_baselines:
        baseline_config = baselines[baseline_id]
        builder = BUILDERS[str(baseline_config["method"])]
        baseline_results[baseline_id] = builder.build(
            baseline_id=baseline_id, config=baseline_config, controls=controls, context=context
        )
    results: dict[str, LfmVariantResult] = {}
    for variant in variants:
        parent = baseline_results[variant["baseline_id"]]
        result = replace(
            parent,
            log_ai=parent.log_ai.copy(),
            valid_mask_model=parent.valid_mask_model.copy(),
            method_fields=dict(parent.method_fields),
            modifier_fields={},
            qc_tables=dict(parent.qc_tables),
            metadata={**parent.metadata, "modifier_chain": []},
        )
        modifier_sources: dict[str, Any] = {}
        for modifier_id in variant["modifier_ids"]:
            modifier_config = dict(modifiers[modifier_id])
            if str(modifier_config.get("method")) == "framework":
                bodies_path = resolve_relative_path(modifier_config["bodies_file"], root=repo_root)
                if not bodies_path.is_file():
                    raise FileNotFoundError(bodies_path)
                modifier_config["bodies_file"] = str(bodies_path)
                modifier_sources[modifier_id] = {
                    "bodies_file": repo_relative_path(bodies_path, root=repo_root),
                }
            result = MODIFIERS[str(modifier_config["method"])].apply(
                modifier_id=modifier_id, config=modifier_config, parent=result, context=context
            )
        if result.modifier_fields:
            result.qc_tables["well_framework_effect_qc"] = _framework_well_effect_qc(
                baseline=baseline_results[variant["baseline_id"]],
                modified=result,
                controls=controls,
                context=context,
            )
        result.metadata.update(
            {
                "variant_id": variant["variant_id"],
                "baseline_id": variant["baseline_id"],
                "baseline_method": result.baseline_method,
                "modifier_chain": list(variant["modifier_ids"]),
                "resolved_baseline_config": baselines[variant["baseline_id"]],
                "resolved_modifier_configs": {key: modifiers[key] for key in variant["modifier_ids"]},
                "modifier_sources": modifier_sources,
            }
        )
        result.validate(context)
        results[variant["variant_id"]] = result
    first = next(iter(results.values()))
    for variant_id, result in results.items():
        if not np.array_equal(result.valid_mask_model, first.valid_mask_model):
            raise ValueError(f"Variant {variant_id!r} valid_mask_model differs from the run-wide authoritative mask.")
    return results, comparisons


def _save_array_sidecar(path: Path, fields: Mapping[str, np.ndarray]) -> None:
    if not fields:
        raise ValueError(f"Cannot write empty sidecar: {path}")
    arrays = {key: np.asarray(value) for key, value in fields.items()}
    if any(value.dtype == object for value in arrays.values()):
        raise ValueError(f"Sidecar contains object array: {path}")
    np.savez_compressed(path, **arrays)


def _write_variant_figures(
    *,
    result: LfmVariantResult,
    context: LfmContext,
    directory: Path,
    final_directory: Path,
    repo_root: Path,
) -> dict[str, dict[str, str]]:
    figure_dir = directory / "qc" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if result.log_ai.ndim == 3:
        section_log_ai = result.log_ai[result.log_ai.shape[0] // 2]
        lateral = context.output_geometry.xlines
        horizon_sections = [
            _surface_for_figure(context, name)[result.log_ai.shape[0] // 2]
            for name in context.target_zone.horizon_names
        ]
    else:
        section_log_ai = result.log_ai
        lateral = np.arange(result.log_ai.shape[0], dtype=np.float64)
        horizon_sections = [_surface_for_figure(context, name) for name in context.target_zone.horizon_names]
    samples = context.output_geometry.samples
    path = figure_dir / "lfm_representative_section.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, values, title in (
        (axes[0], section_log_ai, "log(AI)"),
        (axes[1], np.exp(section_log_ai), "linear AI (m/s*g/cm3)"),
    ):
        mesh = axis.pcolormesh(lateral, samples, values.T, shading="auto", cmap="viridis")
        for name, horizon in zip(context.target_zone.horizon_names, horizon_sections):
            axis.plot(lateral, horizon, linewidth=0.8, label=name)
        axis.set_title(title)
        axis.set_xlabel("xline" if result.log_ai.ndim == 3 else "section trace")
        fig.colorbar(mesh, ax=axis)
    axes[0].set_ylim(float(samples[-1]), float(samples[0]))
    axes[0].set_ylabel(f"{context.sample_axis.domain} ({context.sample_axis.unit})")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs = {
        "lfm_representative_section": {
            "path": repo_relative_path(final_directory / "qc" / "figures" / path.name, root=repo_root),
        }
    }
    class_keys = sorted(key for key in result.modifier_fields if key.startswith("class_probability__"))
    if class_keys:
        framework_path = figure_dir / "framework_map_and_sections.png"
        fig, axes = plt.subplots(len(class_keys), 2, figsize=(12, 4.5 * len(class_keys)), squeeze=False)
        bodies = list(result.metadata.get("framework_bodies", []))
        for row, key in enumerate(class_keys):
            class_name = key.split("__", 1)[1]
            map_values = result.modifier_fields[f"class_map_qc__{class_name}"]
            if context.output_geometry.is_section:
                axes[row, 0].scatter(context.output_geometry.x_m, context.output_geometry.y_m, c=map_values, cmap="magma")
                probability_section = result.modifier_fields[key]
                probability_lateral = np.arange(probability_section.shape[0])
            else:
                axes[row, 0].pcolormesh(
                    context.output_geometry.x_m,
                    context.output_geometry.y_m,
                    map_values,
                    shading="auto",
                    cmap="magma",
                )
                probability_section = result.modifier_fields[key][result.modifier_fields[key].shape[0] // 2]
                probability_lateral = context.output_geometry.xlines
            for body in bodies:
                if body.get("framework_class") != class_name:
                    continue
                vertices = np.asarray(body["vertices_xy_m"], dtype=np.float64)
                closed = np.vstack([vertices, vertices[0]])
                axes[row, 0].plot(closed[:, 0], closed[:, 1], color="cyan", linewidth=1.0)
                axes[row, 0].text(
                    float(np.mean(vertices[:, 0])),
                    float(np.mean(vertices[:, 1])),
                    f"{body['body_id']} [{body['u_top']},{body['u_bottom']}]",
                    fontsize=7,
                )
            axes[row, 0].set_title(f"{class_name} body map")
            mesh = axes[row, 1].pcolormesh(
                probability_lateral,
                samples,
                probability_section.T,
                shading="auto",
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, 1].invert_yaxis()
            axes[row, 1].set_title(f"{class_name} probability section")
            fig.colorbar(mesh, ax=axes[row, 1])
        fig.tight_layout()
        fig.savefig(framework_path, dpi=150)
        plt.close(fig)
        outputs["framework_map_and_sections"] = {
            "path": repo_relative_path(final_directory / "qc" / "figures" / framework_path.name, root=repo_root),
        }
    return outputs


def _surface_for_figure(context: LfmContext, horizon_name: str) -> np.ndarray:
    from cup.seismic.lfm.builders import _surface_for_output

    return _surface_for_output(context, horizon_name)


def _write_variant(
    *,
    variant_id: str,
    result: LfmVariantResult,
    context: LfmContext,
    temp_root: Path,
    final_root: Path,
    repo_root: Path,
    controls_run: Path,
    controls_contract_fingerprint: str,
    horizon_sources: list[dict[str, str]],
) -> dict[str, Any]:
    relative = Path("variants") / variant_id
    directory = temp_root / relative
    final_directory = final_root / relative
    qc_dir = directory / "qc"
    qc_dir.mkdir(parents=True)
    method_path = directory / "method_fields.npz"
    _save_array_sidecar(method_path, result.method_fields)
    modifier_path = directory / "modifier_fields.npz"
    if result.modifier_fields:
        _save_array_sidecar(modifier_path, result.modifier_fields)
    qc_outputs: dict[str, dict[str, str]] = {}
    for name, frame in result.qc_tables.items():
        path = qc_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        qc_outputs[name] = {"path": repo_relative_path(final_directory / "qc" / path.name, root=repo_root)}
    figure_outputs = _write_variant_figures(
        result=result,
        context=context,
        directory=directory,
        final_directory=final_directory,
        repo_root=repo_root,
    )
    input_contracts = {
        "well_control_set": {
            "path": repo_relative_path(controls_run / "run_summary.json", root=repo_root),
            "contract_fingerprint_sha256": controls_contract_fingerprint,
        }
    }
    method_sidecar_path = repo_relative_path(final_directory / method_path.name, root=repo_root)
    modifier_sidecar_path = (
        repo_relative_path(final_directory / modifier_path.name, root=repo_root)
        if result.modifier_fields
        else None
    )
    metadata = {
        "schema_version": VARIANT_SCHEMA,
        "variant_id": variant_id,
        "baseline_id": result.baseline_id,
        "baseline_method": result.baseline_method,
        "modifier_chain": list(result.metadata.get("modifier_chain", [])),
        "sample_domain": context.sample_axis.domain,
        "sample_unit": context.sample_axis.unit,
        "depth_basis": context.depth_basis,
        "value_key": "log_ai",
        "value_domain": "log(AI)",
        "linear_ai_unit": "m/s*g/cm3",
        "valid_mask_key": "valid_mask_model",
        "output_geometry": context.output_geometry.describe(),
        "input_contracts": input_contracts,
        "seismic_path": context.common_sources["seismic"]["path"],
        "horizon_paths": horizon_sources,
        "resolved_baseline_config": result.metadata["resolved_baseline_config"],
        "resolved_modifier_configs": dict(result.metadata["resolved_modifier_configs"]),
        "modifier_source_paths": dict(result.metadata.get("modifier_sources", {})),
        "method_sidecar_path": method_sidecar_path,
        "modifier_sidecar_path": modifier_sidecar_path,
        "method_metadata": {key: value for key, value in result.metadata.items() if not key.startswith("resolved_")},
    }
    lfm_path = directory / "lfm.npz"
    np.savez_compressed(
        lfm_path,
        log_ai=result.log_ai.astype(np.float32),
        valid_mask_model=result.valid_mask_model.astype(bool),
        ilines=context.output_geometry.ilines.astype(np.float64),
        xlines=context.output_geometry.xlines.astype(np.float64),
        samples=context.output_geometry.samples.astype(np.float64),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    primary_artifacts = {
        "lfm": lfm_path,
        "method_fields": method_path,
    }
    if result.modifier_fields:
        primary_artifacts["modifier_fields"] = modifier_path
    variant_config = {
        "baseline": result.metadata["resolved_baseline_config"],
        "modifiers": dict(result.metadata["resolved_modifier_configs"]),
    }
    contract_fingerprint = contract_fingerprint_sha256(
        contract_schema_version=VARIANT_SCHEMA,
        semantics={
            "variant_id": variant_id,
            "sample_domain": context.sample_axis.domain,
            "sample_unit": context.sample_axis.unit,
            "depth_basis": context.depth_basis,
            "value_domain": "log(AI)",
            "linear_ai_unit": "m/s*g/cm3",
            "output_geometry": context.output_geometry.describe(),
        },
        business_config=variant_config,
        input_contracts=input_contracts,
        primary_artifacts=primary_artifacts,
    )
    summary = {
        "schema_version": VARIANT_SCHEMA,
        "status": "ok",
        "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
        "contract_fingerprint_sha256": contract_fingerprint,
        "input_contracts": input_contracts,
        "variant_id": variant_id,
        "metadata": metadata,
        "stats": {
            "valid_sample_count": int(np.count_nonzero(result.valid_mask_model)),
            "log_ai_min": float(np.min(result.log_ai[result.valid_mask_model])),
            "log_ai_median": float(np.median(result.log_ai[result.valid_mask_model])),
            "log_ai_max": float(np.max(result.log_ai[result.valid_mask_model])),
        },
        "outputs": {
            "lfm": {"path": repo_relative_path(final_directory / lfm_path.name, root=repo_root)},
            "method_fields": {"path": method_sidecar_path},
            "modifier_fields": None if modifier_sidecar_path is None else {"path": modifier_sidecar_path},
            "qc": qc_outputs,
            "figures": figure_outputs,
        },
    }
    summary_path = directory / "variant_summary.json"
    write_json(summary_path, summary)
    return {
        "variant_id": variant_id,
        "baseline_id": result.baseline_id,
        "baseline_method": result.baseline_method,
        "modifier_chain": ";".join(result.metadata.get("modifier_chain", [])),
        "status": "ok",
        "lfm_path": summary["outputs"]["lfm"]["path"],
        "method_fields_path": method_sidecar_path,
        "modifier_fields_path": "" if modifier_sidecar_path is None else modifier_sidecar_path,
        "variant_summary_path": repo_relative_path(final_directory / summary_path.name, root=repo_root),
        "contract_fingerprint_sha256": contract_fingerprint,
    }


def _comparison_well_metrics(
    left: LfmVariantResult,
    right: LfmVariantResult,
    *,
    controls: WellControlSet,
    context: LfmContext,
) -> pd.DataFrame:
    rows = []
    output = context.output_geometry
    for control in controls.controls:
        if output.is_section:
            points = np.column_stack([output.x_m, output.y_m])
            control_xy = np.column_stack([control.x_m_by_sample, control.y_m_by_sample])
            trace_index = np.argmin(np.linalg.norm(control_xy[:, None, :] - points[None, :, :], axis=2), axis=1)
            sample_index = np.searchsorted(output.samples, context.sample_axis.values)
            inside = (sample_index >= 0) & (sample_index < output.samples.size)
            sample_index = np.clip(sample_index, 0, output.samples.size - 1)
            left_values = left.log_ai[trace_index, sample_index]
            right_values = right.log_ai[trace_index, sample_index]
            left_mask = left.valid_mask_model[trace_index, sample_index]
            right_mask = right.valid_mask_model[trace_index, sample_index]
        else:
            kwargs = {
                "ilines": output.ilines,
                "xlines": output.xlines,
                "twt_s": output.samples,
                "inline_values": control.inline_by_sample,
                "xline_values": control.xline_by_sample,
                "sample_twt_s": context.sample_axis.values,
            }
            left_values, left_inside = sample_volume_trilinear(left.log_ai, **kwargs)
            right_values, right_inside = sample_volume_trilinear(right.log_ai, **kwargs)
            left_mask_float, left_mask_inside = sample_volume_trilinear(left.valid_mask_model.astype(np.float32), **kwargs)
            right_mask_float, right_mask_inside = sample_volume_trilinear(right.valid_mask_model.astype(np.float32), **kwargs)
            inside = left_inside & right_inside & left_mask_inside & right_mask_inside
            left_mask = left_mask_float > 0.5
            right_mask = right_mask_float > 0.5
        well = np.asarray(control.log_ai.values, dtype=np.float64)
        valid = control.valid_mask & inside & left_mask & right_mask & np.isfinite(left_values) & np.isfinite(right_values)
        rows.append(
            {
                "well_name": control.well_name,
                "n_valid": int(np.count_nonzero(valid)),
                "left_rmse_log_ai": float(np.sqrt(np.mean((well[valid] - left_values[valid]) ** 2))) if np.any(valid) else np.nan,
                "right_rmse_log_ai": float(np.sqrt(np.mean((well[valid] - right_values[valid]) ** 2))) if np.any(valid) else np.nan,
                "mean_right_minus_left_log_ai": float(np.mean(right_values[valid] - left_values[valid])) if np.any(valid) else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _comparison_figure(path: Path, left: LfmVariantResult, right: LfmVariantResult, *, left_id: str, right_id: str) -> None:
    if left.log_ai.ndim == 3:
        index = left.log_ai.shape[0] // 2
        left_section, right_section = left.log_ai[index], right.log_ai[index]
    else:
        left_section, right_section = left.log_ai, right.log_ai
    left_ai, right_ai = np.exp(left_section), np.exp(right_section)
    delta_log = right_section - left_section
    delta_ai = right_ai - left_ai
    percent = 100.0 * (right_ai / left_ai - 1.0)
    panels = [left_section, right_section, left_ai, right_ai, delta_log, delta_ai, percent]
    titles = [f"{left_id} logAI", f"{right_id} logAI", f"{left_id} AI", f"{right_id} AI", "right-left logAI", "right-left AI", "AI percent difference"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 5), sharey=True)
    for axis, values, title in zip(axes, panels, titles):
        cmap = "coolwarm" if "right-left" in title or "percent" in title else "viridis"
        image = axis.imshow(values.T, aspect="auto", origin="upper", cmap=cmap)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("sample index")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _configured_comparison_section(
    result: LfmVariantResult,
    section: Mapping[str, Any],
    *,
    context: LfmContext,
) -> np.ndarray:
    if result.log_ai.ndim != 3:
        raise ValueError("Configured comparison sections require volume/window variants.")
    if set(section) != {"section_id", "direction", "line"}:
        raise ValueError("Comparison section must contain exactly section_id/direction/line.")
    direction = str(section["direction"]).casefold()
    line = float(section["line"])
    if direction == "inline":
        matches = np.flatnonzero(np.isclose(context.output_geometry.ilines, line, rtol=0.0, atol=1e-8))
        if matches.size != 1:
            raise ValueError(f"Configured comparison inline is not on the output axis: {line}")
        return result.log_ai[int(matches[0]), :, :]
    if direction == "xline":
        matches = np.flatnonzero(np.isclose(context.output_geometry.xlines, line, rtol=0.0, atol=1e-8))
        if matches.size != 1:
            raise ValueError(f"Configured comparison xline is not on the output axis: {line}")
        return result.log_ai[:, int(matches[0]), :]
    raise ValueError("Comparison section.direction must be inline or xline.")


def _write_comparisons(
    *,
    comparisons: list[dict[str, Any]],
    results: Mapping[str, LfmVariantResult],
    controls: WellControlSet,
    context: LfmContext,
    temp_root: Path,
    final_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    outputs = {}
    for comparison in comparisons:
        comparison_id = comparison["comparison_id"]
        left = results[comparison["left_variant_id"]]
        right = results[comparison["right_variant_id"]]
        if not np.array_equal(left.valid_mask_model, right.valid_mask_model):
            raise ValueError(f"Comparison {comparison_id!r} variants do not share a mask.")
        directory = temp_root / "comparisons" / comparison_id
        final_directory = final_root / "comparisons" / comparison_id
        directory.mkdir(parents=True)
        valid = left.valid_mask_model
        left_ai, right_ai = np.exp(left.log_ai[valid]), np.exp(right.log_ai[valid])
        metrics = pd.DataFrame(
            [
                {
                    "comparison_id": comparison_id,
                    "left_variant_id": comparison["left_variant_id"],
                    "right_variant_id": comparison["right_variant_id"],
                    "n_valid": int(np.count_nonzero(valid)),
                    "mean_delta_log_ai": float(np.mean(right.log_ai[valid] - left.log_ai[valid])),
                    "rms_delta_log_ai": float(np.sqrt(np.mean((right.log_ai[valid] - left.log_ai[valid]) ** 2))),
                    "mean_delta_linear_ai": float(np.mean(right_ai - left_ai)),
                    "mean_percent_difference": float(np.mean(100.0 * (right_ai / left_ai - 1.0))),
                }
            ]
        )
        metrics_path = directory / "metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        well_path = directory / "well_metrics.csv"
        _comparison_well_metrics(left, right, controls=controls, context=context).to_csv(well_path, index=False)
        figure_path = directory / "figures" / "overview.png"
        _comparison_figure(
            figure_path,
            left,
            right,
            left_id=comparison["left_variant_id"],
            right_id=comparison["right_variant_id"],
        )
        figures = {
            "overview": {
                "path": repo_relative_path(final_directory / "figures" / "overview.png", root=repo_root),
            }
        }
        sections = comparison.get("sections") or []
        if not isinstance(sections, list):
            raise ValueError(f"Comparison {comparison_id!r} sections must be a list.")
        seen_section_ids: set[str] = set()
        for section in sections:
            if not isinstance(section, Mapping):
                raise ValueError(f"Comparison {comparison_id!r} section must be a mapping.")
            section_id = _validate_id(section.get("section_id"), label="comparison section_id")
            if section_id.casefold() in seen_section_ids:
                raise ValueError(f"Comparison {comparison_id!r} repeats section_id={section_id!r}.")
            seen_section_ids.add(section_id.casefold())
            left_section = _configured_comparison_section(left, section, context=context)
            right_section = _configured_comparison_section(right, section, context=context)
            left_proxy = LfmVariantResult(left_section, np.isfinite(left_section), left.baseline_id, left.baseline_method)
            right_proxy = LfmVariantResult(right_section, np.isfinite(right_section), right.baseline_id, right.baseline_method)
            section_path = directory / "figures" / f"{section_id}.png"
            _comparison_figure(
                section_path,
                left_proxy,
                right_proxy,
                left_id=comparison["left_variant_id"],
                right_id=comparison["right_variant_id"],
            )
            figures[section_id] = {
                "path": repo_relative_path(final_directory / "figures" / section_path.name, root=repo_root),
            }
        outputs[comparison_id] = {
            "metrics": {"path": repo_relative_path(final_directory / "metrics.csv", root=repo_root)},
            "well_metrics": {"path": repo_relative_path(final_directory / "well_metrics.csv", root=repo_root)},
            "figures": figures,
        }
    return outputs


def run_lfm_pipeline(
    *,
    config: Mapping[str, Any],
    controls: WellControlSet,
    context: LfmContext,
    controls_run: Path,
    horizon_sources: list[dict[str, str]],
    source_seismic_file: Path,
    source_seismic_type: str,
    seismic_options: Mapping[str, Any],
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    temp_root = output_dir.parent / f".{output_dir.name}.tmp-{uuid4().hex}"
    temp_root.mkdir(parents=True)
    controls_summary_path = controls_run / "run_summary.json"
    with controls_summary_path.open("r", encoding="utf-8") as handle:
        controls_summary = json.load(handle)
    controls_contract_fingerprint = require_contract_fingerprint(
        controls_summary, label=f"WellControlSet {controls_run}"
    )
    input_contracts = {
        "well_control_set": {
            "path": repo_relative_path(controls_summary_path, root=repo_root),
            "contract_fingerprint_sha256": controls_contract_fingerprint,
        }
    }
    try:
        results, comparisons = build_lfm_variants(config=config, controls=controls, context=context, repo_root=repo_root)
        rows = [
            _write_variant(
                variant_id=variant_id,
                result=result,
                context=context,
                temp_root=temp_root,
                final_root=output_dir,
                repo_root=repo_root,
                controls_run=controls_run,
                controls_contract_fingerprint=controls_contract_fingerprint,
                horizon_sources=horizon_sources,
            )
            for variant_id, result in results.items()
        ]
        comparison_outputs = _write_comparisons(
            comparisons=comparisons,
            results=results,
            controls=controls,
            context=context,
            temp_root=temp_root,
            final_root=output_dir,
            repo_root=repo_root,
        )
        export_outputs = {}
        if context.output_geometry.mode == "volume":
            if not (
                np.array_equal(context.output_geometry.ilines, context.line_geometry.inline_axis.values())
                and np.array_equal(context.output_geometry.xlines, context.line_geometry.xline_axis.values())
                and np.array_equal(context.output_geometry.samples, context.sample_axis.values)
            ):
                raise ValueError("volume mode axes must exactly match the source seismic axes before export.")
            for variant_id, result in results.items():
                actual_base = temp_root / "variants" / variant_id / f"{variant_id}_linear_ai"
                payload = export_volume_like_source(
                    output_base=actual_base,
                    volume=log_ai_to_ai_volume(result.log_ai),
                    ilines=context.output_geometry.ilines,
                    xlines=context.output_geometry.xlines,
                    samples=context.output_geometry.samples,
                    source_seismic_file=source_seismic_file,
                    source_seismic_type=source_seismic_type,
                    title=f"Unified LFM v3 linear AI: {variant_id}",
                    details=[
                        f"variant_id={variant_id}",
                        f"domain={context.sample_axis.domain}",
                        "unit=m/s*g/cm3",
                        f"lfm_npz={repo_relative_path(output_dir / 'variants' / variant_id / 'lfm.npz', root=repo_root)}",
                        f"variant_contract_fingerprint={next(row['contract_fingerprint_sha256'] for row in rows if row['variant_id'] == variant_id)}",
                    ],
                    seismic_options=seismic_options,
                    nan_fill=None,
                )
                payload.pop("sha256", None)
                actual_path = Path(payload["path"])
                payload["path"] = repo_relative_path(output_dir / "variants" / variant_id / actual_path.name, root=repo_root)
                export_outputs[variant_id] = payload
        manifest_path = temp_root / "variant_manifest.csv"
        pd.DataFrame.from_records(rows).to_csv(manifest_path, index=False)
        run_contract_fingerprint = contract_fingerprint_sha256(
            contract_schema_version=RUN_SCHEMA,
            semantics={
                "sample_domain": context.sample_axis.domain,
                "sample_unit": context.sample_axis.unit,
                "depth_basis": context.depth_basis,
                "output_geometry": context.output_geometry.describe(),
                "requested_variant_ids": list(results),
            },
            business_config=config,
            input_contracts=input_contracts,
            primary_artifacts={"variant_manifest": manifest_path},
        )
        summary = {
            "schema_version": RUN_SCHEMA,
            "status": "ok",
            "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
            "contract_fingerprint_sha256": run_contract_fingerprint,
            "input_contracts": input_contracts,
            "resolved_config": dict(config),
            "seismic": context.common_sources["seismic"],
            "horizons": horizon_sources,
            "output_geometry": context.output_geometry.describe(),
            "requested_variant_ids": list(results),
            "requested_comparison_ids": [item["comparison_id"] for item in comparisons],
            "outputs": {
                "variant_manifest": {
                    "path": repo_relative_path(output_dir / manifest_path.name, root=repo_root),
                },
                "comparisons": comparison_outputs,
                "volume_exports": export_outputs,
            },
        }
        write_json(temp_root / "lfm_run_summary.json", summary)
        temp_root.replace(output_dir)
        return summary
    except Exception as exc:
        failed = {
            "schema_version": RUN_SCHEMA,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(temp_root / "lfm_run_summary.json", failed)
        failed_dir = output_dir.parent / f"{output_dir.name}_failed_{uuid4().hex[:8]}"
        temp_root.replace(failed_dir)
        raise


__all__ = [
    "RUN_SCHEMA",
    "VARIANT_SCHEMA",
    "build_lfm_context",
    "build_lfm_variants",
    "resolve_output_geometry",
    "run_lfm_pipeline",
]
