"""Run the first conditional residual-texture prototype on frozen GINN sections."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ginn_v2_body_inversion as ginn_entry
from cup.config.workflow import WorkflowConfig
from cup.lfm.artifacts import load_lfm_input
from cup.lfm.math import parse_lowpass_spec
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.seismic.target_zone_io import build_workflow_target_zone
from cup.utils.io import load_yaml_config, resolve_relative_path, write_json
from cup.utils.logging import configure_run_logger
from cup.well.real_field_controls import WellControl, WellControlSet, load_well_control_set
from enhance_v2.artifacts import library_summary, result_summary
from enhance_v2.contracts import ResidualTransferPolicy, ScaleContract, TransferGeometry
from enhance_v2.library import build_residual_library
from enhance_v2.transfer import transfer_residual_texture
from ginn_v2.adapters import DepthDomainAdapter, TimeDomainAdapter
from ginn_v2.checkpoint import load_checkpoint
from ginn_v2.data import PatchReader, SurveyTraceSource, candidate_patch_keys, fit_lfm_normalization
from ginn_v2.inverter import BodyInverter
from ginn_v2.model import CenterTraceBodyNet
from ginn_v2.trainer import BodyInversionConfig, BodyInversionTrainer, build_body_inversion_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/enhance_v2/enhance_v2.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument("--orientations", nargs="+", choices=("inline", "xline"), default=None)
    return parser.parse_args()


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _resolve_output_dir(value: Path | None) -> Path:
    if value is not None:
        return resolve_relative_path(value, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "experiments" / "enhance_v2" / "results" / timestamp


def _trusted_controls(controls: WellControlSet, names: tuple[str, ...]) -> WellControlSet:
    by_name = {item.well_name.casefold(): item for item in controls.controls}
    selected: list[WellControl] = []
    for name in names:
        control = by_name.get(name.casefold())
        if control is None:
            raise ValueError(f"Trusted GINN well is absent from WellControlSet: {name}")
        selected.append(control)
    return WellControlSet(
        sample_axis=controls.sample_axis,
        controls=tuple(selected),
        sample_domain=controls.sample_domain,
        sample_unit=controls.sample_unit,
        depth_basis=controls.depth_basis,
        source_run_type=controls.source_run_type,
        provenance=controls.provenance,
    )


def _well_zone_intervals(
    controls: WellControlSet,
    target_zone: Any,
) -> dict[str, dict[str, tuple[float, float]]]:
    intervals_by_well: dict[str, dict[str, tuple[float, float]]] = {}
    for control in controls.controls:
        axis = np.asarray(control.sample_axis.values, dtype=np.float64)
        position_valid = np.isfinite(control.inline_by_sample) & np.isfinite(control.xline_by_sample)
        markers: list[tuple[str, float]] = []
        for horizon_name in target_zone.horizon_names:
            local_horizon = np.full(axis.shape, np.nan, dtype=np.float64)
            for index in np.flatnonzero(position_valid):
                local_horizon[index] = target_zone.get_horizon_interpretation_at_location(
                    horizon_name,
                    float(control.inline_by_sample[index]),
                    float(control.xline_by_sample[index]),
                )
            candidates = np.flatnonzero(np.isfinite(local_horizon))
            if candidates.size == 0:
                raise ValueError(f"{control.well_name}: horizon {horizon_name!r} has no well-path support.")
            selected = int(candidates[np.argmin(np.abs(axis[candidates] - local_horizon[candidates]))])
            markers.append((horizon_name, float(axis[selected])))
        if any(markers[index + 1][1] <= markers[index][1] for index in range(len(markers) - 1)):
            raise ValueError(f"{control.well_name}: target horizons are not ordered along the well path.")
        intervals_by_well[control.well_name] = {
            f"{top_name}__to__{bottom_name}": (top, bottom)
            for (top_name, top), (bottom_name, bottom) in zip(markers[:-1], markers[1:])
        }
    return intervals_by_well


def _section_geometry(
    keys: tuple[Any, ...],
    body: np.ndarray,
    valid_mask: np.ndarray,
    *,
    reader: PatchReader,
    target_zone: Any,
    orientation: str,
) -> tuple[TransferGeometry, np.ndarray, dict[str, np.ndarray]]:
    axis = np.asarray(reader.sample_axis.values, dtype=np.float64)
    zone_ids = np.empty(body.shape, dtype=object)
    zone_ids[:] = None
    support = np.asarray(valid_mask, dtype=bool) & np.isfinite(body)
    horizon_values = {
        name: np.asarray(
            [target_zone.get_horizon_grid(name)[key.inline_index, key.xline_index] for key in keys],
            dtype=np.float64,
        )
        for name in target_zone.horizon_names
    }
    for zone_index, (top_name, bottom_name) in enumerate(target_zone.iter_zones()):
        zone_id = f"{top_name}__to__{bottom_name}"
        top = horizon_values[top_name]
        bottom = horizon_values[bottom_name]
        for lateral_index in range(len(keys)):
            if not np.isfinite(top[lateral_index]) or not np.isfinite(bottom[lateral_index]) or top[lateral_index] >= bottom[lateral_index]:
                continue
            selected = (axis >= top[lateral_index]) & (
                axis <= bottom[lateral_index] if zone_index == len(target_zone.iter_zones()) - 1 else axis < bottom[lateral_index]
            )
            zone_ids[lateral_index, selected] = zone_id
    support &= np.not_equal(zone_ids, None)
    zone_ids[~support] = None
    xy = np.asarray(
        [
            reader.geometry.line_to_coord(
                reader.ilines[key.inline_index],
                reader.xlines[key.xline_index],
            )
            for key in keys
        ],
        dtype=np.float64,
    )
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    geometry = TransferGeometry(
        sample_axis=reader.sample_axis,
        x_m=xy[:, 0],
        y_m=xy[:, 1],
        support=support,
        zone_ids=zone_ids,
        pinchout_mask=np.zeros(body.shape, dtype=bool),
        orientation=orientation,
    )
    return geometry, distance, horizon_values


def _finite_crop(support: np.ndarray) -> slice:
    columns = np.flatnonzero(np.any(support, axis=0))
    if columns.size == 0:
        raise ValueError("Prototype section has no transfer support.")
    return slice(max(0, int(columns[0]) - 2), min(support.shape[1], int(columns[-1]) + 3))


def _plot_field(
    axis: Any,
    field: np.ndarray,
    support: np.ndarray,
    distance_m: np.ndarray,
    sample_axis: np.ndarray,
    crop: slice,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
) -> Any:
    values = np.where(support[:, crop], field[:, crop], np.nan)
    image = axis.pcolormesh(
        distance_m / 1000.0,
        sample_axis[crop],
        values.T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel("lateral distance (km)")
    axis.set_ylabel("TVDSS (m)")
    return image


def _plot_comparison(
    result: Any,
    distance_m: np.ndarray,
    sample_axis: np.ndarray,
    horizons: Mapping[str, np.ndarray],
    path: Path,
    *,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    support = np.asarray(result.support, dtype=bool)
    crop = _finite_crop(support)
    residuals = [
        ("hard nearest", result.hard_nearest_residual),
        ("uniform zone mixture", result.uniform_residual),
        ("local soft dictionary", result.soft_residual),
        ("spatial soft dictionary", result.predicted_residual),
    ]
    residual_values = np.concatenate([np.abs(np.asarray(values)[support]) for _, values in residuals])
    residual_limit = max(float(np.quantile(residual_values, 0.995)), 1.0e-6)
    enhanced = [np.asarray(result.ginn_body) + np.asarray(values) for _, values in residuals]
    body_values = np.concatenate([values[support] for values in [np.asarray(result.ginn_body), *enhanced]])
    body_min, body_max = np.quantile(body_values, [0.01, 0.99]).astype(float)

    figure, axes = plt.subplots(2, 4, figsize=(19.0, 9.0), sharex=True, sharey=True)
    for column, ((label, residual), enhanced_field) in enumerate(zip(residuals, enhanced)):
        residual_image = _plot_field(
            axes[0, column], np.asarray(residual), support, distance_m, sample_axis, crop,
            cmap="RdBu_r", vmin=-residual_limit, vmax=residual_limit, title=f"{label}\nresidual log-AI",
        )
        body_image = _plot_field(
            axes[1, column], enhanced_field, support, distance_m, sample_axis, crop,
            cmap="viridis", vmin=body_min, vmax=body_max, title=f"{label}\nenhanced log-AI",
        )
        for current in (axes[0, column], axes[1, column]):
            for horizon in horizons.values():
                current.plot(distance_m / 1000.0, horizon, color="black", linewidth=0.8, alpha=0.75)
    figure.colorbar(residual_image, ax=axes[0, :].tolist(), fraction=0.015, pad=0.015, label="residual log-AI")
    figure.colorbar(body_image, ax=axes[1, :].tolist(), fraction=0.015, pad=0.015, label="log-AI")
    figure.suptitle("Conditional residual texture transfer: baselines and spatial result")
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_weights(
    result: Any,
    distance_m: np.ndarray,
    sample_axis: np.ndarray,
    path: Path,
    *,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    records = result.dictionary_weight_summary["node_records"]
    source_wells = tuple(result.dictionary_weight_summary["source_wells"])
    source_index = {name: index for index, name in enumerate(source_wells)}
    node_x: list[float] = []
    node_z: list[float] = []
    node_source: list[int] = []
    node_confidence: list[float] = []
    for record in records:
        weights = dict(record["source_well_weight"])
        dominant, weight = max(weights.items(), key=lambda item: item[1])
        node_x.append(float(distance_m[int(record["lateral_index"])]) / 1000.0)
        node_z.append(float(record["center_m"]))
        node_source.append(source_index[dominant])
        node_confidence.append(float(weight))

    support = np.asarray(result.support, dtype=bool)
    crop = _finite_crop(support)
    difference = np.asarray(result.predicted_residual) - np.asarray(result.soft_residual)
    limit = max(float(np.quantile(np.abs(difference[support]), 0.995)), 1.0e-6)
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.5), sharey=True)
    scatter = axes[0].scatter(
        node_x, node_z, c=node_source, s=10.0 + 30.0 * np.asarray(node_confidence),
        cmap="tab10", vmin=-0.5, vmax=max(0.5, len(source_wells) - 0.5), alpha=0.85,
    )
    axes[0].invert_yaxis()
    axes[0].set_title("dominant source well\n(size = source weight)")
    axes[0].set_xlabel("lateral distance (km)")
    axes[0].set_ylabel("TVDSS (m)")
    colorbar = figure.colorbar(scatter, ax=axes[0], ticks=np.arange(len(source_wells)))
    colorbar.ax.set_yticklabels(source_wells)
    effective = _plot_field(
        axes[1], np.asarray(result.effective_dictionary_count), support, distance_m, sample_axis, crop,
        cmap="magma", vmin=1.0, vmax=max(2.0, float(np.quantile(result.effective_dictionary_count[support], 0.99))),
        title="effective dictionary count",
    )
    difference_image = _plot_field(
        axes[2], difference, support, distance_m, sample_axis, crop,
        cmap="RdBu_r", vmin=-limit, vmax=limit, title="spatial minus local soft residual",
    )
    figure.colorbar(effective, ax=axes[1], fraction=0.045, pad=0.03)
    figure.colorbar(difference_image, ax=axes[2], fraction=0.045, pad=0.03)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_temperature_sweep(
    results: Mapping[float, Any],
    distance_m: np.ndarray,
    sample_axis: np.ndarray,
    path: Path,
    *,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(results.items())
    support = np.asarray(ordered[0][1].support, dtype=bool)
    crop = _finite_crop(support)
    limit = max(
        float(np.quantile(np.concatenate([np.abs(item.predicted_residual[support]) for _, item in ordered]), 0.995)),
        1.0e-6,
    )
    figure, axes = plt.subplots(1, len(ordered), figsize=(6.0 * len(ordered), 5.5), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)
    image = None
    for axis, (temperature, result) in zip(axes_array, ordered):
        rms = float(result.summary["residual_rms"])
        effective = float(result.summary["effective_dictionary_count_median"])
        image = _plot_field(
            axis, result.predicted_residual, support, distance_m, sample_axis, crop,
            cmap="RdBu_r", vmin=-limit, vmax=limit,
            title=f"temperature × {temperature:g}\nRMS={rms:.4f}, Neff={effective:.2f}",
        )
    figure.colorbar(image, ax=axes_array.tolist(), fraction=0.018, pad=0.02, label="residual log-AI")
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    experiment = load_yaml_config(config_path)
    ginn_config_path = resolve_relative_path(str(experiment.get("ginn_config") or ""), root=REPO_ROOT)
    if not ginn_config_path.is_file():
        raise FileNotFoundError(ginn_config_path)
    section = _required_mapping(experiment.get("enhance_v2"), name="enhance_v2")
    inputs = _required_mapping(section.get("inputs"), name="enhance_v2.inputs")
    dictionary_config = _required_mapping(section.get("dictionary"), name="enhance_v2.dictionary")
    transfer_config = _required_mapping(section.get("transfer"), name="enhance_v2.transfer")
    prototype_config = _required_mapping(section.get("prototype"), name="enhance_v2.prototype")
    output_dir = _resolve_output_dir(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Prototype output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    logger = configure_run_logger(output_dir, logger_name="enhance_v2_prototype", file_name="prototype.log")

    raw = ginn_entry._load_composed_config(ginn_config_path)
    workflow = WorkflowConfig.from_mapping(raw)
    ginn_section = _required_mapping(raw.get("ginn_v2_body_inversion"), name="ginn_v2_body_inversion")
    ginn_inputs = _required_mapping(ginn_section.get("inputs"), name="ginn_v2_body_inversion.inputs")
    ginn_config = BodyInversionConfig.from_mapping(_required_mapping(ginn_section.get("training"), name="ginn training"))
    lfm_run_dir = resolve_relative_path(str(ginn_inputs["lfm_run_dir"]), root=REPO_ROOT)
    well_control_run_dir = resolve_relative_path(str(ginn_inputs["well_control_run_dir"]), root=REPO_ROOT)
    forward_inputs_path = resolve_relative_path(str(ginn_inputs["forward_model_inputs"]), root=REPO_ROOT)
    variant_id = str(ginn_inputs["variant_id"])
    ginn_run_dir = resolve_relative_path(str(inputs.get("ginn_run_dir") or ""), root=REPO_ROOT)
    checkpoint_path = ginn_run_dir / "selected_checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    controls = load_well_control_set(well_control_run_dir, repo_root=REPO_ROOT)
    selected_controls = _trusted_controls(controls, tuple(ginn_config.trusted_well_names))
    lfm = load_lfm_input(
        {"lfm_run_dir": str(lfm_run_dir), "variant_id": variant_id, "well_control_run_dir": str(well_control_run_dir)},
        repo_root=REPO_ROOT,
    )
    data_root = resolve_relative_path(workflow.data_root, root=REPO_ROOT)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    survey_options = segy_options_from_config(workflow.seismic.as_dict()) if workflow.seismic.type == "segy" else {}
    survey = open_survey(seismic_path, workflow.seismic.type, segy_options=survey_options or None)
    sample_axis = survey.sample_axis(workflow.seismic.domain)
    if sample_axis.domain != "depth" or sample_axis.depth_basis != "tvdss":
        raise ValueError("The first Enhance V2 prototype requires the current depth/TVDSS workflow.")
    target_zone, horizon_sources = build_workflow_target_zone(
        raw_config=raw, survey=survey, data_root=data_root, repo_root=REPO_ROOT
    )
    if not np.array_equal(sample_axis.values, lfm.sample_axis.values) or not np.array_equal(sample_axis.values, controls.sample_axis.values):
        raise ValueError("Seismic, LFM and well-control SampleAxis values differ.")

    baseline_config = dict(lfm.variant.variant_metadata.get("resolved_baseline_config") or {})
    lfm_lowpass_spec = parse_lowpass_spec(dict(baseline_config.get("filter") or {}), sample_axis)
    wavelet_time_s, wavelet_amplitude, relation, _forward_payload = ginn_entry._load_forward_inputs(
        forward_inputs_path, domain=workflow.seismic.domain, depth_basis=workflow.seismic.depth_basis
    )
    if relation is None:
        raise ValueError("Depth prototype requires the frozen AI--Vp relation.")
    valid = np.asarray(lfm.valid_mask, dtype=bool)
    velocity = np.full(lfm.log_ai.shape, np.nan, dtype=np.float64)
    velocity[valid] = relation.velocity_from_ai(np.exp(np.asarray(lfm.log_ai[valid], dtype=np.float64)))
    adapter = DepthDomainAdapter(
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
        patch_radius=ginn_config.patch_radius,
        domain_extras={"velocity_mps": velocity},
        cache_size=ginn_config.cache_size,
        seismic_feature_mode=ginn_config.seismic_feature_mode,
        seismic_balance_window_samples=ginn_config.seismic_balance_window_samples,
        seismic_balance_floor_fraction=ginn_config.seismic_balance_floor_fraction,
    )
    candidates = candidate_patch_keys(
        lfm.log_ai, lfm.valid_mask, patch_radius=ginn_config.patch_radius, orientations=ginn_config.orientations
    )
    data = build_body_inversion_data(
        reader,
        controls,
        config=ginn_config,
        lfm_lowpass_spec=lfm_lowpass_spec,
        candidate_keys=candidates,
        target_zone_mask=np.asarray(lfm.valid_mask, dtype=bool),
    )
    trainer = BodyInversionTrainer(
        data,
        adapter=adapter,
        config=ginn_config,
        lfm_lowpass_spec=lfm_lowpass_spec,
        output_dir=output_dir / "_ginn_runtime",
        artifact_root=REPO_ROOT,
        logger=logger,
    )
    model = CenterTraceBodyNet(ginn_config.network).to(trainer.device)
    load_checkpoint(checkpoint_path, model=model, expected_network_config=ginn_config.network, map_location=trainer.device)
    inverter = BodyInverter(
        model, reader, adapter, projector=trainer.projector, device=trainer.device, batch_size=ginn_config.batch_size
    )

    zone_intervals_by_well = _well_zone_intervals(selected_controls, target_zone)
    scale_contract = ScaleContract.from_any(
        {
            **dict(dictionary_config),
            "sample_axis": controls.sample_axis,
            "zone_intervals_by_well": zone_intervals_by_well,
        }
    )
    library = build_residual_library(selected_controls, scale_contract)
    write_json(output_dir / "library_summary.json", library_summary(library))
    logger.info(
        "dictionary ready | atoms=%d | zones=%d | wells=%d",
        len(library.atoms), len(library.zone_ids), len(library.source_wells),
    )

    policy = ResidualTransferPolicy.from_any(transfer_config)
    orientations = tuple(args.orientations or prototype_config.get("orientations") or ("inline", "xline"))
    max_traces = int(args.max_traces or prototype_config.get("max_traces") or 64)
    dpi = int(prototype_config.get("figure_dpi") or 180)
    run_summary: dict[str, Any] = {
        "status": "completed",
        "ginn_checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "trusted_wells": list(ginn_config.trusted_well_names),
        "horizon_sources": horizon_sources,
        "library": library_summary(library),
        "sections": {},
    }
    for orientation in orientations:
        section_dir = output_dir / orientation
        section_dir.mkdir(parents=True)
        keys = trainer.validation_section_keys(orientation, max_traces=max_traces)
        logger.info("%s section inference start | traces=%d", orientation, len(keys))
        body_result = inverter.predict(keys, center_visible=True)
        body = body_result.body_log_ai.detach().cpu().numpy().astype(np.float64)
        valid_mask = body_result.valid_mask.detach().cpu().numpy().astype(bool)
        geometry, distance_m, horizons = _section_geometry(
            keys, body, valid_mask, reader=reader, target_zone=target_zone, orientation=orientation
        )
        temperature_results: dict[float, Any] = {}
        for multiplier in policy.temperature_multipliers:
            logger.info("%s transfer | temperature_multiplier=%.3f", orientation, multiplier)
            temperature_results[float(multiplier)] = transfer_residual_texture(
                body,
                geometry,
                library,
                replace(policy, temperature_multiplier=float(multiplier)),
            )
        result = temperature_results[float(policy.temperature_multiplier)]
        _plot_comparison(
            result, distance_m, sample_axis.values, horizons, section_dir / "comparison.png", dpi=dpi
        )
        _plot_weights(
            result, distance_m, sample_axis.values, section_dir / "weights_and_continuity.png", dpi=dpi
        )
        _plot_temperature_sweep(
            temperature_results, distance_m, sample_axis.values, section_dir / "temperature_sweep.png", dpi=dpi
        )
        arrays: dict[str, Any] = {
            "sample_axis": sample_axis.values,
            "lateral_distance_m": distance_m,
            "ginn_body": result.ginn_body,
            "spatial_residual": result.predicted_residual,
            "enhanced_log_ai": result.enhanced_log_ai,
            "soft_residual": result.soft_residual,
            "uniform_residual": result.uniform_residual,
            "hard_nearest_residual": result.hard_nearest_residual,
            "effective_dictionary_count": result.effective_dictionary_count,
            "support": result.support,
        }
        for multiplier, temperature_result in temperature_results.items():
            arrays[f"temperature_{multiplier:g}_residual"] = temperature_result.predicted_residual
        for name, values in horizons.items():
            arrays[f"horizon_{name}"] = values
        np.savez_compressed(section_dir / "section_fields.npz", **arrays)
        write_json(section_dir / "summary.json", result_summary(result))
        run_summary["sections"][orientation] = {
            "trace_count": len(keys),
            "lateral_extent_m": float(distance_m[-1]),
            "result": result_summary(result),
            "temperature_results": {
                str(multiplier): result_summary(item)
                for multiplier, item in temperature_results.items()
            },
            "figures": [
                str((section_dir / name).relative_to(REPO_ROOT))
                for name in ("comparison.png", "weights_and_continuity.png", "temperature_sweep.png")
            ],
        }
        logger.info(
            "%s section complete | residual_rms=%.5f | effective_dictionary_count=%.2f",
            orientation,
            result.summary["residual_rms"],
            result.summary["effective_dictionary_count_median"],
        )
    write_json(output_dir / "run_summary.json", run_summary)
    print("=== Enhance V2 conditional residual texture prototype ===")
    print(f"Output: {output_dir}")
    print("Figures: comparison.png, weights_and_continuity.png, temperature_sweep.png")


if __name__ == "__main__":
    main()
