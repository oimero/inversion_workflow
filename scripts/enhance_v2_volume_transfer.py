"""Transfer Enhance V2 residual texture to a complete GINN body volume."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cup.config.workflow import WorkflowConfig, deep_merge_dict
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.seismic.target_zone_io import build_workflow_target_zone
from cup.seismic.volume_export import export_volume_like_source, log_ai_to_ai_volume
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, write_json
from cup.utils.logging import configure_run_logger
from cup.well.real_field_controls import load_well_control_set
from enhance_v2.artifacts import library_summary
from enhance_v2.contracts import ResidualTransferPolicy, ScaleContract
from enhance_v2.library import build_residual_library
from enhance_v2.volume import (
    ResidualTextureVolumeTransfer,
    VolumeTransferConfig,
    ZoneSurface,
)
from enhance_v2.workflow import select_controls, well_zone_intervals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/enhance_v2/enhance_v2.yaml"))
    parser.add_argument("--ginn-body", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke-size", type=int, default=None)
    parser.add_argument("--skip-segy-export", action="store_true")
    return parser.parse_args()


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _load_composed_config(path: Path) -> dict[str, Any]:
    experiment = load_yaml_config(path)
    workflow_config = str(experiment.get("workflow_config") or "").strip()
    if not workflow_config:
        return experiment
    common = load_yaml_config(resolve_relative_path(workflow_config, root=REPO_ROOT))
    overlay = {key: value for key, value in experiment.items() if key != "workflow_config"}
    return deep_merge_dict(common, overlay)


def _output_dir(value: Path | None) -> Path:
    if value is not None:
        return resolve_relative_path(value, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "experiments" / "enhance_v2" / "results" / f"volume_{timestamp}"


def _load_segy_linear_ai(
    path: Path,
    *,
    domain: str,
    seismic_options: Mapping[str, Any],
) -> tuple[np.ndarray, Any]:
    """Load one regular linear-AI SEG-Y using its scanned geometry."""

    import cigsegy

    survey = open_survey(path, "segy", segy_options=seismic_options)
    sample_axis = survey.sample_axis(domain)
    geometry_indices = np.asarray(survey.geom, dtype=np.int64)
    valid_geometry = geometry_indices >= 0
    if not np.any(valid_geometry):
        raise ValueError("GINN body SEG-Y contains no valid survey traces.")
    trace_stop = int(np.max(geometry_indices[valid_geometry])) + 1
    segy = cigsegy.Pysegy(str(path))
    try:
        traces = np.asarray(
            segy.collect(0, trace_stop, 0, sample_axis.values.size),
            dtype=np.float32,
        ).reshape(trace_stop, sample_axis.values.size)
    finally:
        segy.close()
    volume = np.full((*geometry_indices.shape, sample_axis.values.size), np.nan, dtype=np.float32)
    volume[valid_geometry] = traces[geometry_indices[valid_geometry]]
    finite = np.isfinite(volume)
    if np.any(volume[finite] <= 0.0):
        raise ValueError("GINN body linear AI must be positive wherever finite.")
    log_ai = np.full(volume.shape, np.nan, dtype=np.float32)
    log_ai[finite] = np.log(volume[finite]).astype(np.float32)
    return log_ai, survey


def _crop_bounds(size: int, smoke_size: int | None) -> tuple[int, int]:
    if smoke_size is None:
        return 0, int(size)
    if isinstance(smoke_size, bool) or int(smoke_size) != smoke_size or not 2 <= int(smoke_size) <= size:
        raise ValueError("smoke_size must be an integer in [2, axis size].")
    start = (int(size) - int(smoke_size)) // 2
    return start, start + int(smoke_size)


def _plot_sections(
    body: np.ndarray,
    residual: np.ndarray,
    enhanced: np.ndarray,
    sample_axis: Any,
    line_geometry: Any,
    ilines: np.ndarray,
    xlines: np.ndarray,
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    samples = np.asarray(sample_axis.values, dtype=np.float64)
    for orientation, fixed in (("inline", body.shape[0] // 2), ("xline", body.shape[1] // 2)):
        if orientation == "inline":
            local_body = body[fixed]
            local_residual = residual[fixed]
            local_enhanced = enhanced[fixed]
            xy = np.asarray(
                [line_geometry.line_to_coord(ilines[fixed], value) for value in xlines],
                dtype=np.float64,
            )
        else:
            local_body = body[:, fixed]
            local_residual = residual[:, fixed]
            local_enhanced = enhanced[:, fixed]
            xy = np.asarray(
                [line_geometry.line_to_coord(value, xlines[fixed]) for value in ilines],
                dtype=np.float64,
            )
        support = np.isfinite(local_enhanced)
        vertical = np.flatnonzero(np.any(support, axis=0))
        if vertical.size < 2:
            continue
        start, stop = int(vertical[0]), int(vertical[-1]) + 1
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        body_values = np.concatenate((local_body[support], local_enhanced[support]))
        body_min, body_max = np.quantile(body_values, [0.01, 0.99]).astype(float)
        residual_limit = max(float(np.quantile(np.abs(local_residual[support]), 0.995)), 1.0e-6)
        panels = (
            (local_body, "GINN V2 body log-AI", "viridis", body_min, body_max),
            (local_residual, "Enhance V2 residual log-AI", "RdBu_r", -residual_limit, residual_limit),
            (local_enhanced, "Enhanced log-AI", "viridis", body_min, body_max),
        )
        figure, axes = plt.subplots(1, 3, figsize=(18.0, 6.0), sharex=True, sharey=True)
        extent = [float(distance[0]), float(distance[-1]), float(samples[stop - 1]), float(samples[start])]
        for current, (values, title, cmap, vmin, vmax) in zip(axes, panels):
            image = current.imshow(
                values[:, start:stop].T,
                origin="upper",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            current.set_title(title)
            current.set_xlabel("lateral distance (m)")
            current.set_ylabel(sample_axis.unit)
            figure.colorbar(image, ax=current, fraction=0.04, pad=0.025)
        figure.suptitle(f"Enhance V2 full-volume transfer — center {orientation}")
        figure.tight_layout()
        path = output_dir / f"center_{orientation}.png"
        figure.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(figure)
        generated.append(repo_relative_path(path, root=REPO_ROOT))
    return generated


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    experiment = load_yaml_config(config_path)
    section = _required_mapping(experiment.get("enhance_v2"), name="enhance_v2")
    inputs = _required_mapping(section.get("inputs"), name="enhance_v2.inputs")
    dictionary_config = _required_mapping(section.get("dictionary"), name="enhance_v2.dictionary")
    transfer_config = _required_mapping(section.get("transfer"), name="enhance_v2.transfer")
    volume_config_raw = _required_mapping(section.get("volume"), name="enhance_v2.volume")
    ginn_config_path = resolve_relative_path(str(experiment.get("ginn_config") or ""), root=REPO_ROOT)
    ginn_body_path = resolve_relative_path(
        args.ginn_body or str(inputs.get("ginn_body_linear_ai") or ""),
        root=REPO_ROOT,
    )
    if not ginn_body_path.is_file():
        raise FileNotFoundError(ginn_body_path)
    output_dir = _output_dir(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Enhance V2 volume output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    log = configure_run_logger(output_dir, logger_name="enhance_v2_volume", file_name="volume_transfer.log")

    raw = _load_composed_config(ginn_config_path)
    workflow = WorkflowConfig.from_mapping(raw)
    ginn_section = _required_mapping(raw.get("ginn_v2_body_inversion"), name="ginn_v2_body_inversion")
    ginn_inputs = _required_mapping(ginn_section.get("inputs"), name="ginn_v2_body_inversion.inputs")
    ginn_training = _required_mapping(
        ginn_section.get("training"),
        name="ginn_v2_body_inversion.training",
    )
    trusted_well_names = tuple(str(value) for value in ginn_training.get("trusted_well_names") or ())
    if not trusted_well_names:
        raise ValueError("ginn_v2_body_inversion.training.trusted_well_names must be non-empty.")
    data_root = resolve_relative_path(workflow.data_root, root=REPO_ROOT)
    source_seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    seismic_options = segy_options_from_config(workflow.seismic.as_dict()) if workflow.seismic.type == "segy" else {}
    if workflow.seismic.type != "segy":
        raise ValueError("The current GINN body artifact is SEG-Y; Enhance V2 volume transfer requires a SEG-Y source workflow.")
    source_survey = open_survey(source_seismic_path, workflow.seismic.type, segy_options=seismic_options)
    sample_axis = source_survey.sample_axis(workflow.seismic.domain)
    body, body_survey = _load_segy_linear_ai(
        ginn_body_path,
        domain=workflow.seismic.domain,
        seismic_options=seismic_options,
    )
    if not np.array_equal(body_survey.line_geometry.inline_axis.values(), source_survey.line_geometry.inline_axis.values()):
        raise ValueError("GINN body and source seismic inline axes differ.")
    if not np.array_equal(body_survey.line_geometry.xline_axis.values(), source_survey.line_geometry.xline_axis.values()):
        raise ValueError("GINN body and source seismic xline axes differ.")
    if not np.array_equal(body_survey.sample_axis(workflow.seismic.domain).values, sample_axis.values):
        raise ValueError("GINN body and source seismic sample axes differ.")

    target_zone, horizon_sources = build_workflow_target_zone(
        raw_config=raw,
        survey=source_survey,
        data_root=data_root,
        repo_root=REPO_ROOT,
    )
    well_control_run_dir = resolve_relative_path(str(ginn_inputs["well_control_run_dir"]), root=REPO_ROOT)
    controls = load_well_control_set(well_control_run_dir, repo_root=REPO_ROOT)
    selected_controls = select_controls(controls, trusted_well_names)
    zone_intervals_by_well = well_zone_intervals(selected_controls, target_zone)
    scale_contract = ScaleContract.from_any(
        {
            **dict(dictionary_config),
            "sample_axis": controls.sample_axis,
            "zone_intervals_by_well": zone_intervals_by_well,
        }
    )
    library = build_residual_library(selected_controls, scale_contract)
    write_json(output_dir / "library_summary.json", library_summary(library))
    log.info(
        "dictionary ready | atoms=%d | zones=%d | wells=%d",
        len(library.atoms),
        len(library.zone_ids),
        len(library.source_wells),
    )

    ilines_full = np.asarray(body_survey.line_geometry.inline_axis.values(), dtype=np.float64)
    xlines_full = np.asarray(body_survey.line_geometry.xline_axis.values(), dtype=np.float64)
    il_start, il_stop = _crop_bounds(ilines_full.size, args.smoke_size)
    xl_start, xl_stop = _crop_bounds(xlines_full.size, args.smoke_size)
    body = body[il_start:il_stop, xl_start:xl_stop]
    ilines = ilines_full[il_start:il_stop]
    xlines = xlines_full[xl_start:xl_stop]
    zones = tuple(
        ZoneSurface(
            zone_id=f"{top_name}__to__{bottom_name}",
            top=np.asarray(target_zone.get_horizon_grid(top_name), dtype=np.float64)[il_start:il_stop, xl_start:xl_stop],
            bottom=np.asarray(target_zone.get_horizon_grid(bottom_name), dtype=np.float64)[il_start:il_stop, xl_start:xl_stop],
        )
        for top_name, bottom_name in target_zone.iter_zones()
    )
    policy = ResidualTransferPolicy.from_any(transfer_config)
    config = VolumeTransferConfig(
        orientations=tuple(volume_config_raw.get("orientations") or ("inline", "xline")),
        log_every_sections=int(volume_config_raw.get("log_every_sections") or 10),
        max_workers=int(volume_config_raw.get("max_workers") or 1),
    )
    log.info(
        "volume transfer start | body=%s | shape=%s | orientations=%s",
        ginn_body_path,
        body.shape,
        ",".join(config.orientations),
    )
    result = ResidualTextureVolumeTransfer(library, policy, config, logger=log).transfer(
        body,
        sample_axis=sample_axis,
        line_geometry=source_survey.line_geometry,
        ilines=ilines,
        xlines=xlines,
        zones=zones,
    )
    figures = _plot_sections(
        body,
        result.predicted_residual_log_ai,
        result.enhanced_log_ai,
        sample_axis,
        source_survey.line_geometry,
        ilines,
        xlines,
        output_dir / "figures",
    )

    full_volume = args.smoke_size is None
    exports: dict[str, Any] = {}
    if not args.skip_segy_export:
        if not full_volume:
            raise ValueError("Smoke transfer requires --skip-segy-export.")
        exports["enhanced_linear_ai"] = export_volume_like_source(
            output_base=output_dir / "enhance_v2_linear_ai",
            volume=log_ai_to_ai_volume(result.enhanced_log_ai),
            ilines=ilines,
            xlines=xlines,
            samples=sample_axis.values,
            source_seismic_file=source_seismic_path,
            source_seismic_type=workflow.seismic.type,
            title="Enhance V2 conditional residual texture linear AI",
            details=[
                f"ginn_body={repo_relative_path(ginn_body_path, root=REPO_ROOT)}",
                "enhanced_log_ai=ginn_body_log_ai+predicted_residual_log_ai",
                "orientation_fusion=equal_mean",
            ],
            seismic_options=workflow.seismic.as_dict(),
            nan_fill=None,
        )
        if bool(dict(volume_config_raw.get("exports") or {}).get("residual_log_ai", True)):
            residual_export = np.where(
                result.direction_count > 0,
                result.predicted_residual_log_ai,
                np.nan,
            ).astype(np.float32)
            exports["residual_log_ai"] = export_volume_like_source(
                output_base=output_dir / "enhance_v2_residual_log_ai",
                volume=residual_export,
                ilines=ilines,
                xlines=xlines,
                samples=sample_axis.values,
                source_seismic_file=source_seismic_path,
                source_seismic_type=workflow.seismic.type,
                title="Enhance V2 predicted residual log-AI",
                details=["unit=log-AI", "orientation_fusion=equal_mean"],
                seismic_options=workflow.seismic.as_dict(),
                nan_fill=None,
            )

    summary = {
        "status": "completed",
        "mode": "smoke" if not full_volume else "full_volume",
        "ginn_body_linear_ai": repo_relative_path(ginn_body_path, root=REPO_ROOT),
        "scale_contract": library_summary(library),
        "horizon_sources": horizon_sources,
        "axes": {
            "ilines": [float(ilines[0]), float(ilines[-1]), int(ilines.size)],
            "xlines": [float(xlines[0]), float(xlines[-1]), int(xlines.size)],
            "sample_axis": sample_axis.describe(),
        },
        "result": dict(result.summary),
        "figures": figures,
        "exports": exports,
    }
    write_json(output_dir / "volume_transfer_summary.json", summary)
    log.info("volume transfer finished | summary=%s", json.dumps(result.summary))
    print("=== Enhance V2 full-volume transfer ===")
    print(f"Output: {output_dir}")
    print(f"Mode: {summary['mode']}")
    for name, payload in exports.items():
        print(f"{name}: {payload['path']}")


if __name__ == "__main__":
    main()
