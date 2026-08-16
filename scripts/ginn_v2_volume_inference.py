"""Run deterministic GINN V2 body inference over a depth/time survey volume."""

from __future__ import annotations

import argparse
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

import ginn_v2_body_inversion as training_entry
from cup.config.workflow import WorkflowConfig
from cup.lfm.artifacts import load_lfm_input
from cup.lfm.math import parse_lowpass_spec
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.seismic.volume_export import export_volume_like_source, log_ai_to_ai_volume
from cup.utils.io import repo_relative_path, resolve_relative_path, write_json
from cup.utils.logging import configure_run_logger
from ginn_v2.adapters import DepthDomainAdapter, TimeDomainAdapter
from ginn_v2.checkpoint import load_checkpoint
from ginn_v2.data import PatchReader, SurveyTraceSource, fit_lfm_normalization
from ginn_v2.inverter import BodyInverter
from ginn_v2.model import CenterTraceBodyNet
from ginn_v2.projector import BodyScaleProjector
from ginn_v2.trainer import BodyInversionConfig
from ginn_v2.volume import BodyVolumeInverter, VolumeInferenceConfig, centered_tile_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/ginn_v2/ginn_v2.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--smoke-tile-size", type=int, default=None)
    parser.add_argument(
        "--smoke-tile-origin",
        choices=("center", "northwest"),
        default="center",
        help="Place a smoke tile at survey center or northwest corner.",
    )
    parser.add_argument("--skip-segy-export", action="store_true")
    return parser.parse_args()


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _output_dir(value: Path | None) -> Path:
    if value is not None:
        return resolve_relative_path(value, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "experiments" / "ginn_v2" / "results" / f"volume_{timestamp}"


def _plot_sections(
    result: Any,
    lfm_log_ai: np.ndarray,
    sample_axis: Any,
    geometry: Any,
    ilines: np.ndarray,
    xlines: np.ndarray,
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    body = np.asarray(result.body_log_ai, dtype=np.float32)
    count = np.asarray(result.direction_count, dtype=np.uint8)
    fill_code = np.asarray(result.fill_code, dtype=np.uint8)
    disagreement = np.asarray(result.direction_disagreement_log_ai, dtype=np.float32)
    lfm = np.asarray(lfm_log_ai, dtype=np.float32)
    increment = body - lfm
    files: list[str] = []
    axis = np.asarray(sample_axis.values, dtype=np.float64)
    candidates = (
        ("inline", body.shape[0] // 2),
        ("xline", body.shape[1] // 2),
    )
    for orientation, fixed in candidates:
        if orientation == "inline":
            section_body = body[fixed]
            section_lfm = lfm[fixed]
            section_increment = increment[fixed]
            section_disagreement = disagreement[fixed]
            section_support = fill_code[fixed] > 0
            xy = np.asarray(
                [geometry.line_to_coord(ilines[fixed], value) for value in xlines],
                dtype=np.float64,
            )
        else:
            section_body = body[:, fixed]
            section_lfm = lfm[:, fixed]
            section_increment = increment[:, fixed]
            section_disagreement = disagreement[:, fixed]
            section_support = fill_code[:, fixed] > 0
            xy = np.asarray(
                [geometry.line_to_coord(value, xlines[fixed]) for value in ilines],
                dtype=np.float64,
            )
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        vertical = np.flatnonzero(np.any(section_support, axis=0))
        if vertical.size < 2:
            continue
        start, stop = int(vertical[0]), int(vertical[-1]) + 1
        support = section_support[:, start:stop]
        body_values = np.concatenate(
            (
                section_lfm[:, start:stop][support],
                section_body[:, start:stop][support],
            )
        )
        body_min, body_max = np.quantile(body_values, [0.01, 0.99]).astype(float)
        increment_values = np.abs(section_increment[:, start:stop][support])
        increment_limit = max(float(np.quantile(increment_values, 0.99)), 1.0e-5)
        paired = np.isfinite(section_disagreement[:, start:stop])
        disagreement_limit = (
            max(float(np.quantile(section_disagreement[:, start:stop][paired], 0.99)), 1.0e-5)
            if np.any(paired)
            else 1.0
        )
        panels = (
            (section_lfm, "LFM log-AI", "viridis", body_min, body_max),
            (section_body, "GINN V2 body log-AI", "viridis", body_min, body_max),
            (section_increment, "GINN body minus LFM", "RdBu_r", -increment_limit, increment_limit),
            (section_disagreement, "inline/xline disagreement", "magma", 0.0, disagreement_limit),
        )
        figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), sharex=True, sharey=True)
        extent = [float(distance[0]), float(distance[-1]), float(axis[stop - 1]), float(axis[start])]
        for current, (values, title, cmap, vmin, vmax) in zip(axes.flat, panels):
            image = current.imshow(
                values[:, start:stop].T,
                aspect="auto",
                origin="upper",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            current.set_title(title)
            current.set_xlabel("lateral distance (m)")
            current.set_ylabel(sample_axis.unit)
            figure.colorbar(image, ax=current, fraction=0.035, pad=0.025)
        figure.suptitle(f"GINN V2 volume inference — center {orientation}")
        figure.tight_layout()
        path = output_dir / f"center_{orientation}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        files.append(repo_relative_path(path, root=REPO_ROOT))

    target = fill_code > 0
    target_count = np.count_nonzero(target, axis=-1)

    def target_fraction(mask: np.ndarray) -> np.ndarray:
        values = np.full(target_count.shape, np.nan, dtype=np.float32)
        np.divide(
            np.count_nonzero(mask, axis=-1),
            target_count,
            out=values,
            where=target_count > 0,
        )
        return values

    coverage_panels = (
        (target_fraction(fill_code == 1), "direct prediction / target"),
        (target_fraction(fill_code == 2), "nearest-increment fill / target"),
        (target_fraction(fill_code == 3), "LFM-only fill / target"),
        (target_fraction(count == 2), "two-direction prediction / target"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))
    for current, (values, title) in zip(axes.flat, coverage_panels):
        image = current.imshow(values.T, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        current.set_title(title)
        current.set_xlabel("inline array index")
        current.set_ylabel("xline array index")
        figure.colorbar(image, ax=current, fraction=0.045, pad=0.03)
    figure.tight_layout()
    path = output_dir / "coverage.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    files.append(repo_relative_path(path, root=REPO_ROOT))
    return files


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    raw = training_entry._load_composed_config(config_path)
    workflow = WorkflowConfig.from_mapping(raw)
    training_section = _required_mapping(raw.get("ginn_v2_body_inversion"), name="ginn_v2_body_inversion")
    training_inputs = _required_mapping(training_section.get("inputs"), name="ginn_v2_body_inversion.inputs")
    training_config = BodyInversionConfig.from_mapping(
        _required_mapping(training_section.get("training"), name="ginn_v2_body_inversion.training")
    )
    inference_section = _required_mapping(raw.get("ginn_v2_volume_inference"), name="ginn_v2_volume_inference")
    checkpoint = resolve_relative_path(
        args.checkpoint or str(inference_section.get("checkpoint") or ""),
        root=REPO_ROOT,
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = _output_dir(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Volume inference output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    log = configure_run_logger(output_dir, logger_name="ginn_v2_volume", file_name="volume_inference.log")

    lfm_run_dir = resolve_relative_path(str(training_inputs["lfm_run_dir"]), root=REPO_ROOT)
    well_control_run_dir = resolve_relative_path(str(training_inputs["well_control_run_dir"]), root=REPO_ROOT)
    forward_inputs_path = resolve_relative_path(str(training_inputs["forward_model_inputs"]), root=REPO_ROOT)
    variant_id = str(training_inputs["variant_id"])
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
    seismic_options = segy_options_from_config(workflow.seismic.as_dict()) if workflow.seismic.type == "segy" else {}
    survey = open_survey(seismic_path, workflow.seismic.type, segy_options=seismic_options or None)
    sample_axis = survey.sample_axis(workflow.seismic.domain)
    if not np.array_equal(sample_axis.values, lfm.sample_axis.values):
        raise ValueError("Seismic and LFM SampleAxis values differ.")

    baseline_config = dict(lfm.variant.variant_metadata.get("resolved_baseline_config") or {})
    lfm_lowpass_spec = parse_lowpass_spec(dict(baseline_config.get("filter") or {}), sample_axis)
    wavelet_time_s, wavelet_amplitude, _relation, _payload = training_entry._load_forward_inputs(
        forward_inputs_path,
        domain=workflow.seismic.domain,
        depth_basis=workflow.seismic.depth_basis,
    )
    if workflow.seismic.domain == "depth":
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
    reader = PatchReader(
        SurveyTraceSource(survey=survey, sample_axis=sample_axis, geometry=survey.line_geometry),
        lfm_log_ai=lfm.log_ai,
        lfm_valid_mask=lfm.valid_mask,
        ilines=lfm.ilines,
        xlines=lfm.xlines,
        sample_axis=sample_axis,
        normalization=normalization,
        patch_radius=training_config.patch_radius,
        cache_size=max(training_config.cache_size, 4 * training_config.patch_radius + 2),
        seismic_feature_mode=training_config.seismic_feature_mode,
        seismic_balance_window_samples=training_config.seismic_balance_window_samples,
        seismic_balance_floor_fraction=training_config.seismic_balance_floor_fraction,
    )
    batch_size = int(args.batch_size or inference_section.get("batch_size") or training_config.batch_size)
    device = torch.device(str(inference_section.get("device") or training_config.device))
    model = CenterTraceBodyNet(training_config.network).to(device)
    checkpoint_payload = load_checkpoint(
        checkpoint,
        model=model,
        expected_network_config=training_config.network,
        map_location=device,
    )
    if dict(checkpoint_payload["run_config"]) != training_config.to_json_dict():
        raise ValueError("Checkpoint training/input contract differs from the current GINN configuration.")
    projector = BodyScaleProjector(
        smoothing_fwhm_m=training_config.body_smoothing_fwhm_m,
        sample_step=float(sample_axis.step),
        lowpass_spec=lfm_lowpass_spec,
    )
    inverter = BodyInverter(
        model,
        reader,
        adapter,
        projector=projector,
        device=device,
        batch_size=batch_size,
    )
    orientations = tuple(inference_section.get("orientations") or training_config.orientations)
    volume_config = VolumeInferenceConfig(
        batch_size=batch_size,
        log_every_sections=int(inference_section.get("log_every_sections") or 10),
        min_lfm_support=int(inference_section.get("min_lfm_support") or 8),
        orientations=orientations,
    )
    volume_inverter = BodyVolumeInverter(inverter, volume_config, logger=log)
    inline_bounds = None
    xline_bounds = None
    if args.smoke_tile_size is not None:
        if args.smoke_tile_origin == "northwest":
            inline_bounds = (0, min(int(args.smoke_tile_size), int(lfm.ilines.size)))
            xline_bounds = (0, min(int(args.smoke_tile_size), int(lfm.xlines.size)))
        else:
            inline_bounds = centered_tile_bounds(lfm.ilines.size, args.smoke_tile_size)
            xline_bounds = centered_tile_bounds(lfm.xlines.size, args.smoke_tile_size)
    log.info(
        "volume inference start | checkpoint=%s | batch_size=%d | orientations=%s | inline_bounds=%s | xline_bounds=%s",
        checkpoint,
        batch_size,
        ",".join(orientations),
        inline_bounds,
        xline_bounds,
    )
    result = volume_inverter.predict(inline_bounds=inline_bounds, xline_bounds=xline_bounds)
    local_ilines = np.asarray(lfm.ilines[result.inline_indices], dtype=np.float64)
    local_xlines = np.asarray(lfm.xlines[result.xline_indices], dtype=np.float64)
    local_lfm = np.asarray(
        lfm.log_ai[
            int(result.inline_indices[0]) : int(result.inline_indices[-1]) + 1,
            int(result.xline_indices[0]) : int(result.xline_indices[-1]) + 1,
            :,
        ],
        dtype=np.float32,
    )
    figures = _plot_sections(
        result,
        local_lfm,
        sample_axis,
        survey.line_geometry,
        local_ilines,
        local_xlines,
        output_dir / "figures",
    )

    exports: dict[str, Any] = {}
    full_volume = inline_bounds is None and xline_bounds is None
    if not args.skip_segy_export:
        if not full_volume:
            raise ValueError("SEG-Y export requires full-volume inference; use --skip-segy-export for a smoke tile.")
        export_config = dict(inference_section.get("exports") or {})
        exports["linear_ai"] = export_volume_like_source(
            output_base=output_dir / "ginn_v2_body_linear_ai",
            volume=log_ai_to_ai_volume(result.body_log_ai),
            ilines=local_ilines,
            xlines=local_xlines,
            samples=sample_axis.values,
            source_seismic_file=seismic_path,
            source_seismic_type=workflow.seismic.type,
            title="GINN V2 body-scale linear AI",
            details=[
                f"checkpoint={repo_relative_path(checkpoint, root=REPO_ROOT)}",
                f"domain={sample_axis.domain}",
                f"depth_basis={sample_axis.depth_basis}",
                "orientation_fusion=equal_mean",
            ],
            seismic_options=workflow.seismic.as_dict(),
            nan_fill=None,
        )
        if bool(export_config.get("body_increment_log_ai", True)):
            increment = np.where(
                result.fill_code > 0,
                result.body_log_ai - local_lfm,
                np.nan,
            ).astype(np.float32)
            exports["body_increment_log_ai"] = export_volume_like_source(
                output_base=output_dir / "ginn_v2_body_increment_log_ai",
                volume=increment,
                ilines=local_ilines,
                xlines=local_xlines,
                samples=sample_axis.values,
                source_seismic_file=seismic_path,
                source_seismic_type=workflow.seismic.type,
                title="GINN V2 body increment relative to LFM",
                details=["unit=log-AI", "orientation_fusion=equal_mean"],
                seismic_options=workflow.seismic.as_dict(),
                nan_fill=None,
            )
        if bool(export_config.get("direction_disagreement", False)):
            exports["direction_disagreement"] = export_volume_like_source(
                output_base=output_dir / "ginn_v2_direction_disagreement_log_ai",
                volume=result.direction_disagreement_log_ai,
                ilines=local_ilines,
                xlines=local_xlines,
                samples=sample_axis.values,
                source_seismic_file=seismic_path,
                source_seismic_type=workflow.seismic.type,
                title="GINN V2 inline-xline disagreement",
                details=["unit=absolute log-AI difference"],
                seismic_options=workflow.seismic.as_dict(),
                nan_fill=None,
            )

    summary = {
        "status": "completed",
        "mode": "smoke_tile" if not full_volume else "full_volume",
        "checkpoint": repo_relative_path(checkpoint, root=REPO_ROOT),
        "checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "input_contract": {
            "seismic_feature_mode": training_config.seismic_feature_mode,
            "seismic_balance_window_samples": training_config.seismic_balance_window_samples,
            "seismic_balance_floor_fraction": training_config.seismic_balance_floor_fraction,
            "patch_radius": training_config.patch_radius,
            "orientations": list(orientations),
        },
        "result": result.summary(),
        "axes": {
            "ilines": [float(local_ilines[0]), float(local_ilines[-1]), int(local_ilines.size)],
            "xlines": [float(local_xlines[0]), float(local_xlines[-1]), int(local_xlines.size)],
            "sample_axis": sample_axis.describe(),
        },
        "figures": figures,
        "exports": exports,
    }
    write_json(output_dir / "volume_inference_summary.json", summary)
    log.info("volume inference finished | mode=%s | summary=%s", summary["mode"], json.dumps(result.summary()))
    print("=== GINN V2 volume inference ===")
    print(f"Output: {output_dir}")
    print(f"Mode: {summary['mode']}")
    for name, payload in exports.items():
        print(f"{name}: {payload['path']}")


if __name__ == "__main__":
    main()
