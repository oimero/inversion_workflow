"""Run deterministic GINN v2 body inference over a depth/time survey volume."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for path in (SRC_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cup.seismic.volume_export import export_volume_like_source, log_ai_to_ai_volume
from cup.utils.io import repo_relative_path, resolve_relative_path, write_json
from cup.utils.logging import configure_run_logger
from ginn_v2 import load_body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/ginn_v2/ginn_v2.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lfm-run-dir", type=Path, default=None)
    parser.add_argument("--variant-id", type=str, default=None)
    parser.add_argument("--well-control-run-dir", type=Path, default=None)
    parser.add_argument("--forward-model-inputs", type=Path, default=None)
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
    loaded = load_body(
        config_path,
        checkpoint=args.checkpoint,
        lfm_run_dir=args.lfm_run_dir,
        variant_id=args.variant_id,
        well_control_run_dir=args.well_control_run_dir,
        forward_model_inputs=args.forward_model_inputs,
        batch_size=args.batch_size,
    )
    workflow = loaded.workflow
    training_config = loaded.training_config
    inference_section = loaded.inference_config
    checkpoint = loaded.checkpoint
    lfm = loaded.lfm
    survey = loaded.survey
    sample_axis = loaded.sample_axis
    seismic_path = loaded.seismic_path
    checkpoint_payload = loaded.checkpoint_payload
    output_dir = _output_dir(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Volume inference output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    log = configure_run_logger(output_dir, logger_name="ginn_v2_volume", file_name="volume_inference.log")

    batch_size = int(args.batch_size or inference_section.get("batch_size") or training_config.batch_size)
    orientations = tuple(inference_section.get("orientations") or training_config.orientations)
    log.info(
        "volume inference start | checkpoint=%s | batch_size=%d | orientations=%s | smoke_tile_size=%s | smoke_tile_origin=%s",
        checkpoint,
        batch_size,
        ",".join(orientations),
        args.smoke_tile_size,
        args.smoke_tile_origin,
    )
    result = loaded.predict_volume(
        batch_size=batch_size,
        smoke_tile_size=args.smoke_tile_size,
        smoke_tile_origin=args.smoke_tile_origin,
        logger=log,
    )
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
    full_volume = args.smoke_tile_size is None
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
