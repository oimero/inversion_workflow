"""Run streaming PIAI v3 inference over a survey volume.

Usage::

    python scripts/piai_infer.py --config experiments/ginn_v3/ginn_v3.yaml \
        --checkpoint scripts/output/ginn_v3_piai_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.seismic.volume_export import export_volume_like_source, log_ai_to_ai_volume
from cup.seismic.survey import segy_options_from_config
from cup.utils.io import repo_relative_path, resolve_relative_path, write_json
from ginn_v3.qc import write_wavelet_csv, write_well_qc
from ginn_v3.workflow import load_body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/ginn_v3/ginn_v3.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--smoke-tile-size", type=int, default=None)
    parser.add_argument("--skip-segy-export", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lfm-run-dir", type=Path, default=None)
    parser.add_argument("--variant-id", type=str, default=None)
    parser.add_argument("--well-control-run-dir", type=Path, default=None)
    parser.add_argument("--forward-model-inputs-run-dir", type=Path, default=None)
    parser.add_argument("--wavelet-generation-run-dir", type=Path, default=None)
    parser.add_argument("--trusted-well-name", action="append", dest="trusted_well_names", default=None)
    return parser.parse_args()


def _resolve_output_dir(value: Path | None) -> Path:
    if value is not None:
        return resolve_relative_path(value, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "scripts" / "output" / f"ginn_v3_piai_infer_{timestamp}"


def main() -> None:
    args = parse_args()
    loaded = load_body(
        args.config,
        args.checkpoint,
        lfm_run_dir=args.lfm_run_dir,
        variant_id=args.variant_id,
        well_control_run_dir=args.well_control_run_dir,
        forward_model_inputs_run_dir=args.forward_model_inputs_run_dir,
        wavelet_generation_run_dir=args.wavelet_generation_run_dir,
        trusted_well_names=args.trusted_well_names,
        batch_size=args.batch_size,
        device=args.device,
    )
    output_dir = _resolve_output_dir(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Inference output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tile = args.smoke_tile_size
    inline_slice = xline_slice = None
    if tile is not None:
        if tile <= 0:
            raise ValueError("--smoke-tile-size must be positive.")
        shape = np.asarray(loaded.lfm.log_ai).shape
        inline_start = max(0, (int(shape[0]) - tile) // 2)
        xline_start = max(0, (int(shape[1]) - tile) // 2)
        inline_slice = slice(inline_start, min(int(shape[0]), inline_start + tile))
        xline_slice = slice(xline_start, min(int(shape[1]), xline_start + tile))
    result = loaded.predict_volume(
        output_path=output_dir / "piai_log_ai.npy",
        batch_size=args.batch_size,
        inline_slice=inline_slice,
        xline_slice=xline_slice,
    )
    learned_wavelet_path = output_dir / "learned_wavelet.csv"
    write_wavelet_csv(
        learned_wavelet_path,
        loaded.physics.wavelet_time_s,
        result.wavelet_mean_normalized,
        loaded.normalization.seismic_std,
    )
    well_qc_dir = output_dir / "well_qc"
    well_qc_metrics = write_well_qc(
        loaded.model,
        loaded.data,
        loaded.physics,
        result.wavelet_mean_normalized,
        loaded.reference_wavelet_amplitude,
        well_qc_dir,
        next(loaded.model.parameters()).device,
        wavelet_cohort="predicted_volume_trace_mean",
        reference_wavelet_time_s=loaded.reference_wavelet_time_s,
    )
    summary: dict[str, object] = {
        "schema": "ginn_v3_piai_inference_v1",
        "checkpoint": repo_relative_path(loaded.checkpoint, root=REPO_ROOT),
        "log_ai_memmap": repo_relative_path(result.log_ai_path, root=REPO_ROOT),
        "valid_mask_memmap": repo_relative_path(result.valid_mask_path, root=REPO_ROOT),
        "learned_wavelet_csv": repo_relative_path(learned_wavelet_path, root=REPO_ROOT),
        "well_qc_dir": repo_relative_path(well_qc_dir, root=REPO_ROOT),
        "well_qc_metrics": repo_relative_path(well_qc_dir / "metrics.json", root=REPO_ROOT),
        "shape": list(result.shape),
        "predicted_trace_count": result.predicted_trace_count,
        "wavelet_mean_normalized": result.wavelet_mean_normalized.tolist(),
        "seismic_mean": loaded.normalization.seismic_mean,
        "seismic_std": loaded.normalization.seismic_std,
        "physical_seismic_reconstruction": "F(log_ai, wavelet_raw) + seismic_mean",
        "wavelet_cohort": well_qc_metrics.get("wavelet_cohort", "predicted_volume_trace_mean"),
    }
    if not args.skip_segy_export:
        values = np.load(result.log_ai_path, mmap_mode="r")
        # ``log_ai_to_ai_volume`` iterates in bounded blocks; keep the source
        # memmap instead of materialising a full float64 copy of the volume.
        ai_volume = log_ai_to_ai_volume(values)
        ilines = np.asarray(loaded.lfm.ilines)[inline_slice] if inline_slice is not None else np.asarray(loaded.lfm.ilines)
        xlines = np.asarray(loaded.lfm.xlines)[xline_slice] if xline_slice is not None else np.asarray(loaded.lfm.xlines)
        workflow = loaded.workflow
        data_root = resolve_relative_path(workflow.data_root, root=REPO_ROOT)
        seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
        options = segy_options_from_config(workflow.seismic.as_dict()) if workflow.seismic.type == "segy" else {}
        summary["ai_volume"] = export_volume_like_source(
            output_base=output_dir / "piai_ai",
            volume=ai_volume,
            ilines=ilines,
            xlines=xlines,
            samples=np.asarray(loaded.sample_axis.values),
            source_seismic_file=seismic_path,
            source_seismic_type=workflow.seismic.type,
            sample_domain=workflow.seismic.domain,
            title="GINN v3 PIAI log-AI inversion",
            details=["Unfiltered log-AI = LFM + raw network correction"],
            seismic_options=options,
        )
    write_json(output_dir / "inference_summary.json", summary)
    print("=== PIAI v3 inference ===")
    print(f"Output: {output_dir}")
    print(f"Log-AI: {result.log_ai_path}")
    print(f"Valid mask: {result.valid_mask_path}")


if __name__ == "__main__":
    main()
