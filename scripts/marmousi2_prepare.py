"""Prepare the public Marmousi2 section for the existing GINN workflow.

The command writes only benchmark data under ``opendata/prepared``.  It does
not train a network; run ``marmousi2_wavelet.py`` and
``marmousi2_prepare_lfms.py`` before calling the normal body workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marmousi2.prepare import PrepareSettings, check_prepared_inputs, prepare_marmousi2


def _resolve(path: Path, *, root: Path = ROOT) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vp", type=Path, default=Path("opendata/raw/marmousi2/vp_marmousi-ii.segy"))
    parser.add_argument("--density", type=Path, default=Path("opendata/raw/marmousi2/density_marmousi-ii.segy"))
    parser.add_argument("--seismic", type=Path, default=Path("opendata/raw/marmousi2/Kirchhoff_PoSTM_time.segy"),
                        help="PostM time section; use --synthetic for the ideal convolution smoke input.")
    parser.add_argument("--output-dir", type=Path, default=Path("opendata/prepared/marmousi2_postm_time"))
    parser.add_argument("--synthetic", action="store_true", help="Use the controlled Ricker convolution instead of PostM.")
    parser.add_argument("--wavelet", type=Path, default=None, help="Optional already selected wavelet CSV.")
    parser.add_argument("--wavelet-method", choices=("workflow", "ridge"), default="workflow")
    parser.add_argument("--wavelet-duration-s", type=float, default=0.16)
    parser.add_argument("--wavelet-ridge", type=float, default=0.01)
    parser.add_argument("--trace-stride", type=int, default=None)
    parser.add_argument("--depth-stride", type=int, default=1)
    parser.add_argument("--dx-m", type=float, default=None)
    parser.add_argument("--dz-m", type=float, default=None)
    parser.add_argument("--dt-s", type=float, default=0.004)
    parser.add_argument("--wavelet-hz", type=float, default=30.0)
    parser.add_argument("--lfm-cutoff-hz", type=float, default=5.0)
    parser.add_argument("--body-smoothing-fwhm-s", type=float, default=0.01)
    parser.add_argument("--waveform-qc-window-s", type=float, default=0.06)
    parser.add_argument("--seismic-support-relative-threshold", type=float, default=0.25)
    parser.add_argument("--train-well-fractions", type=float, nargs="+", default=None)
    parser.add_argument("--validation-well-fraction", type=float, default=None)
    parser.add_argument("--test-well-fraction", type=float, default=None)
    parser.add_argument("--target-top-s", type=float, default=0.65)
    parser.add_argument("--target-bottom-buffer-s", type=float, default=0.08)
    parser.add_argument("--check-config", type=Path, default=None,
                        help="Check an existing prepared config without preparing data.")
    args = parser.parse_args()

    if args.check_config is not None:
        config_path = _resolve(args.check_config)
        result = check_prepared_inputs(config_path, repo_root=ROOT)
        print(f"Config: {config_path}")
        print(f"Adapter: {result['status']} | train={result['training_patches']} | validation={result['validation_patches']}")
        return

    output = _resolve(args.output_dir)
    spacing = 1.249 if args.synthetic else 1.25
    kwargs = {
        "trace_stride": args.trace_stride or (8 if args.synthetic else 5),
        "depth_stride": args.depth_stride,
        "dx_m": args.dx_m or spacing,
        "dz_m": args.dz_m or spacing,
        "dt_s": args.dt_s,
        "wavelet_frequency_hz": args.wavelet_hz,
        "lfm_cutoff_hz": args.lfm_cutoff_hz,
        "wavelet_duration_s": args.wavelet_duration_s,
        "wavelet_ridge_fraction": args.wavelet_ridge,
        "wavelet_method": args.wavelet_method,
        "body_smoothing_fwhm_s": args.body_smoothing_fwhm_s,
        "waveform_qc_dynamic_window_s": args.waveform_qc_window_s,
        "seismic_support_relative_threshold": args.seismic_support_relative_threshold,
        "target_top_s": args.target_top_s,
        "target_bottom_buffer_s": args.target_bottom_buffer_s,
    }
    if args.train_well_fractions is not None:
        kwargs["train_well_fractions"] = tuple(args.train_well_fractions)
    if args.validation_well_fraction is not None:
        kwargs["validation_well_fraction"] = args.validation_well_fraction
    if args.test_well_fraction is not None:
        kwargs["test_well_fraction"] = args.test_well_fraction
    settings = PrepareSettings(**kwargs)
    config_path = prepare_marmousi2(
        _resolve(args.vp), _resolve(args.density), output, repo_root=ROOT,
        settings=settings,
        seismic_path=None if args.synthetic else _resolve(args.seismic),
        wavelet_path=None if args.wavelet is None else _resolve(args.wavelet),
    )
    print(f"Prepared config: {config_path}")
    print("Next: scripts/marmousi2_wavelet.py and scripts/marmousi2_prepare_lfms.py")


if __name__ == "__main__":
    main()
