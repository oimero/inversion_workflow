"""Build the numerical Marmousi2 low frequency models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marmousi2.lfm_build import LFM_IDS, prepare_lfm_models


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("opendata/prepared/marmousi2_postm_time"))
    parser.add_argument("--variants", nargs="+", default=["well_huber_trend", "rgt_lowpass_slices"], choices=list(LFM_IDS))
    parser.add_argument("--checkpoint", type=Path, default=Path("opendata/pretrained/rgt/CIG-Bench-RGT.pth"))
    parser.add_argument("--cig-bench-root", type=Path, default=None,
                        help="Optional external CIG-Bench checkout; the vendored adapter is the default.")
    parser.add_argument("--infer-shape", nargs=3, type=int, default=(384, 16, 512), metavar=("T", "H", "W"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cutoff-hz", type=float, default=5.0)
    parser.add_argument("--truth-f-scale", type=float, default=0.1)
    parser.add_argument("--kriging-range-m", type=float, default=None)
    parser.add_argument("--kriging-range-scale", type=float, default=4.0)
    parser.add_argument("--kriging-nugget", type=float, default=0.0)
    parser.add_argument("--default-variant", choices=list(LFM_IDS), default="rgt_lowpass_slices")
    args = parser.parse_args()
    prepared = _resolve(args.prepared_dir)
    checkpoint = _resolve(args.checkpoint)
    cig_root = None if args.cig_bench_root is None else _resolve(args.cig_bench_root)
    default_variant = args.default_variant if args.default_variant in args.variants else args.variants[0]
    summary = prepare_lfm_models(
        prepared,
        repo_root=ROOT,
        checkpoint_path=checkpoint if "rgt_lowpass_slices" in args.variants else None,
        cig_bench_root=cig_root,
        infer_shape=tuple(args.infer_shape),
        device=args.device,
        variants=args.variants,
        lowpass_config={
            "enabled": True,
            "cutoff_hz": args.cutoff_hz,
            "order": 4,
            "buffer_mode": "reflect",
            "buffer_axis_units": 0.2,
        },
        truth_f_scale=args.truth_f_scale,
        nugget=args.kriging_nugget,
        kriging_range_m=args.kriging_range_m,
        kriging_range_scale=args.kriging_range_scale,
        default_variant=default_variant,
    )
    for name, row in summary["variants"].items():
        print(f"{name}: {row['config']}")
    print(f"Default variant: {summary['default_variant']}")


if __name__ == "__main__":
    main()
