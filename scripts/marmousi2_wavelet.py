"""Run the existing Wtie/consensus workflow for Marmousi2 training wells."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marmousi2.tie_workflow import run_wavelet_workflow


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("opendata/prepared/marmousi2_postm_time"))
    parser.add_argument("--pretrained-dir", type=Path, default=None,
                        help="Directory containing the paired Wtie model and parameters.")
    args = parser.parse_args()
    prepared = _resolve(args.prepared_dir)
    if args.pretrained_dir is None:
        pretrained = ROOT / "opendata" / "pretrained" / "wtie"
    else:
        pretrained = _resolve(args.pretrained_dir)
    result = run_wavelet_workflow(
        prepared,
        repo_root=ROOT,
        pretrained_dir=pretrained,
    )
    print(f"Selected wavelet: {result['forward_wavelet']}")
    print(f"Training-well correlations: {result['training_well_correlations_on_original_tdt']}")


if __name__ == "__main__":
    main()
