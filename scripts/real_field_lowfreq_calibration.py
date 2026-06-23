"""Run R2 low-frequency calibration-only for real-field zero-shot outputs.

This is a research-stage script, not a numbered production workflow step.  It
fits one constant log(AI) bias per configured zero-shot model role and writes
calibration evidence without modifying the original R0/R1 outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import load_yaml_config, resolve_relative_path, write_json
from ginn_v2.real_field_calibration import parse_lowfreq_calibration_config, run_lowfreq_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _timestamped_output(prefix: str, explicit: Path | None, *, output_root: Path) -> Path:
    if explicit is not None:
        return resolve_relative_path(explicit, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{prefix}_{timestamp}"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    raw = load_yaml_config(args.config)
    output_root = resolve_relative_path(str(raw.get("output_root", "scripts/output")), root=REPO_ROOT)
    output_dir = _timestamped_output("real_field_lowfreq_calibration", args.output_dir, output_root=output_root)
    cfg = parse_lowfreq_calibration_config(raw, root=REPO_ROOT)
    summary = run_lowfreq_calibration(cfg, output_dir=output_dir, root=REPO_ROOT, git_commit=_git_commit())
    write_json(output_dir / "lowfreq_calibration_summary.json", summary)
    print("=== R2 Low-Frequency Calibration Only ===")
    print(f"Output: {output_dir}")
    print(f"Status: {summary.get('status')}")
    print(f"Evidence rows: {summary.get('n_evidence_rows')}")


if __name__ == "__main__":
    main()
