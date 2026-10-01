"""Run the fixed-input Mero final-curve experiment."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ginn_v2.final_curve_experiment import main


if __name__ == "__main__":
    main()
