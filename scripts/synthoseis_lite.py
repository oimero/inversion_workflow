"""Calibrate and generate a truth-first Synthoseis-lite benchmark.

Usage::

    python scripts/synthoseis_lite.py --config <file> calibrate
    python scripts/synthoseis_lite.py --config <file> generate \
        --impedance-calibration <file>

The ``synthoseis_lite.sample_domain`` and ``benchmark_schema`` keys explicitly
select the primary time-domain or depth-domain extension branch.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.config.sources import load_summary, resolve_source_run  # noqa: E402
from cup.config.workflow import WorkflowConfig  # noqa: E402
from cup.synthetic.schemas import BENCHMARK_SCHEMA_VERSION  # noqa: E402
from cup.synthetic.depth.calibration import run_depth_calibration  # noqa: E402
from cup.synthetic.depth.config import (  # noqa: E402
    load_composed_config,
    parse_depth_config,
    resolve_depth_sources,
)
from cup.synthetic.depth.generation import run_depth_generation  # noqa: E402
from cup.synthetic.time.calibration_pipeline import run_calibration  # noqa: E402
from cup.synthetic.time.config import (  # noqa: E402
    parse_synthoseis_config,
    resolve_sources,
)
from cup.synthetic.time.pipeline import run_generation  # noqa: E402
from cup.utils.io import (  # noqa: E402
    load_yaml_config,
    repo_relative_path,
    resolve_relative_path,
)


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/synthoseis_lite/synthoseis_lite.yaml"),
        help="YAML configuration containing synthoseis_lite.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("calibrate", help="Freeze impedance calibration from authoritative source runs.")
    generate = subparsers.add_parser("generate", help="Generate one configured benchmark suite.")
    generate.add_argument("--impedance-calibration", type=Path, required=True)
    generate.add_argument(
        "--debug-attempt-limit",
        type=_positive_int_arg,
        default=None,
        help="Development-only attempt cap; acceptance thresholds are not evaluated.",
    )
    generate.add_argument(
        "--geometry-family",
        action="append",
        choices=("none", "wedge", "pinchout"),
        default=None,
        help="Restrict field-conditioned generation to one or more geometry families.",
    )
    generate.add_argument(
        "--qc-only",
        action="store_true",
        help="Run full generation and acceptance QC without persisting realization arrays.",
    )
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace, workflow: WorkflowConfig) -> Path:
    if args.output_dir is not None:
        return resolve_relative_path(args.output_dir, root=REPO_ROOT)
    root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"synthoseis_lite_{args.command}_{timestamp}"


def _recorded_time_source_runs_from_wavelet(workflow: WorkflowConfig) -> dict[str, str]:
    output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    wavelet_dir = resolve_source_run(
        None,
        output_root=output_root,
        prefix="wavelet_generation",
        required_files=[
            "run_summary.json",
            "selected_wavelet.csv",
            "selected_wavelet_summary.json",
            "evaluation_well_spatial_clusters.csv",
        ],
        root=REPO_ROOT,
        label="wavelet_generation",
        summary_file="run_summary.json",
    )
    wavelet_summary = load_summary(wavelet_dir / "selected_wavelet_summary.json")
    auto_tie_dir = resolve_relative_path(
        str(wavelet_summary.get("source_auto_tie_dir") or ""),
        root=REPO_ROOT,
    )
    auto_tie_summary = load_summary(auto_tie_dir / "run_summary.json")
    preprocess_status = resolve_relative_path(
        str(dict(auto_tie_summary.get("inputs") or {}).get("preprocess_status_file") or ""),
        root=REPO_ROOT,
    )
    return {
        "well_preprocess_dir": repo_relative_path(preprocess_status.parent, root=REPO_ROOT),
        "well_auto_tie_dir": repo_relative_path(auto_tie_dir, root=REPO_ROOT),
        "wavelet_generation_dir": repo_relative_path(wavelet_dir, root=REPO_ROOT),
    }


def _prepare_synthoseis_config(raw: dict, workflow: WorkflowConfig) -> dict:
    prepared = dict(raw)
    root = dict(prepared.get("synthoseis_lite") or {})
    source_runs = dict(root.get("source_runs") or {})
    if not source_runs:
        source_runs = _recorded_time_source_runs_from_wavelet(workflow)
    root["source_runs"] = source_runs
    prepared["synthoseis_lite"] = root
    return prepared


def _load_time_config(experiment_raw: dict, *, experiment_path: Path) -> tuple[dict, WorkflowConfig, dict[str, str]]:
    """Load the primary time-domain config in direct or common-overlay form."""
    if "workflow_config" not in experiment_raw:
        raw = experiment_raw
        workflow = WorkflowConfig.from_mapping(raw)
        provenance = {
            "experiment_file": str(experiment_path),
            "workflow_config": str(experiment_path),
        }
        return _prepare_synthoseis_config(raw, workflow), workflow, provenance

    allowed = {"workflow_config", "synthoseis_lite"}
    unknown = sorted(set(experiment_raw) - allowed)
    if unknown:
        raise ValueError(f"Time Synthoseis-lite composed config contains unknown top-level keys: {unknown}.")
    common_path = resolve_relative_path(str(experiment_raw["workflow_config"]), root=REPO_ROOT)
    common = load_yaml_config(common_path)
    if not isinstance(common, dict):
        raise ValueError(f"workflow_config must contain a mapping: {common_path}")
    composed = dict(common)
    composed["synthoseis_lite"] = dict(experiment_raw.get("synthoseis_lite") or {})
    workflow = WorkflowConfig.from_mapping(composed)
    if workflow.seismic.domain != "time":
        raise ValueError("Time Synthoseis-lite v5 requires seismic.domain='time'.")
    provenance = {
        "experiment_file": str(experiment_path),
        "workflow_config": str(common_path),
    }
    return _prepare_synthoseis_config(composed, workflow), workflow, provenance


def _dispatch_keys(raw: dict) -> tuple[str, str]:
    root = raw.get("synthoseis_lite")
    if not isinstance(root, dict):
        raise ValueError("Configuration must contain a synthoseis_lite mapping.")
    sample_domain = str(root.get("sample_domain") or "").casefold()
    benchmark_schema = str(root.get("benchmark_schema") or "")
    if not sample_domain or not benchmark_schema:
        raise ValueError(
            "synthoseis_lite.sample_domain and synthoseis_lite.benchmark_schema "
            "must explicitly select the Synthoseis branch."
        )
    return sample_domain, benchmark_schema


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    experiment_raw = load_yaml_config(config_path)
    if not isinstance(experiment_raw, dict):
        raise ValueError("Synthoseis-lite config root must be a mapping.")
    sample_domain, benchmark_schema = _dispatch_keys(experiment_raw)

    if (sample_domain, benchmark_schema) == ("depth", BENCHMARK_SCHEMA_VERSION):
        if "workflow_config" not in experiment_raw:
            raise ValueError("Depth Synthoseis-lite v5 requires workflow_config.")
        raw, workflow, config_provenance = load_composed_config(config_path, repo_root=REPO_ROOT)
        if workflow.seismic.domain != "depth":
            raise ValueError("Composed Synthoseis-lite v5 currently implements the depth branch only.")
        script_cfg = parse_depth_config(raw)
        sources, source_provenance, forward_inputs = resolve_depth_sources(
            script_cfg, workflow=workflow, repo_root=REPO_ROOT
        )
        output_dir = _resolve_output_dir(args, workflow)
        if args.command == "calibrate":
            summary = run_depth_calibration(
                workflow=workflow,
                script_cfg=script_cfg,
                sources=sources,
                source_provenance=source_provenance,
                forward_inputs=forward_inputs,
                config_provenance=config_provenance,
                repo_root=REPO_ROOT,
                output_dir=output_dir,
            )
        else:
            summary = run_depth_generation(
                workflow=workflow,
                script_cfg=script_cfg,
                sources=sources,
                forward_inputs=forward_inputs,
                config_provenance=config_provenance,
                calibration_path=resolve_relative_path(args.impedance_calibration, root=REPO_ROOT),
                repo_root=REPO_ROOT,
                output_dir=output_dir,
                debug_attempt_limit=args.debug_attempt_limit,
                geometry_families=args.geometry_family,
                qc_only=args.qc_only,
            )
        print("=== synthoseis-lite v5 (depth) ===")
        print(f"Command: {args.command}")
        print(f"Output: {output_dir}")
        print(f"Status: {summary.get('status', 'success')}")
        return
    if (sample_domain, benchmark_schema) != ("time", BENCHMARK_SCHEMA_VERSION):
        raise ValueError(
            "Unsupported Synthoseis-lite dispatch: "
            f"sample_domain={sample_domain!r}, benchmark_schema={benchmark_schema!r}."
        )
    raw, workflow, config_provenance = _load_time_config(experiment_raw, experiment_path=config_path)
    script_cfg = parse_synthoseis_config(raw)
    sources = resolve_sources(script_cfg, repo_root=REPO_ROOT)
    output_dir = _resolve_output_dir(args, workflow)
    if args.command == "calibrate":
        summary = run_calibration(
            workflow=workflow,
            script_cfg=script_cfg,
            sources=sources,
            config_provenance=config_provenance,
            repo_root=REPO_ROOT,
            output_dir=output_dir,
        )
    else:
        calibration = resolve_relative_path(args.impedance_calibration, root=REPO_ROOT)
        summary = run_generation(
            workflow=workflow,
            script_cfg=script_cfg,
            sources=sources,
            config_provenance=config_provenance,
            calibration_path=calibration,
            repo_root=REPO_ROOT,
            output_dir=output_dir,
            debug_attempt_limit=args.debug_attempt_limit,
            geometry_families=args.geometry_family,
            qc_only=args.qc_only,
        )
    print("=== synthoseis-lite ===")
    print(f"Command: {args.command}")
    print(f"Output: {output_dir}")
    print(f"Status: {summary.get('status', 'success')}")


if __name__ == "__main__":
    main()
