"""Calibrate and generate a truth-first Synthoseis-lite benchmark.

Usage::

    python scripts/synthoseis_lite.py --config <file> calibrate
    python scripts/synthoseis_lite.py --config <file> generate \
        --suite field_conditioned --impedance-calibration <file>

The ``synthoseis_lite.sample_domain`` and ``benchmark_schema`` keys explicitly
select the time-v2 or depth-v2 branch.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.synthetic.workflow import (  # noqa: E402
    parse_synthoseis_config,
    resolve_sources,
    run_calibration,
    run_generation,
)
from cup.synthetic.depth.calibration import run_depth_calibration  # noqa: E402
from cup.synthetic.depth.config import (  # noqa: E402
    load_composed_config,
    parse_depth_v2_config,
    resolve_depth_v2_sources,
)
from cup.synthetic.depth.generation import run_depth_generation  # noqa: E402
from cup.config.workflow import WorkflowConfig  # noqa: E402
from cup.config.sources import load_summary, resolve_source_run  # noqa: E402
from cup.utils.io import (  # noqa: E402
    load_yaml_config,
    repo_relative_path,
    resolve_relative_path,
)


TIME_SCHEMA = "synthoseis_lite_v3"
DEPTH_SCHEMA = "synthoseis_lite_v3"


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
    subparsers.add_parser(
        "calibrate", help="Freeze impedance calibration from authoritative source runs."
    )
    generate = subparsers.add_parser(
        "generate", help="Generate one configured benchmark suite."
    )
    generate.add_argument("--impedance-calibration", type=Path, required=True)
    generate.add_argument(
        "--suite",
        choices=("canonical", "field_conditioned"),
        required=True,
        help="Depth v3 requires field_conditioned; time v3 also supports canonical.",
    )
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


def _output_dir(args: argparse.Namespace, workflow: WorkflowConfig) -> Path:
    if args.output_dir is not None:
        return resolve_relative_path(args.output_dir, root=REPO_ROOT)
    root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"synthoseis_lite_{args.command}_{timestamp}"


def _prepare_synthoseis_config(raw: dict, workflow: WorkflowConfig) -> dict:
    prepared = dict(raw)
    root = dict(prepared.get("synthoseis_lite") or {})
    source_runs = dict(root.get("source_runs") or {})
    if not source_runs:
        output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
        obs_dir = resolve_source_run(
            None,
            output_root=output_root,
            prefix="forward_observability",
            required_files=[
                "run_summary.json",
                "frequency_evidence_bands.csv",
                "well_frequency_sensitivity.csv",
            ],
            root=REPO_ROOT,
            label="forward_observability",
            summary_file="run_summary.json",
            schema_version="forward_observability_v2",
        )
        summary = load_summary(
            obs_dir / "run_summary.json", schema_version="forward_observability_v2"
        )
        recorded = dict(summary.get("source_runs") or {})
        source_runs = {
            "forward_observability_dir": repo_relative_path(obs_dir, root=REPO_ROOT),
            "well_preprocess_dir": str(recorded.get("well_preprocess_dir") or ""),
            "well_auto_tie_dir": str(recorded.get("well_auto_tie_dir") or ""),
            "wavelet_generation_dir": str(recorded.get("wavelet_generation_dir") or ""),
        }
    root["source_runs"] = source_runs
    prepared["synthoseis_lite"] = root
    return prepared


def _load_time_config(
    experiment_raw: dict, *, experiment_path: Path
) -> tuple[dict, WorkflowConfig, dict[str, str]]:
    """Load time-v2 config, supporting either legacy direct or common-overlay form."""
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
        raise ValueError(
            f"Time Synthoseis-lite composed config contains unknown top-level keys: {unknown}."
        )
    common_path = resolve_relative_path(
        str(experiment_raw["workflow_config"]), root=REPO_ROOT
    )
    common = load_yaml_config(common_path)
    if not isinstance(common, dict):
        raise ValueError(f"workflow_config must contain a mapping: {common_path}")
    composed = dict(common)
    composed["synthoseis_lite"] = dict(experiment_raw.get("synthoseis_lite") or {})
    workflow = WorkflowConfig.from_mapping(composed)
    if workflow.seismic.domain != "time":
        raise ValueError("Time Synthoseis-lite v3 requires seismic.domain='time'.")
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

    if (sample_domain, benchmark_schema) == ("depth", DEPTH_SCHEMA):
        if "workflow_config" not in experiment_raw:
            raise ValueError("Depth Synthoseis-lite v3 requires workflow_config.")
        raw, workflow, config_provenance = load_composed_config(
            config_path, repo_root=REPO_ROOT
        )
        if workflow.seismic.domain != "depth":
            raise ValueError(
                "Composed Synthoseis-lite v3 currently implements the depth branch only."
            )
        script_cfg = parse_depth_v2_config(raw)
        sources, source_provenance, forward_inputs = resolve_depth_v2_sources(
            script_cfg, workflow=workflow, repo_root=REPO_ROOT
        )
        output_dir = _output_dir(args, workflow)
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
            if args.suite != "field_conditioned":
                raise ValueError(
                    "Depth Synthoseis-lite v3 only supports --suite field_conditioned."
                )
            summary = run_depth_generation(
                workflow=workflow,
                script_cfg=script_cfg,
                sources=sources,
                forward_inputs=forward_inputs,
                config_provenance=config_provenance,
                calibration_path=resolve_relative_path(
                    args.impedance_calibration, root=REPO_ROOT
                ),
                repo_root=REPO_ROOT,
                output_dir=output_dir,
                debug_attempt_limit=args.debug_attempt_limit,
                geometry_families=args.geometry_family,
                qc_only=args.qc_only,
            )
        print("=== synthoseis-lite v2 ===")
        print(f"Command: {args.command}")
        print(f"Output: {output_dir}")
        print(f"Status: {summary.get('status', 'success')}")
        return
    if (sample_domain, benchmark_schema) != ("time", TIME_SCHEMA):
        raise ValueError(
            "Unsupported Synthoseis-lite dispatch: "
            f"sample_domain={sample_domain!r}, benchmark_schema={benchmark_schema!r}."
        )
    raw, workflow, config_provenance = _load_time_config(
        experiment_raw, experiment_path=config_path
    )
    script_cfg = parse_synthoseis_config(raw)
    sources = resolve_sources(script_cfg, repo_root=REPO_ROOT)
    output_dir = _output_dir(args, workflow)
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
            suite=args.suite,
        )
    print("=== synthoseis-lite ===")
    print(f"Command: {args.command}")
    print(f"Output: {output_dir}")
    print(f"Status: {summary.get('status', 'success')}")


if __name__ == "__main__":
    main()
