"""Run Structured GINN V2 target preflight and later learning stages."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import load_yaml_config  # noqa: E402
from ginn_v2.contracts import GenerationPolicy  # noqa: E402
from ginn_v2.evidence import (  # noqa: E402
    EvidenceNetworkConfig,
    ObservationPerturbationProfile,
)
from ginn_v2.learning import (  # noqa: E402
    EvaluationConfig,
    LearningConfig,
    TargetPreflightConfig,
    evaluate_generator,
    preflight_evidence_targets,
    train_generator,
)
from ginn_v2.events import (  # noqa: E402
    EventEvaluationConfig,
    EventGeneratorConfig,
    EventLearningConfig,
    EventPolicyCalibrationConfig,
    calibrate_event_generation_policy,
    evaluate_event_generator,
    load_event_generation_policy,
    train_event_generator,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/ginn_v2/ginn_v2.yaml"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser(
        "target-preflight",
        help="Audit target semantics, baselines, and fixed mini-corpus learnability.",
    )
    preflight.add_argument("--corpus", type=Path, required=True)
    preflight.add_argument("--output-dir", type=Path, required=True)
    preflight.add_argument("--parent-count", type=_positive_int, default=None)
    preflight.add_argument("--overfit-steps", type=_positive_int, default=None)
    train = commands.add_parser(
        "train",
        help="Train one full, no-seismic, or single-trace evidence model.",
    )
    train.add_argument("--corpus", type=Path, required=True)
    train.add_argument("--preflight-report", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument(
        "--input-mode",
        choices=("full", "no_seismic", "single_trace"),
        default="full",
    )
    train.add_argument("--epochs", type=_positive_int, default=None)
    train.add_argument("--training-parent-count", type=_positive_int, default=None)
    train.add_argument("--tuning-parent-count", type=_positive_int, default=None)
    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate one frozen evidence checkpoint on a canonical split.",
    )
    evaluate.add_argument("--corpus", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument(
        "--split",
        choices=("tuning", "calibration", "section_gate"),
        default=None,
    )
    evaluate.add_argument("--parent-count", type=_positive_int, default=None)
    train_events = commands.add_parser(
        "train-events",
        help="Train the deterministic section-level EventTrack generator.",
    )
    train_events.add_argument("--corpus", type=Path, required=True)
    train_events.add_argument("--evidence-checkpoint", type=Path, required=True)
    train_events.add_argument("--initial-checkpoint", type=Path, default=None)
    train_events.add_argument("--output-dir", type=Path, required=True)
    train_events.add_argument("--epochs", type=_positive_int, default=None)
    train_events.add_argument("--training-parent-count", type=_positive_int, default=None)
    train_events.add_argument("--tuning-parent-count", type=_positive_int, default=None)
    evaluate_events = commands.add_parser(
        "evaluate-events",
        help="Evaluate one deterministic EventTrack checkpoint.",
    )
    evaluate_events.add_argument("--corpus", type=Path, required=True)
    evaluate_events.add_argument("--evidence-checkpoint", type=Path, required=True)
    evaluate_events.add_argument("--event-checkpoint", type=Path, required=True)
    evaluate_events.add_argument("--output-dir", type=Path, required=True)
    evaluate_events.add_argument(
        "--split",
        choices=("training", "tuning", "calibration", "section_gate"),
        default=None,
    )
    evaluate_events.add_argument("--parent-count", type=_positive_int, default=None)
    evaluate_events.add_argument(
        "--realization-count",
        type=_positive_int,
        default=None,
        help="Enable K-member ensemble evaluation; omit for deterministic MAP.",
    )
    evaluate_events.add_argument("--random-identity", type=int, default=None)
    evaluate_events.add_argument(
        "--event-density-multiplier", type=_positive_float, default=None
    )
    evaluate_events.add_argument(
        "--structure-sampling-temperature", type=_positive_float, default=None
    )
    evaluate_events.add_argument(
        "--profile-sampling-temperature", type=_positive_float, default=None
    )
    evaluate_events.add_argument(
        "--generation-policy",
        type=Path,
        default=None,
        help="Use a calibrated event_generation_policy.json artifact.",
    )
    calibrate_events = commands.add_parser(
        "calibrate-events",
        help="Calibrate EventTrack density and ensemble dispersion on calibration parents.",
    )
    calibrate_events.add_argument("--corpus", type=Path, required=True)
    calibrate_events.add_argument("--evidence-checkpoint", type=Path, required=True)
    calibrate_events.add_argument("--event-checkpoint", type=Path, required=True)
    calibrate_events.add_argument("--output-dir", type=Path, required=True)
    calibrate_events.add_argument("--parent-count", type=_positive_int, default=None)
    calibrate_events.add_argument(
        "--reuse-candidates-from",
        type=Path,
        default=None,
        help=(
            "Reuse evaluator reports referenced by a completed calibration report "
            "and only rerun policy selection/publication."
        ),
    )
    return parser.parse_args()


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping.")
    return dict(value)


def main() -> None:
    args = parse_args()
    root = _mapping(load_yaml_config(args.config), "config")
    config = _mapping(root.get("structured_ginn_v2"), "structured_ginn_v2")
    evidence = EvidenceNetworkConfig.from_mapping(
        _mapping(config.get("evidence"), "structured_ginn_v2.evidence")
    )
    perturbation = ObservationPerturbationProfile(
        **_mapping(
            config.get("observation_perturbation", {}),
            "structured_ginn_v2.observation_perturbation",
        )
    )
    device_name = str(config.get("device", "auto"))
    if args.command == "target-preflight":
        preflight_values = _mapping(
            config.get("target_preflight"),
            "structured_ginn_v2.target_preflight",
        )
        if args.parent_count is not None:
            preflight_values["parent_count"] = args.parent_count
            preflight_values["local_filter_training_parent_count"] = min(
                int(preflight_values["local_filter_training_parent_count"]),
                max(1, args.parent_count - 2),
            )
            preflight_values["overfit_parent_count"] = min(
                int(preflight_values["overfit_parent_count"]), args.parent_count
            )
        if args.overfit_steps is not None:
            preflight_values["overfit_steps"] = args.overfit_steps
        preflight = TargetPreflightConfig.from_mapping(preflight_values)
        report = preflight_evidence_targets(
            args.corpus,
            args.output_dir,
            evidence_config=evidence,
            config=preflight,
            perturbation_profile=perturbation,
            device_name=device_name,
        )
        report_name = "target_preflight_report.json"
        title = "target preflight"
    elif args.command == "train":
        learning = LearningConfig.from_mapping(
            _mapping(config.get("training"), "structured_ginn_v2.training")
        )
        if args.epochs is not None:
            learning = replace(learning, epochs=args.epochs)
        if args.training_parent_count is not None:
            learning = replace(
                learning, training_parent_count=args.training_parent_count
            )
        if args.tuning_parent_count is not None:
            learning = replace(learning, tuning_parent_count=args.tuning_parent_count)
        report = train_generator(
            args.corpus,
            args.preflight_report,
            args.output_dir,
            evidence_config=evidence,
            config=learning,
            perturbation_profile=perturbation,
            input_mode=args.input_mode,
            device_name=device_name,
        )
        report_name = "training_report.json"
        title = "evidence training"
    elif args.command == "evaluate":
        evaluation = EvaluationConfig.from_mapping(
            _mapping(config.get("evaluation"), "structured_ginn_v2.evaluation")
        )
        if args.split is not None:
            evaluation = replace(evaluation, split=args.split)
        if args.parent_count is not None:
            evaluation = replace(evaluation, parent_count=args.parent_count)
        report = evaluate_generator(
            args.corpus,
            args.checkpoint,
            args.output_dir,
            config=evaluation,
            perturbation_profile=perturbation,
            device_name=device_name,
        )
        report_name = "evaluation_report.json"
        title = "evidence evaluation"
    elif args.command == "train-events":
        event_generator = EventGeneratorConfig.from_mapping(
            _mapping(config.get("event_generator"), "structured_ginn_v2.event_generator")
        )
        event_learning = EventLearningConfig.from_mapping(
            _mapping(config.get("event_training"), "structured_ginn_v2.event_training")
        )
        if args.epochs is not None:
            event_learning = replace(event_learning, epochs=args.epochs)
        if args.training_parent_count is not None:
            event_learning = replace(
                event_learning, training_parent_count=args.training_parent_count
            )
        if args.tuning_parent_count is not None:
            event_learning = replace(
                event_learning, tuning_parent_count=args.tuning_parent_count
            )
        report = train_event_generator(
            args.corpus,
            args.evidence_checkpoint,
            args.output_dir,
            dominant_frequency_hz=float(config["dominant_frequency_hz"]),
            generator_config=event_generator,
            learning_config=event_learning,
            perturbation_profile=perturbation,
            device_name=device_name,
            initial_checkpoint=args.initial_checkpoint,
        )
        report_name = "event_training_report.json"
        title = "EventTrack training"
    elif args.command == "calibrate-events":
        calibration = EventPolicyCalibrationConfig.from_mapping(
            _mapping(
                config.get("event_policy_calibration"),
                "structured_ginn_v2.event_policy_calibration",
            )
        )
        if args.parent_count is not None:
            calibration = replace(calibration, parent_count=args.parent_count)
        report = calibrate_event_generation_policy(
            args.corpus,
            args.evidence_checkpoint,
            args.event_checkpoint,
            args.output_dir,
            dominant_frequency_hz=float(config["dominant_frequency_hz"]),
            config=calibration,
            perturbation_profile=perturbation,
            device_name=device_name,
            reuse_candidates_from=args.reuse_candidates_from,
        )
        report_name = "event_policy_calibration_report.json"
        title = "EventTrack policy calibration"
    else:
        event_evaluation = EventEvaluationConfig.from_mapping(
            _mapping(
                config.get("event_evaluation"),
                "structured_ginn_v2.event_evaluation",
            )
        )
        if args.split is not None:
            event_evaluation = replace(event_evaluation, split=args.split)
        if args.parent_count is not None:
            event_evaluation = replace(
                event_evaluation, parent_count=args.parent_count
            )
        generation_policy = None
        if args.generation_policy is not None:
            if any(
                value is not None
                for value in (
                    args.realization_count,
                    args.random_identity,
                    args.event_density_multiplier,
                    args.structure_sampling_temperature,
                    args.profile_sampling_temperature,
                )
            ):
                raise ValueError(
                    "--generation-policy cannot be combined with generation overrides."
                )
            generation_policy = load_event_generation_policy(
                args.generation_policy
            )
        elif args.realization_count is not None:
            generation_values = _mapping(
                config.get("generation"), "structured_ginn_v2.generation"
            )
            generation_values["realization_count"] = args.realization_count
            if args.random_identity is not None:
                generation_values["random_identity"] = args.random_identity
            if args.event_density_multiplier is not None:
                generation_values[
                    "event_density_multiplier"
                ] = args.event_density_multiplier
            if args.structure_sampling_temperature is not None:
                generation_values["structure_sampling_temperature"] = (
                    args.structure_sampling_temperature
                )
            if args.profile_sampling_temperature is not None:
                generation_values["profile_sampling_temperature"] = (
                    args.profile_sampling_temperature
                )
            generation_policy = GenerationPolicy.from_mapping(generation_values)
        report = evaluate_event_generator(
            args.corpus,
            args.evidence_checkpoint,
            args.event_checkpoint,
            args.output_dir,
            dominant_frequency_hz=float(config["dominant_frequency_hz"]),
            config=event_evaluation,
            generation_policy=generation_policy,
            perturbation_profile=perturbation,
            device_name=device_name,
        )
        report_name = "event_evaluation_report.json"
        title = "EventTrack evaluation"
    print(f"=== Structured GINN V2 {title} ===")
    print(f"Status: {report['status']}")
    print(f"Report: {Path(args.output_dir) / report_name}")


if __name__ == "__main__":
    main()
