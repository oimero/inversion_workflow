"""Audit targets, train, or evaluate the Structured GINN V2 generator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cup.utils.io import load_yaml_config, resolve_relative_path, write_json
from ginn_v2.artifacts import (
    calibrate_semi_markov_prior,
    iter_evidence_batches,
    iter_paired_evidence_batches,
    iter_segment_profile_batches,
    load_observable_target_contract,
    load_checkpoint,
    load_corpus,
    load_semi_markov_contract,
    public_checkpoint_metadata,
    save_checkpoint,
)
from ginn_v2.augmentation import load_observation_augmentation_profile
from ginn_v2.evidence import (
    EvidenceNetworkConfig,
    ObservableEvidenceNetwork,
    dominant_frequency_hz,
)
from ginn_v2.contracts import GenerationPolicy
from ginn_v2.generator import ConditionalGenerator, SegmentProfileHeadConfig
from ginn_v2.learning import (
    CoefficientVarianceCalibrationConfig,
    LearningConfig,
    MapProfileProbeConfig,
    SegmentProfileLearningConfig,
    TargetAuditConfig,
    audit_observable_targets,
    calibrate_coefficient_variance,
    calibrate_semi_markov_fusion,
    evaluate_generator,
    evaluate_map_reconstruction,
    evaluate_segment_profile_head,
    evaluate_structured_ensemble,
    seed_training_random_streams,
    train_generator,
    train_map_profile_probe,
    train_segment_profile_head,
)
from ginn_v2.runtime import configure_training_logger, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/ginn_v2/ginn_v2.yaml"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--corpus", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--target-contract", type=Path, required=True)
    train.add_argument(
        "--input-mode",
        choices=("full", "no_seismic"),
        default=None,
    )
    train.add_argument(
        "--smoke",
        action="store_true",
        help="train one epoch on three parents per split",
    )
    train.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume from an epoch checkpoint, normally the output directory's last.pt",
    )
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--corpus", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument(
        "--split", choices=("tuning", "calibration"), default="calibration"
    )
    evaluate.add_argument(
        "--no-seismic-checkpoint",
        type=Path,
        default=None,
        help="optional independently trained no-seismic control checkpoint",
    )
    evaluate.add_argument("--bootstrap-replicates", type=int, default=2000)
    evaluate.add_argument(
        "--parent-limit",
        type=int,
        default=None,
        help="optional bounded evaluator run for interface smoke",
    )
    hsmm = subparsers.add_parser("evaluate-hsmm")
    hsmm.add_argument("--corpus", type=Path, required=True)
    hsmm.add_argument("--checkpoint", type=Path, required=True)
    hsmm.add_argument("--output-dir", type=Path, required=True)
    hsmm.add_argument(
        "--split", choices=("tuning", "calibration"), default="tuning"
    )
    hsmm.add_argument(
        "--parents-per-family",
        type=int,
        default=4,
        help="fixed parent budget for each of none/wedge/pinchout",
    )
    hsmm.add_argument(
        "--prior-parents-per-family",
        type=int,
        default=32,
        help="training-parent budget per family for the model-grid HSMM prior",
    )
    hsmm.add_argument(
        "--smoke",
        action="store_true",
        help="evaluate one parent per geometry family",
    )
    profiles = subparsers.add_parser("train-profiles")
    profiles.add_argument("--corpus", type=Path, required=True)
    profiles.add_argument("--checkpoint", type=Path, required=True)
    profiles.add_argument("--output-dir", type=Path, required=True)
    profiles.add_argument(
        "--smoke",
        action="store_true",
        help="train and evaluate on one parent per geometry family",
    )
    reconstruction = subparsers.add_parser("evaluate-reconstruction")
    reconstruction.add_argument("--corpus", type=Path, required=True)
    reconstruction.add_argument("--checkpoint", type=Path, required=True)
    reconstruction.add_argument("--hsmm-contract-dir", type=Path, required=True)
    reconstruction.add_argument("--output-dir", type=Path, required=True)
    reconstruction.add_argument(
        "--split", choices=("tuning", "calibration"), default="calibration"
    )
    reconstruction.add_argument("--parents-per-family", type=int, default=4)
    reconstruction.add_argument(
        "--smoke",
        action="store_true",
        help="evaluate one parent per geometry family",
    )
    map_profiles = subparsers.add_parser("probe-map-profiles")
    map_profiles.add_argument("--corpus", type=Path, required=True)
    map_profiles.add_argument("--checkpoint", type=Path, required=True)
    map_profiles.add_argument("--hsmm-contract-dir", type=Path, required=True)
    map_profiles.add_argument("--output-dir", type=Path, required=True)
    map_profiles.add_argument(
        "--smoke",
        action="store_true",
        help="exercise one training/tuning/calibration parent per geometry family",
    )
    variance = subparsers.add_parser("calibrate-profile-variance")
    variance.add_argument("--corpus", type=Path, required=True)
    variance.add_argument("--checkpoint", type=Path, required=True)
    variance.add_argument("--output-dir", type=Path, required=True)
    variance.add_argument(
        "--smoke",
        action="store_true",
        help="calibrate on one parent per geometry family",
    )
    ensemble = subparsers.add_parser("evaluate-ensemble")
    ensemble.add_argument("--corpus", type=Path, required=True)
    ensemble.add_argument("--checkpoint", type=Path, required=True)
    ensemble.add_argument("--hsmm-contract-dir", type=Path, required=True)
    ensemble.add_argument("--output-dir", type=Path, required=True)
    ensemble.add_argument(
        "--split", choices=("tuning", "calibration"), default="calibration"
    )
    ensemble.add_argument("--parents-per-family", type=int, default=4)
    ensemble.add_argument("--realization-count", type=int, default=16)
    ensemble.add_argument("--random-identity", type=int, default=20260804)
    ensemble.add_argument("--lateral-correlation-m", type=float, default=900.0)
    ensemble.add_argument("--path-coupling-strength", type=float, default=1.0)
    ensemble.add_argument("--profile-coupling-strength", type=float, default=1.0)
    ensemble.add_argument(
        "--figures-per-family",
        type=int,
        default=0,
        help="publish continuity cards for the first N parents in each geometry family",
    )
    ensemble.add_argument(
        "--smoke",
        action="store_true",
        help="evaluate one parent per geometry family with K=2",
    )
    audit = subparsers.add_parser("audit-targets")
    audit.add_argument("--corpus", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument(
        "--smoke",
        action="store_true",
        help="exercise L0-L2 with tiny budgets without publishing a target contract",
    )
    return parser.parse_args()


def _config(path: Path) -> dict:
    payload = load_yaml_config(resolve_relative_path(path, root=REPO_ROOT))
    if not isinstance(payload, dict) or set(payload) != {"structured_ginn_v2"}:
        raise ValueError("config must contain only structured_ginn_v2.")
    root = payload["structured_ginn_v2"]
    if not isinstance(root, dict):
        raise ValueError("structured_ginn_v2 must be a mapping.")
    allowed = {
        "device",
        "network",
        "learning",
        "target_audit",
        "profile_head",
        "profile_learning",
        "map_profile_probe",
        "coefficient_variance_calibration",
        "augmentation_profile",
    }
    unknown = sorted(set(root).difference(allowed))
    if unknown:
        raise ValueError(f"unknown Structured GINN V2 config keys: {unknown}")
    return root


def _augmentation_profile(config: dict):
    value = config.get("augmentation_profile")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("augmentation_profile must be a non-empty JSON path.")
    return load_observation_augmentation_profile(
        resolve_relative_path(Path(value), root=REPO_ROOT)
    )


def _dominant_frequency(corpus) -> float:
    parent_id = corpus.splits["training"][0]
    context = corpus.benchmark.read_parent(parent_id).forward_context
    return dominant_frequency_hz(
        context["wavelet_time_s"],
        context["wavelet_amplitude"],
    )


def _balanced_parent_ids(corpus, split: str, per_family: int) -> tuple[str, ...]:
    if per_family <= 0:
        raise ValueError("parents_per_family must be positive.")
    index = corpus.benchmark.index
    selected: list[str] = []
    for family in ("none", "wedge", "pinchout"):
        rows = index.loc[
            index["split_role"].eq(split)
            & index["geometry_family"].eq(family)
            & index["corpus_role"].eq("short_patch"),
            "realization_id",
        ].astype(str)
        values = sorted(rows.tolist())[:per_family]
        if len(values) != per_family:
            raise ValueError(
                f"split {split!r} has only {len(values)} {family!r} parents; "
                f"required {per_family}."
            )
        selected.extend(values)
    return tuple(selected)


def main() -> None:
    args = parse_args()
    config = _config(args.config)
    device, runtime = resolve_device(str(config.get("device") or "auto"))
    corpus = load_corpus(resolve_relative_path(args.corpus, root=REPO_ROOT))
    output = resolve_relative_path(args.output_dir, root=REPO_ROOT)
    resume_path: Path | None = None
    if args.command == "train" and args.resume is not None:
        resume_path = resolve_relative_path(args.resume, root=REPO_ROOT)
        if not output.is_dir():
            raise FileNotFoundError(
                f"resume output directory does not exist: {output}"
            )
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
    logger = configure_training_logger(output)
    augmentation_profile = _augmentation_profile(config)
    if args.command == "audit-targets":
        audit_config = TargetAuditConfig.from_mapping(
            config.get("target_audit") or {},
            smoke=bool(args.smoke),
        )
        report = audit_observable_targets(
            corpus,
            output,
            config=audit_config,
            device=device,
            logger=logger,
        )
        report["runtime"] = runtime
        write_json(output / "target_audit.json", report)
        logger.info("observable target audit finished | status=%s", report["status"])
        return
    if args.command == "train-profiles":
        generator, source_metadata = load_checkpoint(
            resolve_relative_path(args.checkpoint, root=REPO_ROOT),
            device=device,
        )
        if generator.network_config.input_mode != "full":
            raise ValueError("segment profile training requires a full checkpoint.")
        if generator.profile_head is not None:
            raise ValueError("source checkpoint already contains a profile head.")
        profile_config = SegmentProfileLearningConfig.from_mapping(
            config.get("profile_learning") or {},
            smoke=bool(args.smoke),
        )
        head_config = SegmentProfileHeadConfig.from_mapping(
            config.get("profile_head") or {}
        )
        training_ids = _balanced_parent_ids(
            corpus,
            "training",
            profile_config.training_parents_per_family,
        )
        tuning_ids = _balanced_parent_ids(
            corpus,
            "tuning",
            profile_config.tuning_parents_per_family,
        )
        corpus_provenance = dict(
            source_metadata.get("corpus_provenance") or {}
        )
        corpus_provenance.update(
            {
                "profile_training_parent_ids": list(training_ids),
                "profile_tuning_parent_ids": list(tuning_ids),
            }
        )

        def save_profile_epoch(
            epoch: int,
            current: ConditionalGenerator,
            state: Mapping[str, Any],
            is_best: bool,
        ) -> None:
            training_state = {
                "schema": "structured_ginn_v2_segment_profile_training_v1",
                "source_checkpoint_metadata": public_checkpoint_metadata(
                    source_metadata
                ),
                **dict(state),
            }
            save_checkpoint(
                output / "last.pt",
                current,
                training_state=training_state,
                corpus_provenance=corpus_provenance,
                overwrite=True,
            )
            if is_best:
                save_checkpoint(
                    output / "best.pt",
                    current,
                    training_state=training_state,
                    corpus_provenance=corpus_provenance,
                    overwrite=True,
                )
            logger.info(
                "profile checkpoint saved | epoch=%d | best=%s",
                epoch,
                is_best,
            )

        result = train_segment_profile_head(
            generator,
            iter_segment_profile_batches(
                corpus,
                "training",
                parent_ids=training_ids,
            ),
            iter_segment_profile_batches(
                corpus,
                "tuning",
                parent_ids=tuning_ids,
            ),
            head_config=head_config,
            config=profile_config,
            logger=logger,
            checkpoint_callback=save_profile_epoch,
        )
        best_generator, best_metadata = load_checkpoint(
            output / "best.pt",
            device=device,
        )
        evaluation = evaluate_segment_profile_head(
            best_generator,
            iter_segment_profile_batches(
                corpus,
                "tuning",
                parent_ids=tuning_ids,
            ),
            prior=result["profile_prior"],
            config=profile_config,
            logger=logger,
        )
        evaluation.update(
            {
                "mode": "smoke" if args.smoke else "formal",
                "runtime": runtime,
                "checkpoint_metadata": public_checkpoint_metadata(best_metadata),
                "training_parent_ids": list(training_ids),
                "tuning_parent_ids": list(tuning_ids),
            }
        )
        write_json(output / "profile_prior.json", result["profile_prior"])
        write_json(output / "profile_evaluation.json", evaluation)
        save_checkpoint(
            output / "generator.pt",
            best_generator,
            training_state=best_metadata["training_state"],
            corpus_provenance=best_metadata["corpus_provenance"],
        )
        write_json(
            output / "run_summary.json",
            {
                "schema": "structured_ginn_v2_segment_profile_run_v1",
                "status": "success",
                "mode": "smoke" if args.smoke else "formal",
                "runtime": runtime,
                "profile_head_config": asdict(head_config),
                "profile_learning_config": asdict(profile_config),
                **result,
                "evaluation": {
                    "learned_beats_deterministic": evaluation[
                        "learned_beats_deterministic"
                    ],
                    "learned_beats_state_conditioned_prior": evaluation[
                        "learned_beats_state_conditioned_prior"
                    ],
                    "learned_beats_strongest_baseline": evaluation[
                        "learned_beats_strongest_baseline"
                    ],
                    "learned_minus_deterministic_profile_rmse": evaluation[
                        "learned_minus_deterministic_profile_rmse"
                    ],
                    "learned_minus_state_conditioned_prior_profile_rmse": evaluation[
                        "learned_minus_state_conditioned_prior_profile_rmse"
                    ],
                },
            },
        )
        logger.info(
            "segment profile training finished | best_epoch=%d | "
            "learned_beats_strongest_baseline=%s",
            result["best_epoch"],
            evaluation["learned_beats_strongest_baseline"],
        )
        return
    if args.command == "probe-map-profiles":
        generator, source_metadata = load_checkpoint(
            resolve_relative_path(args.checkpoint, root=REPO_ROOT),
            device=device,
        )
        if generator.network_config.input_mode != "full":
            raise ValueError("MAP profile probe requires a full checkpoint.")
        if generator.profile_head is None:
            raise ValueError("MAP profile probe requires a V4 profile checkpoint.")
        source_training_state = dict(source_metadata.get("training_state") or {})
        profile_prior = source_training_state.get("profile_prior")
        if not isinstance(profile_prior, Mapping):
            raise ValueError("profile checkpoint metadata lacks profile_prior.")
        prior, conditioning, hsmm_metadata = load_semi_markov_contract(
            resolve_relative_path(args.hsmm_contract_dir, root=REPO_ROOT)
        )
        probe_config = MapProfileProbeConfig.from_mapping(
            config.get("map_profile_probe") or {},
            smoke=bool(args.smoke),
        )
        head_config = SegmentProfileHeadConfig.from_mapping(
            config.get("profile_head") or {}
        )
        training_ids = _balanced_parent_ids(
            corpus,
            "training",
            probe_config.training_parents_per_family,
        )
        tuning_ids = _balanced_parent_ids(
            corpus,
            "tuning",
            probe_config.tuning_parents_per_family,
        )
        calibration_ids = _balanced_parent_ids(
            corpus,
            "calibration",
            probe_config.calibration_parents_per_family,
        )
        corpus_provenance = dict(source_metadata.get("corpus_provenance") or {})
        corpus_provenance.update(
            {
                "map_profile_training_parent_ids": list(training_ids),
                "map_profile_tuning_parent_ids": list(tuning_ids),
                "map_profile_calibration_parent_ids": list(calibration_ids),
            }
        )

        def save_map_profile_epoch(
            epoch: int,
            current: ConditionalGenerator,
            state: Mapping[str, Any],
            is_best: bool,
        ) -> None:
            training_state = {
                "schema": "structured_ginn_v2_map_profile_probe_training_v1",
                "source_checkpoint_metadata": public_checkpoint_metadata(
                    source_metadata
                ),
                "profile_prior": dict(profile_prior),
                "hsmm_contract": dict(hsmm_metadata),
                **dict(state),
            }
            save_checkpoint(
                output / f"epoch_{epoch:04d}.pt",
                current,
                training_state=training_state,
                corpus_provenance=corpus_provenance,
            )
            save_checkpoint(
                output / "last.pt",
                current,
                training_state=training_state,
                corpus_provenance=corpus_provenance,
                overwrite=True,
            )
            if is_best:
                save_checkpoint(
                    output / "best.pt",
                    current,
                    training_state=training_state,
                    corpus_provenance=corpus_provenance,
                    overwrite=True,
                )
            logger.info(
                "MAP profile checkpoint saved | epoch=%d | best=%s",
                epoch,
                is_best,
            )

        logger.info(
            "MAP profile probe start | training_parents=%d | tuning_parents=%d | "
            "calibration_parents=%d",
            len(training_ids),
            len(tuning_ids),
            len(calibration_ids),
        )
        training_result = train_map_profile_probe(
            generator,
            iter_segment_profile_batches(
                corpus, "training", parent_ids=training_ids
            ),
            iter_segment_profile_batches(
                corpus, "tuning", parent_ids=tuning_ids
            ),
            iter_segment_profile_batches(
                corpus, "training", parent_ids=training_ids
            ),
            prior=prior,
            conditioning=conditioning,
            head_config=head_config,
            config=probe_config,
            logger=logger,
            checkpoint_callback=save_map_profile_epoch,
        )
        best_generator, best_metadata = load_checkpoint(
            output / "best.pt", device=device
        )
        reconstruction = evaluate_map_reconstruction(
            best_generator,
            iter_segment_profile_batches(
                corpus, "calibration", parent_ids=calibration_ids
            ),
            prior=prior,
            conditioning=conditioning,
            profile_prior=profile_prior,
            logger=logger,
            log_every_parents=1 if args.smoke else 3,
        )
        aggregate = reconstruction["aggregate"]
        highres = aggregate["highres_log_ai"]
        projected = aggregate["projected_log_ai"]
        family_regression: dict[str, bool] = {}
        for family, row in reconstruction["by_geometry_family"].items():
            family_highres = row["highres_log_ai"]
            family_projected = row["projected_log_ai"]
            family_regression[str(family)] = bool(
                family_highres["learned_profile_head"]["rmse"]
                > family_highres["deterministic_evidence_fit"]["rmse"]
                and family_projected["learned_profile_head"]["rmse"]
                > family_projected["direct_bandlimited_evidence"]["rmse"]
            )
        gate = {
            "learned_beats_deterministic_highres": bool(
                highres["learned_profile_head"]["rmse"]
                < highres["deterministic_evidence_fit"]["rmse"]
            ),
            "learned_not_worse_than_direct_projected": bool(
                projected["learned_profile_head"]["rmse"]
                <= projected["direct_bandlimited_evidence"]["rmse"]
            ),
            "no_family_regresses_both_resolutions": bool(
                not any(family_regression.values())
            ),
            "family_regresses_both_resolutions": family_regression,
        }
        gate["passed"] = bool(
            gate["learned_beats_deterministic_highres"]
            and gate["learned_not_worse_than_direct_projected"]
            and gate["no_family_regresses_both_resolutions"]
        )
        reconstruction["map_profile_probe_gate"] = gate
        write_json(output / "map_reconstruction.json", reconstruction)
        save_checkpoint(
            output / "generator.pt",
            best_generator,
            training_state=best_metadata["training_state"],
            corpus_provenance=best_metadata["corpus_provenance"],
        )
        summary = {
            **training_result,
            "mode": "smoke" if args.smoke else "formal",
            "runtime": runtime,
            "map_profile_probe_config": asdict(probe_config),
            "profile_head_config": asdict(head_config),
            "training_parent_ids": list(training_ids),
            "tuning_parent_ids": list(tuning_ids),
            "calibration_parent_ids": list(calibration_ids),
            "calibration_gate": gate,
            "calibration_metrics": {
                "learned_highres_rmse": highres["learned_profile_head"]["rmse"],
                "deterministic_highres_rmse": highres[
                    "deterministic_evidence_fit"
                ]["rmse"],
                "learned_projected_rmse": projected["learned_profile_head"][
                    "rmse"
                ],
                "direct_projected_rmse": projected[
                    "direct_bandlimited_evidence"
                ]["rmse"],
            },
        }
        write_json(output / "probe_summary.json", summary)
        logger.info(
            "MAP profile probe finished | best_epoch=%d | highres=%.6f vs %.6f | "
            "projected=%.6f vs direct %.6f | gate=%s",
            training_result["best_epoch"],
            summary["calibration_metrics"]["learned_highres_rmse"],
            summary["calibration_metrics"]["deterministic_highres_rmse"],
            summary["calibration_metrics"]["learned_projected_rmse"],
            summary["calibration_metrics"]["direct_projected_rmse"],
            gate["passed"],
        )
        return
    if args.command == "calibrate-profile-variance":
        generator, source_metadata = load_checkpoint(
            resolve_relative_path(args.checkpoint, root=REPO_ROOT),
            device=device,
        )
        if generator.network_config.input_mode != "full":
            raise ValueError("coefficient variance calibration requires a full checkpoint.")
        if generator.profile_head is None:
            raise ValueError("coefficient variance calibration requires a profile head.")
        if generator.coefficient_variance_calibration is not None:
            raise ValueError("source checkpoint already contains variance calibration.")
        variance_config = CoefficientVarianceCalibrationConfig.from_mapping(
            config.get("coefficient_variance_calibration") or {},
            smoke=bool(args.smoke),
        )
        calibration_ids = _balanced_parent_ids(
            corpus,
            "calibration",
            variance_config.parents_per_family,
        )
        logger.info(
            "coefficient variance calibration start | parents=%d | per_family=%d",
            len(calibration_ids),
            variance_config.parents_per_family,
        )
        report = calibrate_coefficient_variance(
            generator,
            iter_segment_profile_batches(
                corpus,
                "calibration",
                parent_ids=calibration_ids,
            ),
            config=variance_config,
            logger=logger,
        )
        report.update(
            {
                "mode": "smoke" if args.smoke else "formal",
                "runtime": runtime,
                "config": asdict(variance_config),
                "calibration_parent_ids": list(calibration_ids),
                "source_checkpoint_metadata": public_checkpoint_metadata(
                    source_metadata
                ),
            }
        )
        training_state = {
            **dict(source_metadata.get("training_state") or {}),
            "coefficient_variance_calibration": report["calibration"],
            "coefficient_variance_calibration_config": asdict(variance_config),
            "coefficient_variance_calibration_parent_ids": list(calibration_ids),
        }
        corpus_provenance = dict(source_metadata.get("corpus_provenance") or {})
        corpus_provenance["coefficient_variance_calibration_parent_ids"] = list(
            calibration_ids
        )
        save_checkpoint(
            output / "generator.pt",
            generator,
            training_state=training_state,
            corpus_provenance=corpus_provenance,
        )
        write_json(output / "coefficient_variance_calibration.json", report)
        write_json(
            output / "run_summary.json",
            {
                "schema": "structured_ginn_v2_coefficient_variance_run_v1",
                "status": "success",
                "mode": report["mode"],
                "parent_count": report["parent_count"],
                "identifiable_segment_count": report[
                    "identifiable_segment_count"
                ],
                "calibration": report["calibration"],
                "coverage_before": report["before"]["aggregate"],
                "coverage_after": report["after"]["aggregate"],
                "generator": str(output / "generator.pt"),
            },
        )
        logger.info(
            "coefficient variance calibration finished | generator=%s",
            output / "generator.pt",
        )
        return
    if args.command == "evaluate-reconstruction":
        generator, metadata = load_checkpoint(
            resolve_relative_path(args.checkpoint, root=REPO_ROOT),
            device=device,
        )
        if generator.network_config.input_mode != "full":
            raise ValueError("MAP reconstruction requires a full checkpoint.")
        if generator.profile_head is None:
            raise ValueError("MAP reconstruction requires a V4 profile checkpoint.")
        training_state = dict(metadata.get("training_state") or {})
        profile_prior = training_state.get("profile_prior")
        if not isinstance(profile_prior, Mapping):
            raise ValueError("profile checkpoint metadata lacks profile_prior.")
        prior, conditioning, hsmm_metadata = load_semi_markov_contract(
            resolve_relative_path(args.hsmm_contract_dir, root=REPO_ROOT)
        )
        per_family = 1 if args.smoke else int(args.parents_per_family)
        parent_ids = _balanced_parent_ids(corpus, args.split, per_family)
        logger.info(
            "MAP reconstruction start | split=%s | parents=%d | per_family=%d",
            args.split,
            len(parent_ids),
            per_family,
        )
        result = evaluate_map_reconstruction(
            generator,
            iter_segment_profile_batches(
                corpus,
                args.split,
                parent_ids=parent_ids,
            ),
            prior=prior,
            conditioning=conditioning,
            profile_prior=profile_prior,
            logger=logger,
            log_every_parents=1 if args.smoke else 3,
        )
        profile_config = SegmentProfileLearningConfig.from_mapping(
            config.get("profile_learning") or {},
            smoke=bool(args.smoke),
        )
        result["truth_segment_profile_control"] = evaluate_segment_profile_head(
            generator,
            iter_segment_profile_batches(
                corpus,
                args.split,
                parent_ids=parent_ids,
            ),
            prior=profile_prior,
            config=profile_config,
            logger=logger,
        )
        result.update(
            {
                "mode": "smoke" if args.smoke else "formal",
                "split": args.split,
                "parent_ids": list(parent_ids),
                "runtime": runtime,
                "checkpoint_metadata": public_checkpoint_metadata(metadata),
                "hsmm_contract": dict(hsmm_metadata),
            }
        )
        write_json(output / "map_reconstruction.json", result)
        logger.info(
            "MAP reconstruction finished | highres_rmse=%.6f | "
            "projected_rmse=%.6f | beats_highres=%s | beats_projected=%s",
            result["aggregate"]["highres_log_ai"]["learned_profile_head"]["rmse"],
            result["aggregate"]["projected_log_ai"]["learned_profile_head"]["rmse"],
            result["gate"]["learned_beats_strongest_highres_baseline"],
            result["gate"]["learned_beats_strongest_projected_baseline"],
        )
        return
    if args.command == "evaluate-ensemble":
        generator, metadata = load_checkpoint(
            resolve_relative_path(args.checkpoint, root=REPO_ROOT),
            device=device,
        )
        if generator.network_config.input_mode != "full":
            raise ValueError("ensemble evaluation requires a full checkpoint.")
        if generator.profile_head is None:
            raise ValueError("ensemble evaluation requires a profile head.")
        if generator.coefficient_variance_calibration is None:
            raise ValueError(
                "ensemble evaluation requires calibrated coefficient variance."
            )
        if generator.semi_markov_prior is not None:
            raise ValueError("source checkpoint already contains a semi-Markov contract.")
        prior, conditioning, hsmm_metadata = load_semi_markov_contract(
            resolve_relative_path(args.hsmm_contract_dir, root=REPO_ROOT)
        )
        generator.set_semi_markov_contract(prior, conditioning)
        per_family = 1 if args.smoke else int(args.parents_per_family)
        parent_ids = _balanced_parent_ids(corpus, args.split, per_family)
        policy = GenerationPolicy(
            realization_count=(2 if args.smoke else int(args.realization_count)),
            random_identity=int(args.random_identity),
            retain_realizations=True,
            lateral_correlation_m=float(args.lateral_correlation_m),
            path_coupling_strength=float(args.path_coupling_strength),
            profile_coupling_strength=float(args.profile_coupling_strength),
        )
        logger.info(
            "ensemble evaluation start | split=%s | parents=%d | K=%d",
            args.split,
            len(parent_ids),
            policy.realization_count,
        )
        result = evaluate_structured_ensemble(
            generator,
            iter_segment_profile_batches(
                corpus,
                args.split,
                parent_ids=parent_ids,
            ),
            policy=policy,
            logger=logger,
            log_every_zones=1 if args.smoke else 3,
            figure_output_dir=output / "figures" / "ensemble",
            figures_per_family=int(args.figures_per_family),
        )
        training_state = {
            **dict(metadata.get("training_state") or {}),
            "semi_markov_contract": dict(hsmm_metadata),
            "ensemble_policy": asdict(policy),
            "spatial_random_key_version": 3,
        }
        corpus_provenance = dict(metadata.get("corpus_provenance") or {})
        corpus_provenance["ensemble_evaluation_parent_ids"] = list(parent_ids)
        checkpoint = save_checkpoint(
            output / "generator.pt",
            generator,
            training_state=training_state,
            runtime_state=metadata.get("runtime_state"),
            corpus_provenance=corpus_provenance,
        )
        result.update(
            {
                "mode": "smoke" if args.smoke else "formal",
                "split": args.split,
                "parent_ids": list(parent_ids),
                "runtime": runtime,
                "source_checkpoint_metadata": public_checkpoint_metadata(metadata),
                "hsmm_contract": dict(hsmm_metadata),
                "generator": str(checkpoint),
            }
        )
        write_json(output / "ensemble_evaluation.json", result)
        write_json(
            output / "run_summary.json",
            {
                "schema": "structured_ginn_v2_ensemble_run_v4",
                "status": "success",
                "mode": result["mode"],
                "parent_count": result["aggregate"]["parent_count"],
                "zone_count": result["aggregate"]["zone_count"],
                "realization_count": policy.realization_count,
                "highres_log_ai": result["aggregate"]["highres_log_ai"],
                "projected_log_ai": result["aggregate"]["projected_log_ai"],
                "lateral_continuity": result["aggregate"]["lateral_continuity"],
                "figure_count": len(result["figures"]),
                "figure_errors": result["figure_errors"],
                "generator": str(checkpoint),
            },
        )
        logger.info(
            "ensemble evaluation finished | highres_rmse=%.6f | "
            "projected_rmse=%.6f | highres_coverage95=%.4f",
            result["aggregate"]["highres_log_ai"]["ensemble_mean_rmse"],
            result["aggregate"]["projected_log_ai"]["ensemble_mean_rmse"],
            result["aggregate"]["highres_log_ai"]["coverage_95"],
        )
        return
    if args.command == "train":
        learning_config = LearningConfig(**dict(config.get("learning") or {}))
        if args.smoke:
            learning_config = replace(
                learning_config,
                epochs=1,
                log_every_batches=1,
            )
        target_contract_path = resolve_relative_path(
            args.target_contract,
            root=REPO_ROOT,
        )
        target_contract = load_observable_target_contract(
            target_contract_path,
            corpus=corpus,
        )
        network_mapping = dict(config.get("network") or {})
        if args.input_mode is not None:
            network_mapping["input_mode"] = args.input_mode
        network_config = EvidenceNetworkConfig.from_mapping(
            network_mapping,
            target_contract=target_contract,
        )
        resume_metadata: Mapping[str, Any] = {}
        if resume_path is None:
            seed_training_random_streams(learning_config.random_seed)
            network = ObservableEvidenceNetwork(network_config)
            generator = ConditionalGenerator(
                network,
                target_contract=target_contract,
                dominant_frequency_hz=_dominant_frequency(corpus),
                sample_domain=corpus.benchmark.sample_domain,
                device=device,
            )
        else:
            generator, resume_metadata = load_checkpoint(
                resume_path,
                device=device,
            )
            if generator.network_config != network_config:
                raise ValueError(
                    "resume checkpoint network/input contract differs from config."
                )
            if generator.target_contract != target_contract:
                raise ValueError(
                    "resume checkpoint target contract differs from --target-contract."
                )
            if not resume_metadata.get("runtime_state"):
                raise ValueError(
                    "resume checkpoint has no runtime_state; use a new training run."
                )

        corpus_provenance = {
            "root": str(corpus.root),
            "recorded_contract_fingerprint_sha256": str(
                corpus.manifest.get("contract_fingerprint_sha256") or ""
            ),
            "observable_target_contract": str(target_contract_path),
        }

        def checkpoint_callback(
            epoch: int,
            checkpoint_generator: ConditionalGenerator,
            runtime_state: Mapping[str, Any],
            is_best: bool,
        ) -> None:
            training_state = {
                "epoch": int(epoch),
                "best_tuning_loss": float(runtime_state["best_tuning_loss"]),
                "best_epoch": int(runtime_state["best_epoch"]),
                "history": list(runtime_state["history"]),
                "learning_config": asdict(learning_config),
                "target_contract": target_contract.to_mapping(),
            }
            epoch_path = output / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                epoch_path,
                checkpoint_generator,
                training_state=training_state,
                runtime_state=runtime_state,
                corpus_provenance=corpus_provenance,
            )
            last_path = output / "last.pt"
            save_checkpoint(
                last_path,
                checkpoint_generator,
                training_state=training_state,
                runtime_state=runtime_state,
                corpus_provenance=corpus_provenance,
                overwrite=True,
            )
            best_path = output / "best.pt"
            if is_best:
                save_checkpoint(
                    best_path,
                    checkpoint_generator,
                    training_state=training_state,
                    runtime_state=runtime_state,
                    corpus_provenance=corpus_provenance,
                    overwrite=True,
                )
            write_json(
                output / "training_progress.json",
                {
                    "schema": "structured_ginn_v2_training_progress_v1",
                    "status": "running",
                    "epoch": int(epoch),
                    "target_epochs": int(learning_config.epochs),
                    "best_epoch": int(runtime_state["best_epoch"]),
                    "best_tuning_loss": float(runtime_state["best_tuning_loss"]),
                    "last_checkpoint": str(last_path),
                    "best_checkpoint": str(best_path),
                    "epoch_checkpoint": str(epoch_path),
                    "history": list(runtime_state["history"]),
                },
            )
            logger.info(
                "checkpoint saved | epoch=%d | last=%s | best=%s",
                epoch,
                last_path.name,
                best_path.name if is_best else "unchanged",
            )

        result = train_generator(
            generator,
            (
                (
                    lambda: iter_evidence_batches(
                        corpus,
                        "training",
                        parent_limit=3 if args.smoke else None,
                    )
                )
                if augmentation_profile is None
                else (
                    lambda: iter_paired_evidence_batches(
                        corpus,
                        "training",
                        augmentation_profile=augmentation_profile,
                        parent_limit=3 if args.smoke else None,
                    )
                )
            ),
            (
                lambda: iter_evidence_batches(
                    corpus,
                    "tuning",
                    parent_limit=3 if args.smoke else None,
                )
            ),
            config=learning_config,
            logger=logger,
            resume_state=(
                resume_metadata.get("runtime_state")
                if resume_path is not None
                else None
            ),
            checkpoint_callback=checkpoint_callback,
        )

        best_path = output / "best.pt"
        if not best_path.is_file():
            raise RuntimeError("training completed without a best checkpoint.")
        best_generator, best_metadata = load_checkpoint(best_path, device=device)
        checkpoint = save_checkpoint(
            output / "generator.pt",
            best_generator,
            training_state=best_metadata["training_state"],
            runtime_state=best_metadata.get("runtime_state"),
            corpus_provenance=best_metadata["corpus_provenance"],
            overwrite=True,
        )
        write_json(
            output / "training_progress.json",
            {
                "schema": "structured_ginn_v2_training_progress_v1",
                "status": "completed",
                "epoch": int(result["epochs_completed"]),
                "target_epochs": int(learning_config.epochs),
                "best_epoch": int(result["best_epoch"]),
                "best_tuning_loss": float(result["best_tuning_loss"]),
                "last_checkpoint": str(output / "last.pt"),
                "best_checkpoint": str(best_path),
                "final_checkpoint": str(checkpoint),
                "history": list(result["history"]),
            },
        )
        write_json(
            output / "run_summary.json",
            {
                "schema": "structured_ginn_v2_training_v1",
                "status": "success",
                "mode": "smoke" if args.smoke else "formal",
                "runtime": runtime,
                "training_conditions": (
                    ["clean"]
                    if augmentation_profile is None
                    else ["clean", "dirty"]
                ),
                "checkpoint": str(checkpoint),
                "last_checkpoint": str(output / "last.pt"),
                "best_checkpoint": str(best_path),
                "resumed_from": str(resume_path) if resume_path else None,
                "network_config": asdict(network_config),
                "target_contract": target_contract.to_mapping(),
                "learning_config": asdict(learning_config),
                **result,
            },
        )
        return
    if args.command == "evaluate-hsmm":
        generator, metadata = load_checkpoint(
            resolve_relative_path(args.checkpoint, root=REPO_ROOT),
            device=device,
        )
        if generator.network_config.input_mode != "full":
            raise ValueError("HSMM oracle requires a full evidence checkpoint.")
        per_family = 1 if args.smoke else int(args.parents_per_family)
        parent_ids = _balanced_parent_ids(corpus, args.split, per_family)
        prior_per_family = (
            1 if args.smoke else int(args.prior_parents_per_family)
        )
        prior_parent_ids = _balanced_parent_ids(
            corpus,
            "training",
            prior_per_family,
        )
        logger.info(
            "HSMM calibration start | split=%s | parents=%d | per_family=%d | "
            "prior_training_parents=%d",
            args.split,
            len(parent_ids),
            per_family,
            len(prior_parent_ids),
        )
        prior = calibrate_semi_markov_prior(
            corpus,
            parent_ids=prior_parent_ids,
        )
        write_json(output / "semi_markov_prior.json", prior.to_mapping())
        result = calibrate_semi_markov_fusion(
            generator,
            iter_evidence_batches(
                corpus,
                args.split,
                parent_ids=parent_ids,
            ),
            prior=prior,
            logger=logger,
            log_every_batches=1 if args.smoke else 5,
        )
        common_metadata = {
            "runtime": runtime,
            "split": args.split,
            "mode": "smoke" if args.smoke else "formal",
            "parent_ids": list(parent_ids),
            "prior_training_parent_ids": list(prior_parent_ids),
            "checkpoint_metadata": public_checkpoint_metadata(metadata),
        }
        oracle = dict(result["oracle"])
        oracle.update(common_metadata)
        calibration = {
            key: value for key, value in result.items() if key != "oracle"
        }
        calibration.update(common_metadata)
        calibration["selected_oracle"] = "hsmm_oracle.json"
        write_json(output / "hsmm_calibration.json", calibration)
        write_json(output / "hsmm_oracle.json", oracle)
        selected = calibration["selected"]["conditioning"]
        logger.info(
            "HSMM calibration finished | status=%s | parents=%d | "
            "state_weight=%.3g | duration_temperature=%.3g | "
            "transition_temperature=%.3g",
            calibration["status"],
            calibration["parent_count"],
            selected["state_evidence_weight"],
            selected["duration_temperature"],
            selected["transition_temperature"],
        )
        return
    generator, metadata = load_checkpoint(
        resolve_relative_path(args.checkpoint, root=REPO_ROOT),
        device=device,
    )
    if generator.network_config.input_mode != "full":
        raise ValueError("the primary evaluation checkpoint must use input_mode=full.")
    controls: dict[str, ConditionalGenerator] = {}
    checkpoint_metadata: dict[str, object] = {
        "full": public_checkpoint_metadata(metadata)
    }
    for name, configured_path in (
        ("no_seismic", args.no_seismic_checkpoint),
    ):
        if configured_path is None:
            continue
        control, control_metadata = load_checkpoint(
            resolve_relative_path(configured_path, root=REPO_ROOT),
            device=device,
        )
        if control.network_config.input_mode != name:
            raise ValueError(
                f"{name} control checkpoint has input_mode="
                f"{control.network_config.input_mode!r}."
            )
        controls[name] = control
        checkpoint_metadata[name] = public_checkpoint_metadata(control_metadata)

    clean_metrics = evaluate_generator(
        generator,
        iter_evidence_batches(
            corpus,
            args.split,
            parent_limit=args.parent_limit,
        ),
        controls=controls,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    dirty_metrics = (
        None
        if augmentation_profile is None
        else evaluate_generator(
            generator,
            iter_evidence_batches(
                corpus,
                args.split,
                condition="dirty",
                augmentation_profile=augmentation_profile,
                parent_limit=args.parent_limit,
            ),
            controls=controls,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    )
    write_json(
        output / "evaluation.json",
        {
            "schema": "structured_ginn_v2_evaluation_v2",
            "status": "success",
            "split": args.split,
            "runtime": runtime,
            "checkpoint_metadata": checkpoint_metadata,
            "conditions": {
                "clean": clean_metrics,
                "dirty": dirty_metrics,
            },
        },
    )


if __name__ == "__main__":
    main()
