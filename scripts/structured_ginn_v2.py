"""Audit targets, train, or evaluate the Structured GINN V2 generator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time
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
    public_checkpoint_metadata,
    save_checkpoint,
    save_synthetic_section_prediction,
    parent_observation_tiles,
)
from ginn_v2.augmentation import load_observation_augmentation_profile
from ginn_v2.evidence import (
    EvidenceNetworkConfig,
    ObservableEvidenceNetwork,
    dominant_frequency_hz,
)
from ginn_v2.generator import ConditionalGenerator, SegmentProfileHeadConfig
from ginn_v2.inference import infer_section
from ginn_v2.learning import (
    CoefficientVarianceCalibrationConfig,
    LearningConfig,
    SegmentProfileLearningConfig,
    TargetAuditConfig,
    audit_observable_targets,
    calibrate_coefficient_variance,
    calibrate_semi_markov_fusion,
    evaluate_generator,
    evaluate_segment_profile_head,
    seed_training_random_streams,
    train_generator,
    train_segment_profile_head,
)
from ginn_v2.runtime import configure_training_logger, resolve_device
from ginn_v2.semi_markov import SemiMarkovConditioning


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
        help="training-parent budget per family for the high-resolution event prior",
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
    variance = subparsers.add_parser("calibrate-profile-variance")
    variance.add_argument("--corpus", type=Path, required=True)
    variance.add_argument("--checkpoint", type=Path, required=True)
    variance.add_argument("--output-dir", type=Path, required=True)
    variance.add_argument(
        "--smoke",
        action="store_true",
        help="calibrate on one parent per geometry family",
    )
    audit = subparsers.add_parser("audit-targets")
    audit.add_argument("--corpus", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument(
        "--smoke",
        action="store_true",
        help="exercise L0-L2 with tiny budgets without publishing a target contract",
    )
    sections = subparsers.add_parser("generate-sections")
    sections.add_argument("--corpus", type=Path, required=True)
    sections.add_argument("--checkpoint", type=Path, required=True)
    sections.add_argument("--output-dir", type=Path, required=True)
    sections.add_argument(
        "--smoke",
        action="store_true",
        help="generate one full section from each geometry family",
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


def _section_parent_ids(corpus, *, smoke: bool) -> tuple[str, ...]:
    index = corpus.benchmark.index
    rows = index.loc[
        index["split_role"].eq("section_gate")
        & index["corpus_role"].eq("full_section")
    ]
    values = tuple(sorted(rows["realization_id"].astype(str).tolist()))
    if not values or set(values) != set(corpus.splits["section_gate"]):
        raise ValueError("section_gate must contain only full-section parents.")
    if not smoke:
        return values
    selected: list[str] = []
    for family in ("none", "wedge", "pinchout"):
        family_values = sorted(
            rows.loc[
                rows["geometry_family"].eq(family),
                "realization_id",
            ].astype(str)
        )
        if not family_values:
            raise ValueError(f"section_gate has no {family!r} parent.")
        selected.append(family_values[0])
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
    if args.command == "generate-sections":
        generator, checkpoint_metadata = load_checkpoint(
            resolve_relative_path(args.checkpoint, root=REPO_ROOT),
            device=device,
        )
        if generator.network_config.input_mode != "full":
            raise ValueError("section generation requires a full checkpoint.")
        if generator.profile_head is None:
            raise ValueError("section generation requires a trained profile head.")
        if generator.semi_markov_prior is None:
            raise ValueError("section generation requires a semi-Markov contract.")
        parent_ids = _section_parent_ids(corpus, smoke=bool(args.smoke))
        index = corpus.benchmark.index.set_index("realization_id", drop=False)
        started = time.perf_counter()
        parent_rows: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        generated_zone_count = 0
        logger.info(
            "section generation start | mode=%s | parents=%d",
            "smoke" if args.smoke else "formal",
            len(parent_ids),
        )
        for parent_number, parent_id in enumerate(parent_ids, start=1):
            parent_started = time.perf_counter()
            row = index.loc[parent_id]
            parent_output = output / "parents" / f"p{parent_number:04d}"
            zone_rows: list[dict[str, Any]] = []
            try:
                parent = corpus.benchmark.read_parent(parent_id)
                tiles = parent_observation_tiles(corpus, parent_id)
            except Exception as error:
                logger.exception(
                    "section parent load failed | parent=%s",
                    parent_id,
                )
                errors.append(
                    {
                        "parent_id": parent_id,
                        "zone_id": None,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                tiles = ()
                parent = None
            for zone_number, tile in enumerate(tiles, start=1):
                zone_id = tile.identity.rsplit(":", maxsplit=1)[-1]
                try:
                    prediction = infer_section(generator, tile)
                    zone_output = save_synthetic_section_prediction(
                        parent_output / f"z{zone_number:02d}",
                        prediction,
                        observation=tile,
                        truth_log_ai_highres=parent.log_ai_highres,
                    )
                except Exception as error:
                    logger.exception(
                        "section zone generation failed | parent=%s | zone=%s",
                        parent_id,
                        zone_id,
                    )
                    errors.append(
                        {
                            "parent_id": parent_id,
                            "zone_id": zone_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    continue
                generated_zone_count += 1
                zone_rows.append(
                    {
                        "zone_id": zone_id,
                        "output": str(zone_output.relative_to(output)),
                        "segment_count": len(prediction.realization.segments),
                        "conditional_log_score": (
                            prediction.realization.conditional_log_score
                        ),
                        "diagnostics": dict(prediction.diagnostics),
                    }
                )
                logger.info(
                    "section zone generated | parent=%d/%d | family=%s | "
                    "zone=%d/%d | segments=%d | elapsed=%.1fs",
                    parent_number,
                    len(parent_ids),
                    row["geometry_family"],
                    zone_number,
                    len(tiles),
                    len(prediction.realization.segments),
                    time.perf_counter() - parent_started,
                )
            parent_status = (
                "success"
                if len(zone_rows) == len(tiles) and len(tiles) > 0
                else "completed_with_warnings"
            )
            parent_report = {
                "parent_id": parent_id,
                "geometry_family": str(row["geometry_family"]),
                "generalization_role": str(row["generalization_role"]),
                "status": parent_status,
                "zone_count": len(tiles),
                "generated_zone_count": len(zone_rows),
                "zones": zone_rows,
                "elapsed_seconds": time.perf_counter() - parent_started,
            }
            parent_rows.append(parent_report)
            write_json(parent_output / "parent_summary.json", parent_report)
            write_json(
                output / "generation_progress.json",
                {
                    "schema": "structured_ginn_v2_section_progress_v1",
                    "status": "running",
                    "mode": "smoke" if args.smoke else "formal",
                    "completed_parent_count": parent_number,
                    "requested_parent_count": len(parent_ids),
                    "generated_zone_count": generated_zone_count,
                    "error_count": len(errors),
                    "elapsed_seconds": time.perf_counter() - started,
                    "last_parent_id": parent_id,
                },
            )
        status = "success" if not errors else "completed_with_warnings"
        report = {
            "schema": "structured_ginn_v2_section_generation_v1",
            "status": status,
            "mode": "smoke" if args.smoke else "formal",
            "runtime": runtime,
            "requested_parent_count": len(parent_ids),
            "completed_parent_count": len(parent_rows),
            "generated_zone_count": generated_zone_count,
            "error_count": len(errors),
            "errors": errors,
            "parents": parent_rows,
            "checkpoint_metadata": public_checkpoint_metadata(
                checkpoint_metadata
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(output / "run_summary.json", report)
        write_json(
            output / "generation_progress.json",
            {
                "schema": "structured_ginn_v2_section_progress_v1",
                "status": status,
                "mode": report["mode"],
                "completed_parent_count": len(parent_rows),
                "requested_parent_count": len(parent_ids),
                "generated_zone_count": generated_zone_count,
                "error_count": len(errors),
                "elapsed_seconds": report["elapsed_seconds"],
            },
        )
        if generated_zone_count == 0:
            raise RuntimeError("section generation produced no usable zones.")
        logger.info(
            "section generation finished | status=%s | parents=%d | zones=%d | "
            "errors=%d | elapsed=%.1fs",
            status,
            len(parent_rows),
            generated_zone_count,
            len(errors),
            report["elapsed_seconds"],
        )
        return
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
        if generator.profile_head is None:
            raise ValueError(
                "HSMM calibration requires a trained segment profile checkpoint."
            )
        if generator.semi_markov_prior is not None:
            raise ValueError("source checkpoint already contains a semi-Markov contract.")
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
            conditioning_candidates=(
                (SemiMarkovConditioning(),) if args.smoke else None
            ),
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
        conditioning = SemiMarkovConditioning.from_mapping(selected)
        generator.set_semi_markov_contract(prior, conditioning)
        training_state = {
            **dict(metadata.get("training_state") or {}),
            "semi_markov_calibration": {
                "schema": calibration["schema"],
                "split": args.split,
                "mode": common_metadata["mode"],
                "conditioning": conditioning.to_mapping(),
                "prior_training_parent_ids": list(prior_parent_ids),
                "calibration_parent_ids": list(parent_ids),
            },
        }
        corpus_provenance = dict(metadata.get("corpus_provenance") or {})
        corpus_provenance.update(
            {
                "semi_markov_prior_parent_ids": list(prior_parent_ids),
                "semi_markov_calibration_parent_ids": list(parent_ids),
            }
        )
        save_checkpoint(
            output / "generator.pt",
            generator,
            training_state=training_state,
            corpus_provenance=corpus_provenance,
        )
        write_json(
            output / "run_summary.json",
            {
                "schema": "structured_ginn_v2_deterministic_generator_run_v1",
                "status": "success",
                "mode": common_metadata["mode"],
                "split": args.split,
                "prior_parent_count": len(prior_parent_ids),
                "calibration_parent_count": len(parent_ids),
                "conditioning": conditioning.to_mapping(),
                "generator": str(output / "generator.pt"),
            },
        )
        logger.info(
            "HSMM calibration finished | status=%s | parents=%d | "
            "state_weight=%.3g | duration_temperature=%.3g | "
            "transition_temperature=%.3g | generator=%s",
            calibration["status"],
            calibration["parent_count"],
            selected["state_evidence_weight"],
            selected["duration_temperature"],
            selected["transition_temperature"],
            output / "generator.pt",
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
