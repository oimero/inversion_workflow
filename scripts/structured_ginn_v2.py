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
    load_observable_target_contract,
    load_checkpoint,
    load_corpus,
    public_checkpoint_metadata,
    save_checkpoint,
)
from ginn_v2.augmentation import load_observation_augmentation_profile
from ginn_v2.evidence import (
    EvidenceNetworkConfig,
    ObservableEvidenceNetwork,
    dominant_frequency_hz,
)
from ginn_v2.generator import ConditionalGenerator
from ginn_v2.learning import (
    LearningConfig,
    TargetAuditConfig,
    audit_observable_targets,
    calibrate_semi_markov_fusion,
    evaluate_generator,
    seed_training_random_streams,
    train_generator,
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
        "--smoke",
        action="store_true",
        help="evaluate one parent per geometry family",
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
        logger.info(
            "HSMM calibration start | split=%s | parents=%d | per_family=%d",
            args.split,
            len(parent_ids),
            per_family,
        )
        prior = calibrate_semi_markov_prior(corpus)
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
