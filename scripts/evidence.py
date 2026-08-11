"""Train or evaluate the band-limited seismic evidence model."""

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
from evidence.artifacts import (
    load_checkpoint,
    public_checkpoint_metadata,
    save_checkpoint,
)
from evidence.augmentation import load_observation_augmentation_profile
from evidence.data import (
    iter_evidence_batches,
    iter_paired_evidence_batches,
    load_corpus,
    load_target_contract,
)
from evidence.learning import (
    EvidenceLearningConfig,
    evaluate_evidence_model,
    seed_random_streams,
    train_evidence_model,
)
from evidence.network import (
    BandlimitedEvidenceNetwork,
    EvidenceModel,
    EvidenceNetworkConfig,
    dominant_frequency_hz,
)
from evidence.runtime import configure_training_logger, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/evidence/evidence.yaml"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--corpus", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument(
        "--target-contract",
        type=Path,
        default=Path("experiments/evidence/target_contract.json"),
    )
    train.add_argument("--input-mode", choices=("full", "no_seismic"), default=None)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--resume", type=Path, default=None)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--corpus", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument(
        "--split",
        choices=("tuning", "calibration"),
        default="calibration",
    )
    evaluate.add_argument("--no-seismic-checkpoint", type=Path, default=None)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=2000)
    evaluate.add_argument("--parent-limit", type=int, default=None)
    return parser.parse_args()


def _config(path: Path) -> dict[str, Any]:
    payload = load_yaml_config(resolve_relative_path(path, root=REPO_ROOT))
    if not isinstance(payload, dict) or set(payload) != {"evidence"}:
        raise ValueError("config must contain only evidence.")
    root = payload["evidence"]
    if not isinstance(root, dict):
        raise ValueError("evidence must be a mapping.")
    allowed = {"device", "network", "learning", "augmentation_profile"}
    unknown = sorted(set(root).difference(allowed))
    if unknown:
        raise ValueError(f"unknown evidence config keys: {unknown}")
    return root


def _augmentation_profile(config: Mapping[str, Any]):
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


def _prepare_output(args: argparse.Namespace) -> tuple[Path, Path | None]:
    output = resolve_relative_path(args.output_dir, root=REPO_ROOT)
    resume_path: Path | None = None
    if args.command == "train" and args.resume is not None:
        resume_path = resolve_relative_path(args.resume, root=REPO_ROOT)
        if not output.is_dir():
            raise FileNotFoundError(f"resume output directory does not exist: {output}")
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
    return output, resume_path


def _train(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    device,
    runtime: Mapping[str, Any],
) -> None:
    corpus = load_corpus(resolve_relative_path(args.corpus, root=REPO_ROOT))
    output, resume_path = _prepare_output(args)
    logger = configure_training_logger(output)
    augmentation_profile = _augmentation_profile(config)
    learning_config = EvidenceLearningConfig(**dict(config.get("learning") or {}))
    if args.smoke:
        learning_config = replace(learning_config, epochs=1, log_every_batches=1)
    target_contract_path = resolve_relative_path(args.target_contract, root=REPO_ROOT)
    target_contract = load_target_contract(target_contract_path, corpus=corpus)
    network_mapping = dict(config.get("network") or {})
    if args.input_mode is not None:
        network_mapping["input_mode"] = args.input_mode
    network_config = EvidenceNetworkConfig.from_mapping(
        network_mapping,
        target_contract=target_contract,
    )
    resume_metadata: Mapping[str, Any] = {}
    if resume_path is None:
        seed_random_streams(learning_config.random_seed)
        model = EvidenceModel(
            BandlimitedEvidenceNetwork(network_config),
            target_contract=target_contract,
            dominant_frequency=_dominant_frequency(corpus),
            device=device,
        )
    else:
        model, resume_metadata = load_checkpoint(resume_path, device=device)
        if model.network_config != network_config:
            raise ValueError("resume checkpoint network contract differs from config.")
        if model.target_contract != target_contract:
            raise ValueError("resume checkpoint target contract differs from input.")
        if not resume_metadata.get("runtime_state"):
            raise ValueError("resume checkpoint has no runtime_state.")

    corpus_provenance = {
        "root": str(corpus.root),
        "recorded_contract_fingerprint_sha256": str(
            corpus.manifest.get("contract_fingerprint_sha256") or ""
        ),
        "target_contract": str(target_contract_path),
    }

    def publish_checkpoint(
        epoch: int,
        checkpoint_model: EvidenceModel,
        runtime_state: Mapping[str, Any],
        is_best: bool,
    ) -> None:
        training_state = {
            "epoch": epoch,
            "best_tuning_loss": float(runtime_state["best_tuning_loss"]),
            "best_epoch": int(runtime_state["best_epoch"]),
            "history": list(runtime_state["history"]),
            "learning_config": asdict(learning_config),
            "target_contract": target_contract.to_mapping(),
        }
        epoch_path = output / f"epoch_{epoch:04d}.pt"
        save_checkpoint(
            epoch_path,
            checkpoint_model,
            training_state=training_state,
            runtime_state=runtime_state,
            corpus_provenance=corpus_provenance,
        )
        last_path = save_checkpoint(
            output / "last.pt",
            checkpoint_model,
            training_state=training_state,
            runtime_state=runtime_state,
            corpus_provenance=corpus_provenance,
            overwrite=True,
        )
        best_path = output / "best.pt"
        if is_best:
            save_checkpoint(
                best_path,
                checkpoint_model,
                training_state=training_state,
                runtime_state=runtime_state,
                corpus_provenance=corpus_provenance,
                overwrite=True,
            )
        write_json(
            output / "training_progress.json",
            {
                "schema": "evidence_training_progress_v1",
                "status": "running",
                "epoch": epoch,
                "target_epochs": learning_config.epochs,
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

    parent_limit = 3 if args.smoke else None
    training_batches = (
        (lambda: iter_evidence_batches(corpus, "training", parent_limit=parent_limit))
        if augmentation_profile is None
        else (
            lambda: iter_paired_evidence_batches(
                corpus,
                "training",
                augmentation_profile=augmentation_profile,
                parent_limit=parent_limit,
            )
        )
    )
    result = train_evidence_model(
        model,
        training_batches,
        lambda: iter_evidence_batches(corpus, "tuning", parent_limit=parent_limit),
        config=learning_config,
        logger=logger,
        resume_state=(resume_metadata.get("runtime_state") if resume_path else None),
        checkpoint_callback=publish_checkpoint,
    )
    best_path = output / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("training completed without a best checkpoint.")
    best_model, best_metadata = load_checkpoint(best_path, device=device)
    final_path = save_checkpoint(
        output / "model.pt",
        best_model,
        training_state=best_metadata["training_state"],
        runtime_state=best_metadata.get("runtime_state"),
        corpus_provenance=best_metadata["corpus_provenance"],
        overwrite=True,
    )
    write_json(
        output / "training_progress.json",
        {
            "schema": "evidence_training_progress_v1",
            "status": "completed",
            "epoch": int(result["epochs_completed"]),
            "target_epochs": learning_config.epochs,
            "best_epoch": int(result["best_epoch"]),
            "best_tuning_loss": float(result["best_tuning_loss"]),
            "last_checkpoint": str(output / "last.pt"),
            "best_checkpoint": str(best_path),
            "final_checkpoint": str(final_path),
            "history": list(result["history"]),
        },
    )
    write_json(
        output / "run_summary.json",
        {
            "schema": "evidence_training_v1",
            "status": "success",
            "mode": "smoke" if args.smoke else "formal",
            "runtime": dict(runtime),
            "checkpoint": str(final_path),
            "network_config": asdict(network_config),
            "target_contract": target_contract.to_mapping(),
            "learning_config": asdict(learning_config),
            **result,
        },
    )


def _evaluate(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    device,
    runtime: Mapping[str, Any],
) -> None:
    corpus = load_corpus(resolve_relative_path(args.corpus, root=REPO_ROOT))
    output, _ = _prepare_output(args)
    configure_training_logger(output)
    model, metadata = load_checkpoint(
        resolve_relative_path(args.checkpoint, root=REPO_ROOT),
        device=device,
    )
    if model.network_config.input_mode != "full":
        raise ValueError("primary evaluation checkpoint must use full input.")
    controls: dict[str, EvidenceModel] = {}
    checkpoint_metadata: dict[str, object] = {
        "full": public_checkpoint_metadata(metadata)
    }
    if args.no_seismic_checkpoint is not None:
        control, control_metadata = load_checkpoint(
            resolve_relative_path(args.no_seismic_checkpoint, root=REPO_ROOT),
            device=device,
        )
        controls["no_seismic"] = control
        checkpoint_metadata["no_seismic"] = public_checkpoint_metadata(control_metadata)
    clean = evaluate_evidence_model(
        model,
        iter_evidence_batches(
            corpus,
            args.split,
            parent_limit=args.parent_limit,
        ),
        controls=controls,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    augmentation_profile = _augmentation_profile(config)
    dirty = (
        None
        if augmentation_profile is None
        else evaluate_evidence_model(
            model,
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
            "schema": "evidence_evaluation_v1",
            "status": "success",
            "split": args.split,
            "runtime": dict(runtime),
            "checkpoint_metadata": checkpoint_metadata,
            "conditions": {"clean": clean, "dirty": dirty},
        },
    )


def main() -> None:
    args = parse_args()
    config = _config(args.config)
    device, runtime = resolve_device(str(config.get("device") or "auto"))
    if args.command == "train":
        _train(args, config, device=device, runtime=runtime)
    elif args.command == "evaluate":
        _evaluate(args, config, device=device, runtime=runtime)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
