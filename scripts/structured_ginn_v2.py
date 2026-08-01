"""Train or evaluate the restarted Structured GINN V2 generator."""

from __future__ import annotations

import argparse
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
    load_checkpoint,
    load_corpus,
    public_checkpoint_metadata,
    save_checkpoint,
)
from ginn_v2.augmentation import load_observation_augmentation_profile
from ginn_v2.evidence import (
    BandlimitedEvidenceNetwork,
    EvidenceNetworkConfig,
    dominant_frequency_hz,
)
from ginn_v2.generator import ConditionalGenerator
from ginn_v2.learning import LearningConfig, evaluate_generator, train_generator
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
    return parser.parse_args()


def _config(path: Path) -> dict:
    payload = load_yaml_config(resolve_relative_path(path, root=REPO_ROOT))
    if not isinstance(payload, dict) or set(payload) != {"structured_ginn_v2"}:
        raise ValueError("config must contain only structured_ginn_v2.")
    root = payload["structured_ginn_v2"]
    if not isinstance(root, dict):
        raise ValueError("structured_ginn_v2 must be a mapping.")
    allowed = {"device", "network", "learning", "augmentation_profile"}
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
    if args.command == "train":
        resume_metadata: Mapping[str, Any] = {}
        if resume_path is None:
            network = BandlimitedEvidenceNetwork(
                EvidenceNetworkConfig.from_mapping(config.get("network") or {})
            )
            generator = ConditionalGenerator(
                network,
                prior=calibrate_semi_markov_prior(corpus),
                dominant_frequency_hz=_dominant_frequency(corpus),
                sample_domain=corpus.benchmark.sample_domain,
                device=device,
            )
        else:
            generator, resume_metadata = load_checkpoint(
                resume_path,
                device=device,
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

        learning_config = LearningConfig(**dict(config.get("learning") or {}))
        result = train_generator(
            generator,
            (
                (lambda: iter_evidence_batches(corpus, "training"))
                if augmentation_profile is None
                else (
                    lambda: iter_paired_evidence_batches(
                        corpus,
                        "training",
                        augmentation_profile=augmentation_profile,
                    )
                )
            ),
            (
                lambda: iter_evidence_batches(corpus, "tuning")
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
                **result,
            },
        )
        return
    generator, metadata = load_checkpoint(
        resolve_relative_path(args.checkpoint, root=REPO_ROOT),
        device=device,
    )
    clean_metrics = evaluate_generator(
        generator,
        iter_evidence_batches(corpus, args.split),
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
            ),
        )
    )
    write_json(
        output / "evaluation.json",
        {
            "schema": "structured_ginn_v2_evaluation_v1",
            "status": "success",
            "split": args.split,
            "runtime": runtime,
            "checkpoint_metadata": public_checkpoint_metadata(metadata),
            "metrics": {
                "clean": clean_metrics,
                "dirty": dirty_metrics,
            },
        },
    )


if __name__ == "__main__":
    main()
