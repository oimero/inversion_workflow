"""Training and prediction helpers for GINN-v2 ablations."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from cup.physics.torch_backend import forward_depth, forward_time, velocity_from_ai
from cup.synthetic.dataset import SynthoseisBenchmark
from cup.synthetic.metrics import regression_metrics
from cup.utils.io import write_json
from cup.utils.logging import configure_run_logger
from ginn_v2.contracts import CHECKPOINT_SCHEMA_VERSION, PATCH_SMOKE_REPORT_SCHEMA_VERSION
from ginn_v2.data import PatchDataset, _aligned_arrays, default_train_kinds, denormalize_delta
from ginn_v2.models import build_model


PROBE_SAMPLE_KINDS = {"frequency_probe", "frequency_probe_seismic_variant"}
PHYSICS_LOSS_MODEL_IDS = {
    "patch_2d_with_physics_loss",
    "trace_1d_dilated_tcn_mismatch_training",
}


def configure_training_logger(output_dir: Path) -> logging.Logger:
    """Create the per-run terminal and file logger."""
    return configure_run_logger(
        output_dir,
        logger_name="ginn_v2.training",
        file_name="training.log",
    )


def resolve_device(device_name: str) -> tuple[torch.device, dict[str, Any]]:
    """Resolve a requested torch device and return auditable metadata.

    ``auto`` keeps the historical behavior: use CUDA when available, otherwise
    CPU.  Explicit values such as ``cuda`` are not softened into a fallback; if
    the runtime cannot satisfy them, PyTorch will fail before the run produces
    outputs.
    """

    requested = str(device_name or "auto")
    cuda_available = bool(torch.cuda.is_available())
    resolved = requested if requested != "auto" else ("cuda" if cuda_available else "cpu")
    if requested.startswith("cuda") and not cuda_available:
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is false.")
    device = torch.device(resolved)
    device_name_text = ""
    if device.type == "cuda" and cuda_available:
        device_name_text = torch.cuda.get_device_name(device)
    metadata = {
        "requested_device": requested,
        "resolved_device": str(device),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_device_name": device_name_text,
        "torch_version": str(torch.__version__),
    }
    return device, metadata


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if torch.count_nonzero(valid) == 0:
        raise ValueError("Batch has no valid samples.")
    residual = prediction[valid] - target[valid]
    return torch.mean(residual * residual)


def train_model(
    *,
    benchmark: SynthoseisBenchmark,
    patch_index: pd.DataFrame,
    normalization: Mapping[str, Any],
    output_dir: Path,
    model_id: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_channels: int,
    depth: int,
    device_name: str,
    lambda_physics: float,
    lambda_real_delta: float,
    seed: int,
    real_delta_support: Any | None = None,
    log_interval_batches: int = 10,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    sample_axis_contract = _benchmark_sample_axis_contract(benchmark.manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logger or configure_training_logger(output_dir)
    if not np.isfinite(lambda_real_delta) or float(lambda_real_delta) < 0.0:
        raise ValueError("lambda_real_delta must be finite and non-negative.")
    if int(log_interval_batches) <= 0:
        raise ValueError("log_interval_batches must be positive.")
    if float(lambda_real_delta) > 0.0 and real_delta_support is None:
        raise ValueError("Non-zero lambda_real_delta requires train.real_delta configuration.")
    if lambda_physics != 0.0:
        if model_id not in PHYSICS_LOSS_MODEL_IDS:
            raise NotImplementedError(
                "Non-zero physics loss is currently only implemented for: "
                f"{sorted(PHYSICS_LOSS_MODEL_IDS)}."
            )
    allowed_kinds = default_train_kinds(model_id)
    unexpected = sorted(set(patch_index["sample_kind"].astype(str)) - allowed_kinds)
    if unexpected:
        raise ValueError(
            f"Patch index contains sample_kind values not allowed for {model_id}: {unexpected}. "
            f"Allowed: {sorted(allowed_kinds)}"
        )
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    device, device_metadata = resolve_device(device_name)
    logger.info(
        "training setup: model=%s device=%s seed=%d epochs=%d batch_size=%d",
        model_id,
        device,
        int(seed),
        int(epochs),
        int(batch_size),
    )
    train_ds = PatchDataset(
        benchmark,
        patch_index,
        split="train",
        normalization=normalization,
    )
    val_ds = PatchDataset(
        benchmark,
        patch_index,
        split="validation",
        normalization=normalization,
    )
    train_sample_kinds = train_ds.frame["sample_kind"].astype(str)
    kind_counts = train_sample_kinds.value_counts().to_dict()
    sample_weights = [1.0 / float(kind_counts[kind]) for kind in train_sample_kinds]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model, info = build_model(model_id, hidden_channels=hidden_channels, depth=depth)
    model.to(device)
    logger.info(
        "model ready: parameters=%d train_patches=%d validation_patches=%d",
        info.parameter_count,
        len(train_ds),
        len(val_ds),
    )
    reconstruction_error = None
    if real_delta_support is not None:
        real_delta_support.configure_model(
            receptive_field_lateral=int(info.receptive_field_lateral)
        )
        logger.info(
            "real-delta: validating %s real-well support",
            (
                "canonical full-patch"
                if real_delta_support.canonical_full_patch
                else "sparse-support/full-patch equivalence"
            ),
        )
        reconstruction_error = real_delta_support.validate_reconstruction(
            model,
            device=device,
        )
        logger.info(
            "real-delta: reconstruction max_abs_log_ai=%.9g",
            reconstruction_error,
        )
    wavelet = (
        _load_nominal_wavelet(benchmark.run_dir, device=device)
        if float(lambda_physics) > 0.0
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch: int | None = None
    best_path = output_dir / "checkpoint_best.pt"
    final_path = output_dir / "checkpoint_final.pt"
    history_path = output_dir / "training_history.csv"
    existing_outputs = [
        path for path in (best_path, final_path, history_path) if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            f"GINN-v2 training outputs already exist: {existing_outputs}"
        )
    sequence = hashlib.sha256()
    total_batches = len(train_loader)
    if total_batches <= 0:
        raise ValueError("GINN-v2 training split has no batches.")
    total_training_steps = int(epochs) * total_batches
    completed_steps = 0
    ema_batch_elapsed: float | None = None
    run_started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        logger.info("epoch %d/%d: training started", epoch, int(epochs))
        model.train()
        train_rows: list[dict[str, float | int]] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            batch_started = time.perf_counter()
            for patch_id in batch["patch_id"]:
                sequence.update(str(patch_id).encode("utf-8"))
                sequence.update(b"\0")
            x = batch["input"].to(device)
            target = batch["target_delta"].to(device)
            mask = batch["valid_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss_ai = masked_mse(pred, target, mask)
            loss = loss_ai
            loss_physics = torch.zeros((), dtype=loss.dtype, device=device)
            if wavelet is not None:
                sample_kinds = list(batch["sample_kind"])
                physics_rows = [
                    idx for idx, kind in enumerate(sample_kinds)
                    if str(kind) == "base"
                ]
                if physics_rows:
                    physics_index = torch.as_tensor(
                        physics_rows,
                        dtype=torch.long,
                        device=device,
                    )
                    lfm = batch["lfm"].to(device).index_select(0, physics_index)
                    seismic = batch["seismic_model_consistent"].to(device).index_select(0, physics_index)
                    physics_mask = mask.index_select(0, physics_index)
                    physics_pred = pred.index_select(0, physics_index)
                    pred_log_ai = lfm + _denormalize_delta_torch(
                        physics_pred[:, 0],
                        normalization,
                    )[:, None]
                    wavelet_time_s, wavelet_amp, relation = wavelet
                    sample_axes = batch["sample_axis"].to(device).index_select(0, physics_index)
                    synthetic = _forward_physics_batch(
                        pred_log_ai[:, 0],
                        sample_axes=sample_axes,
                        sample_domain=str(sample_axis_contract["sample_domain"]),
                        wavelet_time_s=wavelet_time_s,
                        wavelet_amp=wavelet_amp,
                        ai_velocity_relation=relation,
                    )
                    seismic_target = seismic[:, 0]
                    seismic_mask = batch["physics_valid_mask"].to(device).index_select(0, physics_index)[:, 0]
                    loss_physics = masked_mse(
                        synthetic,
                        seismic_target,
                        seismic_mask,
                    )
                    loss = loss + float(lambda_physics) * loss_physics
            loss_real_delta = torch.zeros((), dtype=loss.dtype, device=device)
            real_counts = {
                "selected_real_clusters": 0,
                "selected_real_wells": 0,
                "selected_real_samples": 0,
            }
            if float(lambda_real_delta) > 0.0:
                loss_real_delta, real_counts = real_delta_support.training_loss(
                    model,
                    device=device,
                )
                loss = loss + float(lambda_real_delta) * loss_real_delta
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite GINN-v2 loss at epoch={epoch}, batch={batch_index}."
                )
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_elapsed = time.perf_counter() - batch_started
            ema_batch_elapsed = (
                batch_elapsed
                if ema_batch_elapsed is None
                else 0.9 * ema_batch_elapsed + 0.1 * batch_elapsed
            )
            completed_steps += 1
            values: dict[str, float | int] = {
                "synthetic_loss": float(loss_ai.detach().cpu()),
                "physics_loss": float(loss_physics.detach().cpu()),
                "weighted_physics_loss": float(
                    float(lambda_physics) * loss_physics.detach().cpu()
                ),
                "real_delta_loss": float(loss_real_delta.detach().cpu()),
                "weighted_real_delta_loss": float(
                    float(lambda_real_delta) * loss_real_delta.detach().cpu()
                ),
                "total_loss": float(loss.detach().cpu()),
                **real_counts,
            }
            train_rows.append(values)
            should_log = (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % int(log_interval_batches) == 0
            )
            if should_log:
                epoch_remaining = total_batches - batch_index
                run_remaining = total_training_steps - completed_steps
                logger.info(
                    "epoch=%d/%d batch=%d/%d synthetic=%.6g physics=%.6g "
                    "weighted_physics=%.6g real_delta=%.6g "
                    "weighted_real_delta=%.6g total=%.6g batch_s=%.2f "
                    "ema_batch_s=%.2f epoch_eta_s=%.1f run_eta_s=%.1f "
                    "real_clusters=%d real_wells=%d",
                    epoch,
                    int(epochs),
                    batch_index,
                    total_batches,
                    values["synthetic_loss"],
                    values["physics_loss"],
                    values["weighted_physics_loss"],
                    values["real_delta_loss"],
                    values["weighted_real_delta_loss"],
                    values["total_loss"],
                    batch_elapsed,
                    ema_batch_elapsed,
                    epoch_remaining * ema_batch_elapsed,
                    run_remaining * ema_batch_elapsed,
                    values["selected_real_clusters"],
                    values["selected_real_wells"],
                )
        logger.info("epoch %d/%d: synthetic validation started", epoch, int(epochs))
        val_loss = evaluate_loss(model, val_loader, device)
        epoch_elapsed = time.perf_counter() - epoch_started
        frame = pd.DataFrame.from_records(train_rows)
        row = {
            "epoch": epoch,
            "synthetic_loss": float(frame["synthetic_loss"].mean()),
            "physics_loss": float(frame["physics_loss"].mean()),
            "weighted_physics_loss": float(frame["weighted_physics_loss"].mean()),
            "real_delta_loss": float(frame["real_delta_loss"].mean()),
            "weighted_real_delta_loss": float(
                frame["weighted_real_delta_loss"].mean()
            ),
            "train_loss": float(frame["total_loss"].mean()),
            "validation_loss": val_loss,
            "selected_real_clusters": int(frame["selected_real_clusters"].sum()),
            "selected_real_wells": int(frame["selected_real_wells"].sum()),
            "selected_real_samples": int(frame["selected_real_samples"].sum()),
            "epoch_elapsed_s": float(epoch_elapsed),
            "is_best_checkpoint": False,
            "is_final_checkpoint": epoch == int(epochs),
        }
        history.append(row)
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            row["is_best_checkpoint"] = True
            torch.save(
                _checkpoint_payload(
                    model=model,
                    model_id=model_id,
                    normalization=normalization,
                    model_info=info.__dict__,
                    hidden_channels=hidden_channels,
                    depth=depth,
                    epoch=epoch,
                    validation_loss=val_loss,
                    checkpoint_kind="best",
                    sample_axis_contract=sample_axis_contract,
                ),
                best_path,
            )
            logger.info(
                "checkpoint best updated: epoch=%d validation_loss=%.9g",
                epoch,
                val_loss,
            )
        pd.DataFrame.from_records([row]).to_csv(
            history_path,
            mode="a",
            header=not history_path.exists(),
            index=False,
        )
        logger.info(
            "epoch %d/%d complete: train_loss=%.6g validation_loss=%.6g elapsed_s=%.1f",
            epoch,
            int(epochs),
            row["train_loss"],
            val_loss,
            epoch_elapsed,
        )
    if best_epoch is None or not best_path.is_file():
        raise FloatingPointError("No finite synthetic validation loss; best checkpoint unavailable.")
    final_validation_loss = float(history[-1]["validation_loss"])
    torch.save(
        _checkpoint_payload(
            model=model,
            model_id=model_id,
            normalization=normalization,
            model_info=info.__dict__,
            hidden_channels=hidden_channels,
            depth=depth,
            epoch=int(epochs),
            validation_loss=final_validation_loss,
            checkpoint_kind="final",
            sample_axis_contract=sample_axis_contract,
        ),
        final_path,
    )
    logger.info(
        "checkpoint final written: epoch=%d validation_loss=%.9g",
        int(epochs),
        final_validation_loss,
    )
    for row in history:
        row["is_best_checkpoint"] = int(row["epoch"]) == int(best_epoch)
    pd.DataFrame.from_records(history).to_csv(history_path, index=False)
    logger.info(
        "training complete: best_epoch=%d best_validation_loss=%.9g final_epoch=%d elapsed_s=%.1f",
        best_epoch,
        best_val,
        int(epochs),
        time.perf_counter() - run_started,
    )
    return {
        "status": "ok",
        "device": str(device),
        "device_metadata": device_metadata,
        "checkpoints": {
            "primary": "best",
            "best": {
                "path": best_path,
                "epoch": int(best_epoch),
                "validation_loss": float(best_val),
            },
            "final": {
                "path": final_path,
                "epoch": int(epochs),
                "validation_loss": final_validation_loss,
            },
        },
        "history": history_path,
        "best_validation_loss": best_val,
        "model_info": info.__dict__,
        "synthetic_sequence_sha256": sequence.hexdigest(),
        "real_field_reconstruction_max_abs_log_ai": reconstruction_error,
    }


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    model_id: str,
    normalization: Mapping[str, Any],
    model_info: Mapping[str, Any],
    hidden_channels: int,
    depth: int,
    epoch: int,
    validation_loss: float,
    checkpoint_kind: str,
    sample_axis_contract: Mapping[str, Any],
) -> dict[str, Any]:
    state_dict = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "epoch": int(epoch),
        "validation_loss": float(validation_loss),
        "model_id": model_id,
        "state_dict": state_dict,
        "normalization": dict(normalization),
        "model_info": dict(model_info),
        "architecture": {
            "hidden_channels": int(hidden_channels),
            "depth": int(depth),
        },
        "sample_axis_contract": dict(sample_axis_contract),
    }


def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            target = batch["target_delta"].to(device)
            mask = batch["valid_mask"].to(device)
            losses.append(float(masked_mse(model(x), target, mask).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _benchmark_sample_axis_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    domain = str(manifest.get("sample_domain") or "")
    if domain == "time":
        unit, basis = "s", None
    elif domain == "depth" and manifest.get("depth_basis") == "tvdss":
        unit, basis = "m", "tvdss"
    else:
        raise ValueError(
            "Benchmark must declare sample_domain=time or depth with depth_basis=tvdss."
        )
    return {
        "sample_domain": domain,
        "sample_unit": unit,
        "depth_basis": basis,
    }


def _resolve_manifest_path(value: object, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Benchmark manifest lacks {label}.")
    path = Path(text)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _load_nominal_wavelet(
    benchmark_dir: Path,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
    manifest_path = Path(benchmark_dir) / "benchmark_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"benchmark_manifest.json not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    domain = str(manifest.get("sample_domain") or "")
    if domain == "depth":
        forward_inputs_path = _resolve_manifest_path(
            manifest.get("forward_model_inputs_path"),
            label="forward_model_inputs_path",
        )
        with forward_inputs_path.open("r", encoding="utf-8") as handle:
            forward_inputs = json.load(handle)
        if (
            forward_inputs.get("sample_domain") != "depth"
            or forward_inputs.get("depth_basis") != "tvdss"
        ):
            raise ValueError("Depth forward_model_inputs must declare depth/TVDSS.")
        raw_relation = dict(forward_inputs.get("ai_velocity_relation") or {})
        if raw_relation.get("ai_unit") != "m/s*g/cm3" or raw_relation.get("vp_unit") != "m/s":
            raise ValueError("Depth AI–Vp relation units are invalid.")
        relation: dict[str, float] | None = {
            "a": float(raw_relation["a"]),
            "b": float(raw_relation["b"]),
        }
        csv_path = _resolve_manifest_path(
            dict(forward_inputs.get("wavelet") or {}).get("path"),
            label="forward_model_inputs.wavelet.path",
        )
    elif domain == "time":
        relation = None
        source_runs = manifest.get("source_runs") or {}
        wavelet_dir = source_runs.get("wavelet_generation_dir")
        if not wavelet_dir:
            raise ValueError("Time benchmark lacks source_runs.wavelet_generation_dir.")
        wavelet_path = Path(wavelet_dir)
        if not wavelet_path.is_absolute():
            wavelet_path = Path.cwd() / wavelet_path
        csv_path = wavelet_path / "selected_wavelet.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"selected_wavelet.csv not found: {csv_path}")
    else:
        raise ValueError(f"Unsupported benchmark sample_domain: {domain!r}")
    frame = pd.read_csv(csv_path)
    missing = sorted({"time_s", "amplitude"} - set(frame.columns))
    if missing:
        raise ValueError(f"Wavelet CSV lacks columns {missing}: {csv_path}")
    time_s = frame["time_s"].to_numpy(dtype=np.float64)
    wavelet = frame["amplitude"].to_numpy(dtype=np.float32)
    if wavelet.size < 3 or wavelet.size % 2 == 0:
        raise ValueError("nominal wavelet must have odd length >= 3")
    if not np.all(np.isfinite(wavelet)):
        raise ValueError("nominal wavelet contains non-finite values")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("nominal wavelet time axis contains non-finite values")
    return (
        torch.as_tensor(time_s, dtype=torch.float32, device=device),
        torch.as_tensor(wavelet, dtype=torch.float32, device=device),
        relation,
    )


def _denormalize_delta_torch(values: torch.Tensor, normalization: Mapping[str, Any]) -> torch.Tensor:
    if "delta" in normalization:
        raise ValueError("Normalized-delta checkpoints are obsolete in GINN-v2 checkpoint v4.")
    return values


def _forward_physics_batch(
    log_ai: torch.Tensor,
    *,
    sample_axes: torch.Tensor,
    sample_domain: str,
    wavelet_time_s: torch.Tensor,
    wavelet_amp: torch.Tensor,
    ai_velocity_relation: Mapping[str, float] | None,
) -> torch.Tensor:
    if log_ai.ndim != 3:
        raise ValueError("Physics forward expects [batch, lateral, sample] logAI.")
    if sample_domain == "time":
        return forward_time(log_ai, wavelet_time_s, wavelet_amp)
    if sample_domain != "depth":
        raise ValueError(f"Unsupported physics sample domain: {sample_domain!r}")
    if sample_axes.shape != (log_ai.shape[0], log_ai.shape[-1]):
        raise ValueError("Depth physics axes must have shape [batch, sample].")
    if ai_velocity_relation is None:
        raise ValueError("Depth physics requires the frozen AI–Vp relation.")
    velocity_mps = velocity_from_ai(
        torch.exp(log_ai),
        a=float(ai_velocity_relation["a"]),
        b=float(ai_velocity_relation["b"]),
    )
    return torch.stack(
        [
            forward_depth(
                log_ai[row_index],
                velocity_mps[row_index],
                sample_axes[row_index],
                wavelet_time_s,
                wavelet_amp,
            )
            for row_index in range(log_ai.shape[0])
        ],
        dim=0,
    )


def load_checkpoint(path: Path, *, hidden_channels: int | None = None, depth: int | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("schema_version") or "") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported GINN-v2 checkpoint schema: {path}")
    architecture = dict(checkpoint.get("architecture") or {})
    architecture_id = str(architecture.get("id") or "")
    if not architecture_id:
        raise ValueError(
            "GINN-v2 checkpoint lacks the v4 architecture contract; legacy checkpoints are not supported."
        )
    model, _ = build_model(
        architecture_id,
        hidden_channels=int(hidden_channels or architecture.get("hidden_channels", 32)),
        depth=int(depth or architecture.get("depth", 5)),
        lateral_kernel=architecture.get("lateral_kernel"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def predict_patches(
    *,
    benchmark: SynthoseisBenchmark,
    patch_index: pd.DataFrame,
    checkpoint_path: Path,
    output_dir: Path,
    split: str | None,
    batch_size: int,
    device_name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint(checkpoint_path)
    device, device_metadata = resolve_device(device_name)
    model.to(device)
    model.eval()
    selected = patch_index.copy()
    if split and split != "all":
        selected = selected[selected["split"].astype(str).eq(split)].copy()
    if selected.empty:
        raise ValueError(f"No patches selected for prediction split={split}.")
    dataset = PatchDataset(
        benchmark,
        selected,
        split=sorted(set(selected["split"].astype(str))),
        normalization=checkpoint["normalization"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions = []
    targets = []
    masks = []
    lfm_values = []
    lfm_ideal_values = []
    patch_ids = []
    sample_ids = []
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            pred_delta = model(x).detach().cpu().numpy()[:, 0]
            lfm = batch["lfm"].numpy()[:, 0]
            prediction = lfm + pred_delta
            predictions.append(prediction.astype(np.float32))
            targets.append(batch["target_log_ai"].numpy()[:, 0].astype(np.float32))
            masks.append(batch["valid_mask"].numpy()[:, 0].astype(bool))
            lfm_values.append(lfm.astype(np.float32))
            lfm_ideal_values.append(batch["lfm_ideal"].numpy()[:, 0].astype(np.float32))
            patch_ids.extend([str(value) for value in batch["patch_id"]])
            sample_ids.extend([str(value) for value in batch["sample_id"]])
    pred_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    mask_array = np.concatenate(masks, axis=0)
    lfm_array = np.concatenate(lfm_values, axis=0)
    lfm_ideal_array = np.concatenate(lfm_ideal_values, axis=0)
    npz_path = output_dir / "predictions.npz"
    np.savez_compressed(
        npz_path,
        pred_log_ai=pred_array,
        target_log_ai=target_array,
        valid_mask_model=mask_array,
        lfm_controlled_degraded=lfm_array,
        lfm_ideal=lfm_ideal_array,
        patch_id=np.asarray(patch_ids),
        sample_id=np.asarray(sample_ids),
    )
    selected = selected.set_index("patch_id").loc[patch_ids].reset_index()
    selected["prediction_row"] = np.arange(len(selected), dtype=int)
    selected["architecture_id"] = str(checkpoint["architecture"]["id"])
    index_path = output_dir / "prediction_index.csv"
    selected.to_csv(index_path, index=False)
    return {
        "status": "ok",
        "prediction_npz": npz_path,
        "prediction_index": index_path,
        "n_predictions": int(pred_array.shape[0]),
        "architecture_id": str(checkpoint["architecture"]["id"]),
        "model_info": dict(checkpoint["model_info"]),
        "normalization": dict(checkpoint["normalization"]),
        "device_metadata": device_metadata,
    }


def report_predictions(*, prediction_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(prediction_dir / "prediction_index.csv")
    arrays = np.load(prediction_dir / "predictions.npz", allow_pickle=True)
    pred = arrays["pred_log_ai"]
    target = arrays["target_log_ai"]
    mask = arrays["valid_mask_model"].astype(bool)
    lfm = arrays["lfm_controlled_degraded"]
    lfm_ideal = (
        arrays["lfm_ideal"]
        if "lfm_ideal" in arrays
        else _load_lfm_ideal_patches(prediction_dir, index, target.shape)
    )
    rows = []
    lfm_rows = []
    lfm_ideal_rows = []
    oracle_rows = []
    for _, row in index.iterrows():
        i = int(row["prediction_row"])
        metrics = regression_metrics(target[i], pred[i], valid_mask=mask[i])
        rows.append({**row.to_dict(), "model_id": row.get("model_id", ""), **metrics})
        lfm_metrics = regression_metrics(target[i], lfm[i], valid_mask=mask[i])
        lfm_rows.append({**row.to_dict(), "model_id": "lfm_controlled_degraded", **lfm_metrics})
        lfm_ideal_metrics = regression_metrics(target[i], lfm_ideal[i], valid_mask=mask[i])
        lfm_ideal_rows.append({**row.to_dict(), "model_id": "lfm_ideal", **lfm_ideal_metrics})
        oracle_metrics = regression_metrics(target[i], target[i], valid_mask=mask[i])
        oracle_rows.append({**row.to_dict(), "model_id": "oracle_target", **oracle_metrics})
    metrics_frame = pd.DataFrame.from_records(rows)
    metrics_path = output_dir / "model_patch_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    geometry_path = output_dir / "model_patch_metrics_by_geometry.csv"
    _grouped_patch_metrics(metrics_frame, ["geometry_family"]).to_csv(geometry_path, index=False)
    mismatch_path = output_dir / "model_patch_metrics_by_mismatch_family.csv"
    _grouped_patch_metrics(metrics_frame, ["seismic_mismatch_family"]).to_csv(mismatch_path, index=False)
    geometry_detail_frame = _geometry_patch_metrics(metrics_frame, pred, target, mask, prediction_dir)
    geometry_detail_path = output_dir / "model_geometry_patch_metrics.csv"
    geometry_detail_frame.to_csv(geometry_detail_path, index=False)
    geometry_aggregate_path = output_dir / "model_geometry_metrics_by_family.csv"
    _grouped_geometry_metrics(geometry_detail_frame, ["geometry_family"]).to_csv(
        geometry_aggregate_path,
        index=False,
    )
    stitched = _stitch_realization_metrics(index, pred, target, mask, lfm)
    stitched_uniform_path = output_dir / "model_realization_metrics_uniform.csv"
    stitched["uniform"].to_csv(stitched_uniform_path, index=False)
    stitched_center_path = output_dir / "model_realization_metrics_center_crop.csv"
    stitched["center_crop"].to_csv(stitched_center_path, index=False)
    lfm_frame = pd.DataFrame.from_records(lfm_rows)
    lfm_path = output_dir / "lfm_patch_metrics.csv"
    lfm_frame.to_csv(lfm_path, index=False)
    lfm_ideal_frame = pd.DataFrame.from_records(lfm_ideal_rows)
    lfm_ideal_path = output_dir / "lfm_ideal_patch_metrics.csv"
    lfm_ideal_frame.to_csv(lfm_ideal_path, index=False)
    oracle_frame = pd.DataFrame.from_records(oracle_rows)
    oracle_path = output_dir / "oracle_patch_metrics.csv"
    oracle_frame.to_csv(oracle_path, index=False)
    probe_rows = _paired_probe_rows(metrics_frame, pred, target, mask, prediction_dir)
    probe_frame = pd.DataFrame.from_records(probe_rows)
    probe_path = output_dir / "model_probe_metrics.csv"
    probe_frame.to_csv(probe_path, index=False)
    probe_frequency_path = output_dir / "model_probe_metrics_by_frequency.csv"
    _grouped_probe_metrics(probe_frame, ["probe_frequency_hz"]).to_csv(probe_frequency_path, index=False)
    probe_frequency_amplitude_path = output_dir / "model_probe_metrics_by_frequency_amplitude.csv"
    _grouped_probe_metrics(
        probe_frame,
        ["probe_frequency_hz", "probe_amplitude_multiplier"],
    ).to_csv(probe_frequency_amplitude_path, index=False)
    zero_x_path = output_dir / "model_zero_x_false_prediction_error.csv"
    zero_x_frame = _zero_x_false_prediction_error(metrics_frame, prediction_dir)
    zero_x_frame.to_csv(zero_x_path, index=False)
    zero_x_energy_path = output_dir / "model_zero_x_false_frequency_energy.csv"
    zero_x_energy_frame = _zero_x_false_frequency_energy(metrics_frame, pred, target, mask, prediction_dir)
    zero_x_energy_frame.to_csv(zero_x_energy_path, index=False)
    zero_x_energy_by_frequency_path = output_dir / "model_zero_x_false_frequency_energy_by_frequency.csv"
    _grouped_false_energy(zero_x_energy_frame, ["probe_frequency_hz"]).to_csv(
        zero_x_energy_by_frequency_path,
        index=False,
    )
    model_aggregate = _aggregate(metrics_frame)
    lfm_aggregate = _aggregate(lfm_frame)
    lfm_ideal_aggregate = _aggregate(lfm_ideal_frame)
    oracle_aggregate = _aggregate(oracle_frame)
    report = {
        "schema_version": PATCH_SMOKE_REPORT_SCHEMA_VERSION,
        "report_scope": "patch_smoke",
        "not_synthoseis_lite_report": True,
        "status": "ok",
        "n_patches": int(len(metrics_frame)),
        "aggregate": model_aggregate,
        "lfm_aggregate": lfm_aggregate,
        "lfm_ideal_aggregate": lfm_ideal_aggregate,
        "oracle_aggregate": oracle_aggregate,
        "rmse_improvement_pct_vs_lfm": _rmse_improvement_pct(model_aggregate, lfm_aggregate),
        "probe_aggregate": _aggregate(probe_frame),
        "probe_amplitude_phase_aggregate": _aggregate_amplitude_phase(probe_frame),
        "geometry_aggregate": _aggregate_geometry(geometry_detail_frame),
        "realization_uniform_aggregate": _aggregate(
            stitched["uniform"][stitched["uniform"]["model_id"].astype(str).ne("lfm_controlled_degraded")]
        ),
        "realization_center_crop_aggregate": _aggregate(
            stitched["center_crop"][
                stitched["center_crop"]["model_id"].astype(str).ne("lfm_controlled_degraded")
            ]
        ),
        "zero_x_false_prediction_aggregate": _aggregate(zero_x_frame),
        "unsupported_zero_x_false_prediction_aggregate": _aggregate(_filter_unsupported_zero_x(zero_x_frame)),
        "zero_x_false_energy_aggregate": _aggregate_false_energy(zero_x_energy_frame),
        "unsupported_false_energy_aggregate": _aggregate_false_energy(
            _filter_unsupported_zero_x(zero_x_energy_frame)
        ),
    }
    report_path = output_dir / "model_report_card.json"
    write_json(report_path, report)
    return {
        "status": "ok",
        "model_patch_metrics": metrics_path,
        "model_patch_metrics_by_geometry": geometry_path,
        "model_patch_metrics_by_mismatch_family": mismatch_path,
        "model_geometry_patch_metrics": geometry_detail_path,
        "model_geometry_metrics_by_family": geometry_aggregate_path,
        "model_realization_metrics_uniform": stitched_uniform_path,
        "model_realization_metrics_center_crop": stitched_center_path,
        "lfm_patch_metrics": lfm_path,
        "lfm_ideal_patch_metrics": lfm_ideal_path,
        "oracle_patch_metrics": oracle_path,
        "model_probe_metrics": probe_path,
        "model_probe_metrics_by_frequency": probe_frequency_path,
        "model_probe_metrics_by_frequency_amplitude": probe_frequency_amplitude_path,
        "model_zero_x_false_prediction_error": zero_x_path,
        "model_zero_x_false_frequency_energy": zero_x_energy_path,
        "model_zero_x_false_frequency_energy_by_frequency": zero_x_energy_by_frequency_path,
        "model_report_card": report_path,
    }


def _paired_probe_rows(
    index: pd.DataFrame,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    prediction_dir: Path,
) -> list[dict[str, Any]]:
    by_patch = {str(row["patch_id"]): row for _, row in index.iterrows()}
    rows = []
    dt_s = _model_dt_s(prediction_dir)
    for _, row in index.iterrows():
        pair_id = _clean_report_text(row.get("paired_zero_patch_id"))
        if not pair_id or pair_id not in by_patch:
            continue
        amp = row.get("probe_amplitude_multiplier", "")
        try:
            if float(amp) == 0.0:
                continue
        except (TypeError, ValueError):
            pass
        i = int(row["prediction_row"])
        j = int(by_patch[pair_id]["prediction_row"])
        valid = mask[i] & mask[j]
        target_diff = target[i] - target[j]
        pred_diff = pred[i] - pred[j]
        metrics = regression_metrics(target_diff, pred_diff, valid_mask=valid)
        amp_phase = _amplitude_phase_metrics(
            target_diff=target_diff,
            pred_diff=pred_diff,
            valid=valid,
            frequency_hz=_safe_float(row.get("probe_frequency_hz")),
            dt_s=dt_s,
        )
        rows.append(
            {
                "patch_id": row["patch_id"],
                "paired_zero_patch_id": pair_id,
                "sample_id": row.get("sample_id", ""),
                "paired_zero_sample_id": row.get("paired_zero_sample_id", ""),
                "sample_kind": row.get("sample_kind", ""),
                "source_sample_id": row.get("source_sample_id", ""),
                "source_sample_kind": row.get("source_sample_kind", ""),
                "seismic_variant_id": row.get("seismic_variant_id", ""),
                "seismic_mismatch_family": row.get("seismic_mismatch_family", ""),
                "probe_frequency_hz": row.get("probe_frequency_hz", ""),
                "probe_phase": row.get("probe_phase", ""),
                "probe_lateral_shape": row.get("probe_lateral_shape", ""),
                "probe_amplitude_multiplier": amp,
                "probe_metric_semantics": (
                    "paired_probe_increment_error_under_seismic_mismatch"
                    if str(row.get("sample_kind", "")) == "frequency_probe_seismic_variant"
                    else "paired_probe_increment_error"
                ),
                **metrics,
                **amp_phase,
            }
        )
    return rows


def _clean_report_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "status" not in frame:
        return {"n_ok": 0}
    ok = frame[frame["status"].eq("ok")]
    return {
        "n_ok": int(len(ok)),
        "mean_rmse": float(ok["rmse"].mean()) if not ok.empty else float("nan"),
        "mean_nrmse": float(ok["nrmse"].mean()) if not ok.empty else float("nan"),
        "median_corr": float(ok["corr"].median()) if not ok.empty else float("nan"),
    }


def _grouped_patch_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty or not set(keys).issubset(frame.columns):
        return pd.DataFrame(columns=[*keys, "n_ok", "mean_rmse", "mean_nrmse", "median_corr"])
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(_aggregate(group))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _load_lfm_ideal_patches(
    prediction_dir: Path,
    index: pd.DataFrame,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    benchmark_dir = _benchmark_dir_from_prediction(prediction_dir)
    if benchmark_dir is None:
        raise ValueError("Cannot load lfm_ideal: prediction manifest lacks benchmark_dir.")
    benchmark = SynthoseisBenchmark(benchmark_dir)
    patches: list[np.ndarray] = []
    for _, row in index.iterrows():
        sample = benchmark.load_sample(str(row["sample_id"]))
        target, _, _, _ = _aligned_arrays(sample)
        lfm_ideal = np.asarray(sample.priors["lfm_ideal"], dtype=np.float32)
        if lfm_ideal.shape != target.shape:
            raise ValueError(
                f"lfm_ideal/target shape mismatch for {sample.sample_id}: "
                f"{lfm_ideal.shape} vs {target.shape}"
            )
        patches.append(lfm_ideal[_patch_slice(row, offset=0)].astype(np.float32))
    result = np.stack(patches, axis=0)
    if result.shape != expected_shape:
        raise ValueError(f"lfm_ideal patch shape mismatch: {result.shape} vs {expected_shape}")
    return result


def _geometry_patch_metrics(
    index: pd.DataFrame,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    prediction_dir: Path,
) -> pd.DataFrame:
    """Compute geometry-focused patch metrics from frozen benchmark masks."""

    benchmark_dir = _benchmark_dir_from_prediction(prediction_dir)
    if benchmark_dir is None:
        return pd.DataFrame()
    h5_path = benchmark_dir / "synthetic_benchmark.h5"
    if not h5_path.is_file():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    geometry_cache: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as h5:
        for _, row in index.iterrows():
            prediction_row = int(row["prediction_row"])
            root = _root_from_group_path(str(row.get("hdf5_group", "")))
            if not root:
                continue
            geometry = geometry_cache.get(root)
            if geometry is None:
                geometry = _read_geometry_masks(h5, root)
                geometry_cache[root] = geometry
            sl = _patch_slice(row, offset=0)
            boundary = geometry["boundary_mask_model"][sl]
            event = geometry["geometry_event_mask_model"][sl]
            valid = (
                mask[prediction_row].astype(bool)
                & np.isfinite(pred[prediction_row])
                & np.isfinite(target[prediction_row])
            )
            residual = pred[prediction_row] - target[prediction_row]
            boundary_band = _dilate_mask(boundary, lateral_radius=1, twt_radius=2) & valid
            non_boundary = valid & ~boundary_band
            event_valid = event & valid
            non_event = valid & ~event
            lateral = _lateral_gradient_metrics(
                prediction=pred[prediction_row],
                target=target[prediction_row],
                valid=valid,
            )
            rows.append(
                {
                    "patch_id": row.get("patch_id", ""),
                    "sample_id": row.get("sample_id", ""),
                    "sample_kind": row.get("sample_kind", ""),
                    "split": row.get("split", ""),
                    "geometry_family": row.get("geometry_family", ""),
                    "scenario_id": row.get("scenario_id", ""),
                    "duration_mode": row.get("duration_mode", ""),
                    "geometry_metric_semantics": (
                        "boundary uses dilated truth/boundary_mask_model; "
                        "event uses model-grid projection of truth/geometry_event_mask_highres"
                    ),
                    **_region_metrics("all", residual, valid),
                    **_region_metrics("boundary", residual, boundary_band),
                    **_region_metrics("non_boundary", residual, non_boundary),
                    **_region_metrics("event", residual, event_valid),
                    **_region_metrics("non_event", residual, non_event),
                    **lateral,
                }
            )
    return pd.DataFrame.from_records(rows)


def _benchmark_dir_from_prediction(prediction_dir: Path) -> Path | None:
    import json

    manifest_path = prediction_dir / "prediction_manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    benchmark_dir = manifest.get("benchmark_dir")
    if not benchmark_dir:
        return None
    path = Path(benchmark_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _root_from_group_path(group_path: str) -> str:
    group_path = str(group_path).strip()
    if not group_path:
        return ""
    for marker in ("/probes/", "/seismic_variants/"):
        if marker in group_path:
            return group_path.split(marker, maxsplit=1)[0]
    return group_path


def _read_geometry_masks(h5: h5py.File, root: str) -> dict[str, np.ndarray]:
    truth = h5[f"{root}/truth"]
    boundary = np.asarray(truth["boundary_mask_model"][()], dtype=bool)
    event_highres = np.asarray(truth["geometry_event_mask_highres"][()], dtype=bool)
    event_model = _highres_mask_to_model(event_highres, boundary.shape)
    return {
        "boundary_mask_model": boundary,
        "geometry_event_mask_model": event_model,
    }


def _highres_mask_to_model(mask_highres: np.ndarray, model_shape: tuple[int, int]) -> np.ndarray:
    mask_highres = np.asarray(mask_highres, dtype=bool)
    n_lateral, n_model = model_shape
    if mask_highres.shape[0] != n_lateral:
        return np.zeros(model_shape, dtype=bool)
    if mask_highres.shape[1] == n_model:
        return mask_highres.copy()
    factor = mask_highres.shape[1] / float(n_model)
    result = np.zeros(model_shape, dtype=bool)
    for model_index in range(n_model):
        start = int(np.floor(model_index * factor))
        stop = int(np.floor((model_index + 1) * factor))
        stop = max(stop, start + 1)
        start = min(start, mask_highres.shape[1])
        stop = min(stop, mask_highres.shape[1])
        if start < stop:
            result[:, model_index] = np.any(mask_highres[:, start:stop], axis=1)
    return result


def _patch_slice(row: Mapping[str, Any], *, offset: int = 0) -> tuple[slice, slice]:
    return (
        slice(int(row["lateral_start"]), int(row["lateral_stop"])),
        slice(int(row["twt_start"]) + offset, int(row["twt_stop"]) + offset),
    )


def _dilate_mask(mask: np.ndarray, *, lateral_radius: int, twt_radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros_like(mask, dtype=bool)
    for dl in range(-int(lateral_radius), int(lateral_radius) + 1):
        src_l0 = max(0, -dl)
        src_l1 = min(mask.shape[0], mask.shape[0] - dl)
        dst_l0 = max(0, dl)
        dst_l1 = min(mask.shape[0], mask.shape[0] + dl)
        for dt in range(-int(twt_radius), int(twt_radius) + 1):
            src_t0 = max(0, -dt)
            src_t1 = min(mask.shape[1], mask.shape[1] - dt)
            dst_t0 = max(0, dt)
            dst_t1 = min(mask.shape[1], mask.shape[1] + dt)
            result[dst_l0:dst_l1, dst_t0:dst_t1] |= mask[src_l0:src_l1, src_t0:src_t1]
    return result


def _region_metrics(prefix: str, residual: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(valid, dtype=bool) & np.isfinite(residual)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return {
            f"{prefix}_n_valid": 0,
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_mae": float("nan"),
            f"{prefix}_bias": float("nan"),
        }
    values = np.asarray(residual, dtype=np.float64)[valid]
    return {
        f"{prefix}_n_valid": n_valid,
        f"{prefix}_rmse": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_mae": float(np.mean(np.abs(values))),
        f"{prefix}_bias": float(np.mean(values)),
    }


def _lateral_gradient_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    pair_valid = valid[1:, :] & valid[:-1, :]
    n_valid = int(np.count_nonzero(pair_valid))
    if n_valid == 0:
        return {
            "lateral_gradient_n_valid": 0,
            "lateral_gradient_rmse": float("nan"),
            "lateral_gradient_mae": float("nan"),
            "lateral_gradient_corr": float("nan"),
        }
    pred_grad = np.asarray(prediction[1:, :] - prediction[:-1, :], dtype=np.float64)
    target_grad = np.asarray(target[1:, :] - target[:-1, :], dtype=np.float64)
    residual = pred_grad - target_grad
    metrics = regression_metrics(target_grad, pred_grad, valid_mask=pair_valid)
    return {
        "lateral_gradient_n_valid": n_valid,
        "lateral_gradient_rmse": float(np.sqrt(np.mean(residual[pair_valid] ** 2))),
        "lateral_gradient_mae": float(np.mean(np.abs(residual[pair_valid]))),
        "lateral_gradient_corr": metrics.get("corr", float("nan")),
    }


def _aggregate_geometry(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"n_ok": 0}
    ok = frame[pd.to_numeric(frame.get("all_n_valid", 0), errors="coerce").fillna(0).gt(0)]
    return {
        "n_ok": int(len(ok)),
        "mean_boundary_rmse": _safe_mean(ok, "boundary_rmse"),
        "mean_event_rmse": _safe_mean(ok, "event_rmse"),
        "mean_lateral_gradient_rmse": _safe_mean(ok, "lateral_gradient_rmse"),
        "median_lateral_gradient_corr": _safe_median(ok, "lateral_gradient_corr"),
    }


def _grouped_geometry_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    columns = [
        *keys,
        "n_ok",
        "mean_all_rmse",
        "mean_boundary_rmse",
        "mean_non_boundary_rmse",
        "mean_event_rmse",
        "mean_non_event_rmse",
        "mean_lateral_gradient_rmse",
        "median_lateral_gradient_corr",
    ]
    if frame.empty or not set(keys).issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        ok = group[pd.to_numeric(group.get("all_n_valid", 0), errors="coerce").fillna(0).gt(0)]
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "n_ok": int(len(ok)),
                "mean_all_rmse": _safe_mean(ok, "all_rmse"),
                "mean_boundary_rmse": _safe_mean(ok, "boundary_rmse"),
                "mean_non_boundary_rmse": _safe_mean(ok, "non_boundary_rmse"),
                "mean_event_rmse": _safe_mean(ok, "event_rmse"),
                "mean_non_event_rmse": _safe_mean(ok, "non_event_rmse"),
                "mean_lateral_gradient_rmse": _safe_mean(ok, "lateral_gradient_rmse"),
                "median_lateral_gradient_corr": _safe_median(ok, "lateral_gradient_corr"),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=columns)


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _safe_median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else float("nan")


def _stitch_realization_metrics(
    index: pd.DataFrame,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    lfm: np.ndarray,
) -> dict[str, pd.DataFrame]:
    return {
        "uniform": _stitch_one_strategy(
            index,
            pred,
            target,
            mask,
            lfm,
            strategy="uniform",
        ),
        "center_crop": _stitch_one_strategy(
            index,
            pred,
            target,
            mask,
            lfm,
            strategy="center_crop",
        ),
    }


def _stitch_one_strategy(
    index: pd.DataFrame,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    lfm: np.ndarray,
    *,
    strategy: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if index.empty:
        return pd.DataFrame()
    for sample_id, group in index.groupby("sample_id", dropna=False):
        n_lateral = int(pd.to_numeric(group["lateral_stop"], errors="coerce").max())
        n_twt = int(pd.to_numeric(group["twt_stop"], errors="coerce").max())
        if n_lateral <= 0 or n_twt <= 0:
            continue
        pred_sum = np.zeros((n_lateral, n_twt), dtype=np.float64)
        target_sum = np.zeros((n_lateral, n_twt), dtype=np.float64)
        lfm_sum = np.zeros((n_lateral, n_twt), dtype=np.float64)
        counts = np.zeros((n_lateral, n_twt), dtype=np.float64)
        first_row = group.iloc[0]
        for _, row in group.iterrows():
            patch_index = int(row["prediction_row"])
            local, dest = _stitch_slices(row, strategy=strategy)
            patch_valid = mask[patch_index][local]
            if not np.any(patch_valid):
                continue
            weights = patch_valid.astype(np.float64)
            pred_sum[dest] += np.where(patch_valid, pred[patch_index][local], 0.0)
            target_sum[dest] += np.where(patch_valid, target[patch_index][local], 0.0)
            lfm_sum[dest] += np.where(patch_valid, lfm[patch_index][local], 0.0)
            counts[dest] += weights
        valid = counts > 0.0
        if not np.any(valid):
            continue
        pred_full = np.divide(pred_sum, counts, out=np.zeros_like(pred_sum), where=valid)
        target_full = np.divide(target_sum, counts, out=np.zeros_like(target_sum), where=valid)
        lfm_full = np.divide(lfm_sum, counts, out=np.zeros_like(lfm_sum), where=valid)
        base = {
            "sample_id": sample_id,
            "sample_kind": first_row.get("sample_kind", ""),
            "split": first_row.get("split", ""),
            "geometry_family": first_row.get("geometry_family", ""),
            "scenario_id": first_row.get("scenario_id", ""),
            "duration_mode": first_row.get("duration_mode", ""),
            "stitch_strategy": strategy,
            "n_source_patches": int(len(group)),
            "stitched_lateral_samples": n_lateral,
            "stitched_twt_samples": n_twt,
            "coverage_fraction": float(np.mean(valid)),
            "mean_patch_overlap": float(np.mean(counts[valid])),
        }
        rows.append(
            {
                **base,
                "model_id": first_row.get("model_id", ""),
                **regression_metrics(target_full, pred_full, valid_mask=valid),
            }
        )
        rows.append(
            {
                **base,
                "model_id": "lfm_controlled_degraded",
                **regression_metrics(target_full, lfm_full, valid_mask=valid),
            }
        )
    return pd.DataFrame.from_records(rows)


def _stitch_slices(row: Mapping[str, Any], *, strategy: str) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    lateral_start = int(row["lateral_start"])
    lateral_stop = int(row["lateral_stop"])
    twt_start = int(row["twt_start"])
    twt_stop = int(row["twt_stop"])
    patch_lateral = lateral_stop - lateral_start
    patch_twt = twt_stop - twt_start
    if strategy == "uniform":
        local_l0, local_l1 = 0, patch_lateral
        local_t0, local_t1 = 0, patch_twt
    elif strategy == "center_crop":
        local_l0, local_l1 = _center_crop_bounds(patch_lateral)
        local_t0, local_t1 = _center_crop_bounds(patch_twt)
    else:
        raise ValueError(f"Unsupported stitch strategy: {strategy}")
    return (
        (slice(local_l0, local_l1), slice(local_t0, local_t1)),
        (
            slice(lateral_start + local_l0, lateral_start + local_l1),
            slice(twt_start + local_t0, twt_start + local_t1),
        ),
    )


def _center_crop_bounds(size: int) -> tuple[int, int]:
    if size <= 4:
        return 0, size
    margin = max(1, size // 4)
    start = margin
    stop = size - margin
    if stop <= start:
        return 0, size
    return start, stop


def _filter_unsupported_zero_x(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "operator_support" not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame["operator_support"].astype(str).eq("unsupported")].copy()


def _zero_x_false_frequency_energy(
    metrics_frame: pd.DataFrame,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    prediction_dir: Path,
) -> pd.DataFrame:
    zero = _zero_x_false_prediction_error(metrics_frame, prediction_dir)
    if zero.empty:
        return zero
    dt_s = _model_dt_s(prediction_dir)
    rows: list[dict[str, Any]] = []
    for _, row in zero.iterrows():
        frequency = _safe_float(row.get("probe_frequency_hz"))
        prediction_row = int(row["prediction_row"])
        metrics = _frequency_projection_metrics(
            residual=pred[prediction_row] - target[prediction_row],
            target=target[prediction_row],
            valid=mask[prediction_row],
            frequency_hz=frequency,
            dt_s=dt_s,
        )
        rows.append(
            {
                **row.to_dict(),
                "false_energy_semantics": (
                    "weighted least-squares sin/cos projection of pred-target residual "
                    "at the 0x probe frequency"
                ),
                "model_dt_s": dt_s,
                **metrics,
            }
        )
    return pd.DataFrame.from_records(rows)


def _frequency_projection_metrics(
    *,
    residual: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    frequency_hz: float,
    dt_s: float,
) -> dict[str, Any]:
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        return {"energy_status": "invalid_frequency"}
    residual = np.asarray(residual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(residual) & np.isfinite(target)
    n_valid = int(np.count_nonzero(valid))
    if residual.ndim != 2 or n_valid < 8:
        return {"energy_status": "insufficient_valid_samples", "energy_n_valid": n_valid}
    nt = residual.shape[1]
    t = np.arange(nt, dtype=np.float64) * float(dt_s)
    taper = _tukey_window(nt, alpha=0.5)
    sin_basis = np.sin(2.0 * np.pi * frequency_hz * t)
    cos_basis = np.cos(2.0 * np.pi * frequency_hz * t)
    weights = valid.astype(np.float64) * taper[None, :]
    residual_amp = _weighted_sincos_amplitude(residual, weights, sin_basis, cos_basis)
    target_amp = _weighted_sincos_amplitude(target, weights, sin_basis, cos_basis)
    false_rms = residual_amp / np.sqrt(2.0) if np.isfinite(residual_amp) else float("nan")
    target_rms = target_amp / np.sqrt(2.0) if np.isfinite(target_amp) else float("nan")
    return {
        "energy_status": "ok" if np.isfinite(false_rms) else "invalid_projection",
        "energy_n_valid": n_valid,
        "false_frequency_amplitude": float(residual_amp),
        "false_frequency_rms": float(false_rms),
        "target_frequency_amplitude": float(target_amp),
        "target_frequency_rms": float(target_rms),
        "false_to_target_frequency_rms": (
            float(false_rms / target_rms)
            if np.isfinite(false_rms) and np.isfinite(target_rms) and target_rms > 0.0
            else float("nan")
        ),
    }


def _weighted_sincos_amplitude(
    values: np.ndarray,
    weights: np.ndarray,
    sin_basis: np.ndarray,
    cos_basis: np.ndarray,
) -> float:
    coeffs = _weighted_sincos_coefficients(values, weights, sin_basis, cos_basis)
    if coeffs is None:
        return float("nan")
    return float(np.sqrt(coeffs[0] ** 2 + coeffs[1] ** 2))


def _weighted_sincos_coefficients(
    values: np.ndarray,
    weights: np.ndarray,
    sin_basis: np.ndarray,
    cos_basis: np.ndarray,
) -> tuple[float, float] | None:
    mask = weights > 0.0
    if int(np.count_nonzero(mask)) < 8:
        return None
    y = np.asarray(values, dtype=np.float64)[mask]
    w = weights[mask]
    sin2d = np.broadcast_to(sin_basis[None, :], values.shape)[mask]
    cos2d = np.broadcast_to(cos_basis[None, :], values.shape)[mask]
    basis = np.column_stack([sin2d, cos2d, np.ones_like(sin2d)])
    sw = np.sqrt(w)
    try:
        beta, *_ = np.linalg.lstsq(basis * sw[:, None], y * sw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(beta[0]), float(beta[1])


def _amplitude_phase_metrics(
    *,
    target_diff: np.ndarray,
    pred_diff: np.ndarray,
    valid: np.ndarray,
    frequency_hz: float,
    dt_s: float,
) -> dict[str, Any]:
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        return {"amplitude_phase_status": "invalid_frequency"}
    target_diff = np.asarray(target_diff, dtype=np.float64)
    pred_diff = np.asarray(pred_diff, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(target_diff) & np.isfinite(pred_diff)
    if target_diff.ndim != 2 or int(np.count_nonzero(valid)) < 8:
        return {
            "amplitude_phase_status": "insufficient_valid_samples",
            "amplitude_phase_n_valid": int(np.count_nonzero(valid)),
        }
    nt = target_diff.shape[1]
    t = np.arange(nt, dtype=np.float64) * float(dt_s)
    taper = _tukey_window(nt, alpha=0.5)
    sin_basis = np.sin(2.0 * np.pi * frequency_hz * t)
    cos_basis = np.cos(2.0 * np.pi * frequency_hz * t)
    weights = valid.astype(np.float64) * taper[None, :]
    target_coeff = _weighted_sincos_coefficients(target_diff, weights, sin_basis, cos_basis)
    pred_coeff = _weighted_sincos_coefficients(pred_diff, weights, sin_basis, cos_basis)
    if target_coeff is None or pred_coeff is None:
        return {"amplitude_phase_status": "invalid_projection"}
    target_amp = float(np.sqrt(target_coeff[0] ** 2 + target_coeff[1] ** 2))
    pred_amp = float(np.sqrt(pred_coeff[0] ** 2 + pred_coeff[1] ** 2))
    target_phase = float(np.arctan2(target_coeff[1], target_coeff[0]))
    pred_phase = float(np.arctan2(pred_coeff[1], pred_coeff[0]))
    phase_error = float(np.arctan2(np.sin(pred_phase - target_phase), np.cos(pred_phase - target_phase)))
    return {
        "amplitude_phase_status": "ok",
        "amplitude_phase_n_valid": int(np.count_nonzero(valid)),
        "target_probe_amplitude": target_amp,
        "pred_probe_amplitude": pred_amp,
        "probe_amplitude_error": float(pred_amp - target_amp),
        "probe_abs_amplitude_error": float(abs(pred_amp - target_amp)),
        "probe_amplitude_ratio": (
            float(pred_amp / target_amp) if np.isfinite(target_amp) and target_amp > 0.0 else float("nan")
        ),
        "target_probe_phase_rad": target_phase,
        "pred_probe_phase_rad": pred_phase,
        "probe_phase_error_rad": phase_error,
        "probe_abs_phase_error_deg": float(abs(np.rad2deg(phase_error))),
    }


def _tukey_window(size: int, *, alpha: float) -> np.ndarray:
    if size <= 0:
        return np.zeros(0, dtype=np.float64)
    if size == 1:
        return np.ones(1, dtype=np.float64)
    x = np.linspace(0.0, 1.0, size)
    window = np.ones(size, dtype=np.float64)
    edge = float(alpha) / 2.0
    if edge <= 0.0:
        return window
    left = x < edge
    right = x > 1.0 - edge
    window[left] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[left] / alpha - 1.0)))
    window[right] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[right] / alpha - 2.0 / alpha + 1.0)))
    return window


def _aggregate_false_energy(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "energy_status" not in frame:
        return {"n_ok": 0}
    ok = frame[frame["energy_status"].eq("ok")]
    return {
        "n_ok": int(len(ok)),
        "mean_false_frequency_rms": float(ok["false_frequency_rms"].mean()) if not ok.empty else float("nan"),
        "median_false_frequency_rms": float(ok["false_frequency_rms"].median()) if not ok.empty else float("nan"),
        "median_false_to_target_frequency_rms": (
            float(ok["false_to_target_frequency_rms"].median()) if not ok.empty else float("nan")
        ),
    }


def _aggregate_amplitude_phase(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "amplitude_phase_status" not in frame:
        return {"n_ok": 0}
    ok = frame[frame["amplitude_phase_status"].eq("ok")]
    return {
        "n_ok": int(len(ok)),
        "mean_abs_amplitude_error": (
            float(ok["probe_abs_amplitude_error"].mean()) if not ok.empty else float("nan")
        ),
        "median_amplitude_ratio": (
            float(ok["probe_amplitude_ratio"].median()) if not ok.empty else float("nan")
        ),
        "median_abs_phase_error_deg": (
            float(ok["probe_abs_phase_error_deg"].median()) if not ok.empty else float("nan")
        ),
    }


def _grouped_false_energy(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *keys,
                "n_ok",
                "mean_false_frequency_rms",
                "median_false_frequency_rms",
                "median_false_to_target_frequency_rms",
            ]
        )
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(_aggregate_false_energy(group))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _zero_x_false_prediction_error(metrics_frame: pd.DataFrame, prediction_dir: Path) -> pd.DataFrame:
    if metrics_frame.empty or "probe_amplitude_multiplier" not in metrics_frame:
        return pd.DataFrame()
    probe_amp = pd.to_numeric(metrics_frame["probe_amplitude_multiplier"], errors="coerce")
    zero = metrics_frame[
        metrics_frame["sample_kind"].astype(str).isin(PROBE_SAMPLE_KINDS)
        & probe_amp.eq(0.0)
    ].copy()
    if zero.empty:
        return zero
    zero["zero_x_false_error_semantics"] = (
        "absolute prediction error on 0x frequency-probe samples; "
        "not a full spectral false-energy decomposition"
    )
    catalog = _load_probe_frequency_catalog(prediction_dir)
    if catalog is not None and "probe_frequency_hz" in zero:
        zero["probe_frequency_hz"] = pd.to_numeric(zero["probe_frequency_hz"], errors="coerce")
        zero = zero.merge(
            catalog,
            left_on="probe_frequency_hz",
            right_on="frequency_hz",
            how="left",
            suffixes=("", "_catalog"),
        )
    return zero


def _load_probe_frequency_catalog(prediction_dir: Path) -> pd.DataFrame | None:
    import json

    manifest_path = prediction_dir / "prediction_manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    benchmark_dir = manifest.get("benchmark_dir")
    if not benchmark_dir:
        return None
    catalog_path = Path(benchmark_dir) / "probe_frequency_catalog.csv"
    if not catalog_path.is_absolute():
        catalog_path = Path.cwd() / catalog_path
    if not catalog_path.is_file():
        return None
    catalog = pd.read_csv(catalog_path)
    if "frequency_hz" in catalog:
        catalog["frequency_hz"] = pd.to_numeric(catalog["frequency_hz"], errors="coerce")
    keep = [
        column
        for column in [
            "frequency_hz",
            "evidence_status",
            "operator_support",
            "experiment_class",
            "selection_reason",
            "calibration_status",
        ]
        if column in catalog
    ]
    return catalog[keep].drop_duplicates("frequency_hz")


def _model_dt_s(prediction_dir: Path) -> float:
    import json

    manifest_path = prediction_dir / "prediction_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        benchmark_dir = manifest.get("benchmark_dir")
        if benchmark_dir:
            summary_path = Path(benchmark_dir) / "run_summary.json"
            if not summary_path.is_absolute():
                summary_path = Path.cwd() / summary_path
            if summary_path.is_file():
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                value = _safe_float(summary.get("output_dt_s"))
                if np.isfinite(value) and value > 0.0:
                    return float(value)
    return 0.002


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rmse_improvement_pct(model: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    model_rmse = float(model.get("mean_rmse", float("nan")))
    baseline_rmse = float(baseline.get("mean_rmse", float("nan")))
    if not np.isfinite(model_rmse) or not np.isfinite(baseline_rmse) or baseline_rmse == 0.0:
        return float("nan")
    return float((baseline_rmse - model_rmse) / baseline_rmse * 100.0)


def _grouped_probe_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        columns = [
            *keys,
            "n_ok",
            "mean_rmse",
            "mean_nrmse",
            "median_corr",
            "median_target_rms",
            "mean_abs_amplitude_error",
            "median_amplitude_ratio",
            "median_abs_phase_error_deg",
        ]
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        ok = group[group["status"].eq("ok")]
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "n_ok": int(len(ok)),
                "mean_rmse": float(ok["rmse"].mean()) if not ok.empty else float("nan"),
                "mean_nrmse": float(ok["nrmse"].mean()) if not ok.empty else float("nan"),
                "median_corr": float(ok["corr"].median()) if not ok.empty else float("nan"),
                "median_target_rms": float(ok["target_rms"].median()) if not ok.empty else float("nan"),
                "mean_abs_amplitude_error": (
                    float(ok["probe_abs_amplitude_error"].mean())
                    if not ok.empty and "probe_abs_amplitude_error" in ok
                    else float("nan")
                ),
                "median_amplitude_ratio": (
                    float(ok["probe_amplitude_ratio"].median())
                    if not ok.empty and "probe_amplitude_ratio" in ok
                    else float("nan")
                ),
                "median_abs_phase_error_deg": (
                    float(ok["probe_abs_phase_error_deg"].median())
                    if not ok.empty and "probe_abs_phase_error_deg" in ok
                    else float("nan")
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)
