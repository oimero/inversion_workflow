"""Training and prediction helpers for GINN-v2 ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from cup.synthetic.dataset import SynthoseisBenchmark
from cup.synthetic.metrics import regression_metrics
from cup.utils.io import sha256_file, write_json
from ginn_v2.data import PatchDataset, default_train_kinds, denormalize_delta
from ginn_v2.models import build_model


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
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_id == "patch_2d_with_physics_loss":
        raise NotImplementedError(
            "patch_2d_with_physics_loss is reserved for a future physics-loss ablation. "
            "It is not allowed to run as a supervised-only alias."
        )
    if lambda_physics != 0.0:
        raise NotImplementedError(
            "Non-zero physics loss is intentionally not implemented yet. "
            "Use lambda_physics=0 for supervised and mismatch-training ablations."
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
    device = torch.device(
        device_name
        if device_name != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = output_dir / "checkpoint.pt"
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["input"].to(device)
            target = batch["target_delta"].to(device)
            mask = batch["valid_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = masked_mse(pred, target, mask)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = evaluate_loss(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "validation_loss": val_loss,
        }
        history.append(row)
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_id": model_id,
                    "state_dict": model.state_dict(),
                    "normalization": dict(normalization),
                    "model_info": info.__dict__,
                    "architecture": {
                        "hidden_channels": int(hidden_channels),
                        "depth": int(depth),
                    },
                },
                best_path,
            )
    history_frame = pd.DataFrame.from_records(history)
    history_path = output_dir / "training_history.csv"
    history_frame.to_csv(history_path, index=False)
    return {
        "status": "ok",
        "device": str(device),
        "checkpoint": best_path,
        "history": history_path,
        "best_validation_loss": best_val,
        "model_info": info.__dict__,
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


def load_checkpoint(path: Path, *, hidden_channels: int | None = None, depth: int | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    info = dict(checkpoint["model_info"])
    architecture = dict(checkpoint.get("architecture") or {})
    model, _ = build_model(
        str(checkpoint["model_id"]),
        hidden_channels=int(hidden_channels or architecture.get("hidden_channels", 32)),
        depth=int(depth or architecture.get("depth", 5)),
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
    device = torch.device(
        device_name
        if device_name != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
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
    patch_ids = []
    sample_ids = []
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            pred_delta_n = model(x).detach().cpu().numpy()
            pred_delta = denormalize_delta(pred_delta_n[:, 0], checkpoint["normalization"])
            lfm = batch["lfm"].numpy()[:, 0]
            prediction = lfm + pred_delta
            predictions.append(prediction.astype(np.float32))
            targets.append(batch["target_log_ai"].numpy()[:, 0].astype(np.float32))
            masks.append(batch["valid_mask"].numpy()[:, 0].astype(bool))
            lfm_values.append(lfm.astype(np.float32))
            patch_ids.extend([str(value) for value in batch["patch_id"]])
            sample_ids.extend([str(value) for value in batch["sample_id"]])
    pred_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    mask_array = np.concatenate(masks, axis=0)
    lfm_array = np.concatenate(lfm_values, axis=0)
    npz_path = output_dir / "predictions.npz"
    np.savez_compressed(
        npz_path,
        pred_log_ai=pred_array,
        target_log_ai=target_array,
        valid_mask_model=mask_array,
        lfm_controlled_degraded=lfm_array,
        patch_id=np.asarray(patch_ids),
        sample_id=np.asarray(sample_ids),
    )
    selected = selected.set_index("patch_id").loc[patch_ids].reset_index()
    selected["prediction_row"] = np.arange(len(selected), dtype=int)
    selected["model_id"] = str(checkpoint["model_id"])
    index_path = output_dir / "prediction_index.csv"
    selected.to_csv(index_path, index=False)
    return {
        "status": "ok",
        "prediction_npz": npz_path,
        "prediction_index": index_path,
        "n_predictions": int(pred_array.shape[0]),
        "prediction_sha256": sha256_file(npz_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_id": str(checkpoint["model_id"]),
        "model_info": dict(checkpoint["model_info"]),
        "normalization": dict(checkpoint["normalization"]),
    }


def report_predictions(*, prediction_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(prediction_dir / "prediction_index.csv")
    arrays = np.load(prediction_dir / "predictions.npz", allow_pickle=True)
    pred = arrays["pred_log_ai"]
    target = arrays["target_log_ai"]
    mask = arrays["valid_mask_model"].astype(bool)
    lfm = arrays["lfm_controlled_degraded"]
    rows = []
    lfm_rows = []
    for _, row in index.iterrows():
        i = int(row["prediction_row"])
        metrics = regression_metrics(target[i], pred[i], valid_mask=mask[i])
        rows.append({**row.to_dict(), "model_id": row.get("model_id", ""), **metrics})
        lfm_metrics = regression_metrics(target[i], lfm[i], valid_mask=mask[i])
        lfm_rows.append({**row.to_dict(), "model_id": "lfm_controlled_degraded", **lfm_metrics})
    metrics_frame = pd.DataFrame.from_records(rows)
    metrics_path = output_dir / "model_patch_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    geometry_path = output_dir / "model_patch_metrics_by_geometry.csv"
    _grouped_patch_metrics(metrics_frame, ["geometry_family"]).to_csv(geometry_path, index=False)
    mismatch_path = output_dir / "model_patch_metrics_by_mismatch_family.csv"
    _grouped_patch_metrics(metrics_frame, ["seismic_mismatch_family"]).to_csv(mismatch_path, index=False)
    lfm_frame = pd.DataFrame.from_records(lfm_rows)
    lfm_path = output_dir / "lfm_patch_metrics.csv"
    lfm_frame.to_csv(lfm_path, index=False)
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
    report = {
        "schema_version": "ginn_v2_patch_smoke_report_v1",
        "report_scope": "patch_smoke",
        "not_synthoseis_lite_report": True,
        "status": "ok",
        "n_patches": int(len(metrics_frame)),
        "aggregate": model_aggregate,
        "lfm_aggregate": lfm_aggregate,
        "rmse_improvement_pct_vs_lfm": _rmse_improvement_pct(model_aggregate, lfm_aggregate),
        "probe_aggregate": _aggregate(probe_frame),
        "probe_amplitude_phase_aggregate": _aggregate_amplitude_phase(probe_frame),
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
        "lfm_patch_metrics": lfm_path,
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
        pair_id = str(row.get("paired_zero_patch_id", "") or "").strip()
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
                "probe_frequency_hz": row.get("probe_frequency_hz", ""),
                "probe_phase": row.get("probe_phase", ""),
                "probe_lateral_shape": row.get("probe_lateral_shape", ""),
                "probe_amplitude_multiplier": amp,
                "probe_metric_semantics": "paired_probe_increment_error",
                **metrics,
                **amp_phase,
            }
        )
    return rows


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
        metrics_frame["sample_kind"].astype(str).eq("frequency_probe")
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
