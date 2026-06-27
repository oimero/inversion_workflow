"""Sparse-well real-delta head adaptation for frozen GINN-v2 models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cup.synthetic.dataset import SynthoseisBenchmark
from cup.synthetic.metrics import regression_metrics
from ginn_v2.data import PatchDataset, denormalize_delta
from ginn_v2.real_field import _normalize_with_mask, _stitch_slices, PatchGeometry
from ginn_v2.training import (
    _aggregate,
    _aggregate_geometry,
    _geometry_patch_metrics,
    _stitch_realization_metrics,
    load_checkpoint,
    resolve_device,
)


REQUIRED_WELL_SAMPLE_COLUMNS = {
    "well_name",
    "sample_index",
    "twt_s",
    "inline",
    "xline",
    "x_m",
    "y_m",
    "spatial_cluster_id",
    "spatial_cluster_size",
    "target_log_ai",
    "lfm_log_ai",
    "model_role",
    "r0_pred_log_ai",
    "r0_valid_mask",
    "r0_blend_weight",
    "target_valid",
    "valid_for_fit",
    "valid_reason",
    "sampling_mode",
    "sample_method",
    "wellbore_class",
}


@dataclass(frozen=True)
class HeadState:
    weight: np.ndarray
    bias: float

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float64).reshape(-1)
        if weight.size == 0 or not np.all(np.isfinite(weight)) or not np.isfinite(self.bias):
            raise ValueError("HeadState must contain finite non-empty parameters.")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", float(self.bias))


@dataclass(frozen=True)
class HeadFit:
    status: str
    lambda_anchor: float
    head: HeadState
    feature_mean: np.ndarray
    feature_std: np.ndarray
    active_mask: np.ndarray
    beta: np.ndarray
    beta_anchor: np.ndarray
    training_clusters: tuple[int, ...]
    training_wells: tuple[str, ...]


@dataclass(frozen=True)
class RoleValidationResult:
    fold_metrics: pd.DataFrame
    cluster_metrics: pd.DataFrame
    lambda_selection: pd.DataFrame
    head_parameters: pd.DataFrame
    heads: Mapping[str, HeadFit]
    n_loco_clusters: int
    outer_status: str
    all_wells_status: str


def load_well_samples(path: Path, *, model_roles: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_WELL_SAMPLE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"well_ai_samples.csv missing required columns: {missing}")
    if frame.empty:
        raise ValueError("well_ai_samples.csv is empty.")
    valid_text = frame["valid_for_fit"].astype(str).str.strip().str.casefold()
    frame = frame[valid_text.eq("true")].copy()
    frame = frame[frame["model_role"].astype(str).isin([str(role) for role in model_roles])].copy()
    if frame.empty:
        raise ValueError("well_ai_samples.csv has no valid rows for configured model roles.")
    if not frame["sampling_mode"].astype(str).eq("volume").all():
        raise ValueError("R2 only accepts R1 samples with sampling_mode=volume.")
    numeric = [
        "sample_index",
        "twt_s",
        "inline",
        "xline",
        "spatial_cluster_id",
        "target_log_ai",
        "lfm_log_ai",
        "r0_pred_log_ai",
        "r0_blend_weight",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    finite_columns = [
        "twt_s",
        "inline",
        "xline",
        "target_log_ai",
        "lfm_log_ai",
        "r0_pred_log_ai",
        "r0_blend_weight",
    ]
    if not np.all(np.isfinite(frame[finite_columns].to_numpy(dtype=np.float64))):
        raise ValueError("valid_for_fit rows contain non-finite required values.")
    if np.any(frame["r0_blend_weight"].to_numpy(dtype=np.float64) <= 0.0):
        raise ValueError("valid_for_fit rows contain non-positive r0_blend_weight.")
    duplicate = frame.duplicated(["model_role", "well_name", "sample_index"], keep=False)
    if duplicate.any():
        keys = frame.loc[duplicate, ["model_role", "well_name", "sample_index"]]
        raise ValueError(f"Duplicate R1 sample keys: {keys.head(10).to_dict(orient='records')}")
    _assert_role_sample_alignment(frame, model_roles=model_roles)
    if "filtered_log_ai" in frame.columns:
        raise ValueError("R2 rejects legacy filtered-label well_ai_samples.csv; rerun Step7 -> R0 -> R1.")
    frame["well_delta"] = frame["target_log_ai"] - frame["lfm_log_ai"]
    frame["r0_delta_csv"] = frame["r0_pred_log_ai"] - frame["lfm_log_ai"]
    return frame.sort_values(["model_role", "well_name", "sample_index"]).reset_index(drop=True)


def _assert_role_sample_alignment(frame: pd.DataFrame, *, model_roles: Sequence[str]) -> None:
    roles = [str(role) for role in model_roles]
    if len(roles) < 2:
        return
    key = ["well_name", "sample_index"]
    reference = frame[frame["model_role"].astype(str).eq(roles[0])].set_index(key).sort_index()
    compare_columns = [
        "twt_s",
        "inline",
        "xline",
        "spatial_cluster_id",
        "target_log_ai",
        "lfm_log_ai",
        "sampling_mode",
        "sample_method",
        "wellbore_class",
    ]
    for role in roles[1:]:
        other = frame[frame["model_role"].astype(str).eq(role)].set_index(key).sort_index()
        if not reference.index.equals(other.index):
            raise ValueError(f"R1 sample keys differ between roles {roles[0]} and {role}.")
        for column in compare_columns:
            left = reference[column]
            right = other[column]
            if pd.api.types.is_numeric_dtype(left):
                equal = np.allclose(
                    left.to_numpy(dtype=np.float64),
                    right.to_numpy(dtype=np.float64),
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                )
            else:
                equal = left.fillna("").astype(str).equals(right.fillna("").astype(str))
            if not equal:
                raise ValueError(f"R1 sample column {column} differs between roles {roles[0]} and {role}.")


def get_head_state(model: torch.nn.Module) -> HeadState:
    head = getattr(model, "output_head", None)
    if not isinstance(head, (torch.nn.Conv1d, torch.nn.Conv2d)):
        raise TypeError(f"Model does not expose a supported output_head: {type(head)!r}")
    kernel_shape = tuple(int(value) for value in head.weight.shape[2:])
    if head.out_channels != 1 or any(value != 1 for value in kernel_shape) or head.bias is None:
        raise ValueError(f"R2 requires a biased 1x1 single-output head, got weight={tuple(head.weight.shape)}")
    return HeadState(
        weight=head.weight.detach().cpu().numpy().reshape(-1).astype(np.float64),
        bias=float(head.bias.detach().cpu().numpy().reshape(-1)[0]),
    )


def set_head_state(model: torch.nn.Module, state: HeadState) -> None:
    head = getattr(model, "output_head", None)
    if not isinstance(head, (torch.nn.Conv1d, torch.nn.Conv2d)):
        raise TypeError(f"Model does not expose a supported output_head: {type(head)!r}")
    if int(head.in_channels) != int(state.weight.size):
        raise ValueError(f"Head channel mismatch: model={head.in_channels}, state={state.weight.size}")
    with torch.no_grad():
        head.weight.copy_(
            torch.as_tensor(state.weight, dtype=head.weight.dtype, device=head.weight.device).reshape_as(head.weight)
        )
        head.bias.copy_(torch.as_tensor([state.bias], dtype=head.bias.dtype, device=head.bias.device))


def head_parameter_rows(
    *,
    model_role: str,
    head_id: str,
    fit: HeadFit,
    anchor: HeadState,
    split_type: str,
    fold_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (base, adapted, active) in enumerate(zip(anchor.weight, fit.head.weight, fit.active_mask)):
        rows.append(
            {
                "model_role": model_role,
                "head_id": head_id,
                "split_type": split_type,
                "fold_id": fold_id,
                "lambda_anchor": fit.lambda_anchor,
                "parameter_name": f"weight_{idx:03d}",
                "r0_value": float(base),
                "r2_value": float(adapted),
                "parameter_delta": float(adapted - base),
                "active_feature": bool(active),
                "fit_status": fit.status,
            }
        )
    rows.append(
        {
            "model_role": model_role,
            "head_id": head_id,
            "split_type": split_type,
            "fold_id": fold_id,
            "lambda_anchor": fit.lambda_anchor,
            "parameter_name": "bias",
            "r0_value": anchor.bias,
            "r2_value": fit.head.bias,
            "parameter_delta": fit.head.bias - anchor.bias,
            "active_feature": True,
            "fit_status": fit.status,
        }
    )
    return rows


def hierarchical_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        raise ValueError("Cannot compute hierarchical weights for an empty frame.")
    clusters = sorted(int(value) for value in frame["spatial_cluster_id"].unique())
    weights = np.zeros(len(frame), dtype=np.float64)
    for cluster in clusters:
        cluster_mask = frame["spatial_cluster_id"].to_numpy(dtype=int) == cluster
        wells = sorted(frame.loc[cluster_mask, "well_name"].astype(str).unique())
        for well in wells:
            mask = cluster_mask & (frame["well_name"].astype(str).to_numpy() == well)
            count = int(np.count_nonzero(mask))
            if count <= 0:
                raise ValueError(f"Empty well group while weighting: cluster={cluster}, well={well}")
            weights[mask] = 1.0 / (len(clusters) * len(wells) * count)
    if not np.isclose(float(np.sum(weights)), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"Hierarchical weights do not sum to one: {np.sum(weights)}")
    return weights


def fit_anchored_head(
    *,
    features: np.ndarray,
    target_delta_normalized: np.ndarray,
    frame: pd.DataFrame,
    anchor: HeadState,
    lambda_anchor: float,
    feature_std_epsilon: float = 1e-12,
) -> HeadFit:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target_delta_normalized, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[0] != len(frame):
        raise ValueError(f"Feature/target/frame shape mismatch: {x.shape}, {y.shape}, {len(frame)}")
    if x.shape[1] != anchor.weight.size:
        raise ValueError(f"Feature/head channel mismatch: {x.shape[1]} vs {anchor.weight.size}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Anchored head fit received non-finite features or targets.")
    if not np.isfinite(lambda_anchor) or float(lambda_anchor) <= 0.0:
        raise ValueError("lambda_anchor must be finite and positive.")
    q = hierarchical_sample_weights(frame)
    mean = np.sum(q[:, None] * x, axis=0)
    variance = np.sum(q[:, None] * (x - mean[None, :]) ** 2, axis=0)
    std = np.sqrt(np.maximum(variance, 0.0))
    active = np.isfinite(std) & (std > float(feature_std_epsilon))
    if not np.any(active):
        raise ValueError("degenerate_training_features")
    z = (x[:, active] - mean[active][None, :]) / std[active][None, :]
    design = np.column_stack([z, np.ones(x.shape[0], dtype=np.float64)])
    fixed = x[:, ~active] @ anchor.weight[~active] if np.any(~active) else np.zeros(x.shape[0])
    adjusted_target = y - fixed
    beta_anchor = np.concatenate(
        [
            anchor.weight[active] * std[active],
            [anchor.bias + float(np.dot(anchor.weight[active], mean[active]))],
        ]
    )
    lhs = design.T @ (q[:, None] * design) + float(lambda_anchor) * np.eye(design.shape[1])
    rhs = design.T @ (q * adjusted_target) + float(lambda_anchor) * beta_anchor
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("ridge_solve_failed") from exc
    if not np.all(np.isfinite(beta)):
        raise ValueError("ridge_solve_failed")
    weight = anchor.weight.copy()
    weight[active] = beta[:-1] / std[active]
    bias = float(beta[-1] - np.dot(beta[:-1], mean[active] / std[active]))
    head64 = HeadState(weight=weight, bias=bias)
    standardized_prediction = design @ beta + fixed
    raw_prediction = apply_head(features=x, head=head64)
    if not np.allclose(standardized_prediction, raw_prediction, rtol=0.0, atol=1e-10):
        raise ValueError("head_coordinate_roundtrip_failed")
    head = HeadState(
        weight=head64.weight.astype(np.float32).astype(np.float64),
        bias=float(np.float32(head64.bias)),
    )
    deployed_prediction = apply_head(features=x, head=head)
    if not np.allclose(standardized_prediction, deployed_prediction, rtol=0.0, atol=1e-6):
        raise ValueError("head_float32_roundtrip_failed")
    return HeadFit(
        status="ok",
        lambda_anchor=float(lambda_anchor),
        head=head,
        feature_mean=mean,
        feature_std=std,
        active_mask=active,
        beta=beta,
        beta_anchor=beta_anchor,
        training_clusters=tuple(sorted(int(value) for value in frame["spatial_cluster_id"].unique())),
        training_wells=tuple(sorted(frame["well_name"].astype(str).unique())),
    )


def apply_head(*, features: np.ndarray, head: HeadState) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != head.weight.size:
        raise ValueError(f"Feature/head shape mismatch: {x.shape} vs {head.weight.size}")
    return x @ head.weight + head.bias


def denormalize_delta_values(values: np.ndarray, normalization: Mapping[str, Any]) -> np.ndarray:
    stats = dict(normalization["delta"])
    return np.asarray(values, dtype=np.float64) * float(stats["std"]) + float(stats["mean"])


def normalize_delta_values(values: np.ndarray, normalization: Mapping[str, Any]) -> np.ndarray:
    stats = dict(normalization["delta"])
    std = float(stats["std"])
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("Checkpoint delta normalization std must be finite and positive.")
    return (np.asarray(values, dtype=np.float64) - float(stats["mean"])) / std


def _series_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(prediction)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 8:
        return {"status": "insufficient_valid_samples", "n_valid": n_valid}
    target_values = target[valid]
    prediction_values = prediction[valid]
    target_std = float(np.std(target_values))
    prediction_std = float(np.std(prediction_values))
    if target_std <= 0.0 or prediction_std <= 0.0:
        return {
            "status": "invalid_zero_variance",
            "n_valid": n_valid,
            "target_std": target_std,
            "prediction_std": prediction_std,
        }
    residual = prediction_values - target_values
    return {
        "status": "ok",
        "n_valid": n_valid,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "bias": float(np.mean(residual)),
        "corr": float(np.corrcoef(target_values, prediction_values)[0, 1]),
        "target_std": target_std,
        "prediction_std": prediction_std,
        "prediction_rms": float(np.sqrt(np.mean(prediction_values * prediction_values))),
        "target_rms": float(np.sqrt(np.mean(target_values * target_values))),
    }


def evaluate_head_by_well(
    *,
    frame: pd.DataFrame,
    features: np.ndarray,
    anchor: HeadState,
    adapted: HeadState,
    normalization: Mapping[str, Any],
    model_role: str,
    split_type: str,
    fold_id: str,
    outer_fold: str,
) -> pd.DataFrame:
    r0_delta = denormalize_delta_values(apply_head(features=features, head=anchor), normalization)
    r2_delta = denormalize_delta_values(apply_head(features=features, head=adapted), normalization)
    rows: list[dict[str, Any]] = []
    for well, positions in frame.groupby("well_name", sort=True).indices.items():
        index = np.asarray(positions, dtype=int)
        local = frame.iloc[index]
        order = np.argsort(local["sample_index"].to_numpy(dtype=int))
        index = index[order]
        local = frame.iloc[index]
        well_delta = local["well_delta"].to_numpy(dtype=np.float64)
        lfm = local["lfm_log_ai"].to_numpy(dtype=np.float64)
        r0_delta_metrics = _series_metrics(well_delta, r0_delta[index])
        r2_delta_metrics = _series_metrics(well_delta, r2_delta[index])
        r0_full_metrics = _series_metrics(lfm + well_delta, lfm + r0_delta[index])
        r2_full_metrics = _series_metrics(lfm + well_delta, lfm + r2_delta[index])
        statuses = [
            r0_delta_metrics.get("status"),
            r2_delta_metrics.get("status"),
            r0_full_metrics.get("status"),
            r2_full_metrics.get("status"),
        ]
        status = "ok" if all(value == "ok" for value in statuses) else "invalid_metrics"
        row: dict[str, Any] = {
            "split_type": split_type,
            "model_role": model_role,
            "outer_fold": outer_fold,
            "fold_id": fold_id,
            "held_out_well": str(well),
            "held_out_spatial_cluster_id": int(local["spatial_cluster_id"].iloc[0]),
            "wellbore_class": str(local["wellbore_class"].iloc[0]),
            "n_samples": int(len(local)),
            "status": status,
            "metric_statuses": "|".join(str(value) for value in statuses),
        }
        for prefix, metrics in [
            ("r0_delta", r0_delta_metrics),
            ("r2_delta", r2_delta_metrics),
            ("r0_full_ai", r0_full_metrics),
            ("r2_full_ai", r2_full_metrics),
        ]:
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        if status == "ok":
            row.update(
                {
                    "delta_corr_gain": float(r2_delta_metrics["corr"] - r0_delta_metrics["corr"]),
                    "delta_rmse_delta": float(r2_delta_metrics["rmse"] - r0_delta_metrics["rmse"]),
                    "full_ai_corr_gain": float(r2_full_metrics["corr"] - r0_full_metrics["corr"]),
                    "full_ai_rmse_delta": float(r2_full_metrics["rmse"] - r0_full_metrics["rmse"]),
                    "r2_to_r0_delta_rms_ratio": (
                        float(r2_delta_metrics["prediction_rms"] / r0_delta_metrics["prediction_rms"])
                        if float(r0_delta_metrics["prediction_rms"]) > 0.0
                        else float("nan")
                    ),
                    "r2_to_well_delta_rms_ratio": (
                        float(r2_delta_metrics["prediction_rms"] / r2_delta_metrics["target_rms"])
                        if float(r2_delta_metrics["target_rms"]) > 0.0
                        else float("nan")
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def aggregate_cluster_metrics(well_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["split_type", "model_role", "outer_fold", "fold_id", "held_out_spatial_cluster_id"]
    for values, group in well_metrics.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        ok = group[group["status"].astype(str).eq("ok")]
        row.update(
            {
                "n_held_out_wells": int(group["held_out_well"].nunique()),
                "n_valid_wells": int(ok["held_out_well"].nunique()),
                "cluster_status": "ok" if len(ok) == len(group) and not ok.empty else "invalid_well_metrics",
            }
        )
        for column in [
            "delta_corr_gain",
            "delta_rmse_delta",
            "full_ai_corr_gain",
            "full_ai_rmse_delta",
            "r2_to_r0_delta_rms_ratio",
            "r2_to_well_delta_rms_ratio",
        ]:
            row[f"cluster_median_{column}"] = (
                float(pd.to_numeric(ok[column], errors="coerce").median())
                if not ok.empty and column in ok
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _select_lambda(
    *,
    frame: pd.DataFrame,
    features: np.ndarray,
    anchor: HeadState,
    normalization: Mapping[str, Any],
    lambda_candidates: Sequence[float],
    model_role: str,
    selection_scope: str,
) -> tuple[float | None, str, pd.DataFrame]:
    clusters = sorted(int(value) for value in frame["spatial_cluster_id"].unique())
    if len(clusters) < 2:
        return None, "lambda_selection_invalid_inner_fold", pd.DataFrame()
    target_n = normalize_delta_values(frame["well_delta"].to_numpy(dtype=np.float64), normalization)
    rows: list[dict[str, Any]] = []
    for lambda_anchor in lambda_candidates:
        for held_cluster in clusters:
            train_mask = frame["spatial_cluster_id"].to_numpy(dtype=int) != held_cluster
            held_mask = ~train_mask
            base = {
                "selection_scope": selection_scope,
                "model_role": model_role,
                "lambda_anchor": float(lambda_anchor),
                "inner_held_out_cluster": held_cluster,
                "n_training_clusters": int(frame.loc[train_mask, "spatial_cluster_id"].nunique()),
                "n_training_wells": int(frame.loc[train_mask, "well_name"].nunique()),
            }
            try:
                fit = fit_anchored_head(
                    features=features[train_mask],
                    target_delta_normalized=target_n[train_mask],
                    frame=frame.loc[train_mask].reset_index(drop=True),
                    anchor=anchor,
                    lambda_anchor=float(lambda_anchor),
                )
                metrics = evaluate_head_by_well(
                    frame=frame.loc[held_mask].reset_index(drop=True),
                    features=features[held_mask],
                    anchor=anchor,
                    adapted=fit.head,
                    normalization=normalization,
                    model_role=model_role,
                    split_type="inner_loco",
                    fold_id=f"{selection_scope}__cluster_{held_cluster}",
                    outer_fold=selection_scope,
                )
                cluster = aggregate_cluster_metrics(metrics)
                if len(cluster) != 1 or str(cluster.iloc[0]["cluster_status"]) != "ok":
                    raise ValueError("invalid_inner_metrics")
                value = cluster.iloc[0]
                rows.append(
                    {
                        **base,
                        "status": "ok",
                        "delta_corr_gain": value["cluster_median_delta_corr_gain"],
                        "full_ai_corr_gain": value["cluster_median_full_ai_corr_gain"],
                        "full_ai_rmse_delta": value["cluster_median_full_ai_rmse_delta"],
                    }
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                rows.append({**base, "status": "invalid_inner_fold", "reason": str(exc)})
    result = pd.DataFrame.from_records(rows)
    summaries: list[dict[str, Any]] = []
    for lambda_anchor, group in result.groupby("lambda_anchor", sort=True):
        ok = group[group["status"].astype(str).eq("ok")]
        complete = len(ok) == len(clusters)
        delta_gain = float(ok["delta_corr_gain"].median()) if complete else float("nan")
        full_corr = float(ok["full_ai_corr_gain"].median()) if complete else float("nan")
        full_rmse = float(ok["full_ai_rmse_delta"].median()) if complete else float("nan")
        eligible = bool(complete and delta_gain > 0.0 and full_corr >= 0.0 and full_rmse <= 0.0)
        summaries.append(
            {
                "lambda_anchor": float(lambda_anchor),
                "candidate_complete": complete,
                "candidate_median_delta_corr_gain": delta_gain,
                "candidate_median_full_ai_corr_gain": full_corr,
                "candidate_median_full_ai_rmse_delta": full_rmse,
                "candidate_eligible": eligible,
            }
        )
    summary = pd.DataFrame.from_records(summaries)
    result = result.merge(summary, on="lambda_anchor", how="left")
    eligible = summary[summary["candidate_eligible"]].copy()
    if not eligible.empty:
        selected = eligible.sort_values(
            ["candidate_median_delta_corr_gain", "lambda_anchor"],
            ascending=[False, False],
        ).iloc[0]
        selected_lambda = float(selected["lambda_anchor"])
        result["selected_lambda"] = np.isclose(result["lambda_anchor"], selected_lambda)
        return selected_lambda, "ok", result
    result["selected_lambda"] = False
    complete = summary[summary["candidate_complete"]].copy()
    if complete.empty:
        return None, "lambda_selection_invalid_inner_fold", result
    if bool((complete["candidate_median_delta_corr_gain"] > 0.0).any()):
        return None, "no_eligible_lambda_due_to_full_ai_guardrail", result
    return None, "no_eligible_lambda_due_to_no_transfer_signal", result


def run_role_validation(
    *,
    frame: pd.DataFrame,
    features: np.ndarray,
    anchor: HeadState,
    normalization: Mapping[str, Any],
    lambda_candidates: Sequence[float],
    model_role: str,
) -> RoleValidationResult:
    role_frame = frame.reset_index(drop=True).copy()
    x = np.asarray(features, dtype=np.float64)
    if len(role_frame) != x.shape[0]:
        raise ValueError("Role frame/features length mismatch.")
    clusters = sorted(int(value) for value in role_frame["spatial_cluster_id"].unique())
    target_n = normalize_delta_values(role_frame["well_delta"].to_numpy(dtype=np.float64), normalization)
    fold_frames: list[pd.DataFrame] = []
    lambda_frames: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    heads: dict[str, HeadFit] = {}
    outer_failures: list[str] = []
    for held_cluster in clusters:
        train_mask = role_frame["spatial_cluster_id"].to_numpy(dtype=int) != held_cluster
        held_mask = ~train_mask
        scope = f"outer_loco_cluster_{held_cluster}"
        selected_lambda, selection_status, selection = _select_lambda(
            frame=role_frame.loc[train_mask].reset_index(drop=True),
            features=x[train_mask],
            anchor=anchor,
            normalization=normalization,
            lambda_candidates=lambda_candidates,
            model_role=model_role,
            selection_scope=scope,
        )
        if not selection.empty:
            selection["outer_fold"] = scope
            selection["selection_status"] = selection_status
            lambda_frames.append(selection)
        if selected_lambda is None:
            outer_failures.append(selection_status)
            continue
        fit = fit_anchored_head(
            features=x[train_mask],
            target_delta_normalized=target_n[train_mask],
            frame=role_frame.loc[train_mask].reset_index(drop=True),
            anchor=anchor,
            lambda_anchor=selected_lambda,
        )
        heads[scope] = fit
        metrics = evaluate_head_by_well(
            frame=role_frame.loc[held_mask].reset_index(drop=True),
            features=x[held_mask],
            anchor=anchor,
            adapted=fit.head,
            normalization=normalization,
            model_role=model_role,
            split_type="leave_one_cluster_out",
            fold_id=scope,
            outer_fold=scope,
        )
        metrics["lambda_anchor"] = selected_lambda
        fold_frames.append(metrics)
        parameter_rows.extend(
            head_parameter_rows(
                model_role=model_role,
                head_id=scope,
                fit=fit,
                anchor=anchor,
                split_type="leave_one_cluster_out",
                fold_id=scope,
            )
        )
    for held_well in sorted(role_frame["well_name"].astype(str).unique()):
        train_mask = role_frame["well_name"].astype(str).to_numpy() != held_well
        held_mask = ~train_mask
        scope = f"loo_well_{held_well}"
        selected_lambda, selection_status, selection = _select_lambda(
            frame=role_frame.loc[train_mask].reset_index(drop=True),
            features=x[train_mask],
            anchor=anchor,
            normalization=normalization,
            lambda_candidates=lambda_candidates,
            model_role=model_role,
            selection_scope=scope,
        )
        if not selection.empty:
            selection["outer_fold"] = scope
            selection["selection_status"] = selection_status
            lambda_frames.append(selection)
        if selected_lambda is None:
            continue
        fit = fit_anchored_head(
            features=x[train_mask],
            target_delta_normalized=target_n[train_mask],
            frame=role_frame.loc[train_mask].reset_index(drop=True),
            anchor=anchor,
            lambda_anchor=selected_lambda,
        )
        heads[scope] = fit
        metrics = evaluate_head_by_well(
            frame=role_frame.loc[held_mask].reset_index(drop=True),
            features=x[held_mask],
            anchor=anchor,
            adapted=fit.head,
            normalization=normalization,
            model_role=model_role,
            split_type="leave_one_well_out",
            fold_id=scope,
            outer_fold=scope,
        )
        metrics["lambda_anchor"] = selected_lambda
        fold_frames.append(metrics)
        parameter_rows.extend(
            head_parameter_rows(
                model_role=model_role,
                head_id=scope,
                fit=fit,
                anchor=anchor,
                split_type="leave_one_well_out",
                fold_id=scope,
            )
        )
    all_lambda, all_status, all_selection = _select_lambda(
        frame=role_frame,
        features=x,
        anchor=anchor,
        normalization=normalization,
        lambda_candidates=lambda_candidates,
        model_role=model_role,
        selection_scope="all_wells",
    )
    if not all_selection.empty:
        all_selection["outer_fold"] = "all_wells"
        all_selection["selection_status"] = all_status
        lambda_frames.append(all_selection)
    if all_lambda is not None:
        all_fit = fit_anchored_head(
            features=x,
            target_delta_normalized=target_n,
            frame=role_frame,
            anchor=anchor,
            lambda_anchor=all_lambda,
        )
        heads["all_wells"] = all_fit
        parameter_rows.extend(
            head_parameter_rows(
                model_role=model_role,
                head_id="all_wells",
                fit=all_fit,
                anchor=anchor,
                split_type="all_wells",
                fold_id="all_wells",
            )
        )
    fold_metrics = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    cluster_metrics = aggregate_cluster_metrics(fold_metrics) if not fold_metrics.empty else pd.DataFrame()
    n_outer_heads = sum(head_id.startswith("outer_loco_cluster_") for head_id in heads)
    if n_outer_heads == len(clusters) and not outer_failures:
        outer_status = "ok"
    elif any("invalid" in value for value in outer_failures):
        outer_status = "inconclusive_invalid_fold"
    elif any("full_ai_guardrail" in value for value in outer_failures):
        outer_status = "no_eligible_lambda_due_to_full_ai_guardrail"
    elif outer_failures:
        outer_status = "no_eligible_lambda_due_to_no_transfer_signal"
    else:
        outer_status = "inconclusive_invalid_fold"
    return RoleValidationResult(
        fold_metrics=fold_metrics,
        cluster_metrics=cluster_metrics,
        lambda_selection=pd.concat(lambda_frames, ignore_index=True) if lambda_frames else pd.DataFrame(),
        head_parameters=pd.DataFrame.from_records(parameter_rows),
        heads=heads,
        n_loco_clusters=len(clusters),
        outer_status=outer_status,
        all_wells_status=all_status,
    )


def _axis_index_float(axis: np.ndarray, value: float) -> float:
    values = np.asarray(axis, dtype=np.float64)
    if values.size < 2:
        raise ValueError("Axis must contain at least two samples.")
    step = float(np.median(np.diff(values)))
    if not np.isfinite(step) or step == 0.0:
        raise ValueError("Axis step is invalid.")
    return (float(value) - float(values[0])) / step


def _time_interpolation_nodes(
    *, trace: np.ndarray, twt_s: np.ndarray, sample_twt_s: float
) -> list[tuple[int, float]] | None:
    finite = np.isfinite(trace) & np.isfinite(twt_s)
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size < 2:
        return None
    times = np.asarray(twt_s, dtype=np.float64)[finite_indices]
    value = float(sample_twt_s)
    if value < times[0] or value > times[-1]:
        return None
    position = int(np.searchsorted(times, value, side="left"))
    if position < times.size and times[position] == value:
        return [(int(finite_indices[position]), 1.0)]
    if position <= 0 or position >= times.size:
        return None
    left_time = float(times[position - 1])
    right_time = float(times[position])
    fraction = (value - left_time) / (right_time - left_time)
    return [
        (int(finite_indices[position - 1]), float(1.0 - fraction)),
        (int(finite_indices[position]), float(fraction)),
    ]


def build_volume_query_stencils(
    *,
    samples: pd.DataFrame,
    finite_reference: np.ndarray,
    ilines: np.ndarray,
    xlines: np.ndarray,
    twt_s: np.ndarray,
) -> list[list[tuple[int, int, int, float]]]:
    reference = np.asarray(finite_reference, dtype=np.float64)
    if reference.ndim != 3:
        raise ValueError(f"finite_reference must be [inline, xline, twt], got {reference.shape}")
    stencils: list[list[tuple[int, int, int, float]]] = []
    for _, row in samples.iterrows():
        i_float = _axis_index_float(ilines, float(row["inline"]))
        j_float = _axis_index_float(xlines, float(row["xline"]))
        i0 = int(np.floor(i_float))
        j0 = int(np.floor(j_float))
        i1 = i0 + 1
        j1 = j0 + 1
        if i0 < 0 or j0 < 0 or i1 >= reference.shape[0] or j1 >= reference.shape[1]:
            raise ValueError(f"Well sample lies outside R0 volume support: {row['well_name']}:{row['sample_index']}")
        wi = float(i_float - i0)
        wj = float(j_float - j0)
        corner_nodes: list[tuple[float, list[tuple[int, float]], int, int]] = []
        for i, j, spatial_weight in (
            (i0, j0, (1.0 - wi) * (1.0 - wj)),
            (i0, j1, (1.0 - wi) * wj),
            (i1, j0, wi * (1.0 - wj)),
            (i1, j1, wi * wj),
        ):
            if spatial_weight <= 0.0:
                continue
            time_nodes = _time_interpolation_nodes(
                trace=reference[i, j, :],
                twt_s=twt_s,
                sample_twt_s=float(row["twt_s"]),
            )
            if time_nodes is not None:
                corner_nodes.append((float(spatial_weight), time_nodes, i, j))
        spatial_sum = float(sum(item[0] for item in corner_nodes))
        if spatial_sum <= 0.0:
            raise ValueError(f"No finite R0 interpolation support: {row['well_name']}:{row['sample_index']}")
        combined: dict[tuple[int, int, int], float] = {}
        for spatial_weight, time_nodes, i, j in corner_nodes:
            for k, time_weight in time_nodes:
                key = (i, j, k)
                combined[key] = combined.get(key, 0.0) + spatial_weight / spatial_sum * time_weight
        stencil = [(i, j, k, weight) for (i, j, k), weight in sorted(combined.items())]
        if not np.isclose(sum(item[3] for item in stencil), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("Volume query stencil weights do not sum to one.")
        stencils.append(stencil)
    return stencils


def extract_role_well_features(
    *,
    samples: pd.DataFrame,
    predictions_path: Path,
    prediction_index_path: Path,
    checkpoint_path: Path,
    device_name: str,
    reconstruction_tolerance_log_ai: float,
) -> tuple[np.ndarray, pd.DataFrame, HeadState, dict[str, Any]]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    device, device_metadata = resolve_device(device_name)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    anchor = get_head_state(model)
    normalization = dict(checkpoint["normalization"])
    arrays = np.load(predictions_path, allow_pickle=False)
    required_arrays = {
        "pred_delta_vs_lfm",
        "stitching_weight",
        "lfm_input",
        "seismic_input",
        "valid_mask_model",
        "ilines",
        "xlines",
        "twt_s",
    }
    missing_arrays = sorted(required_arrays - set(arrays.files))
    if missing_arrays:
        raise ValueError(f"R0 predictions.npz missing arrays: {missing_arrays}")
    pred_delta = np.asarray(arrays["pred_delta_vs_lfm"], dtype=np.float32)
    stitching_weight = np.asarray(arrays["stitching_weight"], dtype=np.float32)
    seismic = np.asarray(arrays["seismic_input"], dtype=np.float32)
    lfm = np.asarray(arrays["lfm_input"], dtype=np.float32)
    valid = np.asarray(arrays["valid_mask_model"], dtype=bool)
    ilines = np.asarray(arrays["ilines"], dtype=np.float64)
    xlines = np.asarray(arrays["xlines"], dtype=np.float64)
    twt_s = np.asarray(arrays["twt_s"], dtype=np.float64)
    if not (pred_delta.shape == stitching_weight.shape == seismic.shape == lfm.shape == valid.shape):
        raise ValueError("R0 prediction arrays do not share one volume shape.")
    stencils = build_volume_query_stencils(
        samples=samples,
        finite_reference=pred_delta,
        ilines=ilines,
        xlines=xlines,
        twt_s=twt_s,
    )
    needed_by_inline: dict[int, set[tuple[int, int]]] = {}
    for stencil in stencils:
        for i, j, k, _ in stencil:
            needed_by_inline.setdefault(i, set()).add((j, k))
    index = pd.read_csv(prediction_index_path)
    required_index_columns = {
        "inline_index",
        "xline_start_index",
        "xline_stop_index",
        "twt_start",
        "twt_stop",
        "stitch_strategy",
    }
    missing_index = sorted(required_index_columns - set(index.columns))
    if missing_index:
        raise ValueError(f"R0 prediction_index.csv missing columns: {missing_index}")
    node_sum: dict[tuple[int, int, int], np.ndarray] = {}
    node_count: dict[tuple[int, int, int], float] = {}
    processed_patches = 0
    with torch.no_grad():
        for _, row in index.iterrows():
            inline_idx = int(row["inline_index"])
            required_nodes = needed_by_inline.get(inline_idx)
            if not required_nodes:
                continue
            patch = PatchGeometry(
                lateral_start=int(row["xline_start_index"]),
                lateral_stop=int(row["xline_stop_index"]),
                twt_start=int(row["twt_start"]),
                twt_stop=int(row["twt_stop"]),
                valid_fraction=float(row.get("valid_fraction", float("nan"))),
            )
            local, destination = _stitch_slices(patch, strategy=str(row["stitch_strategy"]))
            j0, j1 = int(destination[0].start), int(destination[0].stop)
            k0, k1 = int(destination[1].start), int(destination[1].stop)
            selected_nodes = [(j, k) for j, k in required_nodes if j0 <= j < j1 and k0 <= k < k1]
            if not selected_nodes:
                continue
            sl = (
                inline_idx,
                slice(patch.lateral_start, patch.lateral_stop),
                slice(patch.twt_start, patch.twt_stop),
            )
            seismic_patch = seismic[sl]
            lfm_patch = lfm[sl]
            valid_patch = valid[sl]
            inputs = np.stack(
                [
                    _normalize_with_mask(seismic_patch, valid_patch, normalization["seismic"]),
                    _normalize_with_mask(lfm_patch, valid_patch, normalization["lfm"]),
                    valid_patch.astype(np.float32),
                ],
                axis=0,
            )[None, ...]
            tensor = torch.as_tensor(inputs, dtype=torch.float32, device=device)
            feature_patch = model.forward_features(tensor).detach().cpu().numpy()[0]
            processed_patches += 1
            for j, k in selected_nodes:
                local_j = j - patch.lateral_start
                local_k = k - patch.twt_start
                if not bool(valid_patch[local_j, local_k]):
                    continue
                key = (inline_idx, j, k)
                values = feature_patch[:, local_j, local_k].astype(np.float64)
                node_sum[key] = node_sum.get(key, np.zeros(anchor.weight.size, dtype=np.float64)) + values
                node_count[key] = node_count.get(key, 0.0) + 1.0
    node_features: dict[tuple[int, int, int], np.ndarray] = {}
    for key, total in node_sum.items():
        count = node_count[key]
        expected = float(stitching_weight[key])
        if not np.isclose(count, expected, rtol=0.0, atol=1e-6):
            raise ValueError(f"Feature stitching count mismatch at {key}: reconstructed={count}, R0={expected}")
        node_features[key] = total / count
    features = np.empty((len(samples), anchor.weight.size), dtype=np.float64)
    for row_idx, stencil in enumerate(stencils):
        value = np.zeros(anchor.weight.size, dtype=np.float64)
        for i, j, k, weight in stencil:
            key = (i, j, k)
            if key not in node_features:
                raise ValueError(f"Missing reconstructed feature node: {key}")
            value += float(weight) * node_features[key]
        features[row_idx] = value
    reconstructed_delta = denormalize_delta_values(apply_head(features=features, head=anchor), normalization)
    csv_delta = samples["r0_delta_csv"].to_numpy(dtype=np.float64)
    error = np.abs(reconstructed_delta - csv_delta)
    detail = samples[
        [
            "model_role",
            "well_name",
            "sample_index",
            "spatial_cluster_id",
            "wellbore_class",
            "sample_method",
        ]
    ].copy()
    detail["r0_delta_csv"] = csv_delta
    detail["r0_delta_reconstructed"] = reconstructed_delta
    detail["feature_reconstruction_abs_error"] = error
    detail["feature_reconstruction_tolerance"] = float(reconstruction_tolerance_log_ai)
    detail["feature_reconstruction_status"] = np.where(
        error <= float(reconstruction_tolerance_log_ai), "ok", "r0_feature_reconstruction_mismatch"
    )
    if bool((error > float(reconstruction_tolerance_log_ai)).any()):
        worst = int(np.nanargmax(error))
        raise ValueError(
            "r0_feature_reconstruction_mismatch: "
            f"max={error[worst]:.9g}, well={samples.iloc[worst]['well_name']}, "
            f"sample={samples.iloc[worst]['sample_index']}"
        )
    metadata = {
        "n_samples": int(len(samples)),
        "n_query_nodes": int(len(node_features)),
        "n_processed_patches": int(processed_patches),
        "feature_channels": int(features.shape[1]),
        "feature_reconstruction_max_abs_error": float(np.max(error)),
        "feature_reconstruction_p99_abs_error": float(np.quantile(error, 0.99)),
        "feature_reconstruction_median_abs_error": float(np.median(error)),
        "device": device_metadata,
    }
    return features, detail, anchor, metadata


def feature_reconstruction_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_specs: list[tuple[str, list[str]]] = [
        ("global", ["model_role"]),
        ("cluster", ["model_role", "spatial_cluster_id"]),
        ("well", ["model_role", "well_name", "spatial_cluster_id"]),
    ]
    for scope, keys in group_specs:
        for values, group in detail.groupby(keys, dropna=False, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            error = group["feature_reconstruction_abs_error"].to_numpy(dtype=np.float64)
            row = {key: value for key, value in zip(keys, values)}
            row.update(
                {
                    "reconstruction_scope": scope,
                    "feature_reconstruction_n_samples": int(error.size),
                    "feature_reconstruction_max_abs_error": float(np.max(error)),
                    "feature_reconstruction_p99_abs_error": float(np.quantile(error, 0.99)),
                    "feature_reconstruction_median_abs_error": float(np.median(error)),
                    "feature_reconstruction_tolerance": float(group["feature_reconstruction_tolerance"].iloc[0]),
                    "feature_reconstruction_status": (
                        "ok"
                        if group["feature_reconstruction_status"].astype(str).eq("ok").all()
                        else "r0_feature_reconstruction_mismatch"
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def evaluate_synthetic_head(
    *,
    checkpoint_path: Path,
    prediction_dir: Path,
    root: Path,
    head: HeadState,
    device_name: str,
    batch_size: int,
) -> dict[str, float | int | str]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    set_head_state(model, head)
    device, _ = resolve_device(device_name)
    model.to(device)
    model.eval()
    index = pd.read_csv(prediction_dir / "prediction_index.csv")
    if index.empty:
        raise ValueError(f"Synthetic prediction index is empty: {prediction_dir}")
    manifest_path = prediction_dir / "prediction_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Synthetic prediction manifest missing: {manifest_path}")
    import json

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    benchmark_dir = Path(str(manifest["benchmark_dir"]))
    if not benchmark_dir.is_absolute():
        benchmark_dir = root / benchmark_dir
    benchmark = SynthoseisBenchmark(benchmark_dir)
    splits = sorted(set(index["split"].astype(str)))
    dataset = PatchDataset(benchmark, index, split=splits, normalization=checkpoint["normalization"])
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    lfms: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            tensor = batch["input"].to(device)
            pred_delta_n = model(tensor).detach().cpu().numpy()[:, 0]
            pred_delta = denormalize_delta(pred_delta_n, checkpoint["normalization"])
            lfm = batch["lfm"].numpy()[:, 0]
            predictions.append((lfm + pred_delta).astype(np.float32))
            targets.append(batch["target_log_ai"].numpy()[:, 0].astype(np.float32))
            masks.append(batch["valid_mask"].numpy()[:, 0].astype(bool))
            lfms.append(lfm.astype(np.float32))
    pred = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    mask = np.concatenate(masks, axis=0)
    lfm = np.concatenate(lfms, axis=0)
    if len(index) != pred.shape[0]:
        raise ValueError(f"Synthetic prediction row mismatch: index={len(index)}, arrays={pred.shape[0]}")
    patch_rows: list[dict[str, Any]] = []
    for row_idx in range(pred.shape[0]):
        metrics = regression_metrics(target[row_idx], pred[row_idx], valid_mask=mask[row_idx])
        patch_rows.append({**index.iloc[row_idx].to_dict(), **metrics})
    metrics_frame = pd.DataFrame.from_records(patch_rows)
    aggregate = _aggregate(metrics_frame)
    geometry = _geometry_patch_metrics(index, pred, target, mask, prediction_dir)
    geometry_aggregate = _aggregate_geometry(geometry)
    stitched = _stitch_realization_metrics(index, pred, target, mask, lfm)
    uniform = _aggregate(
        stitched["uniform"][stitched["uniform"]["model_id"].astype(str).ne("lfm_controlled_degraded")]
    )
    center = _aggregate(
        stitched["center_crop"][
            stitched["center_crop"]["model_id"].astype(str).ne("lfm_controlled_degraded")
        ]
    )
    return {
        "status": "ok",
        "n_patches": int(pred.shape[0]),
        "model_rmse": float(aggregate.get("mean_rmse", float("nan"))),
        "model_nrmse": float(aggregate.get("mean_nrmse", float("nan"))),
        "model_corr": float(aggregate.get("median_corr", float("nan"))),
        "geometry_boundary_rmse": float(geometry_aggregate.get("mean_boundary_rmse", float("nan"))),
        "geometry_event_rmse": float(geometry_aggregate.get("mean_event_rmse", float("nan"))),
        "geometry_lateral_gradient_rmse": float(
            geometry_aggregate.get("mean_lateral_gradient_rmse", float("nan"))
        ),
        "realization_uniform_rmse": float(uniform.get("mean_rmse", float("nan"))),
        "realization_uniform_nrmse": float(uniform.get("mean_nrmse", float("nan"))),
        "realization_uniform_corr": float(uniform.get("median_corr", float("nan"))),
        "realization_center_crop_rmse": float(center.get("mean_rmse", float("nan"))),
        "realization_center_crop_nrmse": float(center.get("mean_nrmse", float("nan"))),
        "realization_center_crop_corr": float(center.get("median_corr", float("nan"))),
    }


def evaluate_synthetic_heads(
    *,
    checkpoint_path: Path,
    prediction_dir: Path,
    root: Path,
    heads: Mapping[str, HeadState],
    device_name: str,
    batch_size: int,
) -> dict[str, dict[str, float | int | str]]:
    if not heads:
        raise ValueError("Synthetic preservation requires at least one head.")
    model, checkpoint = load_checkpoint(checkpoint_path)
    device, _ = resolve_device(device_name)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    anchor = get_head_state(model)
    for head_id, state in heads.items():
        if state.weight.size != anchor.weight.size:
            raise ValueError(f"Synthetic head channel mismatch for {head_id}.")
    head_ids = list(heads)
    weight = torch.as_tensor(
        np.stack([heads[head_id].weight for head_id in head_ids]),
        dtype=torch.float32,
        device=device,
    )
    bias = torch.as_tensor(
        [heads[head_id].bias for head_id in head_ids],
        dtype=torch.float32,
        device=device,
    )
    index = pd.read_csv(prediction_dir / "prediction_index.csv")
    if index.empty:
        raise ValueError(f"Synthetic prediction index is empty: {prediction_dir}")
    prediction_rows = pd.to_numeric(index.get("prediction_row"), errors="raise").to_numpy(dtype=int)
    if not np.array_equal(prediction_rows, np.arange(len(index), dtype=int)):
        raise ValueError(f"Synthetic prediction_index rows are not contiguous and ordered: {prediction_dir}")
    manifest_path = prediction_dir / "prediction_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Synthetic prediction manifest missing: {manifest_path}")
    import json

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    benchmark_dir = Path(str(manifest["benchmark_dir"]))
    if not benchmark_dir.is_absolute():
        benchmark_dir = root / benchmark_dir
    benchmark = SynthoseisBenchmark(benchmark_dir)
    splits = sorted(set(index["split"].astype(str)))
    dataset = PatchDataset(benchmark, index, split=splits, normalization=checkpoint["normalization"])
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    predictions: dict[str, list[np.ndarray]] = {head_id: [] for head_id in head_ids}
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    lfms: list[np.ndarray] = []
    delta_stats = dict(checkpoint["normalization"]["delta"])
    delta_mean = float(delta_stats["mean"])
    delta_std = float(delta_stats["std"])
    with torch.no_grad():
        for batch in loader:
            tensor = batch["input"].to(device)
            features = model.forward_features(tensor)
            pred_delta_n = torch.einsum("nh,bhlt->bnlt", weight, features) + bias[None, :, None, None]
            pred_delta = pred_delta_n * delta_std + delta_mean
            lfm_tensor = batch["lfm"].to(device)[:, 0]
            pred_log_ai = lfm_tensor[:, None] + pred_delta
            pred_numpy = pred_log_ai.detach().cpu().numpy()
            for head_index, head_id in enumerate(head_ids):
                predictions[head_id].append(pred_numpy[:, head_index].astype(np.float32))
            targets.append(batch["target_log_ai"].numpy()[:, 0].astype(np.float32))
            masks.append(batch["valid_mask"].numpy()[:, 0].astype(bool))
            lfms.append(batch["lfm"].numpy()[:, 0].astype(np.float32))
    target = np.concatenate(targets, axis=0)
    mask = np.concatenate(masks, axis=0)
    lfm = np.concatenate(lfms, axis=0)
    results: dict[str, dict[str, float | int | str]] = {}
    for head_id in head_ids:
        pred = np.concatenate(predictions[head_id], axis=0)
        patch_rows: list[dict[str, Any]] = []
        for row_idx in range(pred.shape[0]):
            metrics = regression_metrics(target[row_idx], pred[row_idx], valid_mask=mask[row_idx])
            patch_rows.append({**index.iloc[row_idx].to_dict(), **metrics})
        metrics_frame = pd.DataFrame.from_records(patch_rows)
        aggregate = _aggregate(metrics_frame)
        geometry = _geometry_patch_metrics(index, pred, target, mask, prediction_dir)
        geometry_aggregate = _aggregate_geometry(geometry)
        stitched = _stitch_realization_metrics(index, pred, target, mask, lfm)
        uniform = _aggregate(
            stitched["uniform"][
                stitched["uniform"]["model_id"].astype(str).ne("lfm_controlled_degraded")
            ]
        )
        center = _aggregate(
            stitched["center_crop"][
                stitched["center_crop"]["model_id"].astype(str).ne("lfm_controlled_degraded")
            ]
        )
        results[head_id] = {
            "status": "ok",
            "n_patches": int(pred.shape[0]),
            "model_rmse": float(aggregate.get("mean_rmse", float("nan"))),
            "model_nrmse": float(aggregate.get("mean_nrmse", float("nan"))),
            "model_corr": float(aggregate.get("median_corr", float("nan"))),
            "geometry_boundary_rmse": float(
                geometry_aggregate.get("mean_boundary_rmse", float("nan"))
            ),
            "geometry_event_rmse": float(geometry_aggregate.get("mean_event_rmse", float("nan"))),
            "geometry_lateral_gradient_rmse": float(
                geometry_aggregate.get("mean_lateral_gradient_rmse", float("nan"))
            ),
            "realization_uniform_rmse": float(uniform.get("mean_rmse", float("nan"))),
            "realization_uniform_nrmse": float(uniform.get("mean_nrmse", float("nan"))),
            "realization_uniform_corr": float(uniform.get("median_corr", float("nan"))),
            "realization_center_crop_rmse": float(center.get("mean_rmse", float("nan"))),
            "realization_center_crop_nrmse": float(center.get("mean_nrmse", float("nan"))),
            "realization_center_crop_corr": float(center.get("median_corr", float("nan"))),
        }
        del predictions[head_id]
    return results


def compare_synthetic_metrics(
    *,
    model_role: str,
    head_id: str,
    scope: str,
    r0_metrics: Mapping[str, Any],
    r2_metrics: Mapping[str, Any],
    maximum_error_relative_increase: float,
    maximum_corr_decrease: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = sorted((set(r0_metrics) & set(r2_metrics)) - {"status", "n_patches"})
    for metric in metric_names:
        r0_value = float(r0_metrics[metric])
        r2_value = float(r2_metrics[metric])
        if not np.isfinite(r0_value) or not np.isfinite(r2_value):
            status = "invalid_metric"
            absolute_change = float("nan")
            relative_change = float("nan")
            threshold = float("nan")
        elif metric.endswith("corr"):
            absolute_change = r2_value - r0_value
            relative_change = float("nan")
            threshold = -float(maximum_corr_decrease)
            status = "ok" if absolute_change >= threshold else "synthetic_preservation_failed"
        else:
            absolute_change = r2_value - r0_value
            relative_change = absolute_change / r0_value if r0_value > 0.0 else float("nan")
            threshold = float(maximum_error_relative_increase)
            status = (
                "ok"
                if np.isfinite(relative_change) and relative_change <= threshold
                else "synthetic_preservation_failed"
            )
        rows.append(
            {
                "model_role": model_role,
                "head_id": head_id,
                "synthetic_scope": scope,
                "metric": metric,
                "r0_value": r0_value,
                "r2_value": r2_value,
                "absolute_change": absolute_change,
                "relative_change": relative_change,
                "threshold": threshold,
                "status": status,
            }
        )
    return pd.DataFrame.from_records(rows)


def _band_component(
    values: np.ndarray, valid: np.ndarray, *, dt_s: float, low_hz: float, high_hz: float
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(data)
    prepared = np.zeros_like(data)
    if int(np.count_nonzero(mask)) >= 8:
        prepared = np.where(mask, data - float(np.mean(data[mask])), 0.0)
    spectrum = np.fft.rfft(prepared)
    frequencies = np.fft.rfftfreq(data.size, d=float(dt_s))
    selected = (frequencies >= float(low_hz)) & (frequencies < float(high_hz))
    filtered = np.zeros_like(spectrum)
    filtered[selected] = spectrum[selected]
    return np.fft.irfft(filtered, n=data.size)


def build_band_metrics(
    *,
    fold_metrics: pd.DataFrame,
    frame: pd.DataFrame,
    features: np.ndarray,
    heads: Mapping[str, HeadFit],
    anchor: HeadState,
    normalization: Mapping[str, Any],
    dt_s: float,
    diagnostic_max_hz: float,
    model_role: str,
) -> pd.DataFrame:
    nyquist = 0.5 / float(dt_s)
    high = min(float(diagnostic_max_hz), 0.45 * nyquist)
    bands = [
        ("lowfreq", 0.0, 0.2 * high),
        ("observable_band", 0.2 * high, 0.4 * high),
        ("highfreq_or_nullspace", 0.4 * high, high),
    ]
    rows: list[dict[str, Any]] = []
    r0_delta_all = denormalize_delta_values(apply_head(features=features, head=anchor), normalization)
    for head_id, fit in heads.items():
        if not (head_id.startswith("outer_loco_cluster_") or head_id.startswith("loo_well_")):
            continue
        if head_id.startswith("outer_loco_cluster_"):
            held_cluster = int(head_id.rsplit("_", maxsplit=1)[-1])
            held_mask = frame["spatial_cluster_id"].to_numpy(dtype=int) == held_cluster
            split_type = "leave_one_cluster_out"
        else:
            held_well = head_id.removeprefix("loo_well_")
            held_mask = frame["well_name"].astype(str).to_numpy() == held_well
            split_type = "leave_one_well_out"
        held_positions = np.flatnonzero(held_mask)
        held_frame = frame.iloc[held_positions]
        r2_delta = denormalize_delta_values(
            apply_head(features=features[held_positions], head=fit.head), normalization
        )
        r0_delta = r0_delta_all[held_positions]
        for well, local_positions in held_frame.groupby("well_name", sort=True).indices.items():
            local_idx = np.asarray(local_positions, dtype=int)
            local = held_frame.iloc[local_idx]
            order = np.argsort(local["sample_index"].to_numpy(dtype=int))
            local_idx = local_idx[order]
            local = held_frame.iloc[local_idx]
            well_delta = local["well_delta"].to_numpy(dtype=np.float64)
            lfm = local["lfm_log_ai"].to_numpy(dtype=np.float64)
            valid = np.isfinite(well_delta) & np.isfinite(lfm)
            for band_name, low_hz, high_hz in bands:
                signals = {
                    "well_delta": well_delta,
                    "r0_delta": r0_delta[local_idx],
                    "r2_delta": r2_delta[local_idx],
                    "well_full_ai": lfm + well_delta,
                    "r0_full_ai": lfm + r0_delta[local_idx],
                    "r2_full_ai": lfm + r2_delta[local_idx],
                }
                components = {
                    name: _band_component(values, valid, dt_s=dt_s, low_hz=low_hz, high_hz=high_hz)
                    for name, values in signals.items()
                }
                delta_r0 = _series_metrics(components["well_delta"], components["r0_delta"])
                delta_r2 = _series_metrics(components["well_delta"], components["r2_delta"])
                full_r0 = _series_metrics(components["well_full_ai"], components["r0_full_ai"])
                full_r2 = _series_metrics(components["well_full_ai"], components["r2_full_ai"])
                status = "ok" if all(
                    item.get("status") == "ok" for item in [delta_r0, delta_r2, full_r0, full_r2]
                ) else "invalid_metrics"
                rows.append(
                    {
                        "split_type": split_type,
                        "model_role": model_role,
                        "fold_id": head_id,
                        "held_out_well": str(well),
                        "held_out_spatial_cluster_id": int(local["spatial_cluster_id"].iloc[0]),
                        "band_name": band_name,
                        "low_hz": low_hz,
                        "high_hz": high_hz,
                        "n_samples": int(len(local)),
                        "status": status,
                        "r0_delta_corr": delta_r0.get("corr"),
                        "r2_delta_corr": delta_r2.get("corr"),
                        "delta_corr_gain": (
                            float(delta_r2["corr"] - delta_r0["corr"]) if status == "ok" else float("nan")
                        ),
                        "r0_full_ai_corr": full_r0.get("corr"),
                        "r2_full_ai_corr": full_r2.get("corr"),
                        "full_ai_corr_gain": (
                            float(full_r2["corr"] - full_r0["corr"]) if status == "ok" else float("nan")
                        ),
                        "r0_full_ai_rmse": full_r0.get("rmse"),
                        "r2_full_ai_rmse": full_r2.get("rmse"),
                        "full_ai_rmse_delta": (
                            float(full_r2["rmse"] - full_r0["rmse"]) if status == "ok" else float("nan")
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def build_role_decision(
    *,
    model_role: str,
    validation: RoleValidationResult,
    synthetic_preservation: pd.DataFrame,
    minimum_loco_clusters_for_decision: int,
    minimum_loco_cluster_improvement_fraction: float,
    minimum_median_delta_corr_gain: float,
) -> dict[str, Any]:
    loco = validation.cluster_metrics[
        validation.cluster_metrics.get("split_type", pd.Series(dtype=str)).astype(str).eq("leave_one_cluster_out")
    ].copy()
    k = int(validation.n_loco_clusters)
    minimum_improved = int(math.ceil(float(minimum_loco_cluster_improvement_fraction) * k))
    valid_loco = loco[loco["cluster_status"].astype(str).eq("ok")] if not loco.empty else loco
    improved = int(
        np.count_nonzero(
            pd.to_numeric(valid_loco.get("cluster_median_delta_corr_gain"), errors="coerce").to_numpy()
            > 0.0
        )
    ) if not valid_loco.empty else 0
    median_delta = (
        float(pd.to_numeric(valid_loco["cluster_median_delta_corr_gain"], errors="coerce").median())
        if not valid_loco.empty
        else float("nan")
    )
    median_full_corr = (
        float(pd.to_numeric(valid_loco["cluster_median_full_ai_corr_gain"], errors="coerce").median())
        if not valid_loco.empty
        else float("nan")
    )
    median_full_rmse = (
        float(pd.to_numeric(valid_loco["cluster_median_full_ai_rmse_delta"], errors="coerce").median())
        if not valid_loco.empty
        else float("nan")
    )
    required_head_ids = [f"outer_loco_cluster_{int(value)}" for value in sorted(loco["held_out_spatial_cluster_id"].unique())] if not loco.empty else []
    required_head_ids.append("all_wells")
    synthetic_ok = True
    if synthetic_preservation.empty:
        synthetic_ok = False
    else:
        for head_id in required_head_ids:
            scoped = synthetic_preservation[synthetic_preservation["head_id"].astype(str).eq(head_id)]
            if scoped.empty or not scoped["status"].astype(str).eq("ok").all():
                synthetic_ok = False
                break
    error_rows = synthetic_preservation[
        ~synthetic_preservation.get("metric", pd.Series(dtype=str)).astype(str).str.endswith("corr")
    ] if not synthetic_preservation.empty else pd.DataFrame()
    maximum_synthetic_error_increase = (
        float(pd.to_numeric(error_rows["relative_change"], errors="coerce").max())
        if not error_rows.empty
        else float("nan")
    )
    if k < int(minimum_loco_clusters_for_decision):
        decision = "inconclusive_insufficient_loco_clusters"
    elif validation.outer_status != "ok":
        if "no_transfer" in validation.outer_status:
            decision = "no_transfer_signal"
        elif "guardrail" in validation.outer_status:
            decision = "full_ai_guardrail_failed"
        else:
            decision = "inconclusive_invalid_fold"
    elif len(valid_loco) != k:
        decision = "inconclusive_invalid_fold"
    elif improved < minimum_improved or not median_delta > float(minimum_median_delta_corr_gain):
        decision = "no_transfer_signal"
    elif median_full_corr < 0.0 or median_full_rmse > 0.0:
        decision = "full_ai_guardrail_failed"
    elif validation.all_wells_status != "ok" or "all_wells" not in validation.heads:
        decision = "all_well_head_failed"
    else:
        decision = "r2_positive"
    return {
        "model_role": model_role,
        "n_loco_clusters": k,
        "n_loco_folds_valid": int(len(valid_loco)),
        "minimum_loco_clusters_for_decision": int(minimum_loco_clusters_for_decision),
        "minimum_loco_cluster_improvement_fraction": float(minimum_loco_cluster_improvement_fraction),
        "minimum_improved_clusters": minimum_improved,
        "n_clusters_delta_corr_improved": improved,
        "improved_cluster_fraction": float(improved / k) if k else float("nan"),
        "median_delta_corr_gain": median_delta,
        "minimum_median_delta_corr_gain": float(minimum_median_delta_corr_gain),
        "median_full_ai_corr_gain": median_full_corr,
        "median_full_ai_rmse_delta": median_full_rmse,
        "real_loco_decision": decision,
        "synthetic_preservation_status": "ok" if synthetic_ok else "warning",
        "synthetic_preservation_warning": "" if synthetic_ok else "synthetic_domain_metrics_exceed_diagnostic_thresholds",
        "maximum_synthetic_error_relative_increase_observed": maximum_synthetic_error_increase,
        "all_well_head_status": validation.all_wells_status,
        "decision": decision,
        "decision_reason": decision,
        "eligible_for_r3": decision == "r2_positive",
    }
