"""Composable sources, losses, and stage runner for GINN-v2 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from cup.synthetic.dataset import SynthoseisBenchmark
from cup.utils.io import repo_relative_path, write_json
from ginn_v2.contracts import CHECKPOINT_SCHEMA_VERSION, EXPERIMENT_SCHEMA_VERSION
from ginn_v2.data import PatchDataset, PatchSpec, build_patch_index, compute_normalization
from ginn_v2.experiment import ExperimentConfig
from ginn_v2.models import build_model
from ginn_v2.real_field import RealFieldVolume, build_real_field_patch_index, load_real_field_volume
from ginn_v2.real_delta import prepare_real_delta_support
from ginn_v2.training import _forward_physics_batch, _load_nominal_wavelet, masked_mse, resolve_device


@dataclass
class PreparedBlock:
    config: dict[str, Any]
    train_dataset: Dataset[Any] | None
    validation_dataset: Dataset[Any] | None
    wavelet: tuple[torch.Tensor, torch.Tensor, dict[str, float] | None] | None = None
    real_well_support: Any | None = None


class RealFieldPatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self, model_volume: RealFieldVolume, physics_volume: RealFieldVolume,
        rows: pd.DataFrame, normalization: Mapping[str, Any], patch_spec: PatchSpec,
    ) -> None:
        if model_volume.lfm.shape != physics_volume.lfm.shape:
            raise ValueError("Real-field model-input and physics-target volumes differ in shape.")
        self.model_volume = model_volume
        self.physics_volume = physics_volume
        self.rows = rows.reset_index(drop=True)
        self.normalization = normalization
        self.patch_spec = patch_spec

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[int(index)]
        inline = int(row["inline_index"])
        lateral = slice(int(row["lateral_start"]), int(row["lateral_stop"]))
        vertical = slice(int(row["vertical_start"]), int(row["vertical_stop"]))
        seismic = self.model_volume.seismic[inline, lateral, vertical]
        target = self.physics_volume.seismic[inline, lateral, vertical]
        lfm = self.model_volume.lfm[inline, lateral, vertical]
        valid = self.model_volume.valid_mask[inline, lateral, vertical]
        shape = (self.patch_spec.lateral_samples, self.patch_spec.twt_samples)
        padding = ((0, shape[0] - valid.shape[0]), (0, shape[1] - valid.shape[1]))
        seismic = np.pad(seismic, padding, constant_values=0.0)
        target = np.pad(target, padding, constant_values=0.0)
        lfm = np.pad(lfm, padding, constant_values=0.0)
        valid = np.pad(valid, padding, constant_values=False)
        seismic_n = np.where(valid, (seismic - float(self.normalization["seismic"]["mean"])) / float(self.normalization["seismic"]["std"]), 0.0)
        lfm_n = np.where(valid, (lfm - float(self.normalization["lfm"]["mean"])) / float(self.normalization["lfm"]["std"]), 0.0)
        inputs = np.stack([seismic_n, lfm_n, valid.astype(np.float32)], axis=0).astype(np.float32)
        axis = self.model_volume.sample_axis.values[vertical]
        if len(axis) < shape[1]:
            step = self.model_volume.sample_axis.step
            axis = np.concatenate([axis, axis[-1] + step * np.arange(1, shape[1] - len(axis) + 1)])
        return {
            "input": torch.from_numpy(inputs),
            "lfm": torch.from_numpy(np.where(valid, lfm, 0.0).astype(np.float32))[None],
            "valid_mask": torch.from_numpy(valid.astype(np.float32))[None],
            "physics_valid_mask": torch.from_numpy(valid.astype(np.float32))[None],
            "seismic_model_consistent": torch.from_numpy(np.where(valid, target, 0.0).astype(np.float32))[None],
            "sample_axis": torch.from_numpy(np.asarray(axis, dtype=np.float64)),
            "patch_id": str(row["patch_id"]),
        }


class DeterministicCycler:
    def __init__(self, dataset: Dataset[Any], *, batch_size: int, seed: int) -> None:
        if len(dataset) == 0:
            raise ValueError("Loss block dataset is empty.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        rng = np.random.default_rng(int(seed))
        self.order = rng.permutation(len(dataset)).tolist()
        self.position = 0

    def next(self) -> Any:
        indices = []
        for _ in range(self.batch_size):
            indices.append(self.order[self.position])
            self.position = (self.position + 1) % len(self.order)
        return next(iter(DataLoader(Subset(self.dataset, indices), batch_size=len(indices), shuffle=False)))


def _seed(base: int, stage_index: int, block_index: int, epoch: int) -> int:
    return int(np.random.SeedSequence([base, stage_index, block_index, epoch]).generate_state(1)[0])


def _resolve_auto_benchmark(root: Path) -> Path:
    results = root / "experiments" / "synthoseis_lite" / "results"
    candidates = [
        path / "generate_field_conditioned"
        for path in results.glob("*")
        if (path / "generate_field_conditioned" / "benchmark_manifest.json").is_file()
        and (path / "generate_field_conditioned" / "sample_index.csv").is_file()
        and (path / "generate_field_conditioned" / "synthetic_benchmark.h5").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No complete Synthoseis-lite benchmark under {results}.")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _source_path(value: Any, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _synthetic_sample_kinds(source: Mapping[str, Any]) -> set[str]:
    variant = str(source.get("input_seismic_variant") or "nominal")
    if variant == "nominal":
        return {"base"}
    if variant == "observed_mismatch":
        return {"base", "seismic_variant"}
    raise ValueError(f"Unsupported input_seismic_variant: {variant!r}")


def _prepare_synthetic_indices(
    *, benchmark: SynthoseisBenchmark, source: Mapping[str, Any], patch_spec: PatchSpec,
    blocks: list[Mapping[str, Any]], output_dir: Path,
) -> dict[str, pd.DataFrame]:
    result = {}
    for block in blocks:
        block_id = str(block["block_id"])
        index = build_patch_index(
            benchmark,
            patch_spec=patch_spec,
            sample_kinds=_synthetic_sample_kinds(source),
            min_valid_samples=int(block["min_valid_samples"]),
            max_patches=(int(source["max_patches"]) if source.get("max_patches") is not None else None),
        )
        if block["kind"] == "physics":
            # Physics target is defined only where its final target mask is valid.
            counts = []
            cache: dict[str, Any] = {}
            for _, row in index.iterrows():
                sample_id = str(row["sample_id"])
                sample = cache.setdefault(sample_id, benchmark.load_sample(sample_id))
                mask = np.asarray(sample.physics_valid_mask, dtype=bool)[
                    int(row["lateral_start"]):int(row["lateral_stop"]),
                    int(row["twt_start"]):int(row["twt_stop"]),
                ]
                counts.append(int(np.count_nonzero(mask)))
            index["supervision_valid_samples"] = counts
            index = index[index["supervision_valid_samples"] >= int(block["min_valid_samples"])].copy()
        if source.get("max_patches") is not None and "validation" not in set(index["split"].astype(str)):
            validation = index.head(min(4, len(index))).copy()
            validation["patch_id"] = validation["patch_id"].astype(str) + "__smoke_validation"
            validation["split"] = "validation"
            index = pd.concat([index, validation], ignore_index=True)
        path = output_dir / f"{block_id}_patch_index.csv"
        index.to_csv(path, index=False)
        result[block_id] = index
    return result


def _normalization_for_synthetic(benchmark: SynthoseisBenchmark, index: pd.DataFrame) -> dict[str, Any]:
    normalization = compute_normalization(benchmark, index)
    if "delta" in normalization:
        raise AssertionError("Composable normalization must not contain delta statistics.")
    return normalization


def _real_field_config(source: Mapping[str, Any], *, root: Path, transform: str) -> dict[str, Any]:
    lfm_run_dir = _source_path(source.get("lfm_run_dir"), root)
    with (lfm_run_dir / "lfm_run_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    seismic = dict(summary.get("seismic") or {})
    return {
        "real_field_inputs": {
            "lfm_run_dir": str(lfm_run_dir),
            "variant_id": str(source.get("variant_id") or ""),
            "well_control_run_dir": str(_source_path(source.get("well_control_run_dir"), root)),
            "seismic_file": str(_source_path(seismic.get("path"), root)),
            "seismic_type": str(seismic.get("type") or ""),
            "seismic_value_transform": transform,
            "lfm_value_transform": "identity",
        },
        "volume": dict(source.get("volume") or {}),
    }


def _load_real_source(source: Mapping[str, Any], *, root: Path) -> tuple[RealFieldVolume, RealFieldVolume]:
    model_transform = str(source.get("model_input_seismic_transform") or "identity")
    model_volume = load_real_field_volume(config=_real_field_config(source, root=root, transform=model_transform), root=root, data_root=root)
    if model_transform in {"identity", "raw", "none"}:
        return model_volume, model_volume
    physics_volume = load_real_field_volume(config=_real_field_config(source, root=root, transform=str(source.get("physics_target_seismic_transform") or "identity")), root=root, data_root=root)
    return model_volume, physics_volume


def _normalization_for_real(volume: RealFieldVolume) -> dict[str, Any]:
    valid = volume.valid_mask
    result: dict[str, Any] = {"normalization_scope": "reference_source_train_partition", "normalization_mask": "valid_mask_model"}
    for name, values in (("seismic", volume.seismic), ("lfm", volume.lfm)):
        selected = np.asarray(values, dtype=np.float64)[valid]
        std = float(np.std(selected))
        if selected.size < 2 or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Real normalization reference has invalid {name} values.")
        result[name] = {"mean": float(np.mean(selected)), "std": std}
    return result


def _real_patch_rows(
    volume: RealFieldVolume, *, patch_spec: PatchSpec, min_valid_samples: int,
    validation: Mapping[str, Any],
) -> pd.DataFrame:
    fraction = float(validation.get("fraction", 0.1))
    if not 0 < fraction < 1:
        raise ValueError("real_field.validation_split.fraction must be between zero and one.")
    n_validation = max(1, int(np.ceil(volume.ilines.size * fraction)))
    gap_m = float(validation.get("gap_m", 0.0))
    spatial_step_m = float(validation.get("spatial_step_m", 25.0))
    gap_rows = int(np.ceil(gap_m / spatial_step_m))
    validation_start = volume.ilines.size - n_validation
    training_stop = max(0, validation_start - gap_rows)
    rows: list[dict[str, Any]] = []
    for inline_index in range(volume.ilines.size):
        split = "validation" if inline_index >= validation_start else "train" if inline_index < training_stop else "gap"
        if split == "gap":
            continue
        for patch_index, patch in enumerate(build_real_field_patch_index(
            volume.valid_mask[inline_index], lateral_samples=patch_spec.lateral_samples,
            vertical_samples=patch_spec.twt_samples, lateral_stride=patch_spec.lateral_stride,
            vertical_stride=patch_spec.twt_stride,
        )):
            mask = volume.valid_mask[inline_index, patch.lateral_start:patch.lateral_stop, patch.sample_start:patch.sample_stop]
            count = int(np.count_nonzero(mask))
            if count < int(min_valid_samples):
                continue
            rows.append({
                "patch_id": f"il{inline_index:05d}_p{patch_index:06d}", "split": split,
                "inline_index": inline_index, "lateral_start": patch.lateral_start,
                "lateral_stop": patch.lateral_stop, "vertical_start": patch.sample_start,
                "vertical_stop": patch.sample_stop, "supervision_valid_samples": count,
            })
    frame = pd.DataFrame(rows)
    if frame.empty or not {"train", "validation"}.issubset(set(frame["split"])):
        raise ValueError("Real-field spatial split did not produce both train and validation patches.")
    return frame


def _load_real_wavelet(
    source: Mapping[str, Any], volume: RealFieldVolume, *, root: Path, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
    if volume.sample_axis.domain == "time":
        directory = _source_path(source.get("wavelet_generation_dir"), root)
        path = directory / "selected_wavelet.csv"
        relation = None
    else:
        contract_path = _source_path(source.get("forward_model_inputs_path"), root)
        if not contract_path.is_file():
            raise FileNotFoundError(
                "Depth real-field physics requires source.forward_model_inputs_path."
            )
        with contract_path.open("r", encoding="utf-8") as handle:
            contract = json.load(handle)
        wavelet_ref = str(dict(contract.get("wavelet") or {}).get("path") or "")
        path = _source_path(wavelet_ref, root)
        raw_relation = dict(contract.get("ai_velocity_relation") or {})
        relation = {"a": float(raw_relation["a"]), "b": float(raw_relation["b"])}
    frame = pd.read_csv(path)
    return (
        torch.as_tensor(frame["time_s"].to_numpy(), dtype=torch.float32, device=device),
        torch.as_tensor(frame["amplitude"].to_numpy(), dtype=torch.float32, device=device),
        relation,
    )


def masked_centered_rms(
    values: torch.Tensor, mask: torch.Tensor, *, epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return item-wise standardized tensors and centered RMS values."""
    if values.shape != mask.shape:
        raise ValueError("values/mask shape mismatch for centered RMS.")
    flat_x = values.reshape(values.shape[0], -1)
    flat_m = (mask > 0.5).reshape(mask.shape[0], -1)
    count = flat_m.sum(dim=1)
    if bool(torch.any(count == 0)):
        raise ValueError("Centered RMS item has no valid waveform samples.")
    weights = flat_m.to(values.dtype)
    mean = (flat_x * weights).sum(dim=1) / count
    centered = flat_x - mean[:, None]
    rms = torch.sqrt((centered.square() * weights).sum(dim=1) / count + float(epsilon))
    standardized = torch.where(flat_m, centered / rms[:, None], torch.zeros_like(centered))
    return standardized.reshape_as(values), rms


def forward_support_safe_mask(mask: torch.Tensor, wavelet: torch.Tensor) -> torch.Tensor:
    """Conservatively require a complete valid segment across active wavelet support."""
    active = torch.nonzero(torch.abs(wavelet) >= 0.05 * torch.max(torch.abs(wavelet)), as_tuple=False).flatten()
    if active.numel() == 0:
        raise ValueError("Wavelet has no active support at threshold 0.05.")
    half = int(torch.max(torch.abs(active - wavelet.numel() // 2)).item())
    if half == 0:
        return mask > 0.5
    width = 2 * half + 1
    flat = (mask > 0.5).to(torch.float32).reshape(-1, 1, mask.shape[-1])
    counts = F.conv1d(flat, torch.ones((1, 1, width), device=mask.device), padding=half)
    return (counts == width).reshape_as(mask)


def _block_loss(
    prepared: PreparedBlock, batch: Any, model: torch.nn.Module, *, device: torch.device,
    normalization: Mapping[str, Any], sample_domain: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = prepared.config
    kind = str(cfg["kind"])
    if kind == "real_well_supervised":
        loss, counts = prepared.real_well_support.training_loss(model, device=device)
        return loss, {key: float(value) for key, value in counts.items()}
    x = batch["input"].to(device)
    prediction = model(x)
    if kind == "synthetic_supervised":
        return masked_mse(prediction, batch["target_delta"].to(device), batch["valid_mask"].to(device)), {}
    if kind != "physics" or prepared.wavelet is None:
        raise ValueError(f"Unsupported prepared loss block: {kind}")
    lfm = batch["lfm"].to(device)
    valid = batch["valid_mask"].to(device)
    pred_log_ai = lfm + prediction
    wavelet_time, wavelet_amp, relation = prepared.wavelet
    synthetic = _forward_physics_batch(
        pred_log_ai[:, 0], sample_axes=batch["sample_axis"].to(device),
        sample_domain=sample_domain, wavelet_time_s=wavelet_time,
        wavelet_amp=wavelet_amp, ai_velocity_relation=relation,
    )
    target = batch["seismic_model_consistent"].to(device)[:, 0]
    mask = forward_support_safe_mask(batch["physics_valid_mask"].to(device)[:, 0], wavelet_amp)
    enough_waveform = mask.reshape(mask.shape[0], -1).sum(dim=1) >= 2
    if not bool(torch.any(enough_waveform)):
        raise ValueError("Physics batch has no item with two forward-support-safe waveform samples.")
    synthetic = synthetic[enough_waveform]
    target = target[enough_waveform]
    mask = mask[enough_waveform]
    prediction_for_l2 = prediction[enough_waveform]
    valid_for_l2 = valid[enough_waveform]
    if str(cfg.get("source_kind")) == "real_field":
        pred_n, pred_rms = masked_centered_rms(synthetic, mask, epsilon=float(cfg.get("centered_rms_epsilon", 1e-12)))
        target_n, target_rms = masked_centered_rms(target, mask, epsilon=float(cfg.get("centered_rms_epsilon", 1e-12)))
        minimum = float(cfg.get("min_centered_rms", 1e-6))
        usable = (pred_rms >= minimum) & (target_rms >= minimum)
        if not bool(torch.any(usable)):
            raise ValueError("Real physics batch has no item above min_centered_rms.")
        waveform = masked_mse(pred_n[usable], target_n[usable], mask[usable])
        diagnostics = {
            "observed_centered_rms": float(target_rms[usable].mean().detach().cpu()),
            "predicted_centered_rms": float(pred_rms[usable].mean().detach().cpu()),
            "predicted_to_observed_rms_ratio": float((pred_rms[usable] / target_rms[usable]).mean().detach().cpu()),
        }
    else:
        waveform = masked_mse(synthetic, target, mask)
        diagnostics = {}
    delta_l2 = masked_mse(prediction_for_l2, torch.zeros_like(prediction_for_l2), valid_for_l2)
    diagnostics["waveform"] = float(waveform.detach().cpu())
    diagnostics["delta_l2"] = float(delta_l2.detach().cpu())
    return waveform + float(cfg["delta_l2_weight"]) * delta_l2, diagnostics


def _checkpoint_payload(
    *, model: torch.nn.Module, config: ExperimentConfig, model_info: Mapping[str, Any],
    normalization: Mapping[str, Any], stage_id: str, kind: str, epoch: int,
    metric_name: str, metric_value: float, sample_axis: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "stage_id": stage_id,
        "checkpoint_kind": kind,
        "epoch": int(epoch),
        "selection_metric": metric_name,
        "selection_metric_value": float(metric_value),
        "architecture": config.architecture.as_dict(),
        "model_info": dict(model_info),
        "input_channels": ["seismic", "lfm", "valid_mask"],
        "output_semantics": "physical_delta_log_ai",
        "normalization": dict(normalization),
        "sample_axis_contract": dict(sample_axis),
        "patch_deployment_contract": config.patching.as_dict() | {
            "axis_end_rule": "append_axis_length_minus_window",
            "short_axis_padding": "right",
            "padding_values": [0.0, 0.0, 0.0],
            "stitching": "uniform",
        },
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def _evaluate(
    *, prepared: PreparedBlock, model: torch.nn.Module, device: torch.device,
    normalization: Mapping[str, Any], sample_domain: str, steps: int | None,
) -> dict[str, float]:
    if prepared.config["kind"] == "real_well_supervised":
        support = prepared.real_well_support
        held_out = support.samples[
            support.samples["supervision_role"].astype(str).str.startswith("held_out")
            & support.samples["valid_for_fit"].astype(bool)
        ]
        if held_out.empty:
            raise ValueError("real_well_supervised validation has no held-out well samples.")
        groups = [group.sort_values("sample_index") for _, group in held_out.groupby("well_name", sort=True)]
        with torch.no_grad():
            predictions = support.predictor.predict_delta_n_groups(
                model, groups, device=device, canonical_full_patch=support.canonical_full_patch,
            )
        losses = []
        for rows, prediction in zip(groups, predictions):
            target = torch.as_tensor(
                rows["well_log_ai"].to_numpy(dtype=np.float32) - rows["lfm_log_ai"].to_numpy(dtype=np.float32),
                device=device,
            )
            losses.append(torch.mean((prediction - target) ** 2))
        return {"mse": float(torch.stack(losses).mean().cpu())}
    assert prepared.validation_dataset is not None
    cycler = DeterministicCycler(prepared.validation_dataset, batch_size=int(prepared.config["batch_size"]), seed=0)
    count = int(steps) if steps is not None else int(np.ceil(len(prepared.validation_dataset) / int(prepared.config["batch_size"])))
    rows: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(count):
            loss, _ = _block_loss(prepared, cycler.next(), model, device=device, normalization=normalization, sample_domain=sample_domain)
            rows.append(float(loss.detach().cpu()))
    key = "mse" if prepared.config["kind"] != "physics" else "total"
    return {key: float(np.mean(rows))}


def run_experiment(
    *, config: ExperimentConfig, root: Path, output_dir: Path, logger: logging.Logger,
) -> dict[str, Any]:
    """Run the strict v1 experiment. Synthetic blocks and existing real-well support are composable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "experiment_manifest.json").exists() or (output_dir / "stages").exists():
        raise FileExistsError(f"GINN-v2 experiment outputs already exist: {output_dir}")
    device, device_metadata = resolve_device(config.device)
    model, info = build_model(**({
        "architecture_id": config.architecture.id,
        "hidden_channels": config.architecture.hidden_channels,
        "depth": config.architecture.depth,
        "lateral_kernel": config.architecture.lateral_kernel,
    }))
    model.to(device)
    patch_spec = PatchSpec(
        lateral_samples=config.patching.lateral_samples,
        twt_samples=config.patching.vertical_samples,
        lateral_stride=config.patching.lateral_stride,
        twt_stride=config.patching.vertical_stride,
    )
    synthetic: dict[str, tuple[SynthoseisBenchmark, Path]] = {}
    real_fields: dict[str, tuple[RealFieldVolume, RealFieldVolume]] = {}
    for source_id, source in config.sources.items():
        if source["kind"] == "synthoseis_lite":
            raw_dir = source.get("benchmark_dir")
            benchmark_dir = _resolve_auto_benchmark(root) if str(raw_dir).casefold() == "auto" else _source_path(raw_dir, root)
            synthetic[source_id] = (SynthoseisBenchmark(benchmark_dir), benchmark_dir)
        elif source["kind"] == "real_field":
            real_fields[source_id] = _load_real_source(source, root=root)
    if config.normalization_reference in synthetic:
        reference_benchmark, _ = synthetic[config.normalization_reference]
        reference_blocks = [
            block for stage in config.stages for block in stage["loss_blocks"]
            if block["source"] == config.normalization_reference
        ]
        reference_indices = _prepare_synthetic_indices(
            benchmark=reference_benchmark,
            source=config.sources[config.normalization_reference],
            patch_spec=patch_spec, blocks=reference_blocks, output_dir=output_dir,
        )
        normalization = _normalization_for_synthetic(reference_benchmark, next(iter(reference_indices.values())))
    else:
        normalization = _normalization_for_real(real_fields[config.normalization_reference][0])
    write_json(output_dir / "normalization.json", normalization)
    input_stats_path = output_dir / "input_reference_stats.json"
    write_json(input_stats_path, {"stats": dict(normalization["seismic"])})
    manifest_stages: list[dict[str, Any]] = []
    checkpoints: dict[tuple[str, str], Path] = {}
    sample_axis_contract: dict[str, Any] | None = None
    for stage_index, stage in enumerate(config.stages):
        stage_id = str(stage["stage_id"])
        stage_dir = output_dir / "stages" / stage_id
        stage_dir.mkdir(parents=True, exist_ok=False)
        initialize = str(stage["initialize_from"])
        if initialize != "zero":
            parent_stage, parent_kind = initialize.rsplit(".", 1)
            payload = torch.load(checkpoints[(parent_stage, parent_kind)], map_location="cpu", weights_only=False)
            model.load_state_dict(payload["state_dict"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(stage["optimizer"]["learning_rate"]),
            weight_decay=float(stage["optimizer"]["weight_decay"]),
        )
        prepared_blocks: list[PreparedBlock] = []
        for block in stage["loss_blocks"]:
            source_id = str(block["source"])
            source = config.sources[source_id]
            cfg = dict(block) | {"source_kind": source["kind"]}
            if source["kind"] == "synthoseis_lite":
                benchmark, benchmark_dir = synthetic[source_id]
                indices = _prepare_synthetic_indices(
                    benchmark=benchmark, source=source, patch_spec=patch_spec,
                    blocks=[block], output_dir=stage_dir,
                )[str(block["block_id"])]
                train_dataset = PatchDataset(benchmark, indices, split="train", normalization=normalization)
                validation_dataset = PatchDataset(benchmark, indices, split="validation", normalization=normalization)
                wavelet = _load_nominal_wavelet(benchmark_dir, device=device) if block["kind"] == "physics" else None
                prepared_blocks.append(PreparedBlock(cfg, train_dataset, validation_dataset, wavelet))
                if sample_axis_contract is None:
                    domain = str(benchmark.manifest.get("sample_domain"))
                    sample_axis_contract = {"sample_domain": domain, "sample_unit": "s" if domain == "time" else "m", "depth_basis": benchmark.manifest.get("depth_basis")}
            elif source["kind"] == "real_field":
                model_volume, physics_volume = real_fields[source_id]
                rows = _real_patch_rows(
                    model_volume, patch_spec=patch_spec,
                    min_valid_samples=int(block["min_valid_samples"]),
                    validation=dict(source.get("validation_split") or {}),
                )
                rows.to_csv(stage_dir / f"{block['block_id']}_patch_index.csv", index=False)
                train_dataset = RealFieldPatchDataset(model_volume, physics_volume, rows[rows["split"] == "train"], normalization, patch_spec)
                validation_dataset = RealFieldPatchDataset(model_volume, physics_volume, rows[rows["split"] == "validation"], normalization, patch_spec)
                wavelet = _load_real_wavelet(source, model_volume, root=root, device=device)
                prepared_blocks.append(PreparedBlock(cfg, train_dataset, validation_dataset, wavelet))
                if sample_axis_contract is None:
                    sample_axis_contract = {"sample_domain": model_volume.sample_axis.domain, "sample_unit": model_volume.sample_axis.unit, "depth_basis": model_volume.depth_basis}
            else:
                field_source_id = str(source["field_source"])
                if field_source_id not in real_fields:
                    raise ValueError(f"real_wells source {source_id} references missing real_field source {field_source_id}.")
                support_cfg = {
                    "lfm_run_dir": config.sources[field_source_id]["lfm_run_dir"],
                    "variant_id": config.sources[field_source_id]["variant_id"],
                    "well_control_run_dir": source["well_control_run_dir"],
                    "held_out_well": source["held_out_well"],
                    "exclude_same_cluster": bool(source.get("exclude_same_cluster", True)),
                    "clusters_per_step": int(block["batch_size"]),
                    "cluster_radius_m": float(source.get("cluster_radius_m", 500.0)),
                    "diagnostic_max_hz": float(source.get("diagnostic_max_hz", 80.0)),
                    "reconstruction_tolerance_log_ai": float(source.get("reconstruction_tolerance_log_ai", 1e-5)),
                    "seismic_value_transform": str(config.sources[field_source_id].get("model_input_seismic_transform") or "identity"),
                    "lfm_value_transform": "identity",
                }
                support = prepare_real_delta_support(
                    config=support_cfg, repo_root=root, output_dir=stage_dir,
                    normalization=normalization, patch_spec=config.patching.as_dict(),
                    input_reference_stats_path=input_stats_path, lambda_real_delta=1.0,
                    seed=_seed(config.seed, stage_index, len(prepared_blocks), 0), logger=logger,
                )
                support.configure_model(receptive_field_lateral=info.lateral_receptive_field)
                prepared_blocks.append(PreparedBlock(cfg, None, None, real_well_support=support))
                volume = real_fields[field_source_id][0]
                if sample_axis_contract is None:
                    sample_axis_contract = {"sample_domain": volume.sample_axis.domain, "sample_unit": volume.sample_axis.unit, "depth_basis": volume.depth_basis}
        best_value = float("inf")
        best_epoch = 0
        history: list[dict[str, Any]] = []
        metric_name = str(stage["validation"]["selection_metric"])
        for epoch in range(1, int(stage["epochs"]) + 1):
            cyclers = [
                (DeterministicCycler(block.train_dataset, batch_size=int(block.config["batch_size"]), seed=_seed(config.seed, stage_index, index, epoch)) if block.train_dataset is not None else None)
                for index, block in enumerate(prepared_blocks)
            ]
            model.train()
            totals = {str(block.config["block_id"]): [] for block in prepared_blocks}
            batch_counts = {str(block.config["block_id"]): 0 for block in prepared_blocks}
            for step in range(int(stage["steps_per_epoch"])):
                optimizer.zero_grad(set_to_none=True)
                combined: torch.Tensor | None = None
                for block_index, prepared in enumerate(prepared_blocks):
                    if step % int(prepared.config["update_interval"]) != 0:
                        continue
                    batch = cyclers[block_index].next() if cyclers[block_index] is not None else None
                    loss, _ = _block_loss(prepared, batch, model, device=device, normalization=normalization, sample_domain=str(sample_axis_contract["sample_domain"]))
                    weighted = float(prepared.config["weight"]) * loss
                    combined = weighted if combined is None else combined + weighted
                    totals[str(prepared.config["block_id"])].append(float(loss.detach().cpu()))
                    batch_counts[str(prepared.config["block_id"])] += 1
                if combined is None or not bool(torch.isfinite(combined)):
                    raise FloatingPointError(f"Invalid combined loss at {stage_id} epoch={epoch} step={step}.")
                combined.backward()
                optimizer.step()
            validation_values: dict[str, float] = {}
            validation_steps = stage["validation"].get("steps") if stage["validation"]["mode"] == "fixed_steps" else None
            for prepared in prepared_blocks:
                metrics = _evaluate(prepared=prepared, model=model, device=device, normalization=normalization, sample_domain=str(sample_axis_contract["sample_domain"]), steps=validation_steps)
                for name, value in metrics.items():
                    validation_values[f"{prepared.config['block_id']}.{name}"] = value
            selected = float(validation_values[metric_name])
            row = {"epoch": epoch, "selection_metric": metric_name, "selection_metric_value": selected, "batch_counts": batch_counts, **validation_values}
            history.append(row)
            if selected < best_value:
                best_value, best_epoch = selected, epoch
                path = stage_dir / "checkpoint_best.pt"
                torch.save(_checkpoint_payload(model=model, config=config, model_info=info.__dict__, normalization=normalization, stage_id=stage_id, kind="best", epoch=epoch, metric_name=metric_name, metric_value=selected, sample_axis=sample_axis_contract), path)
                checkpoints[(stage_id, "best")] = path
        final_path = stage_dir / "checkpoint_final.pt"
        torch.save(_checkpoint_payload(model=model, config=config, model_info=info.__dict__, normalization=normalization, stage_id=stage_id, kind="final", epoch=int(stage["epochs"]), metric_name=metric_name, metric_value=float(history[-1]["selection_metric_value"]), sample_axis=sample_axis_contract), final_path)
        checkpoints[(stage_id, "final")] = final_path
        pd.DataFrame(history).to_csv(stage_dir / "training_history.csv", index=False)
        manifest_stages.append({
            **stage, "best_epoch": best_epoch, "best_selection_metric_value": best_value,
            "checkpoints": {kind: repo_relative_path(checkpoints[(stage_id, kind)], root=root) for kind in ("best", "final")},
            "training_history": repo_relative_path(stage_dir / "training_history.csv", root=root),
        })
    deployment_path = checkpoints[(config.deployment_stage_id, config.deployment_checkpoint_kind)]
    deployment_stage_dir = output_dir / "stages" / config.deployment_stage_id
    deployment_indices = sorted(deployment_stage_dir.glob("*_patch_index.csv"))
    first_synthetic = next(iter(synthetic.values()), None)
    manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "status": "ok",
        "experiment_id": config.experiment_id,
        "architecture": config.architecture.as_dict(),
        "model_info": info.__dict__,
        "normalization_reference": {"source": config.normalization_reference},
        "normalization": normalization,
        "normalization_path": repo_relative_path(output_dir / "normalization.json", root=root),
        "sources": config.sources,
        "patching": config.patching.as_dict(),
        "input_channels": ["seismic", "lfm", "valid_mask"],
        "output_semantics": "pred_log_ai = lfm_log_ai + physical_delta_log_ai",
        "sample_axis_contract": sample_axis_contract,
        "stages": manifest_stages,
        "deployment_checkpoint": {
            "stage_id": config.deployment_stage_id,
            "kind": config.deployment_checkpoint_kind,
            "path": repo_relative_path(deployment_path, root=root),
        },
        "device": device_metadata,
    }
    if first_synthetic is not None and deployment_indices:
        manifest["benchmark_dir"] = repo_relative_path(first_synthetic[1], root=root)
        manifest["patch_index"] = repo_relative_path(deployment_indices[0], root=root)
    write_json(output_dir / "experiment_manifest.json", manifest)
    write_json(output_dir / "model_run_manifest.json", manifest)
    return manifest


__all__ = ["DeterministicCycler", "masked_centered_rms", "run_experiment"]
