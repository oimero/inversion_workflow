"""Explicit, offline CIG 3D HRNet inference on an extruded 2D section.

This adapter does not claim native 2D model support. It imports only the
local network/normalization implementation, bypassing all download machinery.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


def _local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_rgt_section(
    seismic: np.ndarray,
    *,
    checkpoint_path: Path,
    cig_bench_root: Path | None = None,
    infer_shape: tuple[int, int, int] = (384, 16, 512),
    device: str = "cuda",
) -> tuple[np.ndarray, dict]:
    """Return raw [profile,time] RGT and reproducible inference provenance."""
    seismic = np.asarray(seismic, dtype=np.float32)
    if seismic.ndim != 2 or min(seismic.shape) < 2 or not np.isfinite(seismic).all():
        raise ValueError("seismic must be a finite [profile,time] section with both dimensions >= 2.")
    if float(np.std(seismic)) <= 0:
        raise ValueError("The CIG normalization requires non-constant seismic.")
    checkpoint_path = Path(checkpoint_path).resolve()
    vendored = cig_bench_root is None
    if vendored:
        cig_bench_root = Path(__file__).resolve().parent / "cig_bench"
    else:
        cig_bench_root = Path(cig_bench_root).resolve()
    package_root = cig_bench_root if vendored else cig_bench_root / "cig_bench"
    network_path = package_root / "networks" / "hrnet.py"
    utils_path = package_root / "predictor" / "utils.py"
    for path in (checkpoint_path, network_path, utils_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required local CIG file is absent: {path}; no automatic downloads.")
    infer_shape = tuple(infer_shape)
    pad = 8
    if len(infer_shape) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 16 or (n + 2 * pad) % 16 for n in infer_shape):
        raise ValueError("infer_shape must contain three integers >= 16, with each padded dimension divisible by 16.")
    selected_device = torch.device(device)
    if selected_device.type not in {"cpu", "cuda"}:
        raise ValueError("Only the native CPU or CUDA environment is supported.")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu explicitly for CPU inference.")
    hrnet = _local_module("marmousi2_local_cig_hrnet", network_path)
    utils = _local_module("marmousi2_local_cig_utils", utils_path)
    model = hrnet.HRNet()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().to(selected_device)
    volume = seismic.T[:, None, :].copy()
    normalized = utils.z_score_clip(volume, clp_s=2.0) * 2 - 1
    tensor = torch.from_numpy(normalized)[None, None]
    tensor = F.interpolate(tensor, infer_shape, mode="trilinear", align_corners=False).to(selected_device)
    inputs = torch.cat((tensor, torch.zeros_like(tensor), torch.zeros_like(tensor)), dim=1)
    use_autocast = selected_device.type == "cuda"
    if use_autocast:
        torch.cuda.synchronize(selected_device)
        torch.cuda.reset_peak_memory_stats(selected_device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(selected_device.type, enabled=use_autocast):
        prediction = model(F.pad(inputs, (pad,) * 6, mode="replicate"))
        prediction = prediction[:, :, pad:-pad, pad:-pad, pad:-pad]
    with torch.inference_mode():
        raw = F.interpolate(prediction.float(), volume.shape, mode="trilinear", align_corners=False)
        raw = raw[0, 0, :, 0, :].T.cpu().numpy().copy()
    if use_autocast:
        torch.cuda.synchronize(selected_device)
    elapsed = time.perf_counter() - start
    if raw.shape != seismic.shape or not np.isfinite(raw).all() or np.ptp(raw) <= 0:
        raise ValueError("CIG produced an invalid RGT field.")
    metadata = {
        "schema": "marmousi2_cig_rgt_2d_extrusion_v1",
        "architecture": "CIG HRNet() default c=48; Conv3d; three input channels",
        "checkpoint": {"path": str(checkpoint_path)},
        "network_source": {"path": str(network_path)},
        "preprocessing_source": {"path": str(utils_path)},
        "source_shape_T_H_W": list(volume.shape),
        "infer_shape_T_H_W": list(infer_shape),
        "replication_pad": pad,
        "spatial_adapter": "2D singleton axis extruded by trilinear resizing to a narrow 3D input",
        "native_2d_support": False,
        "horizon_prompt_channels": "both zero; no interpreted-horizon or well prompts",
        "input_normalization": "whole-section z-score; clip +/-2 sigma; minmax to [-1,1]",
        "output_processing": "trilinear resize back only; raw output without normalization/clipping/smoothing",
        "device": str(selected_device),
        "device_name": torch.cuda.get_device_name(selected_device) if use_autocast else "CPU",
        "torch_version": str(torch.__version__),
        "autocast": use_autocast,
        "autocast_dtype": "float16" if use_autocast else None,
        "inference_and_resize_seconds": elapsed,
        "peak_cuda_allocated_GiB": torch.cuda.max_memory_allocated(selected_device) / 1024 ** 3 if use_autocast else None,
        "raw_min_max": [float(raw.min()), float(raw.max())],
        "fraction_negative_vertical_difference": float(np.mean(np.diff(raw, axis=1) < 0)),
        "limitations": [
            "Non-default inference dimensions and a 3D checkpoint on a 2D section; accuracy is not validated.",
            "Complex fault-zone connections and edge artifacts remain possible; monotonic repair does not correct lateral topology.",
        ],
    }
    del model, inputs, tensor, prediction
    if use_autocast:
        torch.cuda.empty_cache()
    return raw, metadata
