"""Small artifact and diagnostic helpers for residual transfer results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ResidualTextureLibrary, ResidualTransferResult


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_jsonable(item) for item in value.tolist()]
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def library_summary(library: ResidualTextureLibrary) -> dict[str, Any]:
    """Return the published dictionary contract as JSON-compatible values."""

    if not isinstance(library, ResidualTextureLibrary):
        raise TypeError("library must be a ResidualTextureLibrary.")
    return _jsonable(library.describe())


def result_summary(result: ResidualTransferResult) -> dict[str, Any]:
    """Return compact scalar/metadata diagnostics without embedding fields."""

    if not isinstance(result, ResidualTransferResult):
        raise TypeError("result must be a ResidualTransferResult.")
    return _jsonable(result.summary)


def write_result_summary(result: ResidualTransferResult, path: str | Path) -> Path:
    """Write a compact result summary; numerical fields remain in memory/arrays."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_summary(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def plot_transfer_diagnostics(
    result: ResidualTransferResult,
    output_dir: str | Path,
    *,
    prefix: str = "residual_transfer",
) -> tuple[Path, ...]:
    """Create the compact first-round comparison figures when matplotlib is available."""

    if not isinstance(result, ResidualTransferResult):
        raise TypeError("result must be a ResidualTransferResult.")
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fields = [("ginn_body", result.ginn_body), ("predicted_residual", result.predicted_residual), ("enhanced_log_ai", result.enhanced_log_ai)]
    if result.soft_residual is not None:
        fields.insert(1, ("soft_residual", result.soft_residual))
    if result.hard_nearest_residual is not None:
        fields.insert(1, ("hard_nearest_residual", result.hard_nearest_residual))
    if result.uniform_residual is not None:
        fields.insert(1, ("uniform_residual", result.uniform_residual))
    generated: list[Path] = []
    for name, values in fields:
        array = np.asarray(values, dtype=np.float64)
        figure, axes = plt.subplots(1, 2 if array.ndim >= 2 else 1, figsize=(10.0, 4.0), squeeze=False)
        axes_flat = axes.reshape(-1)
        if array.ndim == 1:
            axes_flat[0].plot(array)
        elif array.ndim == 2:
            axes_flat[0].imshow(array, aspect="auto", origin="upper", cmap="viridis")
        else:
            axes_flat[0].imshow(array.reshape((-1, array.shape[-1])), aspect="auto", origin="upper", cmap="viridis")
        axes_flat[0].set_title(name)
        if len(axes_flat) > 1:
            axes_flat[1].imshow(np.asarray(result.support).reshape((-1, result.support.shape[-1])), aspect="auto", origin="upper", cmap="gray_r")
            axes_flat[1].set_title("support")
        figure.tight_layout()
        path = directory / f"{prefix}_{name}.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        generated.append(path)
    return tuple(generated)


__all__ = ["library_summary", "plot_transfer_diagnostics", "result_summary", "write_result_summary"]
