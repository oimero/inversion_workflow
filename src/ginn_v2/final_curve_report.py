"""Compare the old projected-residual GINN run with the final-curve run.

The report evaluates both predictions against the common
``native_filtered_log_ai`` reference stored in ``comparison_traces.npz``.  The
reference is the native filtered well curve after one 25 m Gaussian smoothing;
it is not the training target of either run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


_REQUIRED = (
    "samples",
    "well_names",
    "well_reference_log_ai",
    "well_target_log_ai",
    "well_prediction_log_ai",
    "well_prior_log_ai",
    "section_prediction_log_ai",
    "section_prior_log_ai",
    "section_seismic",
    "section_line_numbers",
    "section_inline",
)


def _load_npz(run: Path) -> dict[str, np.ndarray]:
    path = run / "comparison_traces.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in _REQUIRED if name not in archive.files]
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        return {name: archive[name] for name in archive.files}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(np.asarray(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _curve_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray | None = None,
) -> dict[str, Any]:
    mask = _finite_mask(reference, prediction)
    ref = reference[mask].astype(float, copy=False)
    pred = prediction[mask].astype(float, copy=False)
    result: dict[str, Any] = {"count": int(mask.sum())}
    if ref.size == 0:
        result.update({"log_rmse": None, "ai_relative_rmse": None, "correlation": None})
    else:
        result["log_rmse"] = float(np.sqrt(np.mean((pred - ref) ** 2)))
        ref_ai = np.exp(ref)
        pred_ai = np.exp(pred)
        ai_rmse = np.sqrt(np.mean((pred_ai - ref_ai) ** 2))
        result["ai_relative_rmse"] = float(ai_rmse / np.sqrt(np.mean(ref_ai**2)))
        result["correlation"] = _corr(ref, pred)
    if target is not None:
        target_mask = _finite_mask(target, prediction)
        if target_mask.any():
            result["target_log_rmse"] = float(
                np.sqrt(np.mean((prediction[target_mask] - target[target_mask]) ** 2))
            )
        else:
            result["target_log_rmse"] = None
    return result


def _reference_consistency(
    before: np.ndarray,
    after: np.ndarray,
) -> dict[str, Any]:
    mask = _finite_mask(before, after)
    if not mask.any():
        return {"count": 0, "max_abs_difference": None, "rmse": None}
    difference = before[mask].astype(float) - after[mask].astype(float)
    return {
        "count": int(mask.sum()),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def _run_metrics(
    data: dict[str, np.ndarray],
    common_reference: np.ndarray,
    summary: dict[str, Any],
) -> dict[str, Any]:
    reference = data["well_reference_log_ai"]
    prediction = data["well_prediction_log_ai"]
    target = data["well_target_log_ai"]
    well_metrics = []
    correlations: list[float] = []
    pooled_ref: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    for index, name in enumerate(data["well_names"].astype(str)):
        mask = _finite_mask(common_reference[index], prediction[index])
        ref = common_reference[index][mask]
        pred = prediction[index][mask]
        metrics = _curve_metrics(common_reference[index], prediction[index], target[index])
        metrics["well_name"] = name
        metrics["common_reference_count"] = int(mask.sum())
        metrics["reference_vs_run_reference_max_abs"] = _reference_consistency(
            common_reference[index], reference[index]
        )["max_abs_difference"]
        if ref.size:
            pooled_ref.append(ref)
            pooled_pred.append(pred)
        if metrics.get("correlation") is not None:
            correlations.append(float(metrics["correlation"]))
        well_metrics.append(metrics)

    if pooled_ref:
        pooled_reference = np.concatenate(pooled_ref)
        pooled_prediction = np.concatenate(pooled_pred)
        pooled = _curve_metrics(pooled_reference, pooled_prediction)
    else:
        pooled = {"count": 0, "log_rmse": None, "ai_relative_rmse": None, "correlation": None}

    section_delta = data["section_prediction_log_ai"] - data["section_prior_log_ai"]
    section_mask = np.isfinite(section_delta)
    section_delta_rms = (
        float(np.sqrt(np.mean(section_delta[section_mask] ** 2)))
        if section_mask.any()
        else None
    )
    selected_metrics = summary.get("selected_metrics")
    if not isinstance(selected_metrics, dict) or "visible_correlation" not in selected_metrics:
        raise ValueError("experiment_summary.json is missing selected_metrics.visible_correlation")
    visible = np.asarray(selected_metrics["visible_correlation"], dtype=float)
    visible = visible[np.isfinite(visible)]
    return {
        "common_reference_pooled": pooled,
        "median_visible_correlation": float(np.median(visible)) if visible.size else None,
        "visible_correlation_count": int(visible.size),
        "median_well_reference_correlation": float(np.median(correlations)) if correlations else None,
        "section_delta_rms": section_delta_rms,
        "section_delta_count": int(section_mask.sum()),
        "wells": well_metrics,
    }


def _save_well_plot(
    output: Path,
    samples: np.ndarray,
    names: np.ndarray,
    reference: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    reference_ai = np.exp(reference)
    before_ai = np.exp(before)
    after_ai = np.exp(after)
    all_ai = np.concatenate(
        [array[np.isfinite(array)] for array in (reference_ai, before_ai, after_ai)]
    )
    pad = 0.03 * float(np.ptp(all_ai))
    x_limits = (float(all_ai.min()) - pad, float(all_ai.max()) + pad)
    figure, axes = plt.subplots(1, len(names), figsize=(15, 7), sharey=True)
    axes = np.atleast_1d(axes)
    for index, (axis, name) in enumerate(zip(axes, names.astype(str))):
        axis.plot(reference_ai[index], samples, color="black", lw=1.0, label="common reference")
        axis.plot(before_ai[index], samples, color="#2b6cb0", lw=0.8, label="old prediction")
        axis.plot(after_ai[index], samples, color="#c53030", lw=0.8, label="new prediction")
        axis.set_title(name, loc="left")
        axis.set_xlim(*x_limits)
        axis.set_ylim(6085.0, 5075.0)
        axis.grid(alpha=0.18)
        axis.set_xlabel("AI (m/s*g/cc)")
    axes[0].set_ylabel("TVDSS (m)")
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle("Five wells: common reference and two predictions")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _save_section_plot(output: Path, data_before: dict[str, np.ndarray], data_after: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    samples = data_before["samples"]
    lines = data_before["section_line_numbers"]
    prior = data_before["section_prior_log_ai"]
    old_prediction = data_before["section_prediction_log_ai"]
    new_prediction = data_after["section_prediction_log_ai"]
    depth_mask = np.any(
        np.isfinite(np.stack([prior, old_prediction, new_prediction])), axis=(0, 1)
    )
    if not depth_mask.any():
        raise ValueError("section arrays have no finite depth samples")
    samples = samples[depth_mask]
    prior = prior[:, depth_mask]
    old_prediction = old_prediction[:, depth_mask]
    new_prediction = new_prediction[:, depth_mask]
    combined = np.concatenate(
        [prior[np.isfinite(prior)], old_prediction[np.isfinite(old_prediction)], new_prediction[np.isfinite(new_prediction)]]
    )
    value_limits = (float(np.percentile(combined, 1.0)), float(np.percentile(combined, 99.0)))
    change = new_prediction - old_prediction
    finite_change = change[np.isfinite(change)]
    change_limit = float(np.percentile(np.abs(finite_change), 99.0)) if finite_change.size else 1.0
    change_limit = max(change_limit, 1e-9)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = [
        (axes[0, 0], prior, "Initial model / prior", "viridis", value_limits),
        (axes[0, 1], old_prediction, "Old prediction", "viridis", value_limits),
        (axes[1, 0], new_prediction, "New final-curve prediction", "viridis", value_limits),
        (axes[1, 1], change, "New prediction - old prediction", "RdBu_r", (-change_limit, change_limit)),
    ]
    extent = [float(lines.min()), float(lines.max()), float(samples[-1]), float(samples[0])]
    for axis, values, title, cmap, limits in panels:
        image = axis.imshow(values.T, aspect="auto", origin="upper", extent=extent, cmap=cmap, vmin=limits[0], vmax=limits[1])
        axis.set_title(title)
        axis.set_xlabel("Xline")
        axis.set_ylabel("TVDSS (m)")
        figure.colorbar(image, ax=axis, shrink=0.84, label="log(AI)" if title != "New prediction - old prediction" else "Delta log(AI)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(before: Path, after: Path) -> dict[str, Any]:
    """Write a compact comparison report and figures below ``after``.

    ``before`` is normally the frozen proportional no-gain run.  ``after`` is
    the new final-curve run and must already contain its comparison NPZ.
    The return value is the JSON-compatible report dictionary.
    """

    before = Path(before)
    after = Path(after)
    before_data = _load_npz(before)
    after_data = _load_npz(after)
    before_summary = _load_json(before / "experiment_summary.json")
    after_summary = _load_json(after / "experiment_summary.json")
    if before_data["well_names"].tolist() != after_data["well_names"].tolist():
        raise ValueError("before and after well order does not match")
    if not np.array_equal(before_data["samples"], after_data["samples"]):
        raise ValueError("before and after sample axis does not match")
    if not np.array_equal(before_data["section_line_numbers"], after_data["section_line_numbers"]):
        raise ValueError("before and after section line numbers do not match")

    report_dir = after / "final_curve_comparison"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    common_reference = np.where(
        np.isfinite(before_data["well_reference_log_ai"])
        & np.isfinite(after_data["well_reference_log_ai"]),
        before_data["well_reference_log_ai"],
        np.nan,
    )
    reference_check = _reference_consistency(
        before_data["well_reference_log_ai"], after_data["well_reference_log_ai"]
    )
    old_metrics = _run_metrics(before_data, common_reference, before_summary)
    new_metrics = _run_metrics(after_data, common_reference, after_summary)
    prediction_change = after_data["section_prediction_log_ai"] - before_data["section_prediction_log_ai"]
    change_mask = np.isfinite(prediction_change)
    report: dict[str, Any] = {
        "schema_version": "ginn_v2_final_curve_comparison_v1",
        "before": str(before),
        "after": str(after),
        "reference": {
            "definition": "native_filtered_log_ai smoothed once on native coordinates, then interpolated to model axis (common diagnostic reference)",
            "common_finite_count": int(np.isfinite(common_reference).sum()),
            "consistency_before_vs_after": reference_check,
        },
        "old_projected_residual": old_metrics,
        "new_final_curve": new_metrics,
        "section_prediction_change_rms": (
            float(np.sqrt(np.mean(prediction_change[change_mask] ** 2)))
            if change_mask.any()
            else None
        ),
        "configs": {
            "before": before_summary.get("config", {}),
            "after": after_summary.get("config", {}),
        },
        "outputs": {
            "readme": "final_curve_comparison/README.md",
            "json": "final_curve_comparison/comparison_report.json",
            "well_curves": "final_curve_comparison/figures/well_curves.png",
            "section_comparison": "final_curve_comparison/figures/section_comparison.png",
        },
    }
    _save_well_plot(
        figure_dir / "well_curves.png",
        before_data["samples"],
        before_data["well_names"],
        common_reference,
        before_data["well_prediction_log_ai"],
        after_data["well_prediction_log_ai"],
    )
    _save_section_plot(figure_dir / "section_comparison.png", before_data, after_data)
    (report_dir / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_readme(report_dir / "README.md", report)
    return report


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    old = report["old_projected_residual"]
    new = report["new_final_curve"]
    reference = report["reference"]
    lines = [
        "# GINN v2：旧投影残差与新最终曲线架构对比",
        "",
        "本报告比较旧的 proportional no-gain 运行与新的 final-curve 运行。两组使用同一批五口监督井、相同的井参考定义和共同的剖面坐标。",
        "",
        "共同井参考是原生滤波井曲线在原生坐标上做一次 25 m 高斯平滑，再插值到模型轴的诊断曲线，与上一轮相同。新训练目标则由原生滤波曲线先重采样、再做一次平滑。指标在共同有限区间内计算。",
        "",
        "## 结果",
        "",
        f"共同井参考有限样本数：`{reference['common_finite_count']}`；新旧参考交集上的最大绝对差：`{reference['consistency_before_vs_after']['max_abs_difference']}`。",
        "",
        "| 运行 | pooled log RMSE | AI 相对 RMSE | 空间验证道地震相关性中位数 | 剖面预测相对先验 RMS |",
        "|---|---:|---:|---:|---:|",
        f"| 旧 projected-residual | {old['common_reference_pooled']['log_rmse']:.5f} | {100*old['common_reference_pooled']['ai_relative_rmse']:.2f}% | {old['median_visible_correlation']:.4f} | {old['section_delta_rms']:.5f} |",
        f"| 新 final-curve | {new['common_reference_pooled']['log_rmse']:.5f} | {100*new['common_reference_pooled']['ai_relative_rmse']:.2f}% | {new['median_visible_correlation']:.4f} | {new['section_delta_rms']:.5f} |",
        "",
        "## 解释边界",
        "",
        "两组均没有增益或垂向补偿。相对旧无补偿版本，新架构同时改变了多个位置：删除振幅损失；把 25 m 平滑放到最终 `log(AI)` 曲线；井监督改为完整井曲线只平滑一次；删除网络修正量的 150 m 低频扣除和低频锚定损失。因此差异只能归因于这一整组设计变化，不能单独归因于其中某一个改动。",
        "",
        "新架构的目标形式是 `m = G25(m0 + raw_network)`。旧架构则在低频模型上加经过投影的修正量。共同井参考用于诊断，不等于两组实际训练目标。",
        "",
        "## 图件",
        "",
        "- [五口井：参考、旧预测、新预测](figures/well_curves.png)",
        "- [剖面：初始模型、旧预测、新预测及差异](figures/section_comparison.png)",
        "- [机器可读指标](comparison_report.json)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    report = write_report(args.before, args.after)
    print(json.dumps({"output_dir": str(args.after / "final_curve_comparison"), "schema_version": report["schema_version"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
