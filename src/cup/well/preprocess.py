"""cup.well.preprocess: 时间域测井曲线预处理工具。

本模块提供第三步测井预处理的全部可复用逻辑：单位标准化、常值段替换、
离群值移除、全局分位数阈值计算与曲线可用性判定。

边界说明
--------
- 本模块处理的是单条曲线的数值清洗，不涉及井间逻辑或 LAS 文件 IO。
- 阈值计算基于全局统计数据，不依赖外部模型。

核心公开对象
------------
1. UnitStandardization / CurveThreshold: 单位转换与阈值数据结构。
2. standardize_curve_unit: 按类别和输入单位统一到标准单位。
3. replace_constant_runs / remove_outliers: 常值段与离群值处理。
4. compute_global_quantile_thresholds: 基于全局分位数计算阈值。
5. is_curve_usable / finite_stats: 曲线可用性判定与统计。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from wtie.processing import grid

STANDARD_MNEMONICS: dict[str, str] = {
    "p_sonic": "DT_USM",
    "s_sonic": "DTS_USM",
    "density": "RHO_GCC",
    "gamma_ray": "GR",
    "caliper": "CALI",
    "resistivity": "RT",
    "spontaneous_potential": "SP",
    "porosity": "POR",
    "permeability": "PERM",
    "water_saturation": "SW",
}

DEFAULT_MISSING_SENTINELS = (-999.0, -999.25, -9999.0, -99999.0)

SONIC_CATEGORIES = {"p_sonic", "s_sonic"}


@dataclass(frozen=True)
class UnitStandardization:
    values: np.ndarray
    original_unit: str
    standard_unit: str
    conversion_action: str
    hard_fail_reason: str = ""
    qc_flags: tuple[str, ...] = ()
    input_valid_count: int = 0
    output_valid_count: int = 0
    input_median: float | None = None
    output_median: float | None = None
    input_p01: float | None = None
    input_p99: float | None = None
    output_p01: float | None = None
    output_p99: float | None = None

    def report_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("values")
        row["qc_flags"] = ";".join(self.qc_flags)
        return row


@dataclass(frozen=True)
class ConstantRun:
    start_md: float
    end_md: float
    run_length: int
    constant_value: float
    action: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutlierRemoval:
    values: np.ndarray
    replaced_points: int
    lower: float | None
    upper: float | None
    threshold_source: str


@dataclass(frozen=True)
class CurveThreshold:
    standard_mnemonic: str
    lower: float | None
    upper: float | None
    source: str
    sample_count: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IrregularMdCurve:
    """One curve sampled on an enclosing irregular MD axis."""

    values: np.ndarray
    name: str
    unit: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 1:
            raise ValueError(f"Irregular MD curve {self.name!r} values must be one-dimensional.")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"Irregular MD curve {self.name!r} values must be numeric.")


@dataclass(frozen=True)
class IrregularMdCurveSet:
    """Curves sharing a possibly irregular native LAS MD axis.

    This is the explicit pre-grid boundary type.  ``grid.Log`` is deliberately
    not used until :func:`regularize_md_curve_set` has constructed a regular MD
    axis.
    """

    well_name: str
    md_m: np.ndarray
    curves: Mapping[str, IrregularMdCurve]
    source_las: str = ""

    def __post_init__(self) -> None:
        md = np.asarray(self.md_m)
        if md.ndim != 1 or md.size < 2:
            raise ValueError("Irregular MD basis must be one-dimensional with at least two samples.")
        if not self.curves:
            raise ValueError("IrregularMdCurveSet requires at least one curve.")
        for name, curve in self.curves.items():
            if not isinstance(curve, IrregularMdCurve):
                raise TypeError(f"Curve {name!r} must be an IrregularMdCurve.")
            if np.asarray(curve.values).shape != md.shape:
                raise ValueError(
                    f"Curve {name!r} length does not match the irregular MD basis."
                )

    @property
    def basis(self) -> np.ndarray:
        return np.asarray(self.md_m, dtype=float)

    def require(self, name: str) -> IrregularMdCurve:
        try:
            return self.curves[name]
        except KeyError as exc:
            raise KeyError(
                f"Curve {name!r} is not available in well {self.well_name!r}."
            ) from exc


@dataclass(frozen=True)
class WellCurveSet:
    """单井同一条规则 MD 轴上的任意曲线集合。

    与 ``wtie.processing.grid.LogSet`` 不同，本对象不要求必须包含 Vp/Rho。
    它用于预处理阶段承载 GR、井径、孔隙度、含水饱和度等辅助 LAS 曲线，
    直到后续真正构造岩石物理 LogSet。
    """

    well_name: str
    logs: Mapping[str, grid.Log]
    source_las: str = ""

    def __post_init__(self) -> None:
        validate_shared_basis(self.logs)

    @property
    def basis(self) -> np.ndarray:
        first = next(iter(self.logs.values()))
        return np.asarray(first.basis, dtype=float)

    @property
    def curve_names(self) -> list[str]:
        return list(self.logs.keys())

    def get(self, name: str) -> grid.Log | None:
        return self.logs.get(name)

    def require(self, name: str) -> grid.Log:
        log = self.get(name)
        if log is None:
            raise KeyError(f"Curve {name!r} is not available in well {self.well_name!r}.")
        return log

    def with_log(self, name: str, log: grid.Log) -> "WellCurveSet":
        """返回追加一条共享 MD 轴曲线后的新对象。"""
        if name in self.logs:
            raise KeyError(f"Curve {name!r} already exists in well {self.well_name!r}.")
        logs = dict(self.logs)
        logs[name] = log
        return WellCurveSet(well_name=self.well_name, logs=logs, source_las=self.source_las)


@dataclass(frozen=True)
class MdRegularizationResult:
    """A regular-MD curve set plus well- and curve-level audit records."""

    curve_set: WellCurveSet
    report: Mapping[str, Any]
    curve_reports: Mapping[str, Mapping[str, Any]]


def _is_regular_axis(values: np.ndarray) -> tuple[bool, float]:
    diffs = np.diff(values)
    step = float(np.median(diffs))
    return bool(np.allclose(diffs, step, rtol=1e-5, atol=1e-8)), step


def _interpolate_without_crossing_long_gaps(
    source_md: np.ndarray,
    source_values: np.ndarray,
    target_md: np.ndarray,
    *,
    max_gap_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(source_values, dtype=np.float64)
    valid_indices = np.flatnonzero(np.isfinite(values))
    output = np.full(target_md.shape, np.nan, dtype=np.float64)
    long_gap_count = 0
    supported_interval_count = 0
    tolerance = max(1e-9, max_gap_m * 1e-9)

    for left_index, right_index in zip(valid_indices[:-1], valid_indices[1:]):
        left_md = float(source_md[left_index])
        right_md = float(source_md[right_index])
        gap_m = right_md - left_md
        if gap_m > max_gap_m + tolerance:
            long_gap_count += 1
            continue
        supported_interval_count += 1
        mask = (target_md >= left_md - tolerance) & (target_md <= right_md + tolerance)
        if np.any(mask):
            output[mask] = np.interp(
                target_md[mask],
                [left_md, right_md],
                [float(values[left_index]), float(values[right_index])],
            )

    if valid_indices.size == 1:
        nearest = int(np.argmin(np.abs(target_md - source_md[valid_indices[0]])))
        if abs(float(target_md[nearest] - source_md[valid_indices[0]])) <= tolerance:
            output[nearest] = float(values[valid_indices[0]])

    exact_source = np.zeros(target_md.shape, dtype=bool)
    for source_index in valid_indices:
        exact_source |= np.isclose(target_md, source_md[source_index], rtol=0.0, atol=tolerance)
    finite_output = np.isfinite(output)
    return output, {
        "input_valid_count": int(valid_indices.size),
        "output_valid_count": int(np.count_nonzero(finite_output)),
        "interpolated_sample_count": int(np.count_nonzero(finite_output & ~exact_source)),
        "preserved_null_sample_count": int(np.count_nonzero(~finite_output)),
        "supported_interval_count": int(supported_interval_count),
        "long_gap_count": int(long_gap_count),
    }


def regularize_md_curve_set(
    curve_set: IrregularMdCurveSet,
    *,
    step_m: float,
    max_interpolation_gap_m: float,
) -> MdRegularizationResult:
    """Resample a shared irregular MD axis without bridging long data gaps.

    The output grid starts at the first input MD sample and advances by the
    explicitly configured ``step_m``. Interpolation is only allowed between
    adjacent finite source samples whose MD separation does not exceed
    ``max_interpolation_gap_m``.
    """
    source_md = np.asarray(curve_set.basis, dtype=np.float64)
    if source_md.size < 2 or np.any(~np.isfinite(source_md)) or np.any(np.diff(source_md) <= 0.0):
        raise ValueError("Input MD basis must be finite and strictly increasing.")
    step = float(step_m)
    max_gap = float(max_interpolation_gap_m)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("md_resampling.step_m must be finite and positive.")
    if not np.isfinite(max_gap) or max_gap < step:
        raise ValueError("md_resampling.max_interpolation_gap_m must be finite and >= step_m.")

    span = float(source_md[-1] - source_md[0])
    output_count = int(np.floor(span / step + 1e-9)) + 1
    if output_count < 2:
        raise ValueError("Regularized MD basis would contain fewer than two samples.")
    target_md = float(source_md[0]) + np.arange(output_count, dtype=np.float64) * step
    if target_md[-1] > source_md[-1] + max(1e-8, step * 1e-8):
        raise RuntimeError("Regularized MD basis exceeds the input support.")

    original_regular, original_median_step = _is_regular_axis(source_md)
    output_regular, output_median_step = _is_regular_axis(target_md)
    if not output_regular or not np.isclose(output_median_step, step, rtol=1e-5, atol=1e-8):
        raise RuntimeError("Constructed MD basis is not regularly sampled at step_m.")

    logs: dict[str, grid.Log] = {}
    curve_reports: dict[str, dict[str, Any]] = {}
    for name, curve in curve_set.curves.items():
        output_values, curve_report = _interpolate_without_crossing_long_gaps(
            source_md,
            np.asarray(curve.values, dtype=np.float64),
            target_md,
            max_gap_m=max_gap,
        )
        logs[name] = grid.Log(
            output_values,
            target_md,
            "md",
            name=curve.name,
            unit=curve.unit,
            allow_nan=True,
        )
        curve_reports[name] = {"standard_mnemonic": name, **curve_report}

    diffs = np.diff(source_md)
    report = {
        "well_name": curve_set.well_name,
        "source_las": curve_set.source_las,
        "original_sample_count": int(source_md.size),
        "original_md_start_m": float(source_md[0]),
        "original_md_end_m": float(source_md[-1]),
        "original_step_min_m": float(np.min(diffs)),
        "original_step_median_m": float(original_median_step),
        "original_step_max_m": float(np.max(diffs)),
        "original_md_regular": bool(original_regular),
        "output_sample_count": int(target_md.size),
        "output_md_start_m": float(target_md[0]),
        "output_md_end_m": float(target_md[-1]),
        "output_step_m": float(step),
        "output_md_regular": True,
        "max_interpolation_gap_m": float(max_gap),
    }
    return MdRegularizationResult(
        curve_set=WellCurveSet(
            well_name=curve_set.well_name,
            logs=logs,
            source_las=curve_set.source_las,
        ),
        report=report,
        curve_reports=curve_reports,
    )


def build_native_md_curve_set(curve_set: IrregularMdCurveSet) -> WellCurveSet:
    """Materialize curves on their native MD axis.

    The native axis must already satisfy ``grid.Log`` regular-sampling
    requirements; otherwise construction fails at this boundary.
    """
    source_md = np.asarray(curve_set.basis, dtype=np.float64)
    if source_md.size < 2 or np.any(~np.isfinite(source_md)) or np.any(np.diff(source_md) <= 0.0):
        raise ValueError("Input MD basis must be finite and strictly increasing.")

    logs: dict[str, grid.Log] = {}
    for name, curve in curve_set.curves.items():
        logs[name] = grid.Log(
            np.asarray(curve.values, dtype=np.float64),
            source_md,
            "md",
            name=curve.name,
            unit=curve.unit,
            allow_nan=True,
        )
    return WellCurveSet(
        well_name=curve_set.well_name,
        logs=logs,
        source_las=curve_set.source_las,
    )


def validate_shared_basis(logs: Mapping[str, grid.Log]) -> None:
    """校验所有曲线共享同一条 MD 采样轴。"""
    if not logs:
        raise ValueError("WellCurveSet requires at least one curve.")
    first_name, first_log = next(iter(logs.items()))
    if not isinstance(first_log, grid.Log):
        raise TypeError(f"Curve {first_name!r} is not a grid.Log.")
    basis = np.asarray(first_log.basis, dtype=float)
    for name, log in logs.items():
        if not isinstance(log, grid.Log):
            raise TypeError(f"Curve {name!r} is not a grid.Log.")
        if not log.is_md:
            raise ValueError(f"Curve {name!r} must be in MD domain.")
        if log.basis.shape != basis.shape or not np.allclose(np.asarray(log.basis, dtype=float), basis, atol=1e-6):
            raise ValueError(f"Curve {name!r} does not share the same MD basis as {first_name!r}.")


def derive_acoustic_impedance(
    curve_set: WellCurveSet,
    *,
    dt_mnemonic: str = "DT_USM",
    rho_mnemonic: str = "RHO_GCC",
) -> grid.Log:
    """从标准慢度和密度曲线派生声阻抗，严格传播无效样点。"""
    dt_log = curve_set.require(dt_mnemonic)
    rho_log = curve_set.require(rho_mnemonic)
    if str(dt_log.unit or "").strip().lower() != "us/m":
        raise ValueError(f"{dt_mnemonic} must use unit 'us/m', got {dt_log.unit!r}.")
    if str(rho_log.unit or "").strip().lower() != "g/cm3":
        raise ValueError(f"{rho_mnemonic} must use unit 'g/cm3', got {rho_log.unit!r}.")

    dt_usm = np.asarray(dt_log.values, dtype=float)
    rho_gcc = np.asarray(rho_log.values, dtype=float)
    valid = np.isfinite(dt_usm) & (dt_usm > 0.0) & np.isfinite(rho_gcc) & (rho_gcc > 0.0)
    ai = np.full(dt_usm.shape, np.nan, dtype=float)
    ai[valid] = (1_000_000.0 / dt_usm[valid]) * rho_gcc[valid]
    return grid.Log(
        ai,
        np.asarray(dt_log.basis, dtype=float),
        "md",
        name="AI",
        unit="m/s*g/cm3",
        allow_nan=True,
    )


def with_acoustic_impedance(curve_set: WellCurveSet) -> WellCurveSet:
    """返回固定追加 ``AI`` 派生曲线的单井曲线集合。"""
    return curve_set.with_log("AI", derive_acoustic_impedance(curve_set))


def select_curves_by_category(
    curve_set: WellCurveSet,
    category_to_mnemonic: Mapping[str, str],
    categories: Sequence[str],
) -> dict[str, grid.Log]:
    """按类别到 mnemonic 的映射从曲线集合中选择曲线。"""
    selected: dict[str, grid.Log] = {}
    for category in categories:
        mnemonic = category_to_mnemonic.get(category)
        if mnemonic is None:
            continue
        log = curve_set.get(mnemonic)
        if log is not None:
            selected[category] = log
    return selected


def standard_mnemonic_for_category(category: str) -> str:
    """返回指定曲线类别在工作流中的标准 mnemonic。"""
    try:
        return STANDARD_MNEMONICS[str(category)]
    except KeyError as exc:
        raise ValueError(f"No standard mnemonic configured for category {category!r}.") from exc


def normalize_unit(unit: object) -> str:
    """规范化单位写法以便规则匹配。"""
    text = str(unit or "").strip().lower()
    text = text.replace(" ", "")
    text = text.replace("μ", "u").replace("µ", "u")
    return text


def values_to_nan(
    values: object, *, null_value: float | None = None, sentinels: Sequence[float] = DEFAULT_MISSING_SENTINELS
) -> np.ndarray:
    """将数值转换为 float，并把已知缺失哨兵值替换为 NaN。"""
    out = np.asarray(values, dtype=float).copy()
    missing = ~np.isfinite(out)
    for sentinel in sentinels:
        missing |= np.isclose(out, float(sentinel), equal_nan=False)
    if null_value is not None and np.isfinite(float(null_value)):
        missing |= np.isclose(out, float(null_value), equal_nan=False)
    out[missing] = np.nan
    return out


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    """返回用于报告和 QC 规则的有限值稳健统计。"""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"valid_count": 0, "median": None, "p01": None, "p99": None, "min": None, "max": None}
    return {
        "valid_count": int(finite.size),
        "median": float(np.nanmedian(finite)),
        "p01": float(np.nanquantile(finite, 0.01)),
        "p99": float(np.nanquantile(finite, 0.99)),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
    }


def _with_stats(
    values: np.ndarray,
    converted: np.ndarray,
    *,
    original_unit: str,
    standard_unit: str,
    conversion_action: str,
    hard_fail_reason: str = "",
    qc_flags: Sequence[str] = (),
) -> UnitStandardization:
    before = finite_stats(values)
    after = finite_stats(converted)
    return UnitStandardization(
        values=converted,
        original_unit=original_unit,
        standard_unit=standard_unit,
        conversion_action=conversion_action,
        hard_fail_reason=hard_fail_reason,
        qc_flags=tuple(qc_flags),
        input_valid_count=int(before["valid_count"]),
        output_valid_count=int(after["valid_count"]),
        input_median=before["median"],
        output_median=after["median"],
        input_p01=before["p01"],
        input_p99=before["p99"],
        output_p01=after["p01"],
        output_p99=after["p99"],
    )


def standardize_curve_unit(values: np.ndarray, *, category: str, unit: str) -> UnitStandardization:
    """标准化受支持的曲线单位，并识别明显不可能的单位错配。"""
    original_unit = str(unit or "")
    unit_norm = normalize_unit(unit)
    clean = np.asarray(values, dtype=float)
    stats = finite_stats(clean)
    median = stats["median"]
    qc_flags: list[str] = []

    if category in SONIC_CATEGORIES:
        if stats["valid_count"] == 0:
            return _with_stats(
                clean,
                clean.copy(),
                original_unit=original_unit,
                standard_unit="us/m",
                conversion_action="none",
                hard_fail_reason="no_finite_samples",
            )
        assert median is not None
        positive = clean.copy()
        positive[positive <= 0.0] = np.nan
        if unit_norm in {"us/ft", "usec/ft"}:
            if median > 1000.0:
                return _with_stats(
                    clean,
                    positive * 3.280839895,
                    original_unit=original_unit,
                    standard_unit="us/m",
                    conversion_action="us_ft_to_us_m",
                    hard_fail_reason="sonic_slowness_unit_impossible_median_gt_1000",
                )
            if median > 160.0:
                qc_flags.append("sonic_us_ft_median_looks_like_us_m")
            return _with_stats(
                clean,
                positive * 3.280839895,
                original_unit=original_unit,
                standard_unit="us/m",
                conversion_action="us_ft_to_us_m",
                qc_flags=qc_flags,
            )
        if unit_norm in {"us/m", "usec/m"}:
            if median > 1000.0:
                return _with_stats(
                    clean,
                    positive,
                    original_unit=original_unit,
                    standard_unit="us/m",
                    conversion_action="keep_us_m",
                    hard_fail_reason="sonic_slowness_unit_impossible_median_gt_1000",
                )
            if 40.0 <= median <= 160.0:
                qc_flags.append("sonic_us_m_median_looks_like_us_ft")
            return _with_stats(
                clean,
                positive,
                original_unit=original_unit,
                standard_unit="us/m",
                conversion_action="keep_us_m",
                qc_flags=qc_flags,
            )
        if unit_norm in {"m/s", "mps", "m/sec", "meter/s", "meters/s"}:
            if median < 1000.0:
                return _with_stats(
                    clean,
                    1.0e6 / positive,
                    original_unit=original_unit,
                    standard_unit="us/m",
                    conversion_action="mps_to_us_m",
                    hard_fail_reason="sonic_velocity_unit_impossible_median_lt_1000",
                )
            return _with_stats(
                clean,
                1.0e6 / positive,
                original_unit=original_unit,
                standard_unit="us/m",
                conversion_action="mps_to_us_m",
            )
        return _with_stats(
            clean,
            clean.copy(),
            original_unit=original_unit,
            standard_unit="us/m",
            conversion_action="unsupported_unit",
            hard_fail_reason="unsupported_sonic_unit",
        )

    if category == "density":
        if stats["valid_count"] == 0:
            return _with_stats(
                clean,
                clean.copy(),
                original_unit=original_unit,
                standard_unit="g/cm3",
                conversion_action="none",
                hard_fail_reason="no_finite_samples",
            )
        assert median is not None
        positive = clean.copy()
        positive[positive <= 0.0] = np.nan
        if unit_norm in {"g/cm3", "g/cc", "g/cm^3", "gcc"}:
            if median > 100.0:
                return _with_stats(
                    clean,
                    positive,
                    original_unit=original_unit,
                    standard_unit="g/cm3",
                    conversion_action="keep_g_cm3",
                    hard_fail_reason="density_g_cm3_unit_impossible_median_gt_100",
                )
            if median > 10.0:
                qc_flags.append("density_g_cm3_median_high")
            return _with_stats(
                clean,
                positive,
                original_unit=original_unit,
                standard_unit="g/cm3",
                conversion_action="keep_g_cm3",
                qc_flags=qc_flags,
            )
        if unit_norm in {"kg/m3", "kg/m^3"}:
            if median < 10.0:
                return _with_stats(
                    clean,
                    positive / 1000.0,
                    original_unit=original_unit,
                    standard_unit="g/cm3",
                    conversion_action="kg_m3_to_g_cm3",
                    hard_fail_reason="density_kg_m3_unit_impossible_median_lt_10",
                )
            return _with_stats(
                clean,
                positive / 1000.0,
                original_unit=original_unit,
                standard_unit="g/cm3",
                conversion_action="kg_m3_to_g_cm3",
            )
        return _with_stats(
            clean,
            clean.copy(),
            original_unit=original_unit,
            standard_unit="g/cm3",
            conversion_action="unsupported_unit",
            hard_fail_reason="unsupported_density_unit",
        )

    return _with_stats(
        clean,
        clean.copy(),
        original_unit=original_unit,
        standard_unit=original_unit,
        conversion_action="kept_original_unit",
    )


def replace_constant_runs(
    md: np.ndarray,
    values: np.ndarray,
    *,
    min_run_length: int,
    exclude: bool = False,
) -> tuple[np.ndarray, list[ConstantRun], int]:
    """将严格连续相同值段替换为 NaN，可选跳过特定类别仅做记录。

    Parameters
    ----------
    md : np.ndarray
        MD 深度轴，单位 m。
    values : np.ndarray
        曲线数值数组。
    min_run_length : int
        触发替换的最小连续长度。
    exclude : bool, default=False
        True 时仅记录不做替换，报告 action 标记为 ``skip_caliper``。

    Returns
    -------
    tuple[np.ndarray, list[ConstantRun], int]
        清洗后的值、连续段报告列表、被替换的数据点数。
    """
    if min_run_length < 1:
        raise ValueError("min_run_length must be >= 1.")
    basis = np.asarray(md, dtype=float)
    cleaned = np.asarray(values, dtype=float).copy()
    reports: list[ConstantRun] = []
    replaced = 0
    i = 0
    n = cleaned.size
    while i < n:
        if not np.isfinite(cleaned[i]):
            i += 1
            continue
        j = i + 1
        while j < n and np.isfinite(cleaned[j]) and cleaned[j] == cleaned[i]:
            j += 1
        run_len = j - i
        if run_len >= min_run_length:
            action = "skip_caliper" if exclude else "set_null"
            reports.append(
                ConstantRun(
                    start_md=float(basis[i]),
                    end_md=float(basis[j - 1]),
                    run_length=int(run_len),
                    constant_value=float(cleaned[i]),
                    action=action,
                )
            )
            if not exclude:
                cleaned[i:j] = np.nan
                replaced += run_len
        i = j
    return cleaned, reports, replaced


def compute_global_quantile_thresholds(
    curves_by_standard: Mapping[str, Sequence[np.ndarray]],
    *,
    lower_quantile: float,
    upper_quantile: float,
    min_samples: int,
) -> dict[str, CurveThreshold]:
    """按标准曲线计算全局分位数阈值。"""
    thresholds: dict[str, CurveThreshold] = {}
    for standard, arrays in curves_by_standard.items():
        finite_parts = [np.asarray(values, dtype=float)[np.isfinite(values)] for values in arrays]
        finite_parts = [values for values in finite_parts if values.size]
        if finite_parts:
            merged = np.concatenate(finite_parts)
        else:
            merged = np.asarray([], dtype=float)
        if merged.size < min_samples:
            thresholds[standard] = CurveThreshold(
                standard_mnemonic=standard,
                lower=None,
                upper=None,
                source="skipped_insufficient_samples",
                sample_count=int(merged.size),
            )
            continue
        thresholds[standard] = CurveThreshold(
            standard_mnemonic=standard,
            lower=float(np.nanquantile(merged, lower_quantile)),
            upper=float(np.nanquantile(merged, upper_quantile)),
            source="global_quantile",
            sample_count=int(merged.size),
        )
    return thresholds


def threshold_from_overrides(
    standard_mnemonic: str,
    *,
    well_name: str,
    overrides: Mapping[str, Any] | None,
    auto_thresholds: Mapping[str, CurveThreshold],
) -> CurveThreshold:
    """按井级、全局、自动阈值的优先级解析最终阈值。"""
    overrides = overrides or {}
    well_curve = overrides.get("well_curve", {})
    if isinstance(well_curve, Mapping):
        for candidate_well, curves in well_curve.items():
            if str(candidate_well).strip().casefold() != str(well_name).strip().casefold():
                continue
            if (
                isinstance(curves, Mapping)
                and standard_mnemonic in curves
                and isinstance(curves[standard_mnemonic], Mapping)
            ):
                spec = curves[standard_mnemonic]
                return CurveThreshold(
                    standard_mnemonic=standard_mnemonic,
                    lower=_optional_float(spec.get("min")),
                    upper=_optional_float(spec.get("max")),
                    source="manual_well_curve",
                    sample_count=0,
                )
    global_spec = overrides.get("global", {})
    if (
        isinstance(global_spec, Mapping)
        and standard_mnemonic in global_spec
        and isinstance(global_spec[standard_mnemonic], Mapping)
    ):
        spec = global_spec[standard_mnemonic]
        return CurveThreshold(
            standard_mnemonic=standard_mnemonic,
            lower=_optional_float(spec.get("min")),
            upper=_optional_float(spec.get("max")),
            source="manual_global",
            sample_count=0,
        )
    return auto_thresholds.get(
        standard_mnemonic,
        CurveThreshold(
            standard_mnemonic=standard_mnemonic, lower=None, upper=None, source="missing_threshold", sample_count=0
        ),
    )


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def remove_outliers(values: np.ndarray, threshold: CurveThreshold) -> OutlierRemoval:
    """将解析阈值之外的样点置为 NaN。"""
    cleaned = np.asarray(values, dtype=float).copy()
    if threshold.lower is None and threshold.upper is None:
        return OutlierRemoval(
            values=cleaned, replaced_points=0, lower=None, upper=None, threshold_source=threshold.source
        )
    mask = np.zeros(cleaned.shape, dtype=bool)
    finite = np.isfinite(cleaned)
    if threshold.lower is not None:
        mask |= finite & (cleaned < float(threshold.lower))
    if threshold.upper is not None:
        mask |= finite & (cleaned > float(threshold.upper))
    replaced = int(mask.sum())
    cleaned[mask] = np.nan
    return OutlierRemoval(
        values=cleaned,
        replaced_points=replaced,
        lower=threshold.lower,
        upper=threshold.upper,
        threshold_source=threshold.source,
    )


def is_curve_usable(
    values: np.ndarray,
    *,
    initial_valid_count: int,
    min_valid_samples: int,
    min_valid_fraction_of_initial: float,
) -> tuple[bool, str, int, float]:
    """判断清洗后的曲线是否仍满足可用性要求。"""
    final_count = int(np.isfinite(np.asarray(values, dtype=float)).sum())
    if initial_valid_count <= 0:
        return False, "no_initial_valid_samples", final_count, 0.0
    fraction = float(final_count / initial_valid_count)
    if final_count < int(min_valid_samples):
        return False, "insufficient_valid_samples", final_count, fraction
    if fraction < float(min_valid_fraction_of_initial):
        return False, "valid_fraction_below_threshold", final_count, fraction
    return True, "", final_count, fraction
