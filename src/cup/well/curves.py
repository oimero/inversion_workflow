"""cup.well.curves: LAS 曲线分类与主曲线选择。

本模块提供第二步 LAS 曲线筛选的核心逻辑：曲线简称规范化、按 mnemonic
规则进行类别归属、以及每个类别中选择主曲线的优先级算法。

边界说明
--------
- 本模块不读取 LAS 文件，不进行曲线数值处理。
- LLM 分类为预留接入点，当前实现基于纯规则匹配。

核心公开对象
------------
1. CurveInfo / CurveClassification / CurveSelection: 曲线分类数据结构。
2. classify_curves_by_rules: 按 mnemonic 规则对曲线进行类别归属。
3. select_primary_curves: 从分类结果中选择每类的主曲线。
4. normalize_mnemonic / exact_mnemonic: 曲线简称规范化。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from cup.well.mnemonics import CURVE_CATEGORY_MNEMONICS, CURVE_CATEGORY_PRIORITY


def normalize_mnemonic(mnemonic: object) -> str:
    """为规则匹配规范化 LAS 曲线简称。"""
    text = str(mnemonic).strip().upper()
    return re.sub(r":\d+$", "", text)


def exact_mnemonic(mnemonic: object) -> str:
    """规范化大小写与空白，同时保留 LASIO 重复曲线后缀。"""
    return str(mnemonic).strip().upper()


@dataclass(frozen=True)
class CurveInfo:
    """轻量 LAS 曲线头记录。"""

    mnemonic: str
    unit: str
    description: str
    index: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurveClassification:
    """单条曲线的类别归属结果。"""

    mnemonic: str
    unit: str
    description: str
    index: int
    category: str
    classification_source: str
    confidence: float | None = None
    notes: str = ""
    disabled: bool = False
    is_primary: bool = False

    def to_inventory_row(self, *, well_name: str) -> dict[str, Any]:
        return {
            "well_name": well_name,
            "mnemonic": self.mnemonic,
            "unit": self.unit,
            "description": self.description,
            "category": self.category,
            "is_primary": self.is_primary,
            "classification_source": self.classification_source,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class CurveSelection:
    """单井曲线筛选结果。"""

    well_name: str
    las_file: str
    screen_status: str
    primary_by_category: dict[str, str]
    selected_mnemonics: list[str]
    classifications: list[CurveClassification]
    reasons: list[str]
    exported_las: str | None = None

    def has_category(self, category: str) -> bool:
        return category in self.primary_by_category

    def primary(self, category: str) -> str | None:
        return self.primary_by_category.get(category)


def _schema_lookup(schema: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for category, mnemonics in schema.items():
        for mnemonic in mnemonics:
            key = normalize_mnemonic(mnemonic)
            categories = lookup.setdefault(key, [])
            if str(category) not in categories:
                categories.append(str(category))
    return lookup


def _well_override(overrides: Mapping[str, Any] | None, well_name: str) -> Mapping[str, Any]:
    if not overrides:
        return {}
    wells = overrides.get("wells") if isinstance(overrides, Mapping) else None
    if not isinstance(wells, Mapping):
        return {}
    key = str(well_name).strip()
    if key in wells and isinstance(wells[key], Mapping):
        return wells[key]
    key_fold = key.casefold()
    for candidate, value in wells.items():
        if str(candidate).strip().casefold() == key_fold and isinstance(value, Mapping):
            return value
    return {}


def _split_exact_and_base_mnemonics(values: Any) -> tuple[set[str], set[str]]:
    if values is None:
        return set(), set()
    exact: set[str] = set()
    base: set[str] = set()
    for item in values:
        item_exact = exact_mnemonic(item)
        if re.search(r":\d+$", item_exact):
            exact.add(item_exact)
        else:
            base.add(normalize_mnemonic(item_exact))
    return exact, base


def _disabled_curves(well_override: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    disabled = well_override.get("disabled_curves", [])
    return _split_exact_and_base_mnemonics(disabled)


def _split_forced_categories(value: Any, *, field_name: str) -> tuple[dict[str, str], dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    exact: dict[str, str] = {}
    base: dict[str, str] = {}
    for curve, category in value.items():
        curve_exact = exact_mnemonic(curve)
        if re.search(r":\d+$", curve_exact):
            exact[curve_exact] = str(category)
        else:
            base[normalize_mnemonic(curve_exact)] = str(category)
    return exact, base


def _force_category(
    overrides: Mapping[str, Any] | None,
    well_override: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    global_exact, global_base = _split_forced_categories(
        overrides.get("global_force_category", {}) if overrides else {},
        field_name="global_force_category",
    )
    well_exact, well_base = _split_forced_categories(
        well_override.get("force_category", {}),
        field_name="wells.<well>.force_category",
    )
    global_exact.update(well_exact)
    global_base.update(well_base)
    return global_exact, global_base


def _primary_override(well_override: Mapping[str, Any]) -> dict[str, str]:
    primary = well_override.get("primary", {})
    if not isinstance(primary, Mapping):
        return {}
    return {str(category): exact_mnemonic(mnemonic) for category, mnemonic in primary.items()}


def classify_curves_by_rules(
    curves: Sequence[CurveInfo],
    *,
    schema: Mapping[str, Sequence[str]] | None = None,
    well_name: str = "",
    overrides: Mapping[str, Any] | None = None,
) -> list[CurveClassification]:
    """按本地 mnemonic 规则和单井覆盖配置分类曲线。

    Parameters
    ----------
    curves : Sequence[CurveInfo]
        待分类的 LAS 曲线头列表。
    schema : Mapping[str, Sequence[str]] | None, optional
        分类规则表，默认使用 ``CURVE_CATEGORY_MNEMONICS``。
    well_name : str, default=""
        井名，用于匹配单井 override。
    overrides : Mapping[str, Any] | None, optional
        人工干预配置。

    Returns
    -------
    list[CurveClassification]
        逐条曲线的分类结果。
    """
    schema = CURVE_CATEGORY_MNEMONICS if schema is None else schema
    schema_categories = set(schema)
    lookup = _schema_lookup(schema)
    well_override = _well_override(overrides, well_name)
    disabled_exact, disabled_base = _disabled_curves(well_override)
    forced_exact, forced_base = _force_category(overrides, well_override)
    unknown_forced_categories = sorted((set(forced_exact.values()) | set(forced_base.values())) - schema_categories)
    if unknown_forced_categories:
        raise ValueError(f"Override categories are not in schema: {unknown_forced_categories}")

    classifications: list[CurveClassification] = []
    for curve in curves:
        exact = exact_mnemonic(curve.mnemonic)
        norm = normalize_mnemonic(curve.mnemonic)
        if exact in disabled_exact or norm in disabled_base:
            classifications.append(
                CurveClassification(
                    mnemonic=curve.mnemonic,
                    unit=curve.unit,
                    description=curve.description,
                    index=curve.index,
                    category="disabled",
                    classification_source="override",
                    confidence=1.0,
                    notes="disabled_by_override",
                    disabled=True,
                )
            )
            continue

        if exact in forced_exact or norm in forced_base:
            category = forced_exact.get(exact, forced_base.get(norm))
            if category not in schema_categories:
                raise ValueError(f"Override category {category!r} for {well_name}/{curve.mnemonic} is not in schema.")
            classifications.append(
                CurveClassification(
                    mnemonic=curve.mnemonic,
                    unit=curve.unit,
                    description=curve.description,
                    index=curve.index,
                    category=category,
                    classification_source="override",
                    confidence=1.0,
                    notes="force_category",
                )
            )
            continue

        categories = lookup.get(norm, [])
        if len(categories) == 1:
            classifications.append(
                CurveClassification(
                    mnemonic=curve.mnemonic,
                    unit=curve.unit,
                    description=curve.description,
                    index=curve.index,
                    category=categories[0],
                    classification_source="mnemonic_rule",
                    confidence=1.0,
                )
            )
        elif len(categories) > 1:
            classifications.append(
                CurveClassification(
                    mnemonic=curve.mnemonic,
                    unit=curve.unit,
                    description=curve.description,
                    index=curve.index,
                    category="ambiguous",
                    classification_source="mnemonic_rule",
                    confidence=0.0,
                    notes="multiple_categories:" + ",".join(categories),
                )
            )
        else:
            classifications.append(
                CurveClassification(
                    mnemonic=curve.mnemonic,
                    unit=curve.unit,
                    description=curve.description,
                    index=curve.index,
                    category="unclassified",
                    classification_source="unclassified",
                    confidence=0.0,
                )
            )
    return classifications


def _priority_for_category(
    category: str,
    *,
    overrides: Mapping[str, Any] | None,
    category_priority: Mapping[str, Sequence[str]] | None,
) -> list[str]:
    priority: list[str] = []
    if overrides and isinstance(overrides.get("global_priority"), Mapping):
        priority.extend(normalize_mnemonic(item) for item in overrides["global_priority"].get(category, []))
    priority_source = CURVE_CATEGORY_PRIORITY if category_priority is None else category_priority
    priority.extend(normalize_mnemonic(item) for item in priority_source.get(category, []))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in priority:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _choose_primary(
    candidates: Sequence[CurveClassification],
    *,
    category: str,
    well_override: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
    category_priority: Mapping[str, Sequence[str]] | None,
) -> str | None:
    by_exact = {exact_mnemonic(candidate.mnemonic): candidate for candidate in candidates}
    by_norm: dict[str, list[CurveClassification]] = {}
    for candidate in candidates:
        by_norm.setdefault(normalize_mnemonic(candidate.mnemonic), []).append(candidate)
    primary_overrides = _primary_override(well_override)
    preferred = primary_overrides.get(category)
    if preferred is not None:
        if preferred in by_exact:
            return by_exact[preferred].mnemonic
        preferred_norm = normalize_mnemonic(preferred)
        if preferred_norm not in by_norm:
            raise ValueError(f"Primary override for category {category!r} points to missing curve {preferred!r}.")
        return sorted(by_norm[preferred_norm], key=lambda item: item.index)[0].mnemonic

    for mnemonic in _priority_for_category(category, overrides=overrides, category_priority=category_priority):
        if mnemonic in by_norm:
            return sorted(by_norm[mnemonic], key=lambda item: item.index)[0].mnemonic

    if candidates:
        return sorted(candidates, key=lambda item: item.index)[0].mnemonic
    return None


def select_primary_curves(
    classifications: Sequence[CurveClassification],
    *,
    well_name: str,
    las_file: str,
    selected_categories: Sequence[str],
    required_categories: Sequence[str],
    overrides: Mapping[str, Any] | None = None,
    category_priority: Mapping[str, Sequence[str]] | None = None,
) -> CurveSelection:
    """为每个选中类别选择一条主曲线，并生成单井筛选状态。

    Parameters
    ----------
    classifications : Sequence[CurveClassification]
        逐条曲线分类结果。
    well_name : str
        井名。
    las_file : str
        LAS 文件路径。
    selected_categories : Sequence[str]
        需要选 primary 的类别。
    required_categories : Sequence[str]
        必须同时具备的类别。
    overrides : Mapping[str, Any] | None, optional
        人工干预配置。
    category_priority : Mapping[str, Sequence[str]] | None, optional
        每类主曲线选择优先级。

    Returns
    -------
    CurveSelection
        单井曲线筛选结果。
    """
    well_override = _well_override(overrides, well_name)
    by_category: dict[str, list[CurveClassification]] = {}
    for item in classifications:
        if item.disabled or item.category in {"unclassified", "ambiguous", "disabled"}:
            continue
        if item.category not in selected_categories:
            continue
        by_category.setdefault(item.category, []).append(item)

    primary_by_category: dict[str, str] = {}
    for category in selected_categories:
        primary = _choose_primary(
            by_category.get(category, []),
            category=category,
            well_override=well_override,
            overrides=overrides,
            category_priority=category_priority,
        )
        if primary is not None:
            primary_by_category[category] = primary

    selected_mnemonics = []
    seen: set[str] = set()
    for category in selected_categories:
        primary = primary_by_category.get(category)
        if primary is not None and primary not in seen:
            selected_mnemonics.append(primary)
            seen.add(primary)

    marked: list[CurveClassification] = []
    primary_by_exact_category = {
        (category, exact_mnemonic(mnemonic)) for category, mnemonic in primary_by_category.items()
    }
    for item in classifications:
        marked.append(
            CurveClassification(
                mnemonic=item.mnemonic,
                unit=item.unit,
                description=item.description,
                index=item.index,
                category=item.category,
                classification_source=item.classification_source,
                confidence=item.confidence,
                notes=item.notes,
                disabled=item.disabled,
                is_primary=(item.category, exact_mnemonic(item.mnemonic)) in primary_by_exact_category,
            )
        )

    missing_required = [category for category in required_categories if category not in primary_by_category]
    reasons = [f"missing_{category}" for category in missing_required]
    if not missing_required:
        status = "passed"
    elif selected_mnemonics:
        status = "partial"
    else:
        status = "failed"

    return CurveSelection(
        well_name=well_name,
        las_file=las_file,
        screen_status=status,
        primary_by_category=primary_by_category,
        selected_mnemonics=selected_mnemonics,
        classifications=marked,
        reasons=reasons,
    )
