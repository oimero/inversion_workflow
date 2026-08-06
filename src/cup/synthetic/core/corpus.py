"""Canonical V2 corpus budgeting and domain-neutral parent planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd


FAMILIES = ("none", "wedge", "pinchout")
SPLITS = ("training", "tuning", "calibration")


@dataclass(frozen=True)
class CorpusBudget:
    short_per_family: int
    training_per_family: int
    tuning_per_family: int
    calibration_per_family: int
    full_section_per_family: int
    iid_section_per_family: int
    combination_holdout_per_family: int
    max_candidate_attempts: int

    def __post_init__(self) -> None:
        values = self.__dict__
        if any(isinstance(value, bool) or int(value) <= 0 for value in values.values()):
            raise ValueError("all canonical corpus quotas must be positive integers.")
        if (
            self.training_per_family
            + self.tuning_per_family
            + self.calibration_per_family
            != self.short_per_family
        ):
            raise ValueError("training/tuning/calibration quotas must sum to short quota.")
        if (
            self.iid_section_per_family
            + self.combination_holdout_per_family
            != self.full_section_per_family
        ):
            raise ValueError("IID/combination quotas must sum to full-section quota.")
        if self.planned_parents > self.max_candidate_attempts:
            raise ValueError("canonical quotas exceed max_candidate_attempts.")
        if self.max_candidate_attempts > 4000:
            raise ValueError("max_candidate_attempts may not exceed 4000.")

    @property
    def planned_parents(self) -> int:
        return len(FAMILIES) * (
            self.short_per_family + self.full_section_per_family
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CorpusBudget":
        expected = {
            "short_per_family",
            "training_per_family",
            "tuning_per_family",
            "calibration_per_family",
            "full_section_per_family",
            "iid_section_per_family",
            "combination_holdout_per_family",
            "max_candidate_attempts",
        }
        unknown = sorted(set(value).difference(expected))
        missing = sorted(expected.difference(value))
        if unknown or missing:
            raise ValueError(
                f"canonical corpus budget unknown={unknown}, missing={missing}."
            )
        return cls(**{key: int(value[key]) for key in expected})


def _is_combination_holdout(scenario: Any) -> bool:
    family = str(scenario.geometry_family)
    if float(scenario.correlation_length_m) != 900.0:
        return False
    if float(scenario.coefficient_sigma_multiplier) != 0.5:
        return False
    if float(scenario.thickness_log_sigma) != 0.25:
        return False
    if family == "none":
        return True
    if str(scenario.geometry_direction) != "right_to_left":
        return False
    return family != "pinchout" or str(scenario.geometry_variant_id) == "065"


def build_corpus_v2_plan(
    *,
    short_section_ids: Sequence[str],
    full_section_ids: Sequence[str],
    scenarios: Sequence[Any],
    budget: CorpusBudget,
) -> pd.DataFrame:
    if not short_section_ids or not full_section_ids:
        raise ValueError("canonical V3 requires short-patch and full-section paths.")
    by_family = {
        family: [
            scenario
            for scenario in scenarios
            if str(scenario.geometry_family) == family
        ]
        for family in FAMILIES
    }
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        candidates = by_family[family]
        iid = [item for item in candidates if not _is_combination_holdout(item)]
        held_out = [item for item in candidates if _is_combination_holdout(item)]
        if not iid or not held_out:
            raise ValueError(
                f"scenario catalog cannot form IID and combination sets for {family}."
            )
        split_sequence = (
            ("training", budget.training_per_family),
            ("tuning", budget.tuning_per_family),
            ("calibration", budget.calibration_per_family),
        )
        short_sequence_index = 0
        for split, count in split_sequence:
            for split_index in range(count):
                scenario = iid[short_sequence_index % len(iid)]
                section_id = short_section_ids[
                    short_sequence_index % len(short_section_ids)
                ]
                parent = f"patch__{family}__{split}__{split_index:04d}"
                rows.append(
                    _row(
                        parent,
                        section_id,
                        scenario,
                        attempt_id=short_sequence_index,
                        corpus_role="short_patch",
                        split_role=split,
                        generalization_role="iid",
                    )
                )
                short_sequence_index += 1
        for role_index, (role, count, catalog) in enumerate((
            ("iid", budget.iid_section_per_family, iid),
            (
                "combination_holdout",
                budget.combination_holdout_per_family,
                held_out,
            ),
        )):
            section_offset = (
                family_index * budget.full_section_per_family
                + role_index * count
            )
            for index in range(count):
                scenario = catalog[index % len(catalog)]
                section_id = full_section_ids[
                    (section_offset + index) % len(full_section_ids)
                ]
                parent = f"section__{family}__{role}__{index:03d}"
                rows.append(
                    _row(
                        parent,
                        section_id,
                        scenario,
                        attempt_id=index,
                        corpus_role="full_section",
                        split_role="section_gate",
                        generalization_role=role,
                    )
                )
    for row in rows:
        row["quota_bucket"] = _quota_bucket(row)
    required_rows = list(rows)
    reserve_count = budget.max_candidate_attempts - len(required_rows)
    bucket_rank: dict[str, int] = {}
    for row in required_rows:
        bucket = str(row["quota_bucket"])
        row["candidate_rank"] = bucket_rank.get(bucket, 0)
        bucket_rank[bucket] = int(row["candidate_rank"]) + 1
    rows_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in required_rows:
        rows_by_bucket.setdefault(str(row["quota_bucket"]), []).append(row)
    reserve_buckets = tuple(sorted(rows_by_bucket))
    reserve_allocation = _allocate_reserves(
        rows_by_bucket,
        reserve_count=reserve_count,
    )

    reserve_index = 0
    for bucket in reserve_buckets:
        source_rows = rows_by_bucket[bucket]
        for local_index in range(reserve_allocation[bucket]):
            source = source_rows[local_index % len(source_rows)]
            row = dict(source)
            row["candidate_rank"] = bucket_rank[bucket]
            bucket_rank[bucket] += 1
            row["attempt_id"] = (
                int(source["attempt_id"]) + 10_000 + reserve_index
            )
            row["parent_realization_id"] = (
                f"reserve__{reserve_index:04d}__"
                f"{source['parent_realization_id']}"
            )
            rows.append(row)
            reserve_index += 1
    frame = pd.DataFrame.from_records(rows)
    if len(frame) != budget.max_candidate_attempts:
        raise RuntimeError("canonical corpus planner produced the wrong candidate count.")
    if frame["parent_realization_id"].duplicated().any():
        raise RuntimeError("canonical corpus planner produced duplicate parent IDs.")
    return frame


def _allocate_reserves(
    rows_by_bucket: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reserve_count: int,
) -> dict[str, int]:
    """Protect small section buckets before proportional reserve allocation."""
    buckets = tuple(sorted(rows_by_bucket))
    allocation = {bucket: 0 for bucket in buckets}
    remaining = int(reserve_count)
    if remaining >= len(buckets):
        for bucket in buckets:
            allocation[bucket] = 1
        remaining -= len(buckets)

    full_buckets = tuple(
        bucket for bucket in buckets if bucket.startswith("full_section:")
    )
    full_targets = {
        bucket: max(16, 2 * len(rows_by_bucket[bucket]))
        for bucket in full_buckets
    }
    while remaining > 0 and any(
        allocation[bucket] < full_targets[bucket]
        for bucket in full_buckets
    ):
        for bucket in full_buckets:
            if remaining <= 0:
                break
            if allocation[bucket] < full_targets[bucket]:
                allocation[bucket] += 1
                remaining -= 1

    if remaining <= 0:
        return allocation
    total_weight = sum(len(rows_by_bucket[bucket]) for bucket in buckets)
    exact = {
        bucket: remaining * len(rows_by_bucket[bucket]) / total_weight
        for bucket in buckets
    }
    proportional = {bucket: int(exact[bucket]) for bucket in buckets}
    remainder = remaining - sum(proportional.values())
    remainder_order = sorted(
        buckets,
        key=lambda bucket: (
            -(exact[bucket] - proportional[bucket]),
            bucket,
        ),
    )
    for bucket in remainder_order[:remainder]:
        proportional[bucket] += 1
    for bucket in buckets:
        allocation[bucket] += proportional[bucket]
    return allocation


def _quota_bucket(row: Mapping[str, Any]) -> str:
    if row["corpus_role"] == "short_patch":
        return (
            f"short_patch:{row['geometry_family']}:{row['split_role']}"
        )
    return (
        f"full_section:{row['geometry_family']}:"
        f"{row['generalization_role']}"
    )


def required_quota_counts(budget: CorpusBudget) -> dict[str, int]:
    """Return the canonical parent quota keyed by generation replacement bucket."""
    expected: dict[str, int] = {}
    for family in FAMILIES:
        for split, count in (
            ("training", budget.training_per_family),
            ("tuning", budget.tuning_per_family),
            ("calibration", budget.calibration_per_family),
        ):
            expected[f"short_patch:{family}:{split}"] = count
        expected[f"full_section:{family}:iid"] = budget.iid_section_per_family
        expected[
            f"full_section:{family}:combination_holdout"
        ] = budget.combination_holdout_per_family
    return expected


def published_quota_report(
    index: pd.DataFrame,
    budget: CorpusBudget,
) -> pd.DataFrame:
    """Describe actual canonical counts without turning shortages into I/O errors."""
    required = {
        "geometry_family",
        "corpus_role",
        "split_role",
        "generalization_role",
    }
    missing = sorted(required.difference(index.columns))
    if missing:
        raise ValueError(f"canonical V3 quota index lacks columns: {missing}")
    expected = required_quota_counts(budget)
    actual = {bucket: 0 for bucket in expected}
    for row in index.to_dict(orient="records"):
        bucket = _quota_bucket(row)
        if bucket not in actual:
            raise ValueError(f"published parent has unknown quota bucket: {bucket}")
        actual[bucket] += 1
    rows = []
    for bucket, expected_count in expected.items():
        actual_count = int(actual[bucket])
        rows.append({
            "quota_bucket": bucket,
            "expected": int(expected_count),
            "actual": actual_count,
            "shortfall": max(0, int(expected_count) - actual_count),
            "surplus": max(0, actual_count - int(expected_count)),
            "status": "complete" if actual_count == int(expected_count) else "warning",
        })
    return pd.DataFrame.from_records(rows)


def _row(
    parent: str,
    section_id: str,
    scenario: Any,
    *,
    attempt_id: int,
    corpus_role: str,
    split_role: str,
    generalization_role: str,
) -> dict[str, Any]:
    return {
        "section_id": str(section_id),
        "scenario_id": str(scenario.scenario_id),
        "duration_mode": str(scenario.duration_mode),
        "geometry_family": str(scenario.geometry_family),
        "geometry_direction": str(scenario.geometry_direction),
        "attempt_id": int(attempt_id),
        "parent_realization_id": parent,
        "corpus_role": corpus_role,
        "split_role": split_role,
        "generalization_role": generalization_role,
    }


def validate_published_quota(index: pd.DataFrame, budget: CorpusBudget) -> None:
    report = published_quota_report(index, budget)
    incomplete = report.loc[~report["status"].eq("complete")]
    if not incomplete.empty:
        details = ", ".join(
            f"{row.quota_bucket} expected={row.expected} actual={row.actual}"
            for row in incomplete.itertuples(index=False)
        )
        raise ValueError(f"canonical V3 quota mismatch: {details}")


__all__ = [
    "CorpusBudget",
    "build_corpus_v2_plan",
    "published_quota_report",
    "required_quota_counts",
    "validate_published_quota",
]
