"""Shared preflight, progress, and acceptance-QC support for Synthoseis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from cup.utils.logging import configure_run_logger


PROGRESS_COLUMNS = (
    "timestamp_utc",
    "phase",
    "sequence_index",
    "sequence_total",
    "section_id",
    "scenario_id",
    "geometry_family",
    "attempt_id",
    "parent_realization_id",
    "status",
    "reason",
    "elapsed_s",
    "phase_accepted_count",
    "phase_rejected_count",
    "scenario_completed_count",
    "scenario_total_count",
    "scenario_accepted_count",
    "scenario_rejected_count",
    "scenario_acceptance_fraction",
    "scenario_max_possible_acceptance_fraction",
    "failure_threshold_reachable",
    "warning_threshold_reachable",
)


def configure_generation_logger(
    output_dir: Path,
    *,
    sample_domain: str,
) -> logging.Logger:
    """Create one terminal and file logger without depending on GINN."""
    return configure_run_logger(
        output_dir,
        logger_name=f"cup.synthetic.generation.{sample_domain}",
        file_name="generation.log",
    )


class AttemptProgressLog:
    """Append and flush one auditable row after every attempted realization."""

    def __init__(
        self,
        path: Path,
        *,
        phase: str,
        plan: pd.DataFrame,
        qc_config: Mapping[str, Any],
        logger: logging.Logger,
        append: bool,
    ) -> None:
        self.path = Path(path)
        self.phase = str(phase)
        self.plan = plan.reset_index(drop=True)
        self.qc = dict(qc_config)
        self.logger = logger
        self.total = int(len(self.plan))
        self.accepted = 0
        self.rejected = 0
        self.scenario_totals = {
            (str(section), str(scenario)): int(count)
            for (section, scenario), count in self.plan.groupby(
                ["section_id", "scenario_id"], sort=False
            ).size().items()
        }
        self.scenario_counts: dict[tuple[str, str], dict[str, int]] = {
            key: {"completed": 0, "accepted": 0, "rejected": 0}
            for key in self.scenario_totals
        }
        self._failure_unreachable_reported: set[tuple[str, str]] = set()
        self._warning_unreachable_reported: set[tuple[str, str]] = set()
        mode = "a" if append else "w"
        write_header = (not append) or (not self.path.exists()) or self.path.stat().st_size == 0
        self._handle = self.path.open(mode, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=PROGRESS_COLUMNS)
        if write_header:
            self._writer.writeheader()
            self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> "AttemptProgressLog":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def record(
        self,
        row: Mapping[str, Any],
        *,
        sequence_index: int,
        status: str,
        reason: str,
        elapsed_s: float,
    ) -> dict[str, Any]:
        accepted = str(status) == "accepted"
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        key = (str(row["section_id"]), str(row["scenario_id"]))
        counts = self.scenario_counts[key]
        counts["completed"] += 1
        counts["accepted" if accepted else "rejected"] += 1
        total = self.scenario_totals[key]
        acceptance = counts["accepted"] / counts["completed"]
        maximum = (counts["accepted"] + total - counts["completed"]) / total
        failure = float(self.qc["failure_fraction"])
        warning = float(self.qc["warning_fraction"])
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
            "sequence_index": int(sequence_index),
            "sequence_total": self.total,
            "section_id": key[0],
            "scenario_id": key[1],
            "geometry_family": str(row.get("geometry_family", "")),
            "attempt_id": int(row["attempt_id"]),
            "parent_realization_id": str(row["parent_realization_id"]),
            "status": str(status),
            "reason": str(reason),
            "elapsed_s": float(elapsed_s),
            "phase_accepted_count": self.accepted,
            "phase_rejected_count": self.rejected,
            "scenario_completed_count": counts["completed"],
            "scenario_total_count": total,
            "scenario_accepted_count": counts["accepted"],
            "scenario_rejected_count": counts["rejected"],
            "scenario_acceptance_fraction": acceptance,
            "scenario_max_possible_acceptance_fraction": maximum,
            "failure_threshold_reachable": bool(maximum >= failure),
            "warning_threshold_reachable": bool(maximum >= warning),
        }
        self._writer.writerow(payload)
        self._handle.flush()
        log = self.logger.info if accepted else self.logger.warning
        log(
            "%s %d/%d | scenario=%s | attempt=%s | %s | %.2fs | "
            "scenario accepted=%d/%d, max=%.3f",
            self.phase,
            int(sequence_index),
            self.total,
            key[1],
            row["attempt_id"],
            status if not reason else f"{status}:{reason}",
            float(elapsed_s),
            counts["accepted"],
            counts["completed"],
            maximum,
        )
        if maximum < failure and key not in self._failure_unreachable_reported:
            self._failure_unreachable_reported.add(key)
            self.logger.warning(
                "%s scenario can no longer reach failure threshold: "
                "scenario=%s max_possible=%.3f threshold=%.3f",
                self.phase,
                key[1],
                maximum,
                failure,
            )
        if maximum < warning and key not in self._warning_unreachable_reported:
            self._warning_unreachable_reported.add(key)
            self.logger.warning(
                "%s scenario can no longer reach warning threshold: "
                "scenario=%s max_possible=%.3f threshold=%.3f",
                self.phase,
                key[1],
                maximum,
                warning,
            )
        return payload


def build_acceptance_catalog(
    plan: pd.DataFrame,
    *,
    accepted_parent_ids: Iterable[str],
    qc_config: Mapping[str, Any],
    development_limited: bool,
) -> pd.DataFrame:
    """Evaluate every planned scenario, including scenarios with zero accepts."""
    accepted = {str(value) for value in accepted_parent_ids}
    frame = plan.copy()
    frame["accepted"] = frame["parent_realization_id"].astype(str).isin(accepted)
    group_columns = ["section_id", "scenario_id"]
    metadata_columns = [
        column
        for column in ("geometry_family", "geometry_direction", "duration_mode")
        if column in frame
    ]
    catalog = (
        frame.groupby(group_columns, sort=False, dropna=False)
        .agg(
            attempt_count=("parent_realization_id", "size"),
            accepted_count=("accepted", "sum"),
        )
        .reset_index()
    )
    if metadata_columns:
        metadata = frame[group_columns + metadata_columns].drop_duplicates(group_columns)
        catalog = catalog.merge(metadata, on=group_columns, how="left", validate="one_to_one")
    catalog["accepted_count"] = catalog["accepted_count"].astype(int)
    catalog["rejected_count"] = catalog["attempt_count"] - catalog["accepted_count"]
    catalog["acceptance_fraction"] = (
        catalog["accepted_count"] / catalog["attempt_count"]
    )
    if development_limited:
        catalog["acceptance_status"] = "development_limit_no_verdict"
        return catalog
    minimum = int(qc_config["minimum_attempts_per_scenario"])
    failure = float(qc_config["failure_fraction"])
    warning = float(qc_config["warning_fraction"])
    catalog["acceptance_status"] = np.where(
        catalog["attempt_count"] < minimum,
        "insufficient_attempts",
        np.where(
            catalog["acceptance_fraction"] < failure,
            "failed",
            np.where(
                catalog["acceptance_fraction"] < warning,
                "warning",
                "ok",
            ),
        ),
    )
    return catalog


@dataclass(frozen=True)
class PreflightResult:
    accepted_plan: pd.DataFrame
    attempts: pd.DataFrame
    catalog: pd.DataFrame
    rejection_details: list[dict[str, Any]]

    @property
    def failed(self) -> pd.DataFrame:
        return self.catalog[
            self.catalog["acceptance_status"].isin(
                {"failed", "insufficient_attempts"}
            )
        ].copy()


def run_attempt_preflight(
    plan: pd.DataFrame,
    *,
    validator: Callable[[Mapping[str, Any]], Any],
    rejection_exceptions: Sequence[type[BaseException]],
    qc_config: Mapping[str, Any],
    output_dir: Path,
    logger: logging.Logger,
    development_limited: bool,
) -> PreflightResult:
    """Run cheap structural validation before any HDF5 forward generation."""
    logger.info("preflight started: %d planned attempts", len(plan))
    records: list[dict[str, Any]] = []
    accepted_ids: list[str] = []
    rejection_details: list[dict[str, Any]] = []
    progress_path = output_dir / "attempt_progress.csv"
    with AttemptProgressLog(
        progress_path,
        phase="preflight",
        plan=plan,
        qc_config=qc_config,
        logger=logger,
        append=False,
    ) as progress:
        for sequence_index, row in enumerate(plan.to_dict(orient="records"), start=1):
            started = time.perf_counter()
            try:
                validator(row)
                status = "accepted"
                reason = ""
                accepted_ids.append(str(row["parent_realization_id"]))
            except tuple(rejection_exceptions) as exc:
                status = "rejected"
                reason = f"{type(exc).__name__}:{exc}"
                details = list(getattr(exc, "details", []) or [{}])
                rejection_details.extend(
                    {
                        **dict(row),
                        "realization_id": str(row["parent_realization_id"]),
                        "status": "rejected",
                        "reason": reason,
                        **dict(detail),
                    }
                    for detail in details
                )
            payload = progress.record(
                row,
                sequence_index=sequence_index,
                status=status,
                reason=reason,
                elapsed_s=time.perf_counter() - started,
            )
            records.append(payload)
    attempts = pd.DataFrame.from_records(records, columns=PROGRESS_COLUMNS)
    attempts.to_csv(output_dir / "preflight_attempts.csv", index=False)
    catalog = build_acceptance_catalog(
        plan,
        accepted_parent_ids=accepted_ids,
        qc_config=qc_config,
        development_limited=development_limited,
    )
    catalog.to_csv(output_dir / "preflight_scenario_catalog.csv", index=False)
    accepted_plan = plan[
        plan["parent_realization_id"].astype(str).isin(set(accepted_ids))
    ].reset_index(drop=True)
    failed_count = int(
        catalog["acceptance_status"].isin({"failed", "insufficient_attempts"}).sum()
    )
    logger.info(
        "preflight finished: accepted=%d rejected=%d failed_scenarios=%d",
        len(accepted_plan),
        len(plan) - len(accepted_plan),
        failed_count,
    )
    return PreflightResult(
        accepted_plan=accepted_plan,
        attempts=attempts,
        catalog=catalog,
        rejection_details=rejection_details,
    )


def acceptance_enforcement(qc_config: Mapping[str, Any]) -> str:
    value = str(qc_config.get("enforcement", "warn"))
    if value not in {"warn", "fail_fast"}:
        raise ValueError("acceptance_qc.enforcement must be warn or fail_fast.")
    return value


__all__ = [
    "AttemptProgressLog",
    "PreflightResult",
    "acceptance_enforcement",
    "build_acceptance_catalog",
    "configure_generation_logger",
    "run_attempt_preflight",
]
