"""Shared Synthoseis-lite pipeline implementation.

The scientific differences between the time and depth workflows are exposed
by the two domain adapters.  Everything which gives a benchmark its identity
or its lifecycle lives here: attempt planning, preflight, parent transactions,
publication and acceptance reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import h5py
import numpy as np
import pandas as pd

from cup.synthetic.core.canonical_artifact import publish_realization_index
from cup.synthetic.core.corpus import (
    CorpusBudget,
    FAMILIES,
    build_corpus_v2_plan,
    published_quota_report,
    required_quota_counts,
    select_accepted_quota,
)
from cup.synthetic.core.artifacts import (
    limit_attempt_plan,
    rejection_reason_summary,
    validate_debug_attempt_limit,
)
from cup.synthetic.core.field_runner import (
    AttemptProgressLog,
    build_acceptance_catalog,
    configure_generation_logger,
    run_attempt_preflight,
    stable_records_frame,
)
from cup.synthetic.core.rejections import BenchmarkBuildRejected, StagedRejection
from cup.synthetic.core.writer import (
    validate_structured_truth_tables,
    write_structured_sample,
)
from cup.synthetic.core.records import StructuredSampleRecord
from cup.synthetic.reporting.figures import (
    write_figure_failure_manifest,
    write_generation_figures,
)
from cup.synthetic.schemas import STRUCTURED_ARTIFACT_VERSION
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    repo_relative_path,
    write_json as _write_json,
)


def _new_staging_directory(directory: Path) -> Path:
    """Create a writable sibling without tempfile's restrictive Windows ACL."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(32):
        candidate = directory.parent / f".run-{uuid.uuid4().hex}.staging"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"unable to allocate staging directory beside {directory}")


def _record_staging_failure(staging: Path, exc: BaseException) -> None:
    """Best-effort failure provenance; completed staging artifacts stay inspectable."""
    try:
        _write_json(staging / "run_failure.json", {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "staging_preserved": True,
        })
    except Exception:
        pass


def _rewrite_published_paths(
    value: Any,
    *,
    staging_dir: Path,
    published_dir: Path,
    repo_root: Path | None,
) -> Any:
    """Replace staging-directory aliases in published JSON metadata.

    Scientific files are built in a sibling staging directory so publication
    can be atomic.  Metadata must nevertheless name the final directory; this
    recursive rewrite handles both absolute Windows paths and portable
    repository-relative paths without changing unrelated user values.
    """
    replacements: list[tuple[str, str]] = [
        (str(staging_dir.resolve()), str(published_dir.resolve())),
        (staging_dir.resolve().as_posix(), published_dir.resolve().as_posix()),
    ]
    if repo_root is not None:
        try:
            replacements.append(
                (
                    repo_relative_path(staging_dir, root=repo_root),
                    repo_relative_path(published_dir, root=repo_root),
                )
            )
        except ValueError:
            pass
    replacements = sorted(
        {(str(source), str(target)) for source, target in replacements if source},
        key=lambda item: len(item[0]),
        reverse=True,
    )

    def rewrite_text(text: str) -> str:
        for source, target in replacements:
            if text == source:
                return target
            for separator in ("/", "\\"):
                if text.startswith(source + separator):
                    return target + text[len(source) :]
        return text

    if isinstance(value, dict):
        return {
            key: _rewrite_published_paths(
                child,
                staging_dir=staging_dir,
                published_dir=published_dir,
                repo_root=repo_root,
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_published_paths(
                child,
                staging_dir=staging_dir,
                published_dir=published_dir,
                repo_root=repo_root,
            )
            for child in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _rewrite_published_paths(
                child,
                staging_dir=staging_dir,
                published_dir=published_dir,
                repo_root=repo_root,
            )
            for child in value
        )
    if isinstance(value, str):
        return rewrite_text(value)
    return value


def _rewrite_json_paths_in_directory(
    directory: Path,
    *,
    staging_dir: Path,
    published_dir: Path,
    repo_root: Path | None,
) -> None:
    """Fix JSON provenance emitted by an adapter before atomic publication."""
    for path in directory.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        rewritten = _rewrite_published_paths(
            payload,
            staging_dir=staging_dir,
            published_dir=published_dir,
            repo_root=repo_root,
        )
        if rewritten != payload:
            path.write_text(
                json.dumps(rewritten, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def _portable_figure_summary(
    value: Mapping[str, Any], *, repo_root: Path | None
) -> dict[str, Any]:
    """Store figure artifact paths using the same repository-relative contract."""
    result = dict(value)
    if repo_root is None:
        return result
    for key in ("figure_manifest", "skipped_figures"):
        raw = result.get(key)
        if raw:
            try:
                result[key] = repo_relative_path(Path(str(raw)), root=repo_root)
            except ValueError:
                pass
    generated = result.get("generated")
    if isinstance(generated, list):
        converted: list[Any] = []
        for item in generated:
            try:
                converted.append(repo_relative_path(Path(str(item)), root=repo_root))
            except ValueError:
                converted.append(item)
        result["generated"] = converted
    return result


def _validate_published_artifact(directory: Path, *, qc_only: bool) -> None:
    """Run the canonical reader as the last generation publication gate.

    The reader is intentionally imported here, rather than at module import
    time, so the shared pipeline keeps the writer/reader dependency one-way.
    QC-only runs deliberately do not satisfy the training-consumable
    artifact contract and therefore retain their separate publication path.
    """
    if qc_only:
        return
    from cup.synthetic.readers.structured import StructuredSyntheticBenchmark

    StructuredSyntheticBenchmark(directory)


class SyntheticDomainAdapter(Protocol):
    """The small set of domain-dependent operations at the shared seam."""

    sample_domain: str
    sample_unit: str
    depth_basis: str | None
    generator_family: str

    def validate_axis(self, sample_axis: np.ndarray) -> None:
        """Validate the regular axis consumed by the shared pipeline."""

    # The following methods are the deep lifecycle seam.  They are intentionally
    # methods on the adapter rather than callbacks supplied by each entrypoint.
    # Small test adapters can implement them without importing a domain runner.
    def prepare_generation(
        self,
        config: Mapping[str, Any],
        calibration: Any,
        *,
        output_dir: Path,
        **runtime: Any,
    ) -> Any:
        """Prepare one domain generation session for the shared pipeline."""

    def prepare_calibration(
        self,
        config: Mapping[str, Any],
        *,
        output_dir: Path,
        **runtime: Any,
    ) -> Any:
        """Prepare one domain calibration result for the shared publisher."""


@dataclass
class GenerationAttempt:
    """Result returned by a domain adapter for one parent attempt.

    The adapter returns scientific objects and diagnostic rows only.  It never
    publishes an index or decides run-level acceptance; those decisions belong
    to :class:`SyntheticBenchmarkPipeline`.
    """

    parent_realization_id: str
    sample: StructuredSampleRecord | None
    qc_row: dict[str, Any] = field(default_factory=dict)
    domain_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.sample is not None and not self.reason


@dataclass
class GenerationSession:
    """Domain adapter session consumed by the shared generation lifecycle."""

    # ``plan`` is retained for small test adapters and already-materialised
    # callers.  Production adapters return the neutral plan ingredients below;
    # the shared Pipeline then builds and limits the attempt plan.
    plan: pd.DataFrame | None
    acceptance_qc: Mapping[str, Any]
    development_limited: bool
    sample_domain: str
    sample_unit: str
    depth_basis: str | None
    schema_version: str
    generator_family: str
    hdf5_attributes: Mapping[str, Any]
    section_ids: Sequence[str] = field(default_factory=tuple)
    section_roles: Mapping[str, str] = field(default_factory=dict)
    scenarios: Sequence[Any] = field(default_factory=tuple)
    corpus_budget: CorpusBudget | None = None
    debug_attempt_limit: int | None = None
    input_contracts: Mapping[str, Any] = field(default_factory=dict)
    preflight_summary_prefix: Mapping[str, Any] = field(default_factory=dict)
    manifest_fields: Mapping[str, Any] = field(default_factory=dict)
    validate_attempt: Callable[[Mapping[str, Any]], Any] | None = None
    build_attempt: Callable[[Mapping[str, Any], h5py.File | None, bool], GenerationAttempt] | None = None
    write_domain_outputs: Callable[[Path, Mapping[str, list[dict[str, Any]]]], None] | None = None
    structured_artifact_oracle: Callable[
        [Path, Any, Sequence[str]], Mapping[str, Any]
    ] | None = None

    def resolve_plan(self, debug_attempt_limit: int | None) -> pd.DataFrame:
        """Build the neutral attempt plan exactly once in the shared Pipeline."""
        if self.plan is None:
            if self.corpus_budget is not None:
                short_ids = tuple(
                    section_id
                    for section_id in self.section_ids
                    if self.section_roles.get(section_id) == "short_patch"
                )
                full_ids = tuple(
                    section_id
                    for section_id in self.section_ids
                    if self.section_roles.get(section_id) == "full_section"
                )
                plan = build_corpus_v2_plan(
                    short_section_ids=short_ids,
                    full_section_ids=full_ids,
                    scenarios=self.scenarios,
                    budget=self.corpus_budget,
                )
                return limit_attempt_plan(
                    plan,
                    debug_attempt_limit
                    if debug_attempt_limit is not None
                    else self.debug_attempt_limit,
                )
            raise RuntimeError("generation session lacks canonical V2 corpus budget")
        else:
            plan = self.plan
        return limit_attempt_plan(
            plan,
            debug_attempt_limit
            if debug_attempt_limit is not None
            else self.debug_attempt_limit,
        )

    def validate(self, row: Mapping[str, Any]) -> None:
        if self.validate_attempt is None:
            raise RuntimeError("generation session has no preflight validator")
        self.validate_attempt(row)

    def build(
        self,
        row: Mapping[str, Any],
        h5: h5py.File | None,
        qc_only: bool,
    ) -> GenerationAttempt:
        if self.build_attempt is None:
            raise RuntimeError("generation session has no parent builder")
        return self.build_attempt(row, h5, qc_only)


class SyntheticBenchmarkPipeline:
    """The single domain-neutral calibrate/generate lifecycle.

    Adapters prepare domain science and return a :class:`GenerationSession`;
    this class owns every public benchmark transition after that point.  In
    particular, it owns the parent transaction which includes the base sample
    and all declared outputs, so a failure cannot leave a partial parent.
    """

    def __init__(self, domain_adapter: SyntheticDomainAdapter) -> None:
        self.domain_adapter = domain_adapter

    def _close_generation_logger(self) -> None:
        import logging

        logger = logging.getLogger(
            f"cup.synthetic.generation.{self.domain_adapter.sample_domain}"
        )
        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

    def _validate_config(self, config: Mapping[str, Any]) -> None:
        if str(config.get("sample_domain") or "").casefold() != str(
            self.domain_adapter.sample_domain
        ).casefold():
            raise ValueError("Synthetic config and domain adapter sample_domain differ.")

    def calibrate(
        self,
        config: Mapping[str, Any],
        *,
        output_dir: str | Path,
        **runtime: Any,
    ) -> Any:
        """Run the adapter's scientific calibration through one publisher.

        A domain adapter returns a ``CalibrationResult``-like object with a
        ``publish`` method.  Keeping the method small makes it possible to use
        the same pipeline in lightweight tests while leaving the domain fit in
        the adapter.
        """
        self._validate_config(config)
        prepare = getattr(self.domain_adapter, "prepare_calibration", None)
        if prepare is None:
            raise TypeError("domain adapter must implement prepare_calibration")
        directory = Path(output_dir)
        if directory.exists():
            raise FileExistsError(directory)
        staging = _new_staging_directory(directory)
        try:
            result = prepare(config, output_dir=staging, **runtime)
            publisher = getattr(result, "publish", None)
            if publisher is None:
                raise TypeError("prepare_calibration must return an object with publish()")
            summary = publisher(staging, repo_root=runtime.get("repo_root"))
            if not isinstance(summary, Mapping):
                raise TypeError("calibration publisher must return a mapping summary")
            _rewrite_json_paths_in_directory(
                staging,
                staging_dir=staging,
                published_dir=directory,
                repo_root=runtime.get("repo_root"),
            )
            summary = _rewrite_published_paths(
                dict(summary),
                staging_dir=staging,
                published_dir=directory,
                repo_root=runtime.get("repo_root"),
            )
            staging.replace(directory)
            return dict(summary)
        except Exception as exc:
            _record_staging_failure(staging, exc)
            raise

    def generate(
        self,
        config: Mapping[str, Any],
        calibration: Any,
        *,
        output_dir: str | Path,
        debug_attempt_limit: int | None = None,
        geometry_families: Sequence[str] | None = None,
        qc_only: bool = False,
        **runtime: Any,
    ) -> Any:
        self._validate_config(config)
        parsed_limit = validate_debug_attempt_limit(debug_attempt_limit)
        directory = Path(output_dir)
        if directory.exists():
            raise FileExistsError(directory)
        prepare = getattr(self.domain_adapter, "prepare_generation", None)
        if prepare is None:
            raise TypeError("domain adapter must implement prepare_generation")
        staging = _new_staging_directory(directory)
        try:
            session = prepare(
                config,
                calibration,
                output_dir=staging,
                debug_attempt_limit=parsed_limit,
                geometry_families=geometry_families,
                qc_only=bool(qc_only),
                **runtime,
            )
            if not isinstance(session, GenerationSession):
                raise TypeError("prepare_generation must return GenerationSession")
            oracle_callback = runtime.get("structured_artifact_oracle")
            if oracle_callback is not None:
                if not callable(oracle_callback):
                    raise TypeError("structured_artifact_oracle must be callable")
                session.structured_artifact_oracle = oracle_callback
            summary = self._run_generation_session(
                config,
                session,
                staging,
                calibration=calibration,
                qc_only=bool(qc_only),
                repo_root=runtime.get("repo_root"),
                debug_attempt_limit=parsed_limit,
                published_output_dir=directory,
            )
            staging.replace(directory)
            return summary
        except Exception as exc:
            self._close_generation_logger()
            _record_staging_failure(staging, exc)
            raise

    def _run_generation_session(
        self,
        config: Mapping[str, Any],
        session: GenerationSession,
        output_dir: Path,
        *,
        calibration: Any,
        qc_only: bool,
        repo_root: Path | None,
        debug_attempt_limit: int | None,
        published_output_dir: Path,
    ) -> dict[str, Any]:
        logger = configure_generation_logger(output_dir, sample_domain=session.sample_domain)
        plan = session.resolve_plan(debug_attempt_limit)
        development_limited = bool(
            session.development_limited or debug_attempt_limit is not None
        )
        plan.to_csv(output_dir / "attempt_plan.csv", index=False)
        acceptance_qc = dict(session.acceptance_qc)
        preflight = run_attempt_preflight(
            plan,
            validator=session.validate,
            rejection_exceptions=(StagedRejection,),
            qc_config=acceptance_qc,
            output_dir=output_dir,
            logger=logger,
            development_limited=development_limited,
        )
        preflight_warnings = preflight.warnings
        preflight_has_usable_parent = not preflight.accepted_plan.empty
        preflight_summary = {
            **dict(session.preflight_summary_prefix),
            "sample_domain": session.sample_domain,
            "status": (
                "failed"
                if not preflight_has_usable_parent
                else ("completed_with_warnings" if not preflight_warnings.empty else "success")
            ),
            "usable": preflight_has_usable_parent,
            "planned_attempts": int(len(plan)),
            "accepted_attempts": int(len(preflight.accepted_plan)),
            "rejected_attempts": int(len(plan) - len(preflight.accepted_plan)),
            "severe_warning_scenario_count": int(len(preflight_warnings)),
        }
        _write_json(output_dir / "preflight_summary.json", preflight_summary)
        if not preflight_has_usable_parent:
            raise RuntimeError(f"{session.sample_domain}_generation_preflight_no_accepted_realizations")
        generation_plan = preflight.accepted_plan
        expected_quota: dict[str, int] = {}
        if session.corpus_budget is not None and not development_limited:
            # This is the final hard preflight gate: every bucket must have
            # enough truth-valid candidates before expensive forward work.
            # Formal generation keeps all accepted reserves so a later staged
            # scientific rejection can consume the next candidate in-bucket.
            select_accepted_quota(
                preflight.accepted_plan,
                session.corpus_budget,
            )
            expected_quota = required_quota_counts(session.corpus_budget)

        h5_path = output_dir / "synthetic_benchmark.h5"
        h5_attrs = {
            "artifact_type": "structured_synthetic_benchmark",
            "artifact_version": STRUCTURED_ARTIFACT_VERSION,
            "schema": session.schema_version,
            "schema_version": session.schema_version,
            "sample_domain": session.sample_domain,
            "sample_unit": session.sample_unit,
            "generator_family": session.generator_family,
            "suite": "field_conditioned",
            "global_seed": int(config["global_seed"]),
            "qc_only": bool(qc_only),
        }
        if session.depth_basis:
            h5_attrs["depth_basis"] = session.depth_basis
        h5_attrs.update(dict(session.hdf5_attributes))
        for key in (
            "science_revision",
            "projection_contract_version",
            "random_stream_contract_version",
        ):
            value = session.manifest_fields.get(key)
            if value is not None:
                h5_attrs[key] = value

        index_rows: list[dict[str, Any]] = []
        realization_rows: list[dict[str, Any]] = []
        qc_rows: list[dict[str, Any]] = []
        rejection_rows: list[dict[str, Any]] = list(preflight.rejection_details)
        domain_rows: dict[str, list[dict[str, Any]]] = {}
        generated_trace_count = 0
        generation_elapsed_s = 0.0
        generation_attempt_count = 0
        attempted_generation_rows: list[dict[str, Any]] = []
        accepted_quota = {bucket: 0 for bucket in expected_quota}
        with AttemptProgressLog(
            output_dir / "attempt_progress.csv",
            phase="generation",
            plan=generation_plan,
            qc_config=acceptance_qc,
            logger=logger,
            append=True,
        ) as progress, h5py.File(h5_path, "w") as h5:
            for key, value in h5_attrs.items():
                h5.attrs[key] = value
            for row in generation_plan.to_dict(orient="records"):
                quota_bucket = str(row.get("quota_bucket") or "")
                if expected_quota:
                    if quota_bucket not in expected_quota:
                        raise ValueError(
                            f"generation candidate has unknown quota bucket: {quota_bucket}"
                        )
                    if accepted_quota[quota_bucket] >= expected_quota[quota_bucket]:
                        continue
                generation_attempt_count += 1
                sequence_index = generation_attempt_count
                attempted_generation_rows.append(dict(row))
                started = time.perf_counter()
                parent_id = str(row["parent_realization_id"])
                status = "rejected"
                reason = ""
                try:
                    result = session.build(row, h5, qc_only)
                    if not isinstance(result, GenerationAttempt):
                        raise TypeError("generation session build must return GenerationAttempt")
                    if result.parent_realization_id != parent_id:
                        raise ValueError("generation parent identity changed")
                    if not result.accepted:
                        rejection_reason = (
                            result.reason or "generation attempt returned no sample"
                        )
                        raise BenchmarkBuildRejected(
                            [rejection_reason],
                            diagnostics={"parent_realization_id": parent_id},
                            details=[{"reason": rejection_reason}],
                        )
                    sample = result.sample
                    if sample is None:
                        raise RuntimeError(
                            "accepted generation attempt has no StructuredSampleRecord"
                        )
                    corpus_role = str(row.get("corpus_role") or "")
                    if session.corpus_budget is not None:
                        expected_width = {
                            "short_patch": 25,
                            "full_section": 121,
                        }.get(corpus_role)
                        if expected_width is None:
                            raise ValueError("canonical V2 parent has no corpus_role.")
                        lateral = np.asarray(sample.truth.lateral_m, dtype=np.float64)
                        if not development_limited and (
                            lateral.size != expected_width or (
                            lateral.size > 1
                            and not np.allclose(
                                np.diff(lateral),
                                25.0,
                                rtol=0.0,
                                atol=1.0e-6,
                            )
                            )
                        ):
                            raise ValueError(
                                f"{corpus_role} requires {expected_width} traces at 25 m."
                            )
                    try:
                        validate_structured_truth_tables(sample)
                    except ValueError as exc:
                        raise BenchmarkBuildRejected(
                            ["invalid_structured_truth_tables"],
                            diagnostics={
                                "parent_realization_id": parent_id,
                                "validation_error": str(exc),
                            },
                            details=[{
                                "reason": "invalid_structured_truth_tables",
                                "validation_error": str(exc),
                            }],
                        ) from exc
                    reference = None if qc_only else write_structured_sample(h5, sample)
                    owner_path = "" if reference is None else reference.hdf5_group
                    scenario = sample.truth.scenario
                    local_index_rows: list[dict[str, Any]] = []
                    local_realization_rows: list[dict[str, Any]] = []
                    base_record = {
                        "sample_id": parent_id,
                        "realization_id": parent_id,
                        "parent_realization_id": parent_id,
                        "sample_domain": session.sample_domain,
                        "sample_unit": session.sample_unit,
                        "depth_basis": session.depth_basis or "",
                        "suite": "field_conditioned",
                        "section_id": str(row.get("section_id", "")),
                        "scenario_id": str(row.get("scenario_id", scenario.scenario_id)),
                        "geometry_family": str(row.get("geometry_family", scenario.geometry_family)),
                        "duration_mode": str(row.get("duration_mode", scenario.duration_mode)),
                        "corpus_role": str(row.get("corpus_role", "")),
                        "split_role": str(row.get("split_role", "")),
                        "generalization_role": str(
                            row.get("generalization_role", "")
                        ),
                        "attempt_id": int(row.get("attempt_id", 0)),
                        "status": "ok",
                        "reasons": "",
                        "sample_kind": "base",
                        "hdf5_group": owner_path,
                        "seismic_input_dataset": (
                            "" if reference is None else reference.seismic_input_dataset
                        ),
                        "seismic_model_consistent_dataset": (
                            ""
                            if reference is None
                            else reference.seismic_model_consistent_dataset
                        ),
                        "valid_mask_dataset": (
                            "" if reference is None else reference.valid_mask_dataset
                        ),
                        "valid_sample_count": int(np.count_nonzero(sample.valid_mask)),
                    }
                    local_index_rows.append(base_record)
                    local_realization_rows.append({
                        "realization_id": parent_id,
                        "sample_domain": session.sample_domain,
                        "sample_unit": session.sample_unit,
                        "depth_basis": session.depth_basis or "",
                        "section_id": base_record["section_id"],
                        "scenario_id": base_record["scenario_id"],
                        "geometry_family": base_record["geometry_family"],
                        "duration_mode": base_record["duration_mode"],
                        "suite": "field_conditioned",
                        "corpus_role": base_record["corpus_role"],
                        "split_role": base_record["split_role"],
                        "generalization_role": base_record["generalization_role"],
                        "parent_realization_id": parent_id,
                        "attempt_id": base_record["attempt_id"],
                        "hdf5_group": owner_path,
                        "seismic_dataset": base_record["seismic_input_dataset"],
                        "lfm_dataset": "" if reference is None else reference.lfm_dataset,
                        "model_consistent_seismic_dataset": base_record["seismic_model_consistent_dataset"],
                        "model_log_ai_dataset": (
                            "" if reference is None else reference.model_log_ai_dataset
                        ),
                        "valid_mask_dataset": base_record["valid_mask_dataset"],
                        "n_valid": base_record["valid_sample_count"],
                    })
                    index_rows.extend(local_index_rows)
                    realization_rows.extend(local_realization_rows)
                    qc_rows.append({**base_record, **dict(result.qc_row), **dict(sample.qc)})
                    for name, rows in result.domain_rows.items():
                        domain_rows.setdefault(str(name), []).extend(rows)
                    status = "accepted"
                    if expected_quota:
                        accepted_quota[quota_bucket] += 1
                    generated_trace_count += int(sample.truth.lateral_m.size)
                    generation_elapsed_s += time.perf_counter() - started
                    h5.flush()
                except StagedRejection as exc:
                    failed_group = f"/realizations/{parent_id}"
                    if failed_group in h5:
                        del h5[failed_group]
                    reason = f"{type(exc).__name__}:{exc}"
                    rejection_rows.append({**dict(row), "status": "rejected", "reason": reason})
                    qc_rows.append({**dict(row), "sample_id": parent_id, "status": "rejected", "reasons": reason})
                    logger.warning(
                        "generation candidate rejected; trying the next in quota bucket "
                        "%s | parent=%s | %s",
                        quota_bucket,
                        parent_id,
                        reason,
                    )
                except Exception:
                    failed_group = f"/realizations/{parent_id}"
                    if failed_group in h5:
                        del h5[failed_group]
                    raise
                progress.record(
                    row,
                    sequence_index=sequence_index,
                    status=status,
                    reason=reason,
                    elapsed_s=time.perf_counter() - started,
                )
                if expected_quota and all(
                    accepted_quota[bucket] >= required
                    for bucket, required in expected_quota.items()
                ):
                    break

        corpus_estimate: dict[str, Any] = {}
        if (
            development_limited
            and session.corpus_budget is not None
            and generated_trace_count > 0
            and h5_path.is_file()
        ):
            budget = session.corpus_budget
            full_trace_count = len(FAMILIES) * (
                budget.short_per_family * 25
                + budget.full_section_per_family * 121
            )
            estimated_bytes = int(
                h5_path.stat().st_size
                / generated_trace_count
                * full_trace_count
            )
            estimated_seconds = (
                generation_elapsed_s
                / generated_trace_count
                * full_trace_count
            )
            corpus_estimate = {
                "schema": "structured_synthetic_corpus_estimate_v2",
                "smoke_hdf5_bytes": int(h5_path.stat().st_size),
                "smoke_trace_count": generated_trace_count,
                "estimated_full_hdf5_bytes": estimated_bytes,
                "estimated_full_hdf5_gib": estimated_bytes / 1024**3,
                "estimated_generation_seconds": estimated_seconds,
                "estimated_generation_hours": estimated_seconds / 3600.0,
                "size_limit_gib": 3.5,
                "runtime_limit_hours": 5.0,
                "within_limits": (
                    estimated_bytes <= int(3.5 * 1024**3)
                    and estimated_seconds <= 5.0 * 3600.0
                ),
            }
            _write_json(output_dir / "corpus_estimate.json", corpus_estimate)

        index = stable_records_frame(
            index_rows,
            sort_by=("section_id", "scenario_id", "attempt_id", "sample_kind", "sample_id"),
        )
        realization_frame = stable_records_frame(
            realization_rows,
            sort_by=("corpus_role", "split_role", "realization_id"),
        )
        quota_report = pd.DataFrame()
        quota_shortfall_count = 0
        if session.corpus_budget is not None and not development_limited:
            quota_report = published_quota_report(
                realization_frame,
                session.corpus_budget,
            )
            quota_report.to_csv(output_dir / "quota_report.csv", index=False)
            quota_shortfall_count = int(quota_report["shortfall"].sum())
            if quota_shortfall_count:
                short = quota_report.loc[quota_report["shortfall"].gt(0)]
                logger.warning(
                    "generation exhausted available candidates with %d missing parents "
                    "across %d quota buckets; publishing the usable corpus with warnings",
                    quota_shortfall_count,
                    len(short),
                )
        publish_realization_index(output_dir, realization_rows)
        stable_records_frame(qc_rows, sort_by=("section_id", "scenario_id", "attempt_id", "sample_kind", "sample_id")).to_csv(output_dir / "generation_qc.csv", index=False)
        rejection_frame = stable_records_frame(rejection_rows, sort_by=("section_id", "scenario_id", "attempt_id", "reason"))
        rejection_frame.to_csv(output_dir / "generation_rejection_details.csv", index=False)
        rejection_summary = rejection_reason_summary(rejection_frame, index)
        rejection_summary.to_csv(output_dir / "rejection_reason_summary.csv", index=False)
        successful_parent_ids = sorted(
            {
                str(row.get("realization_id") or row.get("parent_realization_id"))
                for row in realization_rows
                if str(row.get("realization_id") or row.get("parent_realization_id"))
            }
        )
        structured_oracle_report: dict[str, Any] = {}
        if not qc_only and successful_parent_ids:
            _validate_published_artifact(output_dir, qc_only=False)
            if session.structured_artifact_oracle is not None:
                try:
                    candidate_report = session.structured_artifact_oracle(
                        output_dir,
                        calibration,
                        successful_parent_ids,
                    )
                    if not isinstance(candidate_report, Mapping):
                        raise TypeError("structured artifact Oracle must return a mapping")
                    structured_oracle_report = dict(candidate_report)
                except Exception as exc:
                    structured_oracle_report = {
                        "schema": "structured_synthetic_benchmark_oracle_report_v1",
                        "passed": False,
                        "trace_count": 0,
                        "parent_count": 0,
                        "failure_count": 1,
                        "metrics": {},
                        "failures": [{
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }],
                    }
                _write_json(
                    output_dir / "oracle_report.json",
                    structured_oracle_report,
                )
                if not bool(structured_oracle_report.get("passed", False)):
                    raise RuntimeError(
                        "structured_synthetic_benchmark_oracle_failed: "
                        f"{structured_oracle_report.get('failures', [])[:3]}"
                    )
            else:
                structured_oracle_report = {
                    "schema": "structured_synthetic_benchmark_oracle_report_v1",
                    "requested": False,
                    "passed": None,
                    "reason": "oracle callback was not requested",
                }
                _write_json(
                    output_dir / "oracle_report.json",
                    structured_oracle_report,
                )
        parent_quality_warning_count = int(sum(
            int(row.get("quality_warning_count") or 0)
            for row in qc_rows
            if str(row.get("status") or "") != "rejected"
        ))
        attempted_ids = {
            str(row["parent_realization_id"])
            for row in attempted_generation_rows
        }
        preflight_rejected_ids = set(
            preflight.attempts.loc[
                preflight.attempts["status"].eq("rejected"),
                "parent_realization_id",
            ].astype(str)
        )
        catalog_plan = plan.loc[
            plan["parent_realization_id"].astype(str).isin(
                attempted_ids | preflight_rejected_ids
            )
        ].reset_index(drop=True)
        catalog = build_acceptance_catalog(
            catalog_plan,
            accepted_parent_ids=successful_parent_ids,
            qc_config=acceptance_qc,
            development_limited=development_limited,
        )
        catalog.to_csv(output_dir / "scenario_catalog.csv", index=False)
        warning_scenarios = catalog["acceptance_status"].eq("severe_warning")
        failure_reason = f"{session.sample_domain}_generation_no_accepted_realizations" if not successful_parent_ids else ""
        completed_with_warnings = bool(
            not development_limited
            and not failure_reason
            and (
                warning_scenarios.any()
                or parent_quality_warning_count > 0
                or quota_shortfall_count > 0
            )
        )
        diagnostic_warnings: list[dict[str, str]] = []
        if session.write_domain_outputs is not None:
            try:
                session.write_domain_outputs(output_dir, domain_rows)
            except Exception as exc:
                diagnostic_warnings.append({
                    "diagnostic": "domain_outputs",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                logger.warning(
                    "optional domain diagnostic outputs failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        try:
            figure_summary = write_generation_figures(
                output_dir,
                config.get("figures", {}),
                suite="field_conditioned",
                qc_only=qc_only,
            )
        except Exception as exc:
            diagnostic_warnings.append({
                "diagnostic": "figures",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            logger.warning(
                "optional generation figures failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            figure_summary = write_figure_failure_manifest(
                output_dir,
                scope="generation",
                exc=exc,
            )
        if diagnostic_warnings:
            _write_json(output_dir / "diagnostic_warnings.json", {
                "schema": "synthetic_diagnostic_warnings_v1",
                "warnings": diagnostic_warnings,
            })
            if not development_limited and not failure_reason:
                completed_with_warnings = True
        contract_fields: dict[str, Any] = {}
        if not failure_reason and not qc_only:
            contract_fields = {
                "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
                "contract_fingerprint_sha256": contract_fingerprint_sha256(
                    contract_schema_version=session.schema_version,
                    semantics={
                        "science_revision": session.manifest_fields.get("science_revision", ""),
                        "projection_contract_version": session.manifest_fields.get("projection_contract_version", ""),
                        "random_stream_contract_version": session.manifest_fields.get("random_stream_contract_version", ""),
                        "sample_domain": session.sample_domain,
                        "sample_unit": session.sample_unit,
                        "depth_basis": session.depth_basis or "",
                        "generator_family": session.generator_family,
                    },
                    business_config=dict(config),
                    input_contracts=session.input_contracts,
                    primary_artifacts={
                        "synthetic_benchmark": h5_path,
                        "realization_index": output_dir / "realization_index.csv",
                        "oracle_report": output_dir / "oracle_report.json",
                    },
                ),
            }
        manifest = {
            "schema": session.schema_version,
            "schema_version": session.schema_version,
            **dict(session.manifest_fields),
            **contract_fields,
            "status": "failed" if failure_reason else ("development_limited" if development_limited else ("completed_with_warnings" if completed_with_warnings else "success")),
            "failure_reason": failure_reason,
            "input_contracts": dict(session.input_contracts),
            "sample_domain": session.sample_domain,
            "sample_unit": session.sample_unit,
            "depth_basis": session.depth_basis,
            "development_limited": development_limited,
            "qc_only": bool(qc_only),
            "training_consumable": not bool(qc_only),
            "global_seed": int(config["global_seed"]),
            "n_scenarios": int(plan["scenario_id"].nunique()) if "scenario_id" in plan else 0,
            "candidate_attempts": int(len(plan)),
            "selected_parent_attempts": int(generation_attempt_count),
            "accepted_parent_realizations": int(len(successful_parent_ids)),
            "rejected_parent_realizations": int(
                len(preflight_rejected_ids)
                + generation_attempt_count
                - len(successful_parent_ids)
            ),
            "unused_candidate_attempts": int(
                len(plan)
                - len(preflight_rejected_ids)
                - generation_attempt_count
            ),
            "quota_shortfall_count": quota_shortfall_count,
            "quota_report": (
                str(output_dir / "quota_report.csv")
                if not quota_report.empty
                else ""
            ),
            "acceptance_qc": acceptance_qc,
            "preflight": preflight_summary,
            "oracle_report": (
                str(output_dir / "oracle_report.json")
                if not qc_only
                else ""
            ),
            "oracle_requested": bool(
                not qc_only and session.structured_artifact_oracle is not None
            ),
            "oracle_passed": (
                structured_oracle_report.get("passed")
                if not qc_only
                else None
            ),
            "rejection_reason_summary": [] if rejection_summary.empty else rejection_summary.to_dict(orient="records"),
            "usable": not bool(failure_reason),
            "quality_warnings": (
                ([] if not bool(warning_scenarios.any()) else ["scenario_acceptance_qc_warning"])
                + ([] if not parent_quality_warning_count else ["parent_scientific_qc_warning"])
                + ([] if not quota_shortfall_count else ["canonical_quota_shortfall"])
                + ([] if not diagnostic_warnings else ["optional_diagnostic_failure"])
            ),
            "diagnostic_warnings": diagnostic_warnings,
            "corpus_estimate": corpus_estimate,
            "parent_quality_warning_count": parent_quality_warning_count,
            "figures": _portable_figure_summary(
                figure_summary,
                repo_root=repo_root,
            ),
        }
        manifest = _rewrite_published_paths(
            manifest,
            staging_dir=output_dir,
            published_dir=published_output_dir,
            repo_root=repo_root,
        )
        _write_json(output_dir / "benchmark_manifest.json", manifest)
        summary = {
            **manifest,
            "accepted_realizations": int(len(successful_parent_ids)),
            "rejected_realizations": int(
                len(preflight_rejected_ids)
                + generation_attempt_count
                - len(successful_parent_ids)
            ),
            "severe_warning_scenario_count": int(warning_scenarios.sum()),
        }
        _write_json(output_dir / "run_summary.json", summary)
        if failure_reason:
            raise RuntimeError(failure_reason)
        try:
            _validate_published_artifact(output_dir, qc_only=qc_only)
        except Exception as exc:
            raise RuntimeError(
                f"{session.sample_domain}_generation_final_artifact_validation_failed: {exc}"
            ) from exc
        logger.info("Synthoseis generation finished: status=%s accepted=%d rejected=%d", summary["status"], summary["accepted_realizations"], summary["rejected_realizations"])
        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
        return summary

__all__ = [
    "GenerationAttempt",
    "GenerationSession",
    "SyntheticBenchmarkPipeline",
    "SyntheticDomainAdapter",
]
