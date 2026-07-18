"""Shared Synthoseis-lite pipeline implementation.

The scientific differences between the time and depth workflows are exposed
by the two domain adapters.  Everything which gives a benchmark its identity
or its lifecycle lives here: attempt planning, preflight, parent transactions,
view materialisation, publication and acceptance reporting.  Keeping this
module as the only owner of that lifecycle is deliberate; a new seismic view
must not require a second time/depth orchestration implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import h5py
import numpy as np
import pandas as pd

from cup.synthetic.core.view_runner import SeismicViewResult, generate_seismic_views
from cup.synthetic.core.views import SeismicViewSpec, resolve_view_specs
from cup.synthetic.core.v5_artifacts import publish_v5_indexes
from cup.synthetic.core.artifacts import (
    build_attempt_plan,
    limit_attempt_plan,
    rejection_reason_summary,
    validate_debug_attempt_limit,
)
from cup.synthetic.core.field_runner import (
    AttemptProgressLog,
    acceptance_enforcement,
    build_acceptance_catalog,
    configure_generation_logger,
    run_attempt_preflight,
    stable_records_frame,
)
from cup.synthetic.core.rejections import StagedRejection
from cup.synthetic.core.writer import write_benchmark_sample, write_benchmark_view
from cup.synthetic.core.records import BenchmarkSample, BenchmarkView
from cup.synthetic.reporting.figures import write_generation_figures
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
        candidate = directory.parent / f".{directory.name}.{uuid.uuid4().hex}.staging"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"unable to allocate staging directory beside {directory}")


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


class SyntheticDomainAdapter(Protocol):
    """The small set of domain-dependent operations at the shared seam."""

    sample_domain: str
    sample_unit: str
    depth_basis: str | None
    generator_family: str

    def validate_axis(self, sample_axis: np.ndarray) -> None:
        """Validate the regular axis consumed by the shared view pipeline."""

    def forward_with_parameters(
        self, phase_degrees: float, shift: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-run domain forward modelling for one forward-parameter prefix."""

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


@dataclass(frozen=True)
class SeismicViewContext:
    """Parent-realization data needed by :class:`SeismicViewPipeline`."""

    realization_id: str
    base_seismic: np.ndarray
    public_valid_mask: np.ndarray
    operator_source_support: np.ndarray
    lateral_m: np.ndarray
    sample_axis: np.ndarray


class SeismicViewPipeline:
    """Shared ordered operator registry and materialization implementation."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        global_seed: int,
        generator_family: str,
        domain_adapter: SyntheticDomainAdapter,
    ) -> None:
        self.specs: tuple[SeismicViewSpec, ...] = resolve_view_specs(config)
        self.global_seed = int(global_seed)
        self.generator_family = str(generator_family)
        self.domain_adapter = domain_adapter

    def generate(
        self,
        context: SeismicViewContext,
        *,
        perturbed_forward: Callable[[float, float], tuple[np.ndarray, np.ndarray]]
        | None = None,
    ) -> list[SeismicViewResult]:
        axis = np.asarray(context.sample_axis, dtype=np.float64)
        self.domain_adapter.validate_axis(axis)
        callback = perturbed_forward
        if callback is None and any(spec.forward_operator_ids for spec in self.specs):
            callback = getattr(self.domain_adapter, "forward_with_parameters", None)
        return generate_seismic_views(
            base_seismic=context.base_seismic,
            valid_mask=context.public_valid_mask,
            operator_source_support=context.operator_source_support,
            lateral_m=context.lateral_m,
            sample_axis=axis,
            view_specs=self.specs,
            global_seed=self.global_seed,
            generator_family=self.generator_family,
            realization_id=str(context.realization_id),
            axis_unit=str(self.domain_adapter.sample_unit),
            perturbed_forward=callback,
        )


@dataclass
class GenerationAttempt:
    """Result returned by a domain adapter for one parent attempt.

    The adapter returns scientific objects and diagnostic rows only.  It never
    publishes an index or decides run-level acceptance; those decisions belong
    to :class:`SyntheticBenchmarkPipeline`.
    """

    parent_realization_id: str
    sample: BenchmarkSample | None
    index_rows: list[dict[str, Any]] = field(default_factory=list)
    realization_row: dict[str, Any] | None = None
    view_rows: list[dict[str, Any]] = field(default_factory=list)
    view_result_rows: list[dict[str, Any]] = field(default_factory=list)
    qc_row: dict[str, Any] = field(default_factory=dict)
    rejection_rows: list[dict[str, Any]] = field(default_factory=list)
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
    scenarios: Sequence[Any] = field(default_factory=tuple)
    attempts_per_scenario: int | None = None
    held_out_geometry_family: str | None = None
    geometry_families: Sequence[str] | None = None
    debug_attempt_limit: int | None = None
    input_contracts: Mapping[str, Any] = field(default_factory=dict)
    preflight_summary_prefix: Mapping[str, Any] = field(default_factory=dict)
    manifest_fields: Mapping[str, Any] = field(default_factory=dict)
    validate_attempt: Callable[[Mapping[str, Any]], Any] | None = None
    build_attempt: Callable[[Mapping[str, Any], h5py.File | None, bool], GenerationAttempt] | None = None
    view_context: Callable[
        [BenchmarkSample, str],
        tuple[
            SeismicViewContext,
            Callable[[float, float], tuple[np.ndarray, np.ndarray]] | None,
        ],
    ] | None = None
    write_domain_outputs: Callable[[Path, Mapping[str, list[dict[str, Any]]]], None] | None = None

    def resolve_plan(self, debug_attempt_limit: int | None) -> pd.DataFrame:
        """Build the neutral attempt plan exactly once in the shared Pipeline."""
        if self.plan is None:
            if (
                not self.section_ids
                or not self.scenarios
                or self.attempts_per_scenario is None
                or self.held_out_geometry_family is None
            ):
                raise RuntimeError("generation session lacks attempt-plan inputs")
            plan = build_attempt_plan(
                section_ids=tuple(str(value) for value in self.section_ids),
                scenarios=self.scenarios,
                attempts_per_scenario=int(self.attempts_per_scenario),
                held_out_geometry_family=str(self.held_out_geometry_family),
                geometry_families=self.geometry_families,
            )
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
    and all declared views, so a failed view cannot leave a partial parent.
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
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
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
        views = config.get("seismic_views")
        if not isinstance(views, Mapping):
            raise ValueError("Synthoseis v5 requires seismic_views configuration.")
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
        except Exception:
            self._close_generation_logger()
            shutil.rmtree(staging, ignore_errors=True)
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
            rejection_exceptions=(StagedRejection, ValueError, FloatingPointError),
            qc_config=acceptance_qc,
            output_dir=output_dir,
            logger=logger,
            development_limited=development_limited,
        )
        enforcement = acceptance_enforcement(acceptance_qc)
        preflight_summary = {
            **dict(session.preflight_summary_prefix),
            "sample_domain": session.sample_domain,
            "status": "failed" if not preflight.failed.empty else "ok",
            "enforcement": enforcement,
            "planned_attempts": int(len(plan)),
            "accepted_attempts": int(len(preflight.accepted_plan)),
            "rejected_attempts": int(len(plan) - len(preflight.accepted_plan)),
            "failed_scenario_count": int(len(preflight.failed)),
        }
        _write_json(output_dir / "preflight_summary.json", preflight_summary)
        if preflight.accepted_plan.empty:
            raise RuntimeError(f"{session.sample_domain}_generation_preflight_no_accepted_realizations")
        if enforcement == "fail_fast" and not preflight.failed.empty:
            failed = preflight.failed[["section_id", "scenario_id", "acceptance_status"]].to_dict(orient="records")
            raise RuntimeError(f"{session.sample_domain}_generation_preflight_acceptance_qc_failed:{failed}")

        h5_path = output_dir / "synthetic_benchmark.h5"
        view_pipeline = self.build_view_pipeline(config)
        if view_pipeline.specs and session.view_context is None:
            raise TypeError(
                "generation session must provide view_context when views are configured"
            )
        h5_attrs = {
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
        for key in ("science_revision", "projection_contract_version", "seismic_view_contract_version", "seismic_operator_contract_version", "random_stream_contract_version"):
            value = session.manifest_fields.get(key)
            if value is not None:
                h5_attrs[key] = value

        index_rows: list[dict[str, Any]] = []
        realization_rows: list[dict[str, Any]] = []
        view_rows: list[dict[str, Any]] = []
        qc_rows: list[dict[str, Any]] = []
        view_result_rows: list[dict[str, Any]] = []
        rejection_rows: list[dict[str, Any]] = list(preflight.rejection_details)
        domain_rows: dict[str, list[dict[str, Any]]] = {}
        with AttemptProgressLog(
            output_dir / "attempt_progress.csv",
            phase="generation",
            plan=preflight.accepted_plan,
            qc_config=acceptance_qc,
            logger=logger,
            append=True,
        ) as progress, h5py.File(h5_path, "w") as h5:
            for key, value in h5_attrs.items():
                h5.attrs[key] = value
            for sequence_index, row in enumerate(preflight.accepted_plan.to_dict(orient="records"), start=1):
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
                        raise RuntimeError(result.reason or "generation attempt rejected")
                    sample = result.sample
                    if sample is None:
                        raise RuntimeError("accepted generation attempt has no BenchmarkSample")
                    reference = None if qc_only else write_benchmark_sample(h5, sample)
                    owner_path = "" if reference is None else reference.hdf5_group
                    scenario = sample.truth.scenario
                    local_index_rows: list[dict[str, Any]] = []
                    local_realization_rows: list[dict[str, Any]] = []
                    local_view_rows: list[dict[str, Any]] = []
                    local_view_result_rows: list[dict[str, Any]] = []
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
                        "evaluation_role": str(row.get("evaluation_role", "")),
                        "attempt_id": int(row.get("attempt_id", 0)),
                        "status": "ok",
                        "reasons": "",
                        "sample_kind": "base",
                        "hdf5_group": owner_path,
                        "seismic_input_dataset": "" if not owner_path else f"{owner_path}/seismic/seismic_observed",
                        "seismic_model_consistent_dataset": "" if not owner_path else f"{owner_path}/seismic/seismic_model_consistent",
                        "valid_mask_dataset": "" if not owner_path else f"{owner_path}/masks/valid_mask",
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
                        "evaluation_role": base_record["evaluation_role"],
                        "parent_realization_id": parent_id,
                        "hdf5_group": owner_path,
                        "base_seismic_dataset": base_record["seismic_input_dataset"],
                        "model_consistent_seismic_dataset": base_record["seismic_model_consistent_dataset"],
                        "target_log_ai_dataset": "" if not owner_path else f"{owner_path}/truth/model_target_log_ai",
                        "canonical_background_dataset": "" if not owner_path else f"{owner_path}/priors/canonical_background_log_ai",
                        "target_increment_dataset": "" if not owner_path else f"{owner_path}/targets/target_increment_log_ai",
                        "valid_mask_dataset": base_record["valid_mask_dataset"],
                        "n_valid": base_record["valid_sample_count"],
                    })
                    if view_pipeline.specs:
                        if session.view_context is None:
                            raise TypeError(
                                "generation session must provide view_context when views are configured"
                            )
                        view_context, perturbed_forward = session.view_context(sample, parent_id)
                        view_results = view_pipeline.generate(
                            view_context, perturbed_forward=perturbed_forward
                        )
                    else:
                        view_results = []
                    for view in view_results:
                        view_path = "" if not owner_path else f"{owner_path}/seismic_views/{view.view_id}"
                        if not qc_only:
                            public_mask = np.asarray(view_context.public_valid_mask, dtype=bool)
                            positive_gain = np.where(
                                public_mask,
                                np.asarray(view.positive_gain, dtype=np.float64),
                                1.0,
                            )
                            additive_noise = np.where(
                                public_mask,
                                np.asarray(view.additive_noise, dtype=np.float64),
                                0.0,
                            )
                            write_benchmark_view(
                                h5,
                                BenchmarkView(
                                    owner_realization_id=parent_id,
                                    view_id=view.view_id,
                                    seismic_observed=view.seismic_observed,
                                    positive_gain=positive_gain,
                                    additive_noise=additive_noise,
                                    metadata=dict(view.metadata),
                                    qc=view.qc,
                                    sample_domain=session.sample_domain,
                                ),
                            )
                        metadata = dict(view.metadata)
                        local_index_rows.append({
                            **base_record,
                            "sample_id": f"{parent_id}__view__{view.view_id}",
                            "realization_id": parent_id,
                            "sample_kind": "seismic_view",
                            "view_id": view.view_id,
                            "hdf5_group": view_path,
                            "seismic_input_dataset": "" if not view_path else f"{view_path}/seismic_observed",
                            "operator_trace_dataset": "" if not view_path else f"{view_path}/operator_trace_json",
                            "view_spec_sha256": metadata["view_spec_sha256"],
                            "view_spec_canonical_json": metadata["view_spec_canonical_json"],
                            "operator_ids_json": json.dumps(metadata["operator_ids"], sort_keys=True),
                            "operator_kinds_json": json.dumps(metadata["operator_kinds"], sort_keys=True),
                            "operator_parameters_json": json.dumps(metadata.get("operator_parameters", {}), sort_keys=True),
                            "operator_contract_versions_json": json.dumps(metadata.get("operator_contract_versions", {}), sort_keys=True),
                            "random_stream_identity_json": json.dumps(metadata.get("random_stream_identity", {}), sort_keys=True),
                        })
                        local_view_rows.append({
                            "realization_id": parent_id,
                            "parent_realization_id": parent_id,
                            "view_id": view.view_id,
                            "sample_domain": session.sample_domain,
                            "sample_unit": session.sample_unit,
                            "evaluation_role": base_record["evaluation_role"],
                            "hdf5_group": view_path,
                            "seismic_observed_dataset": "" if not view_path else f"{view_path}/seismic_observed",
                            "seismic_input_dataset": "" if not view_path else f"{view_path}/seismic_observed",
                            "seismic_model_consistent_dataset": base_record["seismic_model_consistent_dataset"],
                            "valid_mask_dataset": base_record["valid_mask_dataset"],
                            "view_spec_sha256": metadata["view_spec_sha256"],
                            "view_spec_canonical_json": metadata["view_spec_canonical_json"],
                            "operator_ids_json": json.dumps(metadata["operator_ids"], sort_keys=True),
                            "operator_kinds_json": json.dumps(metadata["operator_kinds"], sort_keys=True),
                            "operator_parameters_json": json.dumps(metadata.get("operator_parameters", {}), sort_keys=True),
                            "operator_contract_versions_json": json.dumps(metadata.get("operator_contract_versions", {}), sort_keys=True),
                            "random_stream_identity_json": json.dumps(metadata.get("random_stream_identity", {}), sort_keys=True),
                            "operator_trace_dataset": "" if not view_path else f"{view_path}/operator_trace_json",
                            "n_valid": base_record["valid_sample_count"],
                        })
                        local_view_result_rows.append({"parent_realization_id": parent_id, **metadata, **dict(view.qc)})
                    index_rows.extend(local_index_rows)
                    realization_rows.extend(local_realization_rows)
                    view_rows.extend(local_view_rows)
                    view_result_rows.extend(local_view_result_rows)
                    qc_rows.append({**base_record, **dict(result.qc_row), **dict(sample.qc)})
                    for name, rows in result.domain_rows.items():
                        domain_rows.setdefault(str(name), []).extend(rows)
                    status = "accepted"
                except (StagedRejection, ValueError, FloatingPointError, RuntimeError) as exc:
                    failed_group = f"/realizations/{parent_id}"
                    if failed_group in h5:
                        del h5[failed_group]
                    reason = f"{type(exc).__name__}:{exc}"
                    rejection_rows.append({**dict(row), "status": "rejected", "reason": reason})
                    qc_rows.append({**dict(row), "sample_id": parent_id, "status": "rejected", "reasons": reason})
                progress.record(
                    row,
                    sequence_index=sequence_index,
                    status=status,
                    reason=reason,
                    elapsed_s=time.perf_counter() - started,
                )

        index = stable_records_frame(
            index_rows,
            sort_by=("section_id", "scenario_id", "attempt_id", "sample_kind", "sample_id"),
        )
        self.publish_indexes(output_dir, realization_rows, view_rows)
        stable_records_frame(qc_rows, sort_by=("section_id", "scenario_id", "attempt_id", "sample_kind", "sample_id")).to_csv(output_dir / "generation_qc.csv", index=False)
        stable_records_frame(view_result_rows, sort_by=("parent_realization_id", "view_id")).to_csv(output_dir / "seismic_view_results.csv", index=False)
        rejection_frame = stable_records_frame(rejection_rows, sort_by=("section_id", "scenario_id", "attempt_id", "reason"))
        rejection_frame.to_csv(output_dir / "generation_rejection_details.csv", index=False)
        rejection_summary = rejection_reason_summary(rejection_frame, index)
        rejection_summary.to_csv(output_dir / "rejection_reason_summary.csv", index=False)
        successful_parent_ids = [str(row.get("realization_id") or row.get("parent_realization_id")) for row in realization_rows]
        catalog = build_acceptance_catalog(plan, accepted_parent_ids=successful_parent_ids, qc_config=acceptance_qc, development_limited=development_limited)
        catalog.to_csv(output_dir / "scenario_catalog.csv", index=False)
        failed_scenarios = catalog["acceptance_status"].isin({"failed", "insufficient_attempts"})
        failure_reason = f"{session.sample_domain}_generation_no_accepted_realizations" if not successful_parent_ids else ""
        if not failure_reason and not development_limited and enforcement == "fail_fast" and bool(failed_scenarios.any()):
            failure_reason = f"{session.sample_domain}_generation_acceptance_qc_failed"
        completed_with_warnings = bool(not development_limited and not failure_reason and failed_scenarios.any())
        if session.write_domain_outputs is not None:
            session.write_domain_outputs(output_dir, domain_rows)
        figure_summary = write_generation_figures(
            output_dir,
            config.get("figures", {}),
            suite="field_conditioned",
            qc_only=qc_only,
        )
        contract_fields: dict[str, Any] = {}
        if not failure_reason:
            contract_fields = {
                "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
                "contract_fingerprint_sha256": contract_fingerprint_sha256(
                    contract_schema_version=session.schema_version,
                    semantics={
                        "science_revision": session.manifest_fields.get("science_revision", ""),
                        "projection_contract_version": session.manifest_fields.get("projection_contract_version", ""),
                        "seismic_view_contract_version": session.manifest_fields.get("seismic_view_contract_version", ""),
                        "seismic_operator_contract_version": session.manifest_fields.get("seismic_operator_contract_version", ""),
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
                        "seismic_view_index": output_dir / "seismic_view_index.csv",
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
            "attempts_per_scenario": int(plan.groupby(["section_id", "scenario_id"], sort=False).size().min()) if not plan.empty else 0,
            "accepted_parent_realizations": int(len(successful_parent_ids)),
            "rejected_parent_realizations": int(len(plan) - len(successful_parent_ids)),
            "acceptance_qc": acceptance_qc,
            "preflight": preflight_summary,
            "seismic_views": dict(config.get("seismic_views") or {}),
            "seismic_view_count": int(len(view_rows)),
            "rejection_reason_summary": [] if rejection_summary.empty else rejection_summary.to_dict(orient="records"),
            "quality_warnings": [] if not completed_with_warnings else ["scenario_acceptance_qc_failed"],
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
            "rejected_realizations": int(len(plan) - len(successful_parent_ids)),
            "failed_scenario_count": int(failed_scenarios.sum()),
        }
        _write_json(output_dir / "run_summary.json", summary)
        if failure_reason:
            raise RuntimeError(failure_reason)
        logger.info("Synthoseis generation finished: status=%s accepted=%d rejected=%d", summary["status"], summary["accepted_realizations"], summary["rejected_realizations"])
        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
        return summary

    def build_view_pipeline(self, config: Mapping[str, Any]) -> SeismicViewPipeline:
        """Build the shared view seam after validating domain-level identity."""
        if str(config.get("sample_domain") or "").casefold() != str(
            self.domain_adapter.sample_domain
        ).casefold():
            raise ValueError("Synthetic config and domain adapter sample_domain differ.")
        views = config.get("seismic_views")
        if not isinstance(views, Mapping):
            raise ValueError("Synthoseis v5 requires seismic_views configuration.")
        return SeismicViewPipeline(
            views,
            global_seed=int(config.get("global_seed")),
            generator_family=str(
                config.get("generator_family")
                or getattr(self.domain_adapter, "generator_family", "")
                or ""
            ),
            domain_adapter=self.domain_adapter,
        )

    def publish_indexes(
        self,
        output_dir: str,
        realization_rows: Sequence[Mapping[str, Any]],
        view_rows: Sequence[Mapping[str, Any]],
    ):
        """Publish the same successful-only dual indexes for either domain."""
        return publish_v5_indexes(output_dir, realization_rows, view_rows)


__all__ = [
    "GenerationAttempt",
    "GenerationSession",
    "SeismicViewContext",
    "SeismicViewPipeline",
    "SyntheticBenchmarkPipeline",
    "SyntheticDomainAdapter",
]
