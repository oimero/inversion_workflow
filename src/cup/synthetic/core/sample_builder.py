"""Ordered construction of a complete in-memory base Benchmark sample."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol

import numpy as np

from cup.synthetic.core.lfm import LfmPolicy, build_lfm_products
from cup.synthetic.core.projection import project_truth_to_model_grid
from cup.synthetic.core.records import (
    BenchmarkResiduals,
    BenchmarkSample,
    DomainPreparation,
    ForwardResult,
    ProjectedTruth,
)
from cup.synthetic.core.rejections import BenchmarkBuildRejected
from cup.synthetic.core.truth import SyntheticTruth


class ForwardAdapter(Protocol):
    def forward(
        self,
        truth: SyntheticTruth,
        projected: ProjectedTruth,
        preparation: DomainPreparation,
    ) -> ForwardResult: ...


@dataclass(frozen=True)
class CanonicalIncrementPolicy:
    contract: object


@dataclass(frozen=True)
class BenchmarkBuildPolicy:
    require_projection_support: bool = True
    require_forward_support: bool = True
    domain_metadata: Mapping[str, Any] = field(default_factory=dict)


class BenchmarkBuilder:
    def build(
        self,
        *,
        truth: SyntheticTruth,
        preparation: DomainPreparation,
        forward_adapter: ForwardAdapter,
        canonical_policy: CanonicalIncrementPolicy,
        lfm_policy: LfmPolicy,
        build_policy: BenchmarkBuildPolicy,
    ) -> BenchmarkSample:
        projected = project_truth_to_model_grid(
            truth,
            preparation.model_axis,
        )
        if lfm_policy.zone_id_model is None:
            lfm_policy = replace(
                lfm_policy, zone_id_model=projected.zone_id_model
            )
        forward = forward_adapter.forward(truth, projected, preparation)
        valid = np.asarray(projected.geometric_valid_mask_model, dtype=bool)
        support_failures: list[str] = []
        if build_policy.require_projection_support and np.any(
            valid & ~projected.projection_support_model
        ):
            support_failures.append("projection")
        if build_policy.require_forward_support:
            if np.any(valid & ~forward.support.observed):
                support_failures.append("observed_forward")
            if np.any(valid & ~forward.support.physics):
                support_failures.append("physics_forward")
        if support_failures:
            raise BenchmarkBuildRejected(
                ["valid_mask_support_not_finite"],
                diagnostics={"failed_support": tuple(support_failures)},
                details=[
                    {
                        "reason": "valid_mask_support_not_finite",
                        "support": name,
                    }
                    for name in support_failures
                ],
            )
        lfm = build_lfm_products(
            projected.model_target_log_ai,
            preparation.model_axis.coordinates,
            canonical_policy.contract,
            lateral_coordinates=truth.lateral_m,
            valid_mask=valid,
            policy=lfm_policy,
        )
        required = {
            "target_log_ai": projected.model_target_log_ai,
            "canonical_background_log_ai": lfm.canonical_background_log_ai,
            "target_increment_log_ai": lfm.target_increment_log_ai,
            "input_lfm_log_ai": lfm.controlled_degraded_log_ai,
            "seismic_observed": forward.seismic_observed,
            "seismic_model_consistent": forward.seismic_model_consistent,
        }
        for name, values in required.items():
            if np.any(valid & ~np.isfinite(values)):
                raise BenchmarkBuildRejected(
                    [f"valid_mask_support_not_finite:{name}"],
                    diagnostics={"array": name},
                    details=[
                        {
                            "reason": f"valid_mask_support_not_finite:{name}",
                            "array": name,
                        }
                    ],
                )
        return BenchmarkSample(
            truth=truth,
            projected=projected,
            forward=forward,
            canonical_background_log_ai=lfm.canonical_background_log_ai,
            target_increment_log_ai=lfm.target_increment_log_ai,
            input_lfm_canonical_log_ai=lfm.canonical_background_log_ai,
            input_lfm_controlled_degraded_log_ai=lfm.controlled_degraded_log_ai,
            residuals=BenchmarkResiduals(
                residual_vs_lfm_ideal=lfm.residual_vs_lfm_ideal,
                residual_vs_lfm_controlled_degraded=(
                    lfm.residual_vs_lfm_controlled_degraded
                ),
            ),
            valid_mask=valid,
            qc={
                **dict(truth.diagnostics),
                **dict(forward.qc),
                **dict(lfm.qc),
            },
            domain_metadata=dict(build_policy.domain_metadata),
        )


__all__ = [
    "BenchmarkBuildPolicy",
    "BenchmarkBuilder",
    "CanonicalIncrementPolicy",
    "ForwardAdapter",
]
