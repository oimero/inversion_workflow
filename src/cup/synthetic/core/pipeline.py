"""Shared Synthoseis-lite pipeline seams.

The domain runners own their scientific inputs and forward kernels, while this
module owns the domain-neutral view contract.  Keeping this seam explicit is
important: adding a view must not require a second time/depth implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from cup.synthetic.core.view_runner import SeismicViewResult, generate_seismic_views
from cup.synthetic.core.views import SeismicViewSpec, resolve_view_specs
from cup.synthetic.core.v5_artifacts import publish_v5_indexes


class SyntheticDomainAdapter(Protocol):
    """The small set of domain-dependent operations at the shared seam."""

    sample_domain: str
    sample_unit: str
    depth_basis: str | None

    def validate_axis(self, sample_axis: np.ndarray) -> None:
        """Validate the regular axis consumed by the shared view pipeline."""

    def forward_with_parameters(
        self, phase_degrees: float, shift: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-run domain forward modelling for one forward-parameter prefix."""


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


class SyntheticBenchmarkPipeline:
    """Domain-neutral orchestration façade used by time and depth adapters.

    Existing runners still supply their calibration and parent-realization
    builders, but all view execution goes through this shared implementation.
    ``calibrate`` and ``generate`` intentionally accept adapter callbacks so
    the seam remains useful to lightweight test adapters as well as production
    domain runners.
    """

    def __init__(self, domain_adapter: SyntheticDomainAdapter) -> None:
        self.domain_adapter = domain_adapter

    def calibrate(
        self,
        config: Mapping[str, Any],
        *,
        calibrator: Callable[[Mapping[str, Any], SyntheticDomainAdapter], Any],
    ) -> Any:
        if str(config.get("sample_domain") or "").casefold() != str(
            self.domain_adapter.sample_domain
        ).casefold():
            raise ValueError("Synthetic config and domain adapter sample_domain differ.")
        return calibrator(config, self.domain_adapter)

    def generate(
        self,
        config: Mapping[str, Any],
        calibration: Any,
        *,
        generator: Callable[
            [Mapping[str, Any], Any, SyntheticDomainAdapter, SeismicViewPipeline],
            Any,
        ],
    ) -> Any:
        if str(config.get("sample_domain") or "").casefold() != str(
            self.domain_adapter.sample_domain
        ).casefold():
            raise ValueError("Synthetic config and domain adapter sample_domain differ.")
        views = config.get("seismic_views")
        if not isinstance(views, Mapping):
            raise ValueError("Synthoseis v5 requires seismic_views configuration.")
        view_pipeline = self.build_view_pipeline(config)
        return generator(config, calibration, self.domain_adapter, view_pipeline)

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
            generator_family=str(config.get("generator_family") or ""),
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
    "SeismicViewContext",
    "SeismicViewPipeline",
    "SyntheticBenchmarkPipeline",
    "SyntheticDomainAdapter",
]
