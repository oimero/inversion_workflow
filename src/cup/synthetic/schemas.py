"""Schema versions shared by synthetic-workflow artifact producers and consumers."""

from __future__ import annotations


BENCHMARK_SCHEMA_VERSION = "synthoseis_lite_v5"
SCIENCE_REVISION = "synthoseis_lite_science_v3"
PROJECTION_CONTRACT_VERSION = "finite_support_projection_v1"
SEISMIC_VIEW_CONTRACT_VERSION = "seismic_views_v1"
SEISMIC_OPERATOR_CONTRACT_VERSION = "seismic_operators_v1"
RANDOM_STREAM_CONTRACT_VERSION = "synthoseis_random_v3"
FROZEN_BENCHMARK_SCHEMA_VERSION = "synthoseis_lite_v3"
LEGACY_BENCHMARK_SCHEMA_VERSION = "synthoseis_lite_v1"
REPORT_SCHEMA_VERSION = "synthoseis_lite_report_v2"
ROCK_PHYSICS_ANALYSIS_SCHEMA_VERSION = "rock_physics_analysis_v4"
DEPTH_FORWARD_MODEL_INPUTS_RUN_SCHEMA_VERSION = "depth_forward_model_inputs_v1"
FORWARD_MODEL_INPUTS_SCHEMA_VERSION = "forward_model_inputs_v3"


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "SCIENCE_REVISION",
    "PROJECTION_CONTRACT_VERSION",
    "SEISMIC_VIEW_CONTRACT_VERSION",
    "SEISMIC_OPERATOR_CONTRACT_VERSION",
    "RANDOM_STREAM_CONTRACT_VERSION",
    "DEPTH_FORWARD_MODEL_INPUTS_RUN_SCHEMA_VERSION",
    "FROZEN_BENCHMARK_SCHEMA_VERSION",
    "FORWARD_MODEL_INPUTS_SCHEMA_VERSION",
    "LEGACY_BENCHMARK_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "ROCK_PHYSICS_ANALYSIS_SCHEMA_VERSION",
]


SCIENCE_CONTRACT = {
    "science_revision": SCIENCE_REVISION,
    "projection_contract_version": PROJECTION_CONTRACT_VERSION,
    "seismic_view_contract_version": SEISMIC_VIEW_CONTRACT_VERSION,
    "seismic_operator_contract_version": SEISMIC_OPERATOR_CONTRACT_VERSION,
    "random_stream_contract_version": RANDOM_STREAM_CONTRACT_VERSION,
}


def require_science_contract(record: object, *, label: str) -> None:
    """Require the complete science-v3 contract on a mapping or HDF5 attrs."""
    for key, expected in SCIENCE_CONTRACT.items():
        try:
            actual = record[key]  # type: ignore[index]
        except (KeyError, TypeError):
            actual = None
        if isinstance(actual, bytes):
            actual = actual.decode("utf-8")
        if str(actual or "") != expected:
            raise ValueError(
                f"{label} requires {key}={expected!r}; got {actual!r}. "
                "Rebuild the artifact with the current Synthoseis-lite science contract."
            )


__all__ += ["SCIENCE_CONTRACT", "require_science_contract"]
