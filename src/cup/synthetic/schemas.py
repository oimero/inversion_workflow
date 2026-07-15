"""Schema versions shared by synthetic-workflow artifact producers and consumers."""

from __future__ import annotations


BENCHMARK_SCHEMA_VERSION = "synthoseis_lite_v4"
FROZEN_BENCHMARK_SCHEMA_VERSION = "synthoseis_lite_v3"
LEGACY_BENCHMARK_SCHEMA_VERSION = "synthoseis_lite_v1"
REPORT_SCHEMA_VERSION = "synthoseis_lite_report_v2"
ROCK_PHYSICS_ANALYSIS_SCHEMA_VERSION = "rock_physics_analysis_v3"
DEPTH_FORWARD_MODEL_INPUTS_RUN_SCHEMA_VERSION = "depth_forward_model_inputs_v1"
FORWARD_MODEL_INPUTS_SCHEMA_VERSION = "forward_model_inputs_v3"


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "DEPTH_FORWARD_MODEL_INPUTS_RUN_SCHEMA_VERSION",
    "FROZEN_BENCHMARK_SCHEMA_VERSION",
    "FORWARD_MODEL_INPUTS_SCHEMA_VERSION",
    "LEGACY_BENCHMARK_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "ROCK_PHYSICS_ANALYSIS_SCHEMA_VERSION",
]
