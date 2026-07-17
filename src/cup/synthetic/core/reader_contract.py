"""Shared manifest and HDF5 ownership checks for v5 readers."""

from __future__ import annotations

from typing import Any, Mapping

from cup.synthetic.schemas import BENCHMARK_SCHEMA_VERSION, require_science_contract


def validate_benchmark_header(
    value: Mapping[str, Any], *, sample_domain: str, label: str
) -> None:
    schema = str(value.get("schema") or value.get("schema_version") or "")
    if schema != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"{label} schema {schema!r} does not match {BENCHMARK_SCHEMA_VERSION!r}"
        )
    if str(value.get("sample_domain") or "").casefold() != sample_domain:
        raise ValueError(f"{label} requires sample_domain={sample_domain}")
    if sample_domain == "depth" and value.get("depth_basis") != "tvdss":
        raise ValueError(f"{label} requires depth_basis=tvdss")
    require_science_contract(value, label=label)


__all__ = ["validate_benchmark_header"]
