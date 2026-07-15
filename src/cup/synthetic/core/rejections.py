"""Stage-aware scientific and benchmark rejection records."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RejectionDetail:
    stage: str
    reason_code: str
    message: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stage or not self.reason_code:
            raise ValueError("rejection stage and reason_code must be non-empty.")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class StagedRejection(ValueError):
    stage = "unknown"

    def __init__(
        self,
        reasons: Sequence[str],
        *,
        diagnostics: Mapping[str, Any],
        details: Sequence[Mapping[str, Any] | RejectionDetail],
    ) -> None:
        self.reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
        self.diagnostics = MappingProxyType(dict(diagnostics))
        normalized: list[RejectionDetail] = []
        legacy_details: list[Mapping[str, Any]] = []
        for item in details:
            if isinstance(item, RejectionDetail):
                normalized.append(item)
                legacy_details.append(item.diagnostics)
                continue
            payload = dict(item)
            reason = str(payload.get("reason") or self.reasons[0])
            normalized.append(
                RejectionDetail(
                    stage=self.stage,
                    reason_code=reason,
                    message=reason,
                    diagnostics=payload,
                )
            )
            legacy_details.append(MappingProxyType(payload))
        self.rejection_details = tuple(normalized)
        self.details = tuple(legacy_details)
        super().__init__(";".join(self.reasons))


class TruthGenerationRejected(StagedRejection):
    stage = "truth"


class ProjectionRejected(StagedRejection):
    stage = "projection"


class ForwardRejected(StagedRejection):
    stage = "forward"


class BenchmarkBuildRejected(StagedRejection):
    stage = "benchmark_build"


def frozen_external_reason(exc: BaseException, *, sample_domain: str) -> str:
    """Map staged internal failures to the frozen Pipeline reason category."""
    if isinstance(exc, TruthGenerationRejected):
        category = "GenerationRejected"
    elif isinstance(exc, StagedRejection):
        category = "ValueError"
    else:
        category = type(exc).__name__
    if str(sample_domain).casefold() == "time" and category == "GenerationRejected":
        return str(exc)
    return f"{category}:{exc}"


__all__ = [
    "BenchmarkBuildRejected",
    "ForwardRejected",
    "frozen_external_reason",
    "ProjectionRejected",
    "RejectionDetail",
    "StagedRejection",
    "TruthGenerationRejected",
]
