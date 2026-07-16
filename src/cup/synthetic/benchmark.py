"""Facade for frozen Synthoseis-lite benchmark artifacts.

Version numbers belong to artifact schemas, not implementation modules.  This
module is the stable reader interface used by evaluators and GINN; it dispatches
to a domain adapter after inspecting the benchmark manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cup.synthetic.schemas import (
    BENCHMARK_SCHEMA_VERSION,
    FROZEN_BENCHMARK_SCHEMA_VERSION,
    LEGACY_BENCHMARK_SCHEMA_VERSION,
    require_science_contract,
)
from cup.synthetic.readers.depth import DepthBenchmark, DepthSyntheticSample
from cup.synthetic.readers.time import TimeBenchmark, TimeSyntheticSample
from cup.synthetic.core.protocols import SyntheticSampleProtocol


SyntheticSample = TimeSyntheticSample | DepthSyntheticSample


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


class SynthoseisBenchmark:
    """Read-only facade around supported Synthoseis-lite artifact schemas."""

    def __init__(
        self,
        run_dir: str | Path,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "benchmark_manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"benchmark_manifest.json not found: {self.manifest_path}"
            )
        manifest = _json(self.manifest_path)
        schema = str(manifest.get("schema") or manifest.get("schema_version") or "")
        sample_domain = str(manifest.get("sample_domain") or "").casefold()

        if schema == LEGACY_BENCHMARK_SCHEMA_VERSION:
            raise ValueError(
                "Legacy Synthoseis artifacts are not accepted by the v4 reader."
            )
        if schema == FROZEN_BENCHMARK_SCHEMA_VERSION:
            raise ValueError(
                "Frozen Synthoseis v3 artifacts are baseline-only and are not accepted "
                "by the v4 canonical-increment reader."
            )
        if schema == BENCHMARK_SCHEMA_VERSION:
            require_science_contract(manifest, label="Synthoseis benchmark manifest")
            if sample_domain == "time":
                self._reader = TimeBenchmark(
                    self.run_dir,
                )
            elif sample_domain == "depth":
                self._reader = DepthBenchmark(
                    self.run_dir,
                )
            else:
                raise ValueError(
                    f"{BENCHMARK_SCHEMA_VERSION} requires sample_domain='time' or 'depth'; got {sample_domain!r}."
                )
        else:
            raise ValueError(
                f"Unsupported Synthoseis schema {schema!r}. "
                f"Supported schema: {BENCHMARK_SCHEMA_VERSION} with sample_domain=time|depth."
            )

        self.schema = schema
        self.sample_domain = getattr(self._reader, "sample_domain")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reader, name)

    def sample_ids(
        self,
        *,
        kinds: set[str] | None = None,
        status: str = "ok",
        split: str | None = None,
    ) -> list[str]:
        return self._reader.sample_ids(kinds=kinds, status=status, split=split)

    def row(self, sample_id: str) -> dict[str, Any]:
        return self._reader.row(sample_id)

    def load_sample(self, sample_id: str) -> SyntheticSample:
        return self._reader.load_sample(sample_id)


__all__ = [
    "DepthSyntheticSample",
    "DepthBenchmark",
    "SyntheticSample",
    "SyntheticSampleProtocol",
    "SynthoseisBenchmark",
    "TimeBenchmark",
    "TimeSyntheticSample",
]
