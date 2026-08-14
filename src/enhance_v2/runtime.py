"""Runtime helpers kept separate from the numerical transfer contract."""

from __future__ import annotations

from typing import Any, Iterable, Iterator


def resolve_device(preferred: str | None = None) -> str:
    """Resolve a requested device without making the analytic path device-dependent."""

    value = "cpu" if preferred is None else str(preferred).casefold()
    if value in {"", "auto"}:
        return "cpu"
    if value != "cpu":
        raise ValueError("enhance_v2 analytic transfer currently supports the CPU/Numpy path only.")
    return "cpu"


def iter_batches(values: Iterable[Any], batch_size: int) -> Iterator[tuple[Any, ...]]:
    """Yield deterministic fixed-size batches for artifact/report callers."""

    if isinstance(batch_size, bool) or int(batch_size) != batch_size or int(batch_size) < 1:
        raise ValueError("batch_size must be a positive integer.")
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == int(batch_size):
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


__all__ = ["iter_batches", "resolve_device"]
