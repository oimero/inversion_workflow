"""Resolve repository Wtie assets independently of private survey data."""

from pathlib import Path

from cup.utils.io import resolve_relative_path


def resolve_wtie_asset(
    value: str | Path | None,
    *,
    filename: str,
    data_root: Path,
    repo_root: Path,
) -> Path:
    """Resolve an explicit asset or the repository's public pretrained asset.

    The former repository-local tutorial location is a path alias for its
    relocated asset. Explicit assets in other data roots keep their location.
    """
    target = repo_root / "opendata" / "pretrained" / "wtie" / filename
    if value is None or not str(value).strip():
        return target.resolve()
    source = resolve_relative_path(value, root=data_root)
    previous = (repo_root / "data" / "tutorial" / filename).resolve()
    return target.resolve() if source.resolve() == previous else source
