from __future__ import annotations

from importlib import metadata
from pathlib import Path

import tomli

_DISTRIBUTION_NAME = "autoware-wrapper"
_PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


def wrapper_version() -> str:
    """Return the installed package version, falling back to this checkout's pyproject."""
    try:
        version = metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        with _PYPROJECT_PATH.open("rb") as pyproject_file:
            version = tomli.load(pyproject_file)["project"]["version"]

    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"Could not determine a non-empty {_DISTRIBUTION_NAME} version")
    return version
