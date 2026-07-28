"""Filesystem locations that Foam-Agent resolves at runtime.

The package lives at ``<root>/src/foamagent`` in a source checkout, while the data it
reads (``database/``) and writes (``runs/``) live at ``<root>``.  Every module used to
recompute that relationship from its own ``__file__``, which silently broke whenever the
package moved.  Resolve it in one place instead, and let ``FOAMAGENT_ROOT`` override it
for installs where the data directory is not a sibling of the source tree.
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent          # <root>/src/foamagent
_SRC_DIR = _PACKAGE_DIR.parent                          # <root>/src


def _env_path(key: str) -> Path | None:
    value = os.getenv(key)
    if value is None or not value.strip():
        return None
    return Path(value.strip()).expanduser().resolve()


def repo_root() -> Path:
    """Directory holding ``database/`` and ``runs/``."""
    override = _env_path("FOAMAGENT_ROOT")
    if override is not None:
        return override
    return _SRC_DIR.parent


def database_dir() -> Path:
    return _env_path("FOAMAGENT_DATABASE_PATH") or (repo_root() / "database")


def runs_dir() -> Path:
    return _env_path("FOAMAGENT_RUN_DIRECTORY") or (repo_root() / "runs")


def package_dir() -> Path:
    """Directory of the installed package, for data files shipped alongside the code."""
    return _PACKAGE_DIR
