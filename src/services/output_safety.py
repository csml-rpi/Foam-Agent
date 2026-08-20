"""Shared safeguards for Foam-Agent-owned output directories.

OpenFOAM cases commonly live next to valuable source data.  A generic
``--overwrite_output`` flag must therefore never be interpreted as permission
to empty an arbitrary directory.  This module centralises the ownership and
path checks used by both generated and imported-case workflows.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Optional


OUTPUT_MARKER_NAME = ".foamagent-output"
_MARKER_CONTENT = "Foam-Agent managed output directory\n"


class OutputDirectorySafetyError(ValueError):
    """Raised when an output target is unsafe to create or clear."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _protected_paths() -> set[Path]:
    """Return roots which are never valid Foam-Agent output directories."""
    repository_root = _repository_root()
    protected = {
        Path("/"),
        Path.home().resolve(),
        Path(tempfile.gettempdir()).resolve(),
        repository_root,
    }
    protected.update(repository_root.parents)
    return protected


def _ensure_not_related_to_source(output_root: Path, source_path: Optional[str | Path]) -> None:
    """Prevent an import target from containing, or being inside, its source."""
    if source_path is None:
        return

    source_root = _resolve_path(source_path)
    try:
        source_root.relative_to(output_root)
    except ValueError:
        pass
    else:
        raise OutputDirectorySafetyError(
            "Import output directory cannot be the source path or an ancestor of it."
        )

    if source_root.is_dir():
        try:
            output_root.relative_to(source_root)
        except ValueError:
            return
        raise OutputDirectorySafetyError(
            "Import output directory cannot be inside the source case directory."
        )


def validate_output_path(
    output_path: str | Path,
    *,
    source_path: Optional[str | Path] = None,
) -> Path:
    """Resolve and validate a dedicated output target without mutating it."""
    raw_path = Path(output_path).expanduser()
    # Check every existing path component.  Checking only the leaf permits an
    # attacker (or a mistaken user) to route a new output through a symlinked
    # parent, then clear/write the target behind that link.
    absolute_raw_path = raw_path.absolute()
    for component in (absolute_raw_path, *absolute_raw_path.parents):
        if component.is_symlink():
            raise OutputDirectorySafetyError(
                "Refusing to use output path through symlinked component: "
                f"{component}"
            )

    output_root = raw_path.resolve()
    # Foundation OpenFOAM v10 rejects case paths containing whitespace in a
    # number of utilities.  Reject early rather than creating an output that
    # can be imported but can never execute.
    if any(character.isspace() for character in str(output_root)):
        raise OutputDirectorySafetyError(
            f"OpenFOAM case output path must not contain whitespace: {output_root}"
        )
    if output_root in _protected_paths():
        raise OutputDirectorySafetyError(
            f"Refusing to use protected broad path as output: {output_root}. "
            "Choose a dedicated child directory."
        )
    _ensure_not_related_to_source(output_root, source_path)
    return output_root


def output_is_owned(output_root: str | Path) -> bool:
    """Return whether *output_root* was previously initialised by Foam-Agent."""
    marker = _resolve_path(output_root) / OUTPUT_MARKER_NAME
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        return marker.read_text(encoding="utf-8") == _MARKER_CONTENT
    except OSError:
        return False


def _write_ownership_marker(output_root: Path) -> None:
    marker = output_root / OUTPUT_MARKER_NAME
    marker.write_text(_MARKER_CONTENT, encoding="utf-8")


def _make_tree_deletable(root: Path) -> None:
    """Restore owner access required to remove an owned directory tree.

    Imported cases deliberately make the original copy read-only. An overwrite
    removes that copy before recreating it, so every real directory in the
    tree needs owner read/write/execute access for shutil.rmtree. Do not
    traverse or modify symlinks while restoring those permissions.
    """
    for directory, directory_names, _ in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        directory_names[:] = [
            name for name in directory_names if not (current / name).is_symlink()
        ]
        mode = current.stat(follow_symlinks=False).st_mode
        current.chmod(mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def _clear_children(output_root: Path) -> None:
    for child in output_root.iterdir():
        if child.is_symlink():
            child.unlink()
        elif child.is_dir():
            _make_tree_deletable(child)
            shutil.rmtree(child)
        else:
            child.unlink()


def validate_output_preparation(
    output_path: str | Path,
    *,
    overwrite: bool,
    source_path: Optional[str | Path] = None,
) -> Path:
    """Check whether an output target may be prepared, without changing it."""
    output_root = validate_output_path(output_path, source_path=source_path)
    if not output_root.exists():
        return output_root
    if not output_root.is_dir():
        raise OutputDirectorySafetyError(
            f"Output path exists and is not a directory: {output_root}"
        )

    meaningful_children = [
        child for child in output_root.iterdir() if child.name != OUTPUT_MARKER_NAME
    ]
    if not meaningful_children:
        return output_root
    if not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Choose a new output directory or pass --overwrite_output."
        )
    if not output_is_owned(output_root):
        raise OutputDirectorySafetyError(
            f"Refusing to clear unowned output directory: {output_root}. "
            "Use a new dedicated directory instead."
        )
    return output_root


def prepare_output_directory(
    output_path: str | Path,
    *,
    overwrite: bool,
    source_path: Optional[str | Path] = None,
) -> Path:
    """Create a managed output directory, safely clearing only owned output.

    A non-empty directory may be cleared only if it contains Foam-Agent's
    ownership marker.  This makes ``--overwrite_output`` safe even if a caller
    accidentally passes a broad existing directory such as ``/tmp``.
    """
    output_root = validate_output_preparation(
        output_path,
        overwrite=overwrite,
        source_path=source_path,
    )
    if output_root.exists():
        meaningful_children = [
            child for child in output_root.iterdir() if child.name != OUTPUT_MARKER_NAME
        ]
        if meaningful_children:
            _clear_children(output_root)
    else:
        output_root.mkdir(parents=True, exist_ok=False)

    _write_ownership_marker(output_root)
    return output_root
