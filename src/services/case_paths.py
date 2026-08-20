"""Contain untrusted generated file names within an OpenFOAM case root."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


class CasePathSafetyError(ValueError):
    """Raised when a case-relative path could escape the intended case root."""


def _normalise_component(value: str, *, label: str, allow_empty: bool = False) -> PurePosixPath:
    if not isinstance(value, str):
        raise CasePathSafetyError(f"{label} must be a string")
    if "\x00" in value:
        raise CasePathSafetyError(f"{label} contains a NUL byte")
    value = value.replace("\\", "/").strip()
    if not value:
        if allow_empty:
            return PurePosixPath(".")
        raise CasePathSafetyError(f"{label} cannot be empty")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CasePathSafetyError(f"{label} must be a safe relative path: {value!r}")
    if path.parts and path.parts[0].endswith(":"):
        raise CasePathSafetyError(f"{label} must not use a Windows drive path: {value!r}")
    return path


def safe_case_relative_path(folder_name: str, file_name: str) -> PurePosixPath:
    """Validate a folder/file pair and return its canonical relative path."""
    folder = _normalise_component(folder_name, label="folder_name", allow_empty=True)
    file_path = _normalise_component(file_name, label="file_name")
    if len(file_path.parts) != 1:
        raise CasePathSafetyError("file_name must be a basename, not a nested path")
    relative = folder / file_path
    if relative == PurePosixPath("."):
        raise CasePathSafetyError("file_name cannot resolve to the case root")
    return relative


def safe_case_path(case_dir: str | Path, folder_name: str, file_name: str) -> Path:
    """Return a validated path guaranteed to remain under ``case_dir``."""
    case_root = Path(case_dir).expanduser().resolve()
    relative = safe_case_relative_path(folder_name, file_name)
    target = (case_root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(case_root)
    except ValueError as exc:
        raise CasePathSafetyError(
            f"Generated path escapes case directory: {relative.as_posix()}"
        ) from exc
    return target


def safe_case_relative_from_text(value: str) -> PurePosixPath:
    """Validate a ``folder/file`` target from a rewrite plan."""
    # Root-level files such as Allrun are legitimate rewrite targets.
    return _normalise_component(value, label="rewrite target")
