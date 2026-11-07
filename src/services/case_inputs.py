"""Bounded current case inputs shared by review and repair, independent of execution mode."""

import hashlib
import gzip
import json
import os
from pathlib import Path

from utils import FoamfilePydantic, FoamPydantic, is_solver_output_dir
from .output_safety import OUTPUT_MARKER_NAME

_METADATA_NAMES = {".foamagent", OUTPUT_MARKER_NAME}


def format_case_inputs(foamfiles) -> str:
    """Give model prompts explicit case-relative paths instead of object reprs."""
    value = foamfiles.model_dump() if hasattr(foamfiles, "model_dump") else foamfiles
    if isinstance(value, dict) and "list_foamfile" in value:
        entries = [entry.model_dump() if hasattr(entry, "model_dump") else entry for entry in value["list_foamfile"]]
        value = {
            "/".join(part for part in (entry["folder_name"], entry["file_name"]) if part): entry["content"]
            for entry in entries
        }
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def is_case_artifact(relative: Path, *, directory: bool = False) -> bool:
    """Identify run outputs and Foam-Agent metadata, shared with input snapshots."""
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in _METADATA_NAMES or (
        is_solver_output_dir(parts[0]) and (directory or len(parts) > 1)
    ):
        return True
    if directory or len(parts) > 1:
        return False
    # These root files are produced by orchestration. An arbitrary .log/.out
    # file may instead be a user input, particularly inside an asset directory.
    return (
        relative.name.startswith("log")
        or relative.name in {
            "Allrun.out", "Allrun.err", "workflow.log", "review.log", "similar_case.txt",
            "Allmesh.out", "Allmesh.err", "submit_job.slurm",
        }
    )


def _input_paths(case_dir: str) -> tuple[Path, list[Path], dict[str, str]]:
    """Discover inputs once for both model context and complete content identity."""
    if not case_dir or not str(case_dir).strip():
        raise ValueError("Case directory must not be empty or whitespace")
    root = Path(case_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {root}")
    paths: list[Path] = []
    omitted: dict[str, str] = {}

    def walk_error(error):
        raise error

    for directory, names, files in os.walk(root, followlinks=False, onerror=walk_error):
        parent = Path(directory)
        kept = []
        for name in sorted(names):
            path = parent / name
            if is_case_artifact(path.relative_to(root), directory=True):
                continue
            if name.startswith("."):
                omitted[path.relative_to(root).as_posix()] = "Hidden input not inspected"
            elif path.is_symlink():
                omitted[path.relative_to(root).as_posix()] = "Symbolic link not inspected"
            else:
                kept.append(name)
        names[:] = kept
        for name in files:
            if name.startswith("."):
                if parent != root or name not in _METADATA_NAMES:
                    omitted[(parent / name).relative_to(root).as_posix()] = "Hidden input not inspected"
                continue
            if is_case_artifact((parent / name).relative_to(root)):
                continue
            paths.append(parent / name)
    return root, paths, omitted


def case_input_fingerprint(case_dir: str) -> str | None:
    """Hash complete inputs, including binary/large files, excluding run artifacts.

    Unknown or linked inputs cannot prove equality; leave those trials uncached.
    """
    if not case_dir:
        return None
    try:
        root, paths, omitted = _input_paths(case_dir)
        if omitted:
            return None
        digests = {}
        for path in paths:
            if path.is_symlink() or not path.is_file():
                return None
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digests[path.relative_to(root).as_posix()] = digest.hexdigest()
        return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    except (OSError, ValueError):
        return None


def read_case_inputs(case_dir: str) -> tuple[FoamPydantic, dict[str, str]]:
    """Read bounded current inputs, disclosing anything not inspected in full."""
    root, paths, omitted = _input_paths(case_dir)

    remaining = 512_000
    inputs = []
    # Root scripts first, then shallow dictionaries before bulk nested data.
    for path in sorted(paths, key=lambda path: (len(path.relative_to(root).parts), path.as_posix())):
        relative = path.relative_to(root)
        name = relative.as_posix()
        if path.is_symlink():
            omitted[name] = "Symbolic link not inspected"
            continue
        if not path.is_file():
            omitted[name] = "Non-regular input not inspected"
            continue
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rb") as handle:
                data = handle.read(min(128_000, remaining) + 1)
        except (EOFError, gzip.BadGzipFile):
            omitted[name] = "Unreadable gzip input not inspected"
            continue
        if len(data) > min(128_000, remaining):
            omitted[name] = "File exceeds the per-file or total review context limit"
            continue
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            omitted[name] = "Non-UTF-8 input not inspected"
            continue
        if "\x00" in content:
            omitted[name] = "Binary input not inspected"
            continue
        remaining -= len(data)
        inputs.append(FoamfilePydantic(
            folder_name="" if relative.parent == Path(".") else relative.parent.as_posix(),
            file_name=relative.name,
            content=content,
        ))
    return FoamPydantic(list_foamfile=inputs), omitted
