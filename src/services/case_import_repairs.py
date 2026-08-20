"""Narrow, numeric-invariant repairs for imported OpenFOAM dictionaries."""

from __future__ import annotations

from difflib import unified_diff
from pathlib import Path
import re
from typing import Any, Optional

from .case_import_source import _read_text, iter_regular_files


_FOAM_NUMBER_RE = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z_])"
)
_NON_NUMERIC_SEMICOLON_KEYS = frozenset(
    {
        "application",
        "continuousPhase",
        "continuousPhaseName",
        "runTimeModifiable",
        "simulationType",
        "startFrom",
        "stopAt",
        "transportModel",
        "viscosityModel",
        "writeControl",
        "writeFormat",
    }
)


def numeric_signature(content: str) -> dict[str, Any]:
    """Capture every numeric token and its source-line binding."""
    tokens = _FOAM_NUMBER_RE.findall(content)
    bindings: list[tuple[str, tuple[str, ...]]] = []
    for line in content.splitlines():
        values = tuple(_FOAM_NUMBER_RE.findall(line))
        if values:
            bindings.append((_FOAM_NUMBER_RE.sub("<number>", line), values))
    return {"tokens": tokens, "bindings": bindings}


def numeric_snapshot(case_dir: Path) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for path in iter_regular_files(case_dir):
        if ".foamagent" in path.parts:
            continue
        snapshot[path.relative_to(case_dir).as_posix()] = numeric_signature(_read_text(path))
    return snapshot


def _foamfile_object_match(content: str) -> Optional[re.Match[str]]:
    """Find ``object`` only inside the actual FoamFile header."""
    foam_file = re.search(r"\bFoamFile\b", content)
    if not foam_file:
        return None
    opening = content.find("{", foam_file.end())
    if opening < 0:
        return None
    depth = 0
    closing = -1
    for index in range(opening, len(content)):
        if content[index] == "{":
            depth += 1
        elif content[index] == "}":
            depth -= 1
            if depth == 0:
                closing = index
                break
    if closing < 0:
        return None

    header = content[opening + 1 : closing]
    comment_free = re.sub(
        r"/\*.*?\*/|//[^\n]*",
        lambda match: " " * len(match.group(0)),
        header,
        flags=re.DOTALL,
    )
    match = re.search(r"\bobject\s+([^\s;]+)(\s*;)", comment_free)
    if match is None:
        return None
    offset = opening + 1
    return re.compile(r"\bobject\s+([^\s;]+)(\s*;)").search(
        content,
        offset + match.start(),
        offset + match.end(),
    )


def _repair_object_headers(case_dir: Path) -> list[tuple[Path, str, str]]:
    repairs: list[tuple[Path, str, str]] = []
    for path in iter_regular_files(case_dir):
        if ".foamagent" in path.parts or path.name in {"Allrun", "Allclean"}:
            continue
        content = _read_text(path)
        if "FoamFile" not in content:
            continue
        match = _foamfile_object_match(content)
        if not match or match.group(1) == path.name:
            continue
        repaired = content[: match.start(1)] + path.name + content[match.end(1) :]
        repairs.append((path, content, repaired))
    return repairs


def _repair_missing_semicolons(case_dir: Path) -> list[tuple[Path, str, str]]:
    key_pattern = "|".join(sorted(_NON_NUMERIC_SEMICOLON_KEYS))
    statement = re.compile(
        rf"^(?P<prefix>\s*(?:{key_pattern})\s+[^;{{}}()\n]+?)(?P<comment>\s*//.*)?$",
        re.MULTILINE,
    )
    repairs: list[tuple[Path, str, str]] = []
    for path in iter_regular_files(case_dir):
        if ".foamagent" in path.parts or path.name in {"Allrun", "Allclean"}:
            continue
        content = _read_text(path)
        if not content:
            continue
        repaired = statement.sub(
            lambda match: f"{match.group('prefix').rstrip()};{match.group('comment') or ''}",
            content,
        )
        if repaired != content:
            repairs.append((path, content, repaired))
    return repairs


def apply_safe_repairs(work_dir: str | Path, errors: list[Any]) -> list[dict[str, Any]]:
    """Apply only error-gated repairs that preserve all source numeric inputs."""
    case_dir = Path(work_dir)
    error_text = "\n".join(str(error) for error in errors).lower()
    candidates: list[tuple[Path, str, str]] = []
    if "object" in error_text or "foamfile" in error_text:
        candidates.extend(_repair_object_headers(case_dir))
    if "expected ';'" in error_text or "expected ;" in error_text:
        candidates.extend(_repair_missing_semicolons(case_dir))

    before = numeric_snapshot(case_dir)
    records: list[dict[str, Any]] = []
    for path, old, new in candidates:
        relative = path.relative_to(case_dir).as_posix()
        if numeric_signature(old) != numeric_signature(new):
            records.append(
                {
                    "file": relative,
                    "status": "rejected",
                    "reason": "repair would alter numeric tokens or their line bindings",
                }
            )
            continue
        path.write_text(new, encoding="utf-8")
        records.append(
            {
                "file": relative,
                "status": "applied",
                "reason": "deterministic non-numeric syntax/header repair",
                "diff": "".join(
                    unified_diff(
                        old.splitlines(keepends=True),
                        new.splitlines(keepends=True),
                        fromfile=f"before/{relative}",
                        tofile=f"after/{relative}",
                    )
                ),
            }
        )
    if before != numeric_snapshot(case_dir):
        raise RuntimeError("Safe repair violated the numeric-invariant check; work copy must be inspected.")
    return records
