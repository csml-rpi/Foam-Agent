from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Optional
import zipfile

from models import CaseImportError, CaseManifest
from openfoam_target import ESI_V2006, FOUNDATION_V10, normalise_openfoam_target
from .output_safety import (
    OutputDirectorySafetyError,
    prepare_output_directory,
)


_APPLICATION_RE = re.compile(r"\bapplication\s+([^\s;]+)\s*;")
_VERSION_RE = re.compile(r"\bVersion\s*:\s*([vV]?\d+(?:\.\d+)?)")
_TIME_DIRECTORY_RE = re.compile(
    r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$"
)


def _read_text(path: Path, *, limit: Optional[int] = None) -> str:
    try:
        with path.open("rb") as handle:
            data = handle.read(limit) if limit is not None else handle.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _strip_comments(content: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    return re.sub(r"//.*$", "", without_blocks, flags=re.MULTILINE)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _safe_zip_member(entry: zipfile.ZipInfo) -> Optional[PurePosixPath]:
    """Return a contained relative ZIP path or reject an escaping member."""
    name = entry.filename.replace("\\", "/")
    if stat.S_ISLNK(entry.external_attr >> 16):
        raise CaseImportError(f"ZIP symbolic links are not supported: {entry.filename!r}")
    if not name or name.endswith("/"):
        return None
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise CaseImportError(f"ZIP member escapes the import root: {entry.filename!r}")
    if relative.parts and re.fullmatch(r"[A-Za-z]:", relative.parts[0]):
        raise CaseImportError(f"ZIP member uses an absolute Windows path: {entry.filename!r}")
    return relative


# Bound zip-bomb exposure: a small, crafted archive can otherwise expand to an
# arbitrary size (or entry count) inside a tmpfs-backed TemporaryDirectory before
# any case-content validation runs.
_MAX_ZIP_ENTRIES = 200_000
_MAX_ZIP_UNCOMPRESSED_BYTES = 10 * 1024 ** 3  # 10 GiB: generous for mesh-heavy cases
_ZIP_COPY_CHUNK_SIZE = 1024 * 1024


def _extract_zip(archive: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(archive) as zip_file:
            infolist = zip_file.infolist()
            if len(infolist) > _MAX_ZIP_ENTRIES:
                raise CaseImportError(
                    f"ZIP archive has too many entries ({len(infolist)} > {_MAX_ZIP_ENTRIES})."
                )
            declared_total = sum(entry.file_size for entry in infolist)
            if declared_total > _MAX_ZIP_UNCOMPRESSED_BYTES:
                raise CaseImportError(
                    f"ZIP archive's declared uncompressed size ({declared_total} bytes) "
                    f"exceeds the {_MAX_ZIP_UNCOMPRESSED_BYTES}-byte limit."
                )
            written_total = 0
            for entry in infolist:
                relative = _safe_zip_member(entry)
                if relative is None:
                    continue
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(entry) as source, target.open("wb") as output:
                    while True:
                        chunk = source.read(_ZIP_COPY_CHUNK_SIZE)
                        if not chunk:
                            break
                        written_total += len(chunk)
                        # Track actual bytes written rather than trusting the entry's
                        # declared file_size, which a crafted archive can understate.
                        if written_total > _MAX_ZIP_UNCOMPRESSED_BYTES:
                            raise CaseImportError(
                                "ZIP archive exceeded the "
                                f"{_MAX_ZIP_UNCOMPRESSED_BYTES}-byte uncompressed size limit "
                                "during extraction."
                            )
                        output.write(chunk)
    except zipfile.BadZipFile as exc:
        raise CaseImportError(f"Invalid ZIP archive: {archive}") from exc


def _copy_source(source: Path, destination: Path) -> None:
    if source.is_dir():
        # Source links are rejected before this copy; keep the materialised
        # trees free of links as an additional boundary.
        shutil.copytree(source, destination, symlinks=False)
        return
    if source.suffix.lower() != ".zip":
        raise CaseImportError("--case_path must reference a directory or ZIP archive.")
    destination.mkdir(parents=True)
    _extract_zip(source, destination)


def _reject_source_symlinks(source: Path) -> None:
    """Do not follow links while materialising an imported case."""
    if source.is_dir():
        for path in source.rglob("*"):
            if path.is_symlink():
                raise CaseImportError(
                    f"Imported case contains a symbolic link: {path.relative_to(source)}"
                )


def _candidate_cases(root: Path) -> list[str]:
    candidates: list[str] = []
    for control_dict in root.rglob("controlDict"):
        if control_dict.parent.name != "system" or not control_dict.is_file():
            continue
        case_root = control_dict.parent.parent
        candidates.append(case_root.relative_to(root).as_posix())
    return sorted(set(candidates))


def _select_case_root(
    root: Path,
    case_subdir: Optional[str],
) -> tuple[Path, list[str], list[str]]:
    candidates = _candidate_cases(root)
    issues: list[str] = []
    if case_subdir:
        relative = PurePosixPath(case_subdir.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise CaseImportError("--case_subdir must remain inside the imported input.")
        selected = root.joinpath(*relative.parts)
        if not selected.is_dir():
            raise CaseImportError(f"--case_subdir={case_subdir!r} is not a directory.")
        return selected, candidates, issues
    if len(candidates) == 1:
        return root.joinpath(*PurePosixPath(candidates[0]).parts), candidates, issues
    if len(candidates) > 1:
        raise CaseImportError(
            "Multiple OpenFOAM cases were found; specify --case_subdir. Candidates: "
            + ", ".join(candidates)
        )

    children = [path for path in root.iterdir() if path.is_dir()]
    selected = children[0] if len(children) == 1 else root
    return selected, candidates, issues


def _case_evidence(case_root: Path) -> str:
    paths = [case_root / "system" / "controlDict"]
    for name in ("blockMeshDict", "fvSchemes", "fvSolution"):
        paths.append(case_root / "system" / name)
    return "\n".join(_read_text(path, limit=256_000) for path in paths if path.is_file())


def _detect_platform(case_root: Path) -> tuple[str, Optional[str]]:
    text = _case_evidence(case_root)
    match = _VERSION_RE.search(text)
    version = match.group(1) if match else None
    lowered = text.lower()
    if "openfoam.com" in lowered:
        if not version:
            return "esi-unknown-version", None
        if version and version.lstrip("vV") == "2006":
            return ESI_V2006, version
        return "esi-other", version
    if "openfoam.org" in lowered:
        if not version:
            return "foundation-unknown-version", None
        if version and version.lstrip("vV") == "10":
            return FOUNDATION_V10, version
        return "foundation-other", version
    # Dictionary names alone do not identify a distribution or release.
    return "unknown", version


def _resolve_platform(
    detected: str,
    configured_target: str,
) -> tuple[str, list[str]]:
    target = normalise_openfoam_target(configured_target)
    issues: list[str] = []
    if target:
        compatible = {"unknown", target}
        if target == FOUNDATION_V10:
            compatible.add("foundation-unknown-version")
        elif target == ESI_V2006:
            compatible.add("esi-unknown-version")
        if detected not in compatible:
            issues.append(
                f"Configured target {target!r} differs from detected case platform {detected!r}."
            )
        return target, issues
    if detected in {FOUNDATION_V10, ESI_V2006}:
        return detected, issues
    issues.append(
        "The case platform could not be resolved to Foundation v10 or ESI v2006; "
        "provide --openfoam_target."
    )
    return detected, issues


def _application(case_root: Path) -> Optional[str]:
    content = _strip_comments(_read_text(case_root / "system" / "controlDict"))
    match = _APPLICATION_RE.search(content)
    return match.group(1) if match else None


def _mesh_state(case_root: Path) -> str:
    if (case_root / "constant" / "polyMesh").is_dir():
        return "existing-polyMesh"
    if (case_root / "system" / "blockMeshDict").is_file():
        return "blockMesh"
    return "missing"


def _discovery_issues(case_root: Path) -> list[str]:
    """Describe incomplete or unreadable case inputs for Planner/Reviewer."""
    issues: list[str] = []
    required = [
        case_root / "system" / "controlDict",
        case_root / "system" / "fvSchemes",
        case_root / "system" / "fvSolution",
    ]
    for path in required:
        if not path.is_file():
            issues.append(f"Missing {path.relative_to(case_root).as_posix()}.")
            continue
        try:
            path.open("rb").close()
        except OSError as exc:
            issues.append(
                f"Cannot read {path.relative_to(case_root).as_posix()}: {exc}"
            )
    if (case_root / "system" / "controlDict").is_file() and not _application(case_root):
        issues.append("system/controlDict does not declare an application solver.")
    if not (case_root / "Allrun").is_file():
        issues.append("Missing Allrun.")
    if _mesh_state(case_root) == "missing":
        issues.append("No existing polyMesh or system/blockMeshDict was found.")
    if case_root.is_dir() and not any(
        path.is_dir() and _TIME_DIRECTORY_RE.fullmatch(path.name)
        for path in case_root.iterdir()
    ):
        issues.append("No initial or restart time directory was found.")
    return issues


def snapshot_files(root: str | Path) -> dict[str, str]:
    root = Path(root)
    snapshot: dict[str, str] = {}
    if not root.is_dir():
        return snapshot
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                snapshot[path.relative_to(root).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            except OSError:
                continue
    return snapshot


def _make_read_only(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        try:
            path.chmod(path.stat().st_mode & ~0o222)
        except OSError:
            continue


def _make_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        try:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        except OSError:
            continue


def build_case_context(
    case_dir: str | Path,
    *,
    manifest: Optional[CaseManifest] = None,
) -> dict[str, Any]:
    root = Path(case_dir)
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    ) if root.is_dir() else []
    time_directories = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and _TIME_DIRECTORY_RE.fullmatch(path.name)
    ) if root.is_dir() else []
    platform, version = _detect_platform(root)
    application = _application(root)
    return {
        "case_dir": str(root),
        "platform": manifest.platform if manifest else platform,
        "detected_platform": platform,
        "version": version or (manifest.version if manifest else None),
        "application": application or (manifest.application if manifest else None),
        "files": files,
        "has_control_dict": (root / "system" / "controlDict").is_file(),
        "has_allrun": (root / "Allrun").is_file(),
        "mesh_state": _mesh_state(root),
        "time_directories": time_directories,
        "has_existing_results": any(name not in {"0", "0.0"} for name in time_directories)
        or (root / "postProcessing").exists(),
        "candidate_cases": list(manifest.candidate_cases) if manifest else [],
        "issues": [
            *(list(manifest.issues) if manifest else []),
            *_discovery_issues(root),
        ],
    }


def import_case(
    case_path: str | Path,
    output_dir: str | Path,
    *,
    case_subdir: Optional[str] = None,
    overwrite: bool = False,
    openfoam_target: str = "",
) -> CaseManifest:
    """Copy an existing case into immutable ``original`` and mutable ``work`` trees."""
    source_input = Path(case_path).expanduser()
    source_absolute = source_input.absolute()
    if any(path.is_symlink() for path in (source_absolute, *source_absolute.parents)):
        raise CaseImportError("Imported case paths must not use symbolic-link components.")
    source = source_input.resolve()
    if not source.exists():
        raise CaseImportError(f"Imported case does not exist: {source}")
    if not source.is_dir() and source.suffix.lower() != ".zip":
        raise CaseImportError("--case_path must reference a directory or ZIP archive.")
    _reject_source_symlinks(source)
    try:
        output_root = prepare_output_directory(
            output_dir,
            overwrite=overwrite,
            source_path=source,
        )
    except (OutputDirectorySafetyError, FileExistsError) as exc:
        raise CaseImportError(str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix="foamagent-case-import-") as temporary:
        staging = Path(temporary) / "source"
        _copy_source(source, staging)
        selected, candidates, selection_issues = _select_case_root(staging, case_subdir)
        detected, version = _detect_platform(selected)
        if selected == staging and candidates:
            detected_candidates = [
                _detect_platform(staging.joinpath(*PurePosixPath(candidate).parts))
                for candidate in candidates
            ]
            platforms = {item[0] for item in detected_candidates}
            if len(platforms) == 1:
                detected = detected_candidates[0][0]
                versions = {item[1] for item in detected_candidates}
                version = versions.pop() if len(versions) == 1 else None
        platform, platform_issues = _resolve_platform(detected, openfoam_target)

        original = output_root / "original"
        work = output_root / "work"
        report = output_root / "report"
        shutil.copytree(selected, original, symlinks=False)
        shutil.copytree(selected, work, symlinks=False)
        report.mkdir(parents=True, exist_ok=True)

    manifest = CaseManifest(
        source=str(source),
        case_root="." if selected == staging else selected.relative_to(staging).as_posix(),
        output_root=str(output_root),
        platform=platform,
        version=version,
        application=_application(original),
        allrun_provided=(original / "Allrun").is_file(),
        mesh_state=_mesh_state(original),
        candidate_cases=[] if case_subdir else candidates,
        issues=[*selection_issues, *platform_issues],
        target_mismatch=any(
            issue.startswith("Configured target ") and "differs from detected" in issue
            for issue in platform_issues
        ),
    )
    _make_read_only(original)
    _make_writable(work)
    _write_json(report / "case_manifest.json", manifest.to_dict())
    _write_json(report / "case_context.json", build_case_context(work, manifest=manifest))
    return manifest
