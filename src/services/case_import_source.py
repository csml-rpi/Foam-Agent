"""Source validation and materialisation for existing-case import mode."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Iterable, Optional
import zipfile

from .case_import_allrun import (
    ensure_mesh_check,
    parse_allrun,
    synthesise_execution_plan,
)
from .case_import_models import CaseImportError, CaseManifest
from .output_safety import (
    OutputDirectorySafetyError,
    prepare_output_directory,
    validate_output_path,
)


MAX_ARCHIVE_FILES = 10_000
MAX_ARCHIVE_BYTES = 1_000_000_000
TEXT_SAMPLE_BYTES = 128_000
_APPLICATION_RE = re.compile(r"\bapplication\s+([^\s;]+)\s*;")
_VERSION_RE = re.compile(r"\bVersion\s*:\s*([vV]?\d+(?:\.\d+)?)")
_CONTROL_ENTRY_RE = re.compile(r"\b(?P<key>startFrom|startTime)\s+(?P<value>[^\s;]+)\s*;")
_TIME_DIRECTORY_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")
_SAFE_COMMAND_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_FORBIDDEN_DYNAMIC_CODE_RE = re.compile(
    r"#\s*(?:codeStream|include|includeEtc|includeFunc)\b|"
    r"\b(?:coded|codeExecute|codeInclude|codeLibs|codeOptions)\b",
    re.IGNORECASE,
)
_LIBRARY_DECLARATION_RE = re.compile(r"\blibs\s*\(", re.IGNORECASE)


def _read_text(path: Path, *, limit: Optional[int] = None) -> str:
    try:
        with path.open("rb") as handle:
            data = handle.read(limit) if limit is not None else handle.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _strip_foam_comments(content: str) -> str:
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    return re.sub(r"//.*$", "", content, flags=re.MULTILINE)


def _control_dict_application(case_root: Path) -> str:
    content = _strip_foam_comments(_read_text(case_root / "system" / "controlDict"))
    match = _APPLICATION_RE.search(content)
    if not match:
        raise CaseImportError(
            "system/controlDict does not define an application; cannot identify the solver."
        )
    application = match.group(1)
    if not _SAFE_COMMAND_RE.fullmatch(application):
        raise CaseImportError(f"Unsupported application token in system/controlDict: {application!r}.")
    return application


def _synthesised_startup_issue(case_root: Path) -> Optional[str]:
    """Reject inferred plans with an unsatisfied restart dependency."""
    content = _strip_foam_comments(_read_text(case_root / "system" / "controlDict"))
    entries = {match.group("key"): match.group("value") for match in _CONTROL_ENTRY_RE.finditer(content)}
    start_from = entries.get("startFrom")
    start_time = entries.get("startTime")
    time_directories = [
        path.name
        for path in case_root.iterdir()
        if path.is_dir() and _TIME_DIRECTORY_RE.fullmatch(path.name)
    ]

    if start_from in {"latestTime", "firstTime"}:
        if not time_directories:
            return (
                "Allrun is missing, but controlDict requests "
                f"startFrom {start_from} with no numeric time directory. "
                "Provide an explicit Allrun/setup stage."
            )
        return None
    if start_from not in {None, "startTime"} or not start_time:
        return None
    try:
        expected_time = float(start_time)
    except ValueError:
        return (
            "Allrun is missing and controlDict has an unsupported startTime "
            f"token {start_time!r}; provide an explicit Allrun."
        )
    if expected_time == 0 or any(
        abs(float(directory) - expected_time) <= 1e-12 for directory in time_directories
    ):
        return None
    return (
        "Allrun is missing, but controlDict starts from time "
        f"{start_time} and that numeric time directory is absent. "
        "Provide the required setup stage or an explicit Allrun."
    )


def iter_regular_files(root: Path) -> Iterable[Path]:
    """Yield case files while rejecting links instead of following them."""
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CaseImportError(f"Symbolic links are not supported in imported cases: {path}")
        if path.is_file():
            yield path


def _hash_files(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in iter_regular_files(root):
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise CaseImportError(f"Unable to hash imported file {path}: {exc}") from exc
        hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
    return hashes


def _validate_source_tree(root: Path) -> None:
    if not root.is_dir():
        raise CaseImportError(f"Case path is not a directory: {root}")
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CaseImportError(f"Symbolic links are not allowed in imported cases: {path}")
        if path.name == ".foamagent":
            raise CaseImportError(
                "Imported cases cannot contain the reserved .foamagent control directory."
            )
        if path.is_file():
            file_count += 1
            try:
                total_bytes += path.stat().st_size
            except OSError as exc:
                raise CaseImportError(f"Unable to inspect imported file {path}: {exc}") from exc
            if file_count > MAX_ARCHIVE_FILES:
                raise CaseImportError(
                    f"Case contains more than {MAX_ARCHIVE_FILES} files; the import safety limit was exceeded."
                )
            if total_bytes > MAX_ARCHIVE_BYTES:
                raise CaseImportError(
                    "Case size exceeds the 1 GB import safety limit."
                )


def _normalise_zip_member(member: zipfile.ZipInfo) -> Optional[PurePosixPath]:
    """Validate one ZIP member path before anything is written to disk."""
    filename = member.filename
    if (
        not filename
        or "\x00" in filename
        or "\\" in filename
        or re.match(r"^[A-Za-z]:", filename)
    ):
        raise CaseImportError(f"ZIP entry uses an unsupported Windows path: {filename!r}.")
    member_path = PurePosixPath(filename)
    if member_path.is_absolute() or ".." in member_path.parts or filename.startswith(("/", "\\")):
        raise CaseImportError(f"ZIP entry escapes the import directory: {filename!r}.")
    if stat.S_ISLNK(member.external_attr >> 16):
        raise CaseImportError(f"ZIP symbolic links are not supported: {filename!r}.")
    if not member_path.parts:
        if member.is_dir():
            return None
        raise CaseImportError(f"ZIP entry has an empty path: {filename!r}.")
    return member_path


def _validated_zip_members(
    members: list[zipfile.ZipInfo],
) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    if len(members) > MAX_ARCHIVE_FILES:
        raise CaseImportError(
            f"ZIP contains {len(members)} entries; the limit is {MAX_ARCHIVE_FILES}."
        )
    if sum(member.file_size for member in members) > MAX_ARCHIVE_BYTES:
        raise CaseImportError("ZIP uncompressed size exceeds the 1 GB import safety limit.")

    normalised_members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    seen_members: set[PurePosixPath] = set()
    for member in members:
        member_path = _normalise_zip_member(member)
        if member_path is None:
            continue
        if member_path in seen_members:
            raise CaseImportError(f"ZIP contains duplicate member path: {member.filename!r}.")
        seen_members.add(member_path)
        normalised_members.append((member, member_path))
    return normalised_members


def _extract_zip_member(
    zip_file: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    member_path: PurePosixPath,
    destination: Path,
    destination_root: Path,
    extracted_bytes: int,
) -> int:
    """Extract one validated member and enforce the runtime archive-size limit."""
    target = destination.joinpath(*member_path.parts)
    try:
        target.resolve().relative_to(destination_root)
    except ValueError as exc:
        raise CaseImportError(
            f"ZIP entry escapes the import directory: {member.filename!r}."
        ) from exc
    if member.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return extracted_bytes
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise CaseImportError(f"ZIP member conflicts with an existing path: {member.filename!r}.")
        with zip_file.open(member, "r") as source, target.open("xb") as output:
            while chunk := source.read(64 * 1024):
                extracted_bytes += len(chunk)
                if extracted_bytes > MAX_ARCHIVE_BYTES:
                    raise CaseImportError("ZIP uncompressed size exceeds the 1 GB import safety limit.")
                output.write(chunk)
        return extracted_bytes
    except CaseImportError:
        target.unlink(missing_ok=True)
        raise
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise CaseImportError(f"Unable to extract ZIP member {member.filename!r}: {exc}") from exc


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract an archive with traversal, link, count, and size checks."""
    try:
        with zipfile.ZipFile(archive) as zip_file:
            members = _validated_zip_members(zip_file.infolist())
            extracted_bytes = 0
            destination_root = destination.resolve()
            for member, member_path in members:
                extracted_bytes = _extract_zip_member(
                    zip_file,
                    member,
                    member_path,
                    destination,
                    destination_root,
                    extracted_bytes,
                )
    except zipfile.BadZipFile as exc:
        raise CaseImportError(f"Invalid ZIP archive: {archive}") from exc


def _find_case_root(source_root: Path, case_subdir: Optional[str]) -> Path:
    if case_subdir:
        relative = PurePosixPath(case_subdir.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise CaseImportError("--case_subdir must be a relative path inside the import.")
        candidate = (source_root / Path(*relative.parts)).resolve()
        try:
            candidate.relative_to(source_root.resolve())
        except ValueError as exc:
            raise CaseImportError("--case_subdir escapes the imported source.") from exc
        if not (candidate / "system" / "controlDict").is_file():
            raise CaseImportError(f"--case_subdir={case_subdir!r} does not contain system/controlDict.")
        return candidate

    candidates = sorted(
        control.parent.parent
        for control in source_root.rglob("controlDict")
        if control.parent.name == "system" and control.is_file()
    )
    if not candidates:
        raise CaseImportError("No system/controlDict found in the imported case.")
    if len(candidates) > 1:
        listed = ", ".join(str(candidate.relative_to(source_root)) for candidate in candidates[:8])
        raise CaseImportError(
            "Multiple OpenFOAM cases found in the import. Specify --case_subdir. "
            f"Candidates: {listed}"
        )
    return candidates[0]


def _detect_platform(case_root: Path) -> tuple[str, Optional[str]]:
    evidence_paths = [case_root / "system" / "controlDict"]
    evidence_paths.extend(
        path
        for path in iter_regular_files(case_root)
        if path.name in {"blockMeshDict", "fvSolution", "fvSchemes"} and path not in evidence_paths
    )
    contents = "\n".join(_read_text(path, limit=TEXT_SAMPLE_BYTES) for path in evidence_paths)
    lowered = contents.lower()
    version_match = _VERSION_RE.search(contents)
    version = version_match.group(1) if version_match else None
    if "openfoam.com" in lowered:
        return "esi", version
    if "openfoam.org" in lowered:
        return ("foundation-v10", version) if version and version.lstrip("vV") == "10" else ("foundation-other", version)

    constant_dir = case_root / "constant"
    if any(constant_dir.glob("momentumTransport*")):
        return "foundation-v10-compatible", version
    if (constant_dir / "turbulenceProperties").exists() or any(constant_dir.glob("turbulenceProperties.*")):
        return "unknown", version
    return "unknown", version


def _mesh_state(case_root: Path) -> str:
    if (case_root / "constant" / "polyMesh").is_dir():
        return "existing-polyMesh"
    if (case_root / "system" / "blockMeshDict").is_file():
        return "blockMesh"
    return "missing"


def _extract_libraries(case_root: Path) -> list[str]:
    libraries: set[str] = set()
    for path in _dependency_scan_files(case_root):
        content = _strip_foam_comments(_read_text(path, limit=TEXT_SAMPLE_BYTES))
        for block in re.findall(r"\blibs\s*\((.*?)\)", content, re.DOTALL):
            libraries.update(re.findall(r'["\']([^"\']+\.so)["\']', block))
    return sorted(libraries)


def _dependency_scan_files(case_root: Path) -> Iterable[Path]:
    """Yield only dictionary inputs which can affect imported-case execution.

    Imported archives often contain README files, notes, and patch artefacts.
    Those files are not interpreted by OpenFOAM, so wording such as ``coded``
    or ``#include`` in them must not block an otherwise safe case.
    """
    for path in iter_regular_files(case_root):
        relative = path.relative_to(case_root)
        if path.name == "Allrun":
            yield path
            continue
        if relative.parts and relative.parts[0] in {"0", "constant", "system"}:
            yield path
            continue
        if relative.parts and _TIME_DIRECTORY_RE.fullmatch(relative.parts[0]):
            yield path


def _iter_comment_free_chunks(path: Path) -> Iterable[str]:
    """Stream OpenFOAM text with ``//`` and ``/* ... */`` comments removed."""
    in_block_comment = False
    in_line_comment = False
    pending = ""
    try:
        with path.open("rb") as handle:
            while raw_chunk := handle.read(64 * 1024):
                text = pending + raw_chunk.decode("utf-8", errors="replace")
                pending = ""
                output: list[str] = []
                index = 0
                # Retain one unprocessed character for the next chunk so a
                # comment opener or closer split at the I/O boundary is still
                # recognised as a two-character token.
                while index < len(text) - 1:
                    character = text[index]
                    following = text[index + 1]
                    if in_line_comment:
                        if character == "\n":
                            in_line_comment = False
                            output.append(character)
                        index += 1
                    elif in_block_comment:
                        if character == "*" and following == "/":
                            in_block_comment = False
                            index += 2
                        else:
                            if character == "\n":
                                output.append(character)
                            index += 1
                    elif character == "/" and following == "/":
                        in_line_comment = True
                        index += 2
                    elif character == "/" and following == "*":
                        in_block_comment = True
                        index += 2
                    else:
                        output.append(character)
                        index += 1
                pending = text[index:]
                if output:
                    yield "".join(output)
            if pending:
                if in_line_comment:
                    if pending == "\n":
                        yield pending
                elif not in_block_comment:
                    yield pending
    except OSError as exc:
        raise CaseImportError(f"Unable to inspect imported file {path}: {exc}") from exc


def _file_contains_pattern(path: Path, pattern: re.Pattern[str]) -> bool:
    overlap = ""
    for chunk in _iter_comment_free_chunks(path):
        text = overlap + chunk
        if pattern.search(text):
            return True
        overlap = text[-512:]
    return False


def _detect_custom_dependencies(case_root: Path, allrun_content: str) -> list[str]:
    issues: list[str] = []
    if any(path.name in {"Make", "Allwmake"} for path in case_root.rglob("*")):
        issues.append("case contains a Make/ or Allwmake build dependency")
    if re.search(r"\b(?:wmake|make|cmake)\b", allrun_content):
        issues.append("Allrun requests compilation")

    for path in _dependency_scan_files(case_root):
        if _file_contains_pattern(path, _FORBIDDEN_DYNAMIC_CODE_RE):
            issues.append(f"dynamic code found in {path.relative_to(case_root)}")
            break
    libraries = _extract_libraries(case_root)
    if libraries or any(
        _file_contains_pattern(path, _LIBRARY_DECLARATION_RE)
        for path in _dependency_scan_files(case_root)
    ):
        issues.append(
            "case declares dynamically loaded libraries, which are not supported "
            "by safe case-import mode: "
            + (", ".join(libraries) if libraries else "unparsed declaration")
        )
    return issues


def _write_json(path: Path, value: object) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_tree_read_only(root: Path) -> None:
    # Copy ``work/`` before this function is called.  Marking directories as
    # read-only matters too: a read-only file may still be deleted or replaced
    # when its parent directory remains writable.
    paths = [root, *root.rglob("*")]
    for path in reversed(paths):
        if path.is_symlink():
            continue
        try:
            path.chmod(path.stat().st_mode & ~0o222)
        except OSError:
            pass


def import_case(
    case_path: str | Path,
    output_dir: str | Path,
    *,
    case_subdir: Optional[str] = None,
    overwrite: bool = False,
) -> CaseManifest:
    """Validate a source then materialise ``original/``, ``work/``, and ``report/``."""
    source_path = Path(case_path).expanduser().resolve()
    if not source_path.exists():
        raise CaseImportError(f"Case input does not exist: {source_path}")
    if not source_path.is_dir() and source_path.suffix.lower() != ".zip":
        raise CaseImportError("--case_path must reference a directory or a .zip archive.")
    try:
        output_root = validate_output_path(output_dir, source_path=source_path)
    except OutputDirectorySafetyError as exc:
        raise CaseImportError(str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix="foamagent-case-import-") as temporary:
        staging = Path(temporary) / "source"
        if source_path.is_dir():
            _validate_source_tree(source_path)
            shutil.copytree(source_path, staging)
        else:
            staging.mkdir(parents=True)
            _safe_extract_zip(source_path, staging)
            _validate_source_tree(staging)
        selected_root = _find_case_root(staging, case_subdir)
        selected_root_relative = selected_root.relative_to(staging)

        platform, version = _detect_platform(selected_root)
        application = _control_dict_application(selected_root)
        source_allrun = selected_root / "Allrun"
        allrun_content = _read_text(source_allrun) if source_allrun.is_file() else ""
        blocking_issues = _detect_custom_dependencies(selected_root, allrun_content)
        try:
            mesh_state = _mesh_state(selected_root)
            plan = (
                parse_allrun(allrun_content, application)
                if source_allrun.is_file()
                else synthesise_execution_plan(application, mesh_state)
            )
            if not source_allrun.is_file() and (startup_issue := _synthesised_startup_issue(selected_root)):
                blocking_issues.append(startup_issue)
            plan = ensure_mesh_check(plan, application)
        except CaseImportError as exc:
            blocking_issues.append(str(exc))
            plan = []
        if platform not in {"foundation-v10", "foundation-v10-compatible"}:
            blocking_issues.append(
                "Only Foundation OpenFOAM v10 cases are supported by case-import mode; "
                f"detected platform is {platform}{f' ({version})' if version else ''}."
            )

        try:
            output_root = prepare_output_directory(output_root, overwrite=overwrite, source_path=source_path)
        except (FileExistsError, OutputDirectorySafetyError) as exc:
            raise CaseImportError(str(exc)) from exc

        original_dir = output_root / "original"
        work_dir = output_root / "work"
        report_dir = output_root / "report"
        original_staging_dir = output_root / ".original-staging"
        try:
            shutil.copytree(selected_root, original_staging_dir)
            original_staging_dir.rename(original_dir)
        except OSError as exc:
            shutil.rmtree(original_staging_dir, ignore_errors=True)
            raise CaseImportError(f"Unable to materialize imported case: {exc}") from exc

        manifest = CaseManifest(
            source=str(source_path),
            case_root=str(selected_root_relative),
            output_root=str(output_root),
            platform=platform,
            version=version,
            application=application,
            allrun_provided=source_allrun.is_file(),
            mesh_state=_mesh_state(original_dir),
            execution_plan=plan,
            detected_libraries=_extract_libraries(original_dir),
            blocking_issues=list(dict.fromkeys(blocking_issues)),
            original_hashes=_hash_files(original_dir),
        )
        shutil.copytree(original_dir, work_dir)
        _make_tree_read_only(original_dir)
        _write_json(report_dir / "case_manifest.json", manifest.to_dict())
        return manifest
