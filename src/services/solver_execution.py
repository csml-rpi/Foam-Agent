"""Check native solver evidence after a freshly cleaned local or HPC attempt."""

import gzip
import os
from pathlib import Path
import re
import subprocess

from utils import check_foam_errors, run_openfoam_utility


_FATAL = re.compile(r"FOAM FATAL(?: IO)? ERROR|\bERROR\s*:|Floating point exception\s*\(core dumped\)|Segmentation fault", re.I)
_SKIPPED = re.compile(r"already run.*(?:remove|re-run)", re.I)
_EXEC = re.compile(r"^Exec\s*:\s*(\S+)[^\n]*$", re.M)
_END = re.compile(r"^End\s*$", re.M)


def check_solver_execution(case_dir: str, *, fields_before: dict[str, tuple[int, ...]],
                           openfoam_target: str = "", timeout: int = 3600) -> list[dict]:
    """Require the configured solver's normal end and a written result field.

    Compare fields before and after the whole Allrun, including time-zero writes.
    This checks execution, not the requested physical end time or accuracy.
    """
    root = Path(case_dir).resolve()
    errors = check_foam_errors(case_dir)
    # Foundation v10 requires expansion; ESI expands entries by default and its
    # -expand option prints the entire dictionary instead of the selected value.
    expand = openfoam_target == "foundation-v10" or (
        not openfoam_target and os.getenv("WM_PROJECT_VERSION") == "10")
    command = ["foamDictionary", *(["-expand"] if expand else []),
               "system/controlDict", "-entry", "application", "-value"]
    try:
        parsed = run_openfoam_utility(
            command,
            working_dir=str(root), timeout=timeout, openfoam_target=openfoam_target,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as error:
        details = "\n".join(str(part) for part in
                            (error, getattr(error, "stdout", ""), getattr(error, "stderr", "")) if part)
        return errors + [{"file": "system/controlDict", "error_content":
                          f"Cannot resolve application with foamDictionary:\n{details}"}]
    try:
        solver = _application_name(parsed.stdout)
    except ValueError as error:
        return errors + [{"file": "system/controlDict", "error_content":
                          f"Cannot resolve application from foamDictionary: {error}"}]
    solver_runs = []
    paths = sorted(p for p in root.iterdir() if p.is_file() and (
        p.name.startswith("log") or p.name in {"Allrun.out", "Allrun.err"}
    ))
    for path in paths:
        content = path.read_text(encoding="utf-8", errors="replace")
        for pattern in (_FATAL, _SKIPPED):
            match = pattern.search(content)
            if match:
                errors.append({"file": path.name, "error_content": content[max(0, match.start() - 200):match.start() + 4000]})
        headers = list(_EXEC.finditer(content))
        for index, header in enumerate(headers):
            if solver and Path(header.group(1)).name == solver:
                end = headers[index + 1].start() if index + 1 < len(headers) else len(content)
                solver_runs.append(bool(_END.search(content[header.end():end])))
    completed = bool(solver_runs) and all(solver_runs)
    if not completed:
        errors.append({"file": "Allrun", "error_content": f"No normal completion evidence for solver {solver}: expected its native Exec header and End in this attempt's output."})
    if completed:
        try:
            has_result = next(changed_result_fields(root, fields_before), None) is not None
        except (OSError, ValueError, TypeError, EOFError) as error:
            errors.append({"file": "Allrun", "error_content": f"Cannot verify this attempt's result fields: {error}"})
        else:
            if not has_result:
                errors.append({"file": "Allrun", "error_content": f"Solver {solver} ended but this attempt produced no new or rewritten result field."})
    return errors


def _application_name(value: str) -> str:
    """Read one native executable word/string, without shell or JSON interpretation."""
    atom = r'''[^\s"'\\$;{}]+'''
    match = re.fullmatch(rf'"({atom})"|({atom})', value.strip())
    if not match:
        raise ValueError(f"Expected one executable word or double-quoted string, got {value!r}")
    name = Path(match[1] or match[2]).name
    if not name:
        raise ValueError("The application does not name an executable")
    return name


PROCESSOR_DIRECTORY_RE = re.compile(r"processor(?:\d+|s\d+(?:_\d+-\d+)?)")
_FOAMFILE_HEADER = re.compile(r"\bFoamFile\s*\{([^{}]*)\}", re.S)


def result_files(root):
    roots = [root, *(p for p in root.iterdir()
                     if p.is_dir() and PROCESSOR_DIRECTORY_RE.fullmatch(p.name))]
    for case in roots:
        for time_dir in case.iterdir():
            if not time_dir.is_dir():
                continue
            try:
                float(time_dir.name)
            except ValueError:
                continue
            for field in time_dir.rglob("*"):
                if field.is_file():
                    yield field


def file_state(path):
    stat = path.stat()
    # Reading a field must not count as a write. Metadata includes same-content rewrites.
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def snapshot_result_fields(case_dir):
    root = Path(case_dir)
    return {path.relative_to(root).as_posix(): file_state(path)
            for path in result_files(root)}


def is_result_field(field):
    opener = gzip.open if field.suffix == ".gz" else open
    with opener(field, "rb") as stream:
        header = stream.read(4096).decode("utf-8", errors="replace")
    headers = [match.group(1) for match in _FOAMFILE_HEADER.finditer(header)]
    if not headers:
        return False
    field_header = headers[0]
    if re.search(r"\bclass\s+decomposedBlockData\s*;", field_header):
        if len(headers) < 2:
            return False
        field_header = headers[1]
    return bool(re.search(r"\bclass\s+\w*Field\s*;", field_header))


def changed_result_fields(root, fields_before):
    for field in result_files(root):
        relative = field.relative_to(root)
        if tuple(fields_before.get(relative.as_posix(), ())) == file_state(field):
            continue
        if is_result_field(field):
            yield relative.as_posix()
