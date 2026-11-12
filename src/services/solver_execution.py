"""Check native solver evidence after a freshly cleaned local or HPC attempt."""

import gzip
import os
from pathlib import Path
import re
import subprocess
import zlib

from utils import read_case_text, run_openfoam_utility, is_processor_dir, _is_numeric_name
from . import case_inputs
from .execution_logs import current_execution_logs, scan_execution_log


def check_solver_execution(case_dir: str, *, fields_before: dict[str, tuple[int, ...]],
                           logs_before: dict[str, tuple[int, ...]], protected_inputs=(),
                           openfoam_target: str = "", timeout: int = 3600,
                           job_info_dir: str | None = None) -> list[dict]:
    """Require the configured solver's normal end and a written result field.

    Compare fields before and after the whole Allrun, including time-zero writes.
    This checks execution, not the requested physical end time or accuracy.
    """
    root = Path(case_dir).resolve()
    errors = []
    try:
        application = native_application_value(
            str(root), timeout=timeout, openfoam_target=openfoam_target,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as error:
        details = "\n".join(str(part) for part in
                            (error, getattr(error, "stdout", ""), getattr(error, "stderr", "")) if part)
        return errors + [{"file": "system/controlDict", "error_content":
                          f"Cannot resolve application with foamDictionary:\n{details}"}]
    try:
        solver = _application_name(application)
    except ValueError as error:
        return errors + [{"file": "system/controlDict", "error_content":
                          f"Cannot resolve application from foamDictionary: {error}"}]
    solver_runs = []
    for path in current_execution_logs(root, logs_before, protected_inputs=protected_inputs):
        scanned = scan_execution_log(path, solver=solver, job_info_dir=job_info_dir)
        errors.extend({"file": path.relative_to(root).as_posix(), "error_content": text}
                      for text in scanned["errors"][:max(0, 32 - len(errors))])
        solver_runs.extend(scanned["runs"])
        if solver_runs:
            solver_runs[:] = [all(solver_runs)]
    completed = bool(solver_runs) and all(solver_runs)
    if not completed:
        errors.append({"file": "Allrun", "error_content": f"No normal completion evidence for solver {solver}: expected its native Exec header and a normal termination record or unambiguous End in this attempt's output."})
    if completed:
        try:
            has_result = next(changed_result_fields(root, fields_before), None) is not None
        except (OSError, ValueError, TypeError, EOFError, zlib.error) as error:
            errors.append({"file": "Allrun", "error_content": f"Cannot verify this attempt's result fields: {error}"})
        else:
            if not has_result:
                errors.append({"file": "Allrun", "error_content": f"Solver {solver} ended but this attempt produced no new or rewritten result field."})
    return errors


def native_application_value(case_dir: str, *, openfoam_target: str = "", timeout: int = 3600) -> str:
    """Read the expanded application with the selected runtime's dictionary API."""
    # Foundation v10 requires expansion; ESI expands entries by default and its
    # -expand option prints the entire dictionary instead of the selected value.
    expand = openfoam_target == "foundation-v10" or (
        not openfoam_target and os.getenv("WM_PROJECT_VERSION") == "10")
    command = ["foamDictionary", *(["-expand"] if expand else []),
               "system/controlDict", "-entry", "application"]
    parsed = run_openfoam_utility(
        command, working_dir=str(Path(case_dir).resolve()), timeout=timeout,
        openfoam_target=openfoam_target,
    )
    # Expansion may print diagnostics. Both targets terminate this query with
    # the complete named primitive entry, rather than an unframed value.
    # The native query also resolves quoted/pattern keys and prints their
    # original spelling. Its successful lookup establishes the selected key.
    key = r'application|"(?:\\.|[^"\\\n])+"'
    entry = re.search(rf'(?:^|\n)(?:{key})\s+([^;\n]*);\s*\Z', parsed.stdout)
    if entry is None:
        details = '\n'.join(part for part in (parsed.stdout, parsed.stderr) if part)
        raise RuntimeError('foamDictionary did not return a terminal application entry:\n' + details)
    return entry[1].strip()


def read_case_application(case_dir: str, *, openfoam_target: str = "", timeout: int = 3600) -> str:
    """Return the effective executable, retaining a supplied executable path."""
    content = read_case_text(Path(case_dir) / "system/controlDict")
    if not application_requires_native(content):
        entry = application_entry(content)
        if entry is None:
            raise ValueError("system/controlDict does not declare an application")
        return application_executable(entry[1])
    return application_executable(native_application_value(
        case_dir, openfoam_target=openfoam_target, timeout=timeout,
    ))


def application_requires_native(content: str) -> bool:
    """Resolve root preprocessing and quoted native keys before trusting a literal."""
    keys = [key[0] for key, _, _ in case_inputs.dictionary_entries(content)]
    if any(key.startswith(('#', '$', '"')) for key in keys):
        return True
    entry = application_entry(content)
    return entry is not None and ('$' in entry[1] or entry[1].lstrip().startswith('#'))


def application_entry(content: str):
    """Locate the actual top-level application value without resolving directives."""
    return case_inputs.dictionary_primitive_entry(content, 'application')


def application_executable(value: str) -> str:
    """Read one native executable word/string, without shell or JSON interpretation."""
    atom = r'''[^\s"'\\$;{}]+'''
    match = re.fullmatch(rf'"({atom})"|({atom})', value.strip())
    if not match:
        raise ValueError(f"Expected one executable word or double-quoted string, got {value!r}")
    executable = match[1] or match[2]
    if not Path(executable).name:
        raise ValueError("The application does not name an executable")
    return executable


def _application_name(value: str) -> str:
    """Native Exec headers identify the executable by its basename."""
    return Path(application_executable(value)).name


def result_files(root):
    roots = [root, *(p for p in root.iterdir()
                     if p.is_dir() and is_processor_dir(p.name))]
    for case in roots:
        for time_dir in case.iterdir():
            if not time_dir.is_dir() or not _is_numeric_name(time_dir.name):
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
    field_header = case_inputs.native_foamfile_header(header)
    if field_header is None:
        return False
    if _header_class(field_header[0]) == 'decomposedBlockData':
        payload = header[field_header[1]:]
        framing = re.match(r'\s*\d+\s*\(', case_inputs.dictionary_code(payload))
        field_header = case_inputs.native_foamfile_header(
            payload[framing.end():] if framing is not None else payload)
        if field_header is None:
            return False
    return bool(re.fullmatch(r'\w*Field', _header_class(field_header[0]) or ''))


def _header_class(body):
    try:
        entry = case_inputs.dictionary_primitive_entry(body, 'class')
    except ValueError:
        return None
    value = re.fullmatch(r'"(\w+)"|(\w+)', entry[1].strip()) if entry is not None else None
    return (value[1] or value[2]) if value is not None else None


def changed_result_fields(root, fields_before):
    for field in result_files(root):
        relative = field.relative_to(root)
        if tuple(fields_before.get(relative.as_posix(), ())) == file_state(field):
            continue
        if is_result_field(field):
            yield relative.as_posix()
