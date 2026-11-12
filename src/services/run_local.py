from logger import print
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, List

from utils import is_initial_time_dir, _is_numeric_name, run_command, is_processor_dir
from .solver_execution import check_solver_execution, snapshot_result_fields
from .execution_logs import archive_native_logs, snapshot_execution_logs, current_execution_logs, check_execution_outputs
from .case_inputs import is_declared_input
from .case_paths import safe_case_path


def run_allrun_and_collect_errors(
    case_dir: str,
    timeout: int = 3600,
    max_retries: int = 1,
    *,
    openfoam_target: str = "",
    protected_inputs=(),
) -> List[Any]:
    """
    Execute the Allrun script and collect any error logs from the simulation.
    
    This function runs the Allrun script in the specified case directory,
    captures the output and error streams, and parses the results to identify
    any OpenFOAM errors that occurred during execution.
    
    Args:
        case_dir (str): Directory path containing the OpenFOAM case and Allrun script
        timeout (int, optional): Maximum execution time in seconds. Defaults to 3600.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 1.

        openfoam_target: Optional native runtime target.
    
    Returns:
        List[Any]: Structured execution errors and errors found in simulation logs.
                 Empty list indicates successful execution with no errors.
    
    Execution failures, missing Allrun, and timeouts are returned as errors.
    
    Example:
        >>> errors = run_allrun_and_collect_errors(
        ...     case_dir="/path/to/case",
        ...     timeout=1800,
        ...     max_retries=2,
        ... )
        >>> if not errors:
        ...     print("Simulation completed successfully")
        >>> else:
        ...     print(f"Found {len(errors)} errors")
    """
    allrun_file_path = str(Path(case_dir).resolve() / "Allrun")
    if not os.path.isfile(allrun_file_path):
        return [
            {
                "file": "Allrun",
                "error_content": f"Allrun script not found at {allrun_file_path}",
            }
        ]
    
    out_file = os.path.join(case_dir, "Allrun.out")
    err_file = os.path.join(case_dir, "Allrun.err")
    conflicts = check_execution_outputs(["Allrun.out", "Allrun.err"], protected_inputs)
    if conflicts:
        return conflicts
    _cleanup_run_artifacts(case_dir, protected_inputs=protected_inputs)

    last_error_logs: List[Any] = []
    for attempt in range(1, max_retries + 1):
        print(f"Running Allrun (attempt {attempt}/{max_retries})")
        command_kwargs = (
            {"openfoam_target": openfoam_target}
            if openfoam_target
            else {}
        )
        fields_before = snapshot_result_fields(case_dir)
        logs_before = snapshot_execution_logs(case_dir)
        jobs = safe_case_path(case_dir, '.foamagent', 'solver-native-jobs')
        jobs.mkdir(parents=True, exist_ok=True)
        job_info_dir = tempfile.mkdtemp(prefix='attempt-', dir=jobs)
        command_result = run_command(
            allrun_file_path,
            out_file,
            err_file,
            case_dir,
            timeout,
            fail_fast=True,
            job_info_dir=job_info_dir,
            **command_kwargs,
        )

        error_logs: List[Any] = []
        timed_out = command_result["timed_out"]
        returncode = command_result["returncode"]
        if timed_out:
            output = _execution_output(case_dir, logs_before)
            error_logs.append(
                {
                    "file": "Allrun",
                    "error_content": f"Allrun exceeded the {timeout} second execution timeout.",
                    "diagnostic_context": output,
                }
            )
        elif returncode != 0:
            output = _execution_output(case_dir, logs_before)
            error_logs.append(
                {
                    "file": "Allrun",
                    "error_content": f"Allrun exited with non-zero return code {returncode}.",
                    "diagnostic_context": output,
                }
            )

        error_logs.extend(check_solver_execution(
            case_dir, fields_before=fields_before, logs_before=logs_before,
            protected_inputs=protected_inputs, openfoam_target=openfoam_target, timeout=timeout,
            job_info_dir=job_info_dir))
        error_logs = _deduplicate_errors(error_logs)
        if not error_logs:
            return []

        last_error_logs = error_logs
        if attempt < max_retries:
            print("Allrun reported errors; retrying after cleanup...")
            _cleanup_run_artifacts(case_dir, protected_inputs=protected_inputs)

    return last_error_logs


def _deduplicate_errors(errors: List[Any]) -> List[Any]:
    deduplicated: List[Any] = []
    seen: set[Any] = set()
    for error in errors:
        if isinstance(error, dict):
            key = (error.get("file"), error.get("error_content"))
        else:
            key = str(error)
        if key not in seen:
            seen.add(key)
            deduplicated.append(error)
    return deduplicated


def _cleanup_run_artifacts(case_dir: str, *, protected_inputs=()) -> None:
    """Remove disposable solver outputs before a fresh execution attempt."""
    root = Path(case_dir)
    archive_native_logs(root, protected_inputs=protected_inputs)
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        if is_declared_input(entry.name, protected_inputs):
            continue
        try:
            if entry.is_symlink():
                if entry.name in {"Allrun.out", "Allrun.err"} or _is_numeric_name(entry.name) and not is_initial_time_dir(entry.name):
                    entry.unlink()
                continue
            if entry.is_file() and (
                entry.name in {"Allrun.out", "Allrun.err"}
            ):
                entry.unlink()
            elif entry.is_dir() and (
                is_processor_dir(entry.name)
                or entry.name in {"postProcessing", "VTK"}
                or _is_numeric_name(entry.name) and not is_initial_time_dir(entry.name)
            ):
                shutil.rmtree(entry)
        except OSError as exc:
            raise RuntimeError(f"Unable to clean prior run artifact {entry}: {exc}") from exc


def _execution_output(case_dir: str, logs_before: dict) -> str:
    """Return recent process output as diagnostic context for the Reviewer."""
    root = Path(case_dir)
    command_logs = sorted(
        current_execution_logs(root, logs_before, include_unverified=True),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )[:3]
    paths = command_logs
    excerpts: List[str] = []
    remaining = 32_000
    for path in paths:
        if remaining <= 0:
            break
        if path.is_symlink():
            continue
        try:
            with path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - min(8_000, remaining)))
                data = stream.read(min(8_000, remaining))
        except OSError:
            continue
        content = data.decode("utf-8", errors="replace")
        if content.strip():
            remaining -= len(data)
            excerpts.append(f"[{path.name}: recent output]\n{content}")
    return "\n\n".join(excerpts)
