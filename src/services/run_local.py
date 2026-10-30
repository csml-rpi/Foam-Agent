import os
import re
import shutil
from pathlib import Path
from typing import Any, List

from utils import check_foam_errors, remove_numeric_folders, run_command


def run_allrun_and_collect_errors(
    case_dir: str,
    timeout: int = 3600,
    max_retries: int = 1,
    *,
    openfoam_target: str = "",
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
        ...     max_retries=2
        ... )
        >>> if not errors:
        ...     print("Simulation completed successfully")
        >>> else:
        ...     print(f"Found {len(errors)} errors")
    """
    allrun_file_path = os.path.join(case_dir, "Allrun")
    if not os.path.isfile(allrun_file_path):
        return [
            {
                "file": "Allrun",
                "error_content": f"Allrun script not found at {allrun_file_path}",
            }
        ]
    
    out_file = os.path.join(case_dir, "Allrun.out")
    err_file = os.path.join(case_dir, "Allrun.err")
    _cleanup_run_artifacts(case_dir)

    last_error_logs: List[Any] = []
    for attempt in range(1, max_retries + 1):
        print(f"Running Allrun (attempt {attempt}/{max_retries})")
        command_kwargs = (
            {"openfoam_target": openfoam_target}
            if openfoam_target
            else {}
        )
        command_result = run_command(
            allrun_file_path,
            out_file,
            err_file,
            case_dir,
            timeout,
            **command_kwargs,
        )

        error_logs: List[Any] = []
        timed_out = bool(_result_field(command_result, "timed_out", False))
        returncode = _result_field(command_result, "returncode", 0)
        if timed_out:
            output = _execution_output(case_dir)
            error_logs.append(
                {
                    "file": "Allrun",
                    "error_content": f"Allrun exceeded the {timeout} second execution timeout."
                    + (f"\n{output}" if output else ""),
                }
            )
        elif returncode not in (None, 0):
            output = _execution_output(case_dir)
            error_logs.append(
                {
                    "file": "Allrun",
                    "error_content": f"Allrun exited with non-zero return code {returncode}."
                    + (f"\n{output}" if output else ""),
                }
            )

        error_logs.extend(check_foam_errors(case_dir))
        error_logs = _deduplicate_errors(error_logs)
        if not error_logs:
            return []

        last_error_logs = error_logs
        if attempt < max_retries:
            print("Allrun reported errors; retrying after cleanup...")
            _cleanup_run_artifacts(case_dir)

    return last_error_logs


def _result_field(result: Any, name: str, default: Any = None) -> Any:
    """Read a run result while remaining compatible with older monkeypatches."""
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


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


def _cleanup_run_artifacts(case_dir: str) -> None:
    """Remove disposable outputs before retrying a newly generated case."""
    root = Path(case_dir)
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_symlink():
                if entry.name.startswith("log") or entry.name in {"Allrun.out", "Allrun.err"}:
                    entry.unlink()
                continue
            if entry.is_file() and (
                entry.name.startswith("log")
                or entry.name in {"Allrun.out", "Allrun.err"}
            ):
                entry.unlink()
            elif entry.is_dir() and (
                re.fullmatch(r"processor\d+", entry.name)
                or entry.name in {"postProcessing", "VTK"}
            ):
                shutil.rmtree(entry)
        except OSError as exc:
            raise RuntimeError(f"Unable to clean prior run artifact {entry}: {exc}") from exc
    remove_numeric_folders(case_dir)


def _execution_output(case_dir: str) -> str:
    """Return recent process output as diagnostic context for the Reviewer."""
    excerpts: List[str] = []
    for name in ("Allrun.err", "Allrun.out"):
        path = Path(case_dir) / name
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if content.strip():
            excerpts.append(f"[{name}]\n" + "\n".join(content.splitlines()[-200:]))
    return "\n\n".join(excerpts)
