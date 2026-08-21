"""Public orchestration API for safely importing existing OpenFOAM cases.

The detailed responsibilities live in focused modules:

* :mod:`case_import_source` validates and materialises an input;
* :mod:`case_import_allrun` converts an allowed Allrun into a safe plan;
* :mod:`case_import_repairs` owns the strict non-numeric repair policy.

Keeping this module as the orchestration boundary makes the CLI-facing API
small without hiding the safety rules behind a single 1300-line service.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import stat
from typing import Any

from logger import close_logging, setup_logging
from utils import check_foam_errors, run_command
from .case_import_allrun import render_controlled_allrun
from .case_import_models import (
    CaseImportError,
    CaseManifest,
    ExecutionStep,
    ImportRunResult,
)
from .case_import_repairs import (
    apply_safe_repairs,
    numeric_signature,
)
from .case_import_source import import_case, iter_regular_files
from .run_local import (
    validate_openfoam_case_postflight,
    validate_openfoam_case_preflight,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clear_attempt_artifacts(case_dir: Path) -> None:
    """Clear only import-attempt logs, never source dictionaries or time folders."""
    for path in case_dir.iterdir():
        if path.is_file() and (
            path.name.startswith("log.")
            or path.name in {"Allrun.import.out", "Allrun.import.err"}
        ):
            path.unlink(missing_ok=True)


def _capture_safe_work_overrides(
    original_dir: Path,
    work_dir: Path,
) -> dict[str, bytes]:
    """Keep only numeric-invariant input edits for a clean retry."""
    overrides: dict[str, bytes] = {}
    for work_path in iter_regular_files(work_dir):
        if ".foamagent" in work_path.parts:
            continue
        relative = work_path.relative_to(work_dir)
        original_path = original_dir / relative
        if not original_path.is_file():
            continue
        try:
            old_bytes = original_path.read_bytes()
            new_bytes = work_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"Unable to capture safe repair {relative}: {exc}") from exc
        if old_bytes == new_bytes:
            continue
        old_text = old_bytes.decode("utf-8", errors="replace")
        new_text = new_bytes.decode("utf-8", errors="replace")
        if numeric_signature(old_text) != numeric_signature(new_text):
            raise RuntimeError(f"Repair for {relative} changed numeric inputs and cannot be replayed.")
        overrides[relative.as_posix()] = new_bytes
    return overrides


def _make_work_tree_writable(work_dir: Path) -> None:
    for path in (work_dir, *work_dir.rglob("*")):
        if path.is_symlink():
            continue
        try:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        except OSError as exc:
            raise RuntimeError(f"Unable to prepare mutable work copy {path}: {exc}") from exc


def _restore_clean_work_copy(
    original_dir: Path,
    work_dir: Path,
    overrides: dict[str, bytes],
) -> None:
    """Restore immutable input and replay only prior approved repairs."""
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.copytree(original_dir, work_dir)
    _make_work_tree_writable(work_dir)
    for relative, content in overrides.items():
        destination = work_dir / Path(relative)
        try:
            destination.relative_to(work_dir)
        except ValueError as exc:
            raise RuntimeError(f"Unsafe stored repair path: {relative}") from exc
        if not destination.is_file():
            raise RuntimeError(f"Stored repair target disappeared: {relative}")
        destination.write_bytes(content)


def execute_imported_case(
    work_dir: str | Path,
    manifest: CaseManifest,
    *,
    timeout: int,
) -> list[Any]:
    """Execute the generated controlled script in an imported case work copy."""
    case_dir = Path(work_dir)
    controlled_dir = case_dir / ".foamagent"
    controlled_dir.mkdir(exist_ok=True)
    controlled_script = controlled_dir / "Allrun.controlled"
    script_content = render_controlled_allrun(manifest.execution_plan)
    controlled_script.write_text(script_content, encoding="utf-8")

    preflight = validate_openfoam_case_preflight(str(case_dir), script_content)
    if preflight:
        return preflight

    _clear_attempt_artifacts(case_dir)
    out_file = case_dir / "Allrun.import.out"
    err_file = case_dir / "Allrun.import.err"
    try:
        command_result = run_command(
            str(controlled_script),
            str(out_file),
            str(err_file),
            str(case_dir),
            timeout,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return [{"file": "Allrun", "error_content": str(exc)}]

    errors: list[Any] = []
    if command_result.get("timed_out"):
        errors.append(
            {
                "file": "Allrun",
                "error_content": f"Controlled execution exceeded the {timeout} second timeout.",
            }
        )
    elif command_result.get("returncode") not in (None, 0):
        errors.append(
            {
                "file": "Allrun",
                "error_content": "Controlled execution exited with non-zero return code "
                f"{command_result.get('returncode')}.",
            }
        )
    errors.extend(check_foam_errors(str(case_dir)))
    errors.extend(validate_openfoam_case_postflight(str(case_dir), script_content))

    deduplicated: list[Any] = []
    seen: set[str] = set()
    for error in errors:
        marker = json.dumps(error, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            deduplicated.append(error)
    return deduplicated


def write_import_attempt_report(report_dir: str | Path, attempts: list[dict[str, Any]]) -> None:
    """Persist the complete current history of import attempts."""
    report_dir = Path(report_dir)
    _write_json(report_dir / "attempts.json", attempts)


def prepare_imported_work_tree(work_dir: str | Path) -> None:
    """Make the disposable work copy writable before a controlled run."""
    _make_work_tree_writable(Path(work_dir))


def restore_imported_work_tree(
    original_dir: str | Path,
    work_dir: str | Path,
    overrides: dict[str, bytes],
) -> None:
    """Recreate ``work/`` from ``original/`` and replay approved safe repairs."""
    _restore_clean_work_copy(Path(original_dir), Path(work_dir), overrides)


def capture_safe_import_overrides(
    original_dir: str | Path,
    work_dir: str | Path,
) -> dict[str, bytes]:
    """Return only numeric-invariant repairs that may be replayed on retry."""
    return _capture_safe_work_overrides(Path(original_dir), Path(work_dir))


def _error_fingerprint(errors: list[Any]) -> str:
    """Produce a stable marker so an unchanged failed retry stops promptly."""
    return hashlib.sha256(
        json.dumps(errors, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _blocked_import_attempt(
    attempts: list[dict[str, Any]],
    report_dir: str | Path,
    reason: str,
    error_fingerprints: list[str],
) -> dict[str, Any]:
    """Persist a terminal import attempt in the format shared by both runners."""
    if attempts:
        attempts[-1] = {**attempts[-1], "terminal_reason": reason}
    write_import_attempt_report(report_dir, attempts)
    return {
        "attempts": attempts,
        "status": "blocked",
        "termination_reason": reason,
        "error_fingerprints": error_fingerprints,
    }


def execute_imported_case_attempt(
    manifest: CaseManifest,
    *,
    original_dir: str | Path,
    work_dir: str | Path,
    report_dir: str | Path,
    attempts: list[dict[str, Any]],
    overrides: dict[str, bytes],
    timeout: int,
    executor: Any = None,
) -> dict[str, Any]:
    """Run and record one controlled imported-case attempt.

    This is the one shared transition used by the direct public API and the
    LangGraph runner adapter.  The optional executor keeps the adapter easy to
    test while production callers use :func:`execute_imported_case`.
    """
    attempts = list(attempts)
    try:
        if attempts:
            restore_imported_work_tree(original_dir, work_dir, overrides)
        run_case = executor if executor is not None else execute_imported_case
        errors = run_case(work_dir, manifest, timeout=timeout)
    except Exception as exc:  # Keep every runner failure in the audited history.
        errors = [{"file": "Allrun", "error_content": str(exc)}]

    record = {"attempt": len(attempts) + 1, "errors": errors, "repairs": []}
    attempts.append(record)
    write_import_attempt_report(report_dir, attempts)
    return {
        "attempts": attempts,
        "status": "success" if not errors else "running",
        "errors": errors,
    }


def repair_imported_case_attempt(
    *,
    original_dir: str | Path,
    work_dir: str | Path,
    report_dir: str | Path,
    attempts: list[dict[str, Any]],
    errors: list[Any],
    error_fingerprints: list[str],
    loop_count: int,
    max_repairs: int,
) -> dict[str, Any]:
    """Apply at most one approved repair and return the next import state.

    Both execution entry points now share fingerprint handling, retry limits,
    report updates, and numeric-invariant override capture through this
    transition rather than maintaining parallel retry implementations.
    """
    attempts = list(attempts)
    known_fingerprints = list(error_fingerprints)
    if not errors:
        return {
            "attempts": attempts,
            "status": "success",
            "error_fingerprints": known_fingerprints,
        }

    fingerprint = _error_fingerprint(errors)
    if fingerprint in known_fingerprints:
        return _blocked_import_attempt(
            attempts,
            report_dir,
            "repeated_error_fingerprint",
            known_fingerprints,
        )

    known_fingerprints.append(fingerprint)
    if loop_count >= max_repairs:
        return _blocked_import_attempt(
            attempts,
            report_dir,
            "maximum_safe_repair_attempts_reached",
            known_fingerprints,
        )

    repairs = apply_safe_repairs(work_dir, errors)
    if attempts:
        attempts[-1] = {**attempts[-1], "repairs": repairs}
    write_import_attempt_report(report_dir, attempts)
    if not any(repair.get("status") == "applied" for repair in repairs):
        result = _blocked_import_attempt(
            attempts,
            report_dir,
            "no_safe_non_numeric_repair_available",
            known_fingerprints,
        )
        result["repairs"] = repairs
        return result

    return {
        "attempts": attempts,
        "overrides": capture_safe_import_overrides(original_dir, work_dir),
        "error_fingerprints": known_fingerprints,
        "loop_count": loop_count + 1,
        "repairs": repairs,
        "status": "ready",
    }


def _run_imported_manifest(
    manifest: CaseManifest,
    *,
    timeout: int,
    max_repairs: int,
) -> ImportRunResult:
    """Run a materialised case until it succeeds or no safe repair remains."""
    output_root = Path(manifest.output_root)
    original_dir = output_root / "original"
    work_dir = output_root / "work"
    report_dir = output_root / "report"
    print("<case_import>")
    print(f"<platform>{manifest.platform}</platform>")
    print(f"<application>{manifest.application}</application>")
    print(f"<allrun_provided>{manifest.allrun_provided}</allrun_provided>")
    print("</case_import>")

    attempts: list[dict[str, Any]] = []
    if not manifest.supported:
        errors = [
            {"file": "case_manifest", "error_content": issue}
            for issue in manifest.blocking_issues
        ]
        write_import_attempt_report(report_dir, attempts)
        return ImportRunResult(
            "blocked", str(original_dir), str(work_dir), str(report_dir), manifest, attempts, errors
        )

    # ``copytree`` deliberately retains the source mode bits.  The immutable
    # ``original`` copy is useful for auditability, but the first execution
    # must have the same writable work tree as later repair attempts.  Without
    # this, a read-only uploaded case fails before the controlled runner can
    # create its private ``.foamagent`` directory.
    prepare_imported_work_tree(work_dir)

    prior_fingerprints: list[str] = []
    approved_overrides: dict[str, bytes] = {}
    errors: list[Any] = []
    loop_count = 0
    while True:
        execution = execute_imported_case_attempt(
            manifest,
            original_dir=original_dir,
            work_dir=work_dir,
            report_dir=report_dir,
            attempts=attempts,
            overrides=approved_overrides,
            timeout=timeout,
        )
        attempts = execution["attempts"]
        errors = execution["errors"]
        if execution["status"] == "success":
            return ImportRunResult(
                "success", str(original_dir), str(work_dir), str(report_dir), manifest, attempts, []
            )

        repair = repair_imported_case_attempt(
            original_dir=original_dir,
            work_dir=work_dir,
            report_dir=report_dir,
            attempts=attempts,
            errors=errors,
            error_fingerprints=prior_fingerprints,
            loop_count=loop_count,
            max_repairs=max_repairs,
        )
        attempts = repair["attempts"]
        prior_fingerprints = repair["error_fingerprints"]
        if repair["status"] != "ready":
            return ImportRunResult(
                "blocked",
                str(original_dir),
                str(work_dir),
                str(report_dir),
                manifest,
                attempts,
                errors,
            )
        approved_overrides = repair["overrides"]
        loop_count = repair["loop_count"]


def run_imported_case(
    case_path: str | Path,
    output_dir: str | Path,
    *,
    case_subdir: str | None = None,
    overwrite: bool = False,
    timeout: int = 3600,
    max_repairs: int = 25,
) -> ImportRunResult:
    """Import, execute, and safely repair a Foundation v10 case."""
    manifest = import_case(
        case_path,
        output_dir,
        case_subdir=case_subdir,
        overwrite=overwrite,
    )
    setup_logging(str(Path(manifest.output_root) / "report"))
    try:
        return _run_imported_manifest(
            manifest,
            timeout=timeout,
            max_repairs=max_repairs,
        )
    finally:
        close_logging()


__all__ = [
    "CaseImportError",
    "CaseManifest",
    "ExecutionStep",
    "ImportRunResult",
    "apply_safe_repairs",
    "capture_safe_import_overrides",
    "execute_imported_case",
    "execute_imported_case_attempt",
    "import_case",
    "prepare_imported_work_tree",
    "repair_imported_case_attempt",
    "restore_imported_work_tree",
    "run_imported_case",
    "write_import_attempt_report",
]
