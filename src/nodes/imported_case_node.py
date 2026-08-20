"""Imported-case preparation plus adapters used by common workflow nodes.

This branch deliberately never calls Planner, Meshing, Input Writer, or the
LLM reviewer.  An imported case is user-owned input: its dictionaries are
executed only through a validated command plan, and retries may apply only the
deterministic numeric-invariant repairs defined by ``case_import_repairs``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from logger import log_review, setup_logging
from services.case_import import (
    CaseImportError,
    execute_imported_case,
    execute_imported_case_attempt,
    import_case,
    prepare_imported_work_tree,
    repair_imported_case_attempt,
    write_import_attempt_report,
)


def case_import_node(state: dict[str, Any]) -> dict[str, Any]:
    """Validate and materialise an input case without running user shell code."""
    print("<case_import>")
    try:
        manifest = import_case(
            state["case_import_path"],
            state["config"].case_dir,
            case_subdir=state.get("case_import_subdir"),
            overwrite=state["config"].overwrite_case_dir,
        )
    except CaseImportError as exc:
        error = {"file": "case_import", "error_content": str(exc)}
        print(f"<case_import_error>{exc}</case_import_error>")
        print("</case_import>")
        return {
            "case_import_status": "blocked",
            "error_logs": [error],
            "termination_reason": "case_import_blocked",
        }

    original_dir = Path(manifest.output_root) / "original"
    work_dir = Path(manifest.output_root) / "work"
    report_dir = Path(manifest.output_root) / "report"
    setup_logging(str(report_dir))
    print(f"<platform>{manifest.platform}</platform>")
    print(f"<application>{manifest.application}</application>")
    print(f"<allrun_provided>{manifest.allrun_provided}</allrun_provided>")
    print("</case_import>")

    requested_visualization = state.get("requires_visualization") is True
    base_result = {
        "case_import_manifest": manifest,
        "case_import_original_dir": str(original_dir),
        "case_import_report_dir": str(report_dir),
        "case_import_attempts": [],
        "case_import_overrides": {},
        "case_import_error_fingerprints": [],
        "case_dir": str(work_dir),
        "case_name": Path(manifest.case_root).name or manifest.application,
        "case_solver": manifest.application,
        # The common runner/reviewer route by these explicit policies, rather
        # than by the fact that this case happened to come from a CLI import.
        "execution_policy": "controlled_import",
        "repair_policy": "numeric_invariant_only",
        # Imported execution is local. Visualization is explicitly requested
        # only and uses the deterministic renderer on the disposable work copy;
        # it never invokes the LLM fallback or changes original inputs.
        "requires_hpc": False,
        "requires_visualization": requested_visualization,
    }
    if not manifest.supported:
        errors = [
            {"file": "case_manifest", "error_content": issue}
            for issue in manifest.blocking_issues
        ]
        write_import_attempt_report(report_dir, [])
        log_review(str(errors), "case_import_blockers")
        return {
            **base_result,
            "case_import_status": "blocked",
            "error_logs": errors,
            "termination_reason": "case_import_blocked",
        }

    prepare_imported_work_tree(work_dir)
    return {**base_result, "case_import_status": "ready", "error_logs": []}


def run_imported_case_attempt(state: dict[str, Any]) -> dict[str, Any]:
    """Execute one controlled import attempt for the common runner node."""
    print("<imported_case_runner>")
    result = execute_imported_case_attempt(
        state["case_import_manifest"],
        original_dir=state["case_import_original_dir"],
        work_dir=state["case_dir"],
        report_dir=state["case_import_report_dir"],
        attempts=state.get("case_import_attempts") or [],
        overrides=state.get("case_import_overrides") or {},
        timeout=state["config"].max_time_limit,
        executor=execute_imported_case,
    )
    print("</imported_case_runner>")
    if result["errors"]:
        log_review(str(result["errors"]), "error_logs")
    return {
        "case_import_attempts": result["attempts"],
        "case_import_status": result["status"],
        "error_logs": result["errors"],
    }


def repair_imported_case(state: dict[str, Any]) -> dict[str, Any]:
    """Apply an error-gated safe repair for the common reviewer node."""
    errors = list(state.get("error_logs") or [])
    if not errors:
        return {"case_import_status": "success"}

    print("<imported_case_reviewer>")
    result = repair_imported_case_attempt(
        original_dir=state["case_import_original_dir"],
        work_dir=state["case_dir"],
        report_dir=state["case_import_report_dir"],
        attempts=state.get("case_import_attempts") or [],
        errors=errors,
        error_fingerprints=state.get("case_import_error_fingerprints") or [],
        loop_count=state.get("loop_count", 0),
        max_repairs=state["config"].max_loop,
    )
    print("</imported_case_reviewer>")
    if "repairs" in result:
        log_review(str(result["repairs"]), "safe_import_repairs")
    update = {
        "case_import_attempts": result["attempts"],
        "case_import_error_fingerprints": result["error_fingerprints"],
        "case_import_status": result["status"],
    }
    if result["status"] == "ready":
        update["case_import_overrides"] = result["overrides"]
        update["loop_count"] = result["loop_count"]
    elif result.get("termination_reason"):
        update["termination_reason"] = result["termination_reason"]
    return update
