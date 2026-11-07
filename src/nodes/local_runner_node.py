"""Common local execution node for generated and imported cases."""

from typing import Any

from logger import log_review
from openfoam_target import runtime_openfoam_target
from services.run_local import run_allrun_and_collect_errors
from services.review_history import ReviewHistory
from services.case_inputs import case_input_fingerprint
from services.case_workspace import prepare_case_workspace


def local_runner_node(state: dict[str, Any]) -> dict[str, Any]:
    case_dir = state["case_dir"]
    history = ReviewHistory(case_dir, state.get("case_import_report_dir"))
    workspace = prepare_case_workspace(
        case_dir, state.get("case_import_report_dir"), state.get("case_input_source_dir"),
    )
    state = {**state, "case_input_source_dir": str(workspace.source_dir) if workspace else None}
    fingerprint = case_input_fingerprint(case_dir)
    failed_inputs = dict(state.get("failed_run_inputs") or {})
    if fingerprint and fingerprint in failed_inputs:
        message = "Repair restored inputs that already failed in this workflow; execution was not repeated."
        history.append("run_skipped", reason="revisited_failed_inputs", input_fingerprint=fingerprint,
                       runtime_loop=state.get("loop_count", 0))
        print(f"<runner>{message}</runner>")
        return {**state, "workflow_status": "failed", "termination_reason": "repair_made_no_progress",
                "workflow_message": message, "error_logs": failed_inputs[fingerprint]}
    history.append("run_started", runner="local", runtime_loop=state.get("loop_count", 0),
                   input_fingerprint=fingerprint)
    print("<runner>")
    error_logs = run_allrun_and_collect_errors(
        case_dir,
        state["config"].max_time_limit,
        openfoam_target=runtime_openfoam_target(state["config"]),
    )

    if error_logs:
        print("Errors detected in the Allrun execution.")
        log_review(str(error_logs), "error_logs")
        if fingerprint:
            failed_inputs[fingerprint] = error_logs
    else:
        print("Allrun executed successfully without errors.")
        state["workflow_status"] = "success"
        state["termination_reason"] = None

    history.append("run_result", runner="local", runtime_loop=state.get("loop_count", 0),
                   error_logs=error_logs, input_fingerprint=fingerprint)

    print("</runner>")
    return {
        **state,
        # The workflow's one initial static-review opportunity is now consumed.
        "static_review_complete": True,
        "failed_run_inputs": failed_inputs,
        "last_run_input_fingerprint": fingerprint,
        "error_logs": error_logs
    }
