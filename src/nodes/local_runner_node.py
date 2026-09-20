"""Common local execution node for generated and imported cases."""

from typing import Any

from logger import log_review
from openfoam_target import runtime_openfoam_target
from services.run_local import run_allrun_and_collect_errors


def local_runner_node(state: dict[str, Any]) -> dict[str, Any]:
    case_dir = state["case_dir"]
    print("<runner>")
    error_logs = run_allrun_and_collect_errors(
        case_dir,
        state["config"].max_time_limit,
        openfoam_target=runtime_openfoam_target(state["config"]),
    )

    if error_logs:
        print("Errors detected in the Allrun execution.")
        log_review(str(error_logs), "error_logs")
    else:
        print("Allrun executed successfully without errors.")
        state["workflow_status"] = "success"
        state["termination_reason"] = None

    print("</runner>")
    return {
        **state,
        "error_logs": error_logs
    }
