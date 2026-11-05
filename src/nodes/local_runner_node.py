"""Common local execution node for generated and imported cases."""

from typing import Any

from logger import log_review
from openfoam_target import runtime_openfoam_target
from services.run_local import run_allrun_and_collect_errors
from services.stage_plan import check_stage_inputs, check_stage_script, stage_outcome


def local_runner_node(state: dict[str, Any]) -> dict[str, Any]:
    case_dir = state["case_dir"]
    print("<runner>")
    errors = check_stage_inputs(case_dir, state["solver_operations"], "solver")
    if not errors:
        errors = check_stage_script(case_dir, state['solver_operations'], 'solver')
    if errors:
        return {**stage_outcome(state, "solver", {"executed": False, "error_logs": errors}),
                "error_logs": errors, "solver_execution": {"executed": False, "error_logs": errors}}
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
        **stage_outcome(state, "solver", {"executed": True, "error_logs": error_logs}),
        "solver_execution": {"executed": True, "error_logs": error_logs},
        "error_logs": error_logs
    }
