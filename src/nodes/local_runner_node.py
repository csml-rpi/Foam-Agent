"""Common local execution node for generated and imported cases."""
from logger import print

from typing import Any

from logger import log_review
from openfoam_target import runtime_openfoam_target
from services.run_local import run_allrun_and_collect_errors
from services.stage_plan import check_stage_inputs, check_stage_script, stage_outcome
from services.execution_logs import declared_startup_inputs
from services.solver_attempts import prepare_solver_attempt, record_solver_attempt
from services.case_inputs import canonical_input_protections


def local_runner_node(state: dict[str, Any]) -> dict[str, Any]:
    state, skipped = prepare_solver_attempt(state, 'local')
    if skipped:
        return state
    case_dir = state["case_dir"]
    protected = canonical_input_protections(
        state.get('case_input_source_dir'), declared_startup_inputs(state['solver_operations']))
    print("<runner>")
    errors = check_stage_inputs(case_dir, state["solver_operations"], "solver")
    if not errors:
        errors = check_stage_script(case_dir, state['solver_operations'], 'solver')
    if errors:
        return record_solver_attempt(state, {
            **stage_outcome(state, "solver", {"executed": False, "error_logs": errors}),
            "error_logs": errors, "solver_execution": {"executed": False, "error_logs": errors}}, 'local')
    error_logs = run_allrun_and_collect_errors(
        case_dir,
        state["config"].max_time_limit,
        openfoam_target=runtime_openfoam_target(state["config"]),
        protected_inputs=protected,
    )

    if error_logs:
        print("Errors detected in the Allrun execution.")
        log_review(str(error_logs), "error_logs")
    else:
        print("Allrun executed successfully without errors.")
        state["workflow_status"] = "success"
        state["termination_reason"] = None

    print("</runner>")
    return record_solver_attempt(state, {
        **state,
        **stage_outcome(state, "solver", {"executed": True, "error_logs": error_logs}),
        "solver_execution": {"executed": True, "error_logs": error_logs},
        "error_logs": error_logs
    }, 'local')
