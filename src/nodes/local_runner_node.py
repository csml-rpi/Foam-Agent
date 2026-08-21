"""Common execution node with policy-specific execution adapters."""

from typing import Any

from nodes.imported_case_node import run_imported_case_attempt
from services.run_local import run_allrun_and_collect_errors
from logger import log_review


def local_runner_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a prepared case locally and return normalized error records.

    Both input modes meet here only after their preparation stages:
    generated cases execute Foam-Agent's ``Allrun``; imported cases execute a
    data-only, validated command plan in their disposable work copy.  Keeping
    the distinction as a policy prevents an imported user ``Allrun`` from
    becoming executable merely because it reaches the shared graph node.
    """
    execution_policy = state.get("execution_policy", "generated_allrun")
    if execution_policy == "controlled_import":
        return run_imported_case_attempt(state)
    if execution_policy != "generated_allrun":
        return {
            "error_logs": [
                {
                    "file": "workflow",
                    "error_content": (
                        "Unsupported execution policy: "
                        f"{execution_policy!r}. Refusing to execute the case."
                    ),
                }
            ],
            "termination_reason": "unsupported_execution_policy",
        }

    case_dir = state["case_dir"]
    max_time_limit = state["config"].max_time_limit

    print("<runner>")

    # Execute using service and collect errors
    error_logs = run_allrun_and_collect_errors(case_dir, max_time_limit)

    if len(error_logs) > 0:
        print("Errors detected in the Allrun execution.")
        log_review(str(error_logs), "error_logs")
    else:
        print("Allrun executed successfully without errors.")

    print("</runner>")

    # Return updated state
    return {
        **state,
        "error_logs": error_logs
    }
        