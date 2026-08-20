"""Common repair-decision node with policy-specific repair adapters."""

from typing import Any

from nodes.imported_case_node import repair_imported_case
from services.review import review_error_logs, generate_rewrite_plan
from logger import log_review


def reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Select the allowed repair strategy for a failed prepared case.

    Generated cases receive an LLM analysis plus a constrained rewrite plan.
    Imported cases never reach that LLM path: they receive only deterministic
    non-numeric repairs approved by the import policy.
    """
    repair_policy = state.get("repair_policy", "llm_rewrite")
    if repair_policy == "numeric_invariant_only":
        return repair_imported_case(state)
    if repair_policy != "llm_rewrite":
        return {
            "termination_reason": "unsupported_repair_policy",
        }

    print("<reviewer>")
    if len(state["error_logs"]) == 0:
        print("No error to review.")
        print("</reviewer>")
        return state

    # Log error logs to review.log
    log_review(str(state["error_logs"]), "error_logs")

    # Stateless review via service
    history_text = state.get("history_text") or []
    review_content, updated_history = review_error_logs(
        tutorial_reference=state.get('tutorial_reference', ''),
        foamfiles=state.get('foamfiles'),
        error_logs=state.get('error_logs'),
        user_requirement=state.get('user_requirement', ''),
        similar_case_advice=state.get('similar_case_advice'),
        history_text=history_text,
        llm_service=state.get("llm_service"),
    )

    log_review(review_content, "review_analysis")

    rewrite_plan = generate_rewrite_plan(
        foamfiles=state.get('foamfiles'),
        error_logs=state.get('error_logs', []),
        review_analysis=review_content,
        user_requirement=state.get('user_requirement', ''),
        llm_service=state.get("llm_service"),
    )
    log_review(str(rewrite_plan), "rewrite_plan")

    print("</reviewer>")

    next_loop_count = state.get("loop_count", 0) + 1
    result = {
        "history_text": updated_history,
        "review_analysis": review_content,
        "rewrite_plan": rewrite_plan,
        "loop_count": next_loop_count,
        "input_writer_mode": "rewrite",
    }
    # Conditional-edge router mutations are not persisted by LangGraph. Store
    # the terminal reason in this node update so the CLI can return a non-zero
    # exit status when the retry budget is exhausted.
    if next_loop_count >= state["config"].max_loop:
        result["termination_reason"] = "max_review_loop_reached"
    return result
