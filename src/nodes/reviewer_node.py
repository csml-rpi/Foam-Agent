# reviewer_node.py
from pydantic import BaseModel, Field
from typing import List
from services.review import review_error_logs, generate_rewrite_plan
from logger import log_review


def reviewer_node(state):
    """
    Reviewer node: single-call review (FA 1.1.0 style).
    No rewrite_plan generation, no snapshot, no oscillation detection.
    """
    print("<reviewer>")
    if len(state["error_logs"]) == 0:
        print("No error to review.")
        print("</reviewer>")
        return state

    # Log error logs to review.log
    log_review(str(state["error_logs"]), "error_logs")

    error_logs = state.get("error_logs", [])
    loop_count = state.get("loop_count", 0)
    history_text = state.get("history_text") or []

    review_content, updated_history = review_error_logs(
        tutorial_reference=state.get("tutorial_reference", ""),
        foamfiles=state.get("foamfiles"),
        error_logs=error_logs,
        user_requirement=state.get("user_requirement", ""),
        similar_case_advice=state.get("similar_case_advice"),
        history_text=history_text,
        loop_count=loop_count,
        pending_rewrite_plan=None,
        pending_file_diff=None,
        oscillation_hint="",
    )

    log_review(review_content, "review_analysis")

    print("</reviewer>")

    return {
        "history_text": updated_history,
        "review_analysis": review_content,
        "loop_count": loop_count + 1,
        "input_writer_mode": "rewrite",
    }
