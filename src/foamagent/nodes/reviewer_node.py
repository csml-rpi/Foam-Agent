# reviewer_node.py
from pydantic import BaseModel, Field
from typing import List
from foamagent.services.review import review_error_logs, generate_rewrite_plan
from foamagent.logger import log_review

from foamagent.logger import get_logger

logger = get_logger(__name__)


def reviewer_node(state):
    """
    Reviewer node: Reviews the error logs and provides analysis and suggestions
    for fixing the errors. This node only focuses on analysis, not file modification.
    """
    logger.info("<reviewer>")
    if len(state["error_logs"]) == 0:
        logger.info("No error to review.")
        logger.info("</reviewer>")
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
    )

    log_review(review_content, "review_analysis")

    rewrite_plan = generate_rewrite_plan(
        foamfiles=state.get('foamfiles'),
        error_logs=state.get('error_logs', []),
        review_analysis=review_content,
        user_requirement=state.get('user_requirement', ''),
    )
    log_review(str(rewrite_plan), "rewrite_plan")

    logger.info("</reviewer>")

    return {
        "history_text": updated_history,
        "review_analysis": review_content,
        "rewrite_plan": rewrite_plan,
        "loop_count": state.get("loop_count", 0) + 1,
        "input_writer_mode": "rewrite",
    }
