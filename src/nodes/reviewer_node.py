# reviewer_node.py
from services.review import review_error_logs, generate_rewrite_plan, error_fingerprint
from logger import log_review
from openfoam_target import generation_convention


def reviewer_node(state):
    """
    Reviewer node: Reviews the error logs and provides analysis and suggestions
    for fixing the errors. This node only focuses on analysis, not file modification.
    """
    print("<reviewer>")
    if not state.get("error_logs"):
        print("No error to review.")
        print("</reviewer>")
        return state

    # Log error logs to review.log
    log_review(str(state["error_logs"]), "error_logs")

    fingerprint = error_fingerprint(state)
    fingerprints = list(state.get("error_fingerprints") or [])
    repeated = bool(fingerprints and fingerprint == fingerprints[-1])
    fingerprints.append(fingerprint)
    if repeated:
        print("Repair made no progress: errors and case inputs are unchanged.")
        print("</reviewer>")
        return {
            "error_fingerprints": fingerprints,
            "workflow_status": "failed",
            "termination_reason": "repair_made_no_progress",
        }

    # Stateless review via service
    history_text = state.get("history_text") or []
    review_content, updated_history = review_error_logs(
        tutorial_reference=state.get('tutorial_reference') or '',
        foamfiles=state.get('foamfiles'),
        error_logs=state.get('error_logs'),
        user_requirement=state.get('user_requirement', ''),
        similar_case_advice=state.get('similar_case_advice'),
        history_text=history_text,
        llm_service=state.get("llm_service"),
        openfoam_target=generation_convention(state["config"]),
    )

    log_review(review_content, "review_analysis")

    if state.get("repairing_mesh"):
        review_content = (
            "Mesh preprocessing failed. After any file edits, the workflow will retry "
            "meshing with this feedback before running the solver. "
            "Return an empty target_files list if only mesh regeneration is needed.\n"
            + review_content
        )

    rewrite_plan = generate_rewrite_plan(
        foamfiles=state.get('foamfiles'),
        error_logs=state.get('error_logs', []),
        review_analysis=review_content,
        llm_service=state.get("llm_service"),
        openfoam_target=generation_convention(state["config"]),
        user_requirement=state.get('user_requirement', ''),
    )
    log_review(str(rewrite_plan), "rewrite_plan")

    print("</reviewer>")

    next_loop_count = state.get("loop_count", 0) + 1
    result = {
        "error_fingerprints": fingerprints,
        "history_text": updated_history,
        "review_analysis": review_content,
        "rewrite_plan": rewrite_plan,
        "loop_count": next_loop_count,
        "input_writer_mode": "rewrite",
    }
    if next_loop_count >= state["config"].max_loop:
        result["termination_reason"] = "max_review_loop_reached"
        result["workflow_status"] = "failed"
    return result
