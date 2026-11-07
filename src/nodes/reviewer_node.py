# reviewer_node.py
from services.review import review_error_logs, error_fingerprint
from logger import log_review
from openfoam_target import generation_convention
from services.review_history import ReviewHistory
from services.case_inputs import read_case_inputs


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
    history = ReviewHistory(state["case_dir"], state.get("case_import_report_dir"))

    fingerprint = error_fingerprint(state)
    fingerprints = list(state.get("error_fingerprints") or [])
    repeated = bool(fingerprint and fingerprints and fingerprint == fingerprints[-1])
    if fingerprint:
        fingerprints.append(fingerprint)
    if repeated:
        history.append("runtime_review_exit", reason="no_progress", error_logs=state["error_logs"])
        print("Repair made no progress: errors and case inputs are unchanged.")
        print("</reviewer>")
        return {
            "error_fingerprints": fingerprints,
            "workflow_status": "failed",
            "termination_reason": "repair_made_no_progress",
        }

    # Stateless review via service
    foamfiles, omitted = read_case_inputs(state["case_dir"])
    source_files, source_omitted = foamfiles, omitted
    source_context = None
    if state.get("case_input_source_dir"):
        source_files, source_omitted = read_case_inputs(state["case_input_source_dir"])
        executed = {(f.folder_name, f.file_name): f.content for f in foamfiles.list_foamfile}
        source_context = {
            "files": {
                "/".join(part for part in (f.folder_name, f.file_name) if part): f.content
                for f in source_files.list_foamfile
                if executed.get((f.folder_name, f.file_name)) != f.content
            },
            "omitted_files": source_omitted,
        }
    history_text = state.get("history_text") or []
    review_content, updated_history = review_error_logs(
        tutorial_reference=state.get('tutorial_reference') or '',
        foamfiles=foamfiles,
        error_logs=state.get('error_logs'),
        user_requirement=state.get('user_requirement', ''),
        similar_case_advice=state.get('similar_case_advice'),
        history_text=history_text,
        llm_service=state.get("llm_service"),
        openfoam_target=generation_convention(state["config"]),
        last_repair=state.get("last_repair"),
        omitted_files=omitted,
        **({"source_inputs": source_context} if source_context is not None else {}),
    )

    log_review(review_content, "review_analysis")
    history.append(
        "runtime_review", runtime_loop=state.get("loop_count", 0) + 1,
        error_logs=state["error_logs"], review_analysis=review_content,
        applied_changes=state.get("last_repair"), omitted_files=omitted,
        input_source_dir=state.get("case_input_source_dir"), source_inputs=source_context,
    )

    if state.get("repairing_mesh"):
        review_content = (
            "Mesh preprocessing failed. After any file edits, the workflow will retry "
            "meshing with this feedback before running the solver. "
            "Preserve all user-specified physical conditions.\n"
            + review_content
        )

    print("</reviewer>")

    next_loop_count = state.get("loop_count", 0) + 1
    result = {
        "error_fingerprints": fingerprints,
        "history_text": updated_history,
        "review_analysis": review_content,
        # Clear any initial import plan so repair can address the current error.
        "rewrite_plan": None,
        "loop_count": next_loop_count,
        "input_writer_mode": "rewrite",
        "repair_source": "runtime",
        "foamfiles": source_files,
        "input_omissions": source_omitted,
    }
    if next_loop_count >= state["config"].max_loop:
        result["termination_reason"] = "max_review_loop_reached"
        result["workflow_status"] = "failed"
    return result
