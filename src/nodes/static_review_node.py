"""Review before the first execution and reuse the shared Writer for corrections."""

import json
from functools import partial

from logger import log_review
from openfoam_target import generation_convention
from services.static_review import review_case_inputs
from services.review_history import ReviewHistory


def static_review_node(state):
    retries = state.get("static_review_retries", 0)
    history = ReviewHistory(state["case_dir"], state.get("case_import_report_dir"))
    report_path = history.directory / "static_review.json"
    reference_path = history.directory / "static_review_references.json"
    advice = state.get("similar_case_advice")
    references = {
        "tutorial_reference": state.get("tutorial_reference") or "",
        "allrun_reference": state.get("allrun_reference") or "",
        "similar_case_advice": advice.model_dump() if hasattr(advice, "model_dump") else advice,
    }
    try:
        report, foamfiles = review_case_inputs(
            case_dir=state["case_dir"],
            user_requirement=state.get("user_requirement", ""),
            openfoam_fork=generation_convention(state["config"]),
            llm_service=state.get("llm_service"),
            review_context=history.context(events={"static_review", "file_repair", "static_review_exit"}),
            record_event=partial(history.append, runtime_loop=state.get("loop_count", 0), iteration=retries + 1),
            **references,
        )
        history.append(
            "static_review", runtime_loop=state.get("loop_count", 0), iteration=retries + 1,
            user_requirement=state.get("user_requirement", ""), report=report.model_dump(),
            references_path=str(reference_path),
        )
        reference_path.write_text(json.dumps(references, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        log_review(report.model_dump_json(), "static_review")
    except Exception as exc:
        message = f"Static review failed; execution was not started: {exc}"
        log_review(message, "static_review_failed")
        try:
            history.append("static_review_failed", error=message, references=references)
            report_path.write_text(json.dumps({"error": message}, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass  # The workflow message and logs still retain the diagnosis.
        return {
            "static_review_report": None,
            "workflow_status": "failed",
            "termination_reason": "static_review_failed",
            "workflow_message": message,
        }

    result = {"static_review_report": report.model_dump(), "foamfiles": foamfiles,
              "input_omissions": report.omitted_files}
    if not report.errors:
        history.append("static_review_exit", reason="passed", runtime_loop=state.get("loop_count", 0))
        result.update({"static_review_complete": True, "repair_source": None})
        return result

    analysis = "\n".join(
        f"{finding.file} ({finding.location}): {finding.message}\n"
        f"Evidence: {finding.evidence}\nCorrection: {finding.suggestion}"
        for finding in report.errors
    )
    if retries >= state["config"].max_static_review_retries:
        result.update({
            "static_review_complete": True,
            "repair_source": None,
        })
        history.append("static_review_exit", reason="iteration_limit", runtime_loop=state.get("loop_count", 0))
        return result

    # The verified review already provides the repair analysis; keep it out of
    # the runtime-review loop.
    result.update({
        "review_analysis": (
            "This is a pre-execution static repair. Correct only definite, necessary "
            "defects supported by the current files and original requirement. The "
            "review may be mistaken. Leave runtime-dependent questions for execution; "
            "return an empty file list if no justified static correction remains.\n" + analysis
        ),
        "rewrite_plan": None,
        "input_writer_mode": "rewrite",
        "repair_source": "static_review",
        "static_review_retries": retries + 1,
        "static_review_complete": False,
    })
    return result
