import hashlib
import json
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from models import PlannedFileChange
from .case_inputs import case_input_fingerprint, format_case_inputs
from openfoam_target import generation_contract
from . import global_llm_service


class RewritePlan(BaseModel):
    target_files: List[PlannedFileChange] = Field(description="Files to modify and required changes")


REVIEWER_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM simulation and numerical modeling. "
    "Diagnose the latest execution failure using the current files, actual applied changes, "
    "and target-version references. History contains earlier evidence and fallible opinions; "
    "a past error is not a current blocker unless the latest evidence still supports it. "
    "If an edit disappears or the executed dictionary differs from the repair, trace Allrun "
    "and its templates to the producing definition; repair that source instead of repeatedly "
    "editing an output the script regenerates. "
    "Distinguish the fatal error from its preceding numerical symptoms. "
    "Give one minimal, coherent repair or controlled numerical trial supported by that evidence, "
    "naming the affected files and the hypothesis the next run will test. Numerical instability "
    "does not require proof of a unique root cause before trying a compatible change to "
    "unspecified numerical controls; explain the uncertainty and expected effect. "
    "Missing keywords must retain the exact spelling from "
    "the error, including punctuation. Preserve all physical conditions and parameters "
    "specified by the user. References demonstrate compatible syntax, not replacement physics. "
    "Separate necessary repairs from questions requiring another run and optional diagnostics. "
    "Do not assert an uncertain diagnosis as fact, add monitoring configuration "
    "as a substitute for a repair, or change a resolved setting merely to try a different approach. "
    "If the evidence does not justify a file correction, say so. Do not modify files or ask questions."
)


def review_error_logs(
    tutorial_reference: str,
    foamfiles: Any,
    error_logs: List[Any],
    user_requirement: str,
    similar_case_advice: Optional[Any] = None,
    history_text: Optional[List[str]] = None,
    llm_service: Optional[Any] = None,
    openfoam_target: str = "",
    last_repair: Optional[dict] = None,
    omitted_files: Optional[dict[str, str]] = None,
    source_inputs: Optional[dict] = None,
) -> Tuple[str, List[str]]:
    """Stateless reviewer: returns (review_analysis, updated_history)."""
    advice_text = ""
    if similar_case_advice:
        advice = similar_case_advice.model_dump() if hasattr(similar_case_advice, "model_dump") else similar_case_advice
        advice_text = f"<similar_case_advice>{advice}</similar_case_advice>\n"

    applied_changes = json.dumps(last_repair or {}, ensure_ascii=False)
    if len(applied_changes) > 64_000:
        applied_changes = "[Earlier edit details omitted; complete diffs remain in the journal.]\n" + applied_changes[-64_000:]
    source_block = (
        f"<source_input_overrides>{json.dumps(source_inputs, ensure_ascii=False)}</source_input_overrides>\n"
        if source_inputs is not None else ""
    )
    reviewer_user_prompt = (
        f"<history>\n{chr(10).join(history_text or [])}\n</history>\n"
        f"<similar_case_reference>{tutorial_reference}</similar_case_reference>\n"
        f"{advice_text}"
        f"<foamfiles>{format_case_inputs(foamfiles)}</foamfiles>\n"
        f"{source_block}"
        f"<omitted_files>{json.dumps(omitted_files or {})}</omitted_files>\n"
        f"<actual_applied_changes>{applied_changes}</actual_applied_changes>\n"
        f"<current_error_logs>{error_logs}</current_error_logs>\n"
        f"<user_requirement>{user_requirement}</user_requirement>\n"
        "Diagnose this latest attempt. Current files and errors take precedence over previous suggestions."
    )

    llm_client = llm_service if llm_service is not None else global_llm_service
    system_prompt = _reviewer_system_prompt(openfoam_target)
    if source_inputs is not None:
        system_prompt += (
            "\nSource input overrides contain canonical files that differ from the executed inputs above. "
            "Writer edits these canonical sources; before the next Run the workflow restores them "
            "and reruns preprocessing. Use the executed files to diagnose the failure and fix the "
            "source or producing script, rather than copying expanded runtime arrays into source inputs."
        )
    review_response = llm_client.invoke(reviewer_user_prompt, system_prompt)
    review_content = review_response

    updated_history = list(history_text) if history_text else []
    attempt_num = sum(item.startswith("<Attempt ") for item in updated_history) + 1
    current_attempt = [
        f"<Attempt {attempt_num}>\n",
        f"<Error_Logs>\n{error_logs}\n</Error_Logs>",
        f"<Applied_Changes>\n{applied_changes}\n</Applied_Changes>",
        f"<Review_Analysis>\n{review_content}\n</Review_Analysis>",
        "</Attempt>\n",
    ]
    updated_history.extend(current_attempt)
    return review_content, updated_history


def generate_rewrite_plan(
    foamfiles: Any,
    error_logs: List[str],
    review_analysis: str,
    user_requirement: str,
    llm_service: Optional[Any] = None,
    openfoam_target: str = "",
) -> dict:
    """Generate a minimal, explicit rewrite plan for downstream rewrite step."""
    planner_system_prompt = (
        "You are an OpenFOAM debugging planner. "
        "Given current foam files, error logs and reviewer analysis, create a minimal rewrite plan. "
        "Output MUST be strict JSON only, with this exact schema: "
        "{\"target_files\": [{\"file\": \"relative/path\", \"changes\": \"change1; change2\"}]}. "
        "Rules: "
        "1) Do not use markdown, backticks, or comments. "
        "2) Use double quotes for all strings. "
        "3) In changes, use short plain text actions separated by semicolons. "
        "4) Do not include parentheses, backticks, or quote characters inside changes text. "
        "5) Do not include run steps; only file edits."
    )
    if openfoam_target == "esi-v2006":
        planner_system_prompt += (
            " The target is native ESI/OpenCFD OpenFOAM v2006; preserve its "
            "dictionary conventions and do not request Foundation v10 translations."
        )

    planner_user_prompt = (
        f"<foamfiles>{format_case_inputs(foamfiles)}</foamfiles>\n"
        f"<error_logs>{error_logs}</error_logs>\n"
        f"<review_analysis>{review_analysis}</review_analysis>\n"
        f"<user_requirement>{user_requirement}</user_requirement>\n"
        "Return strict JSON now with key target_files only."
    )

    llm_client = llm_service if llm_service is not None else global_llm_service
    response = llm_client.invoke(
        planner_user_prompt,
        planner_system_prompt,
        pydantic_obj=RewritePlan,
    )
    return response.model_dump()


def error_fingerprint(state: dict[str, Any]) -> Optional[str]:
    """Compare errors and case inputs, excluding changing runtime outputs."""
    # Compare actual attempts, not just their post-run files: Allrun may regenerate
    # a dictionary and erase a repair before the solver reads it.
    if "last_run_input_fingerprint" in state and not state.get("repairing_mesh"):
        inputs = state["last_run_input_fingerprint"]
    else:
        inputs = case_input_fingerprint(state.get("case_dir") or "")
    if inputs is None:
        return None
    payload = {
        # Diagnostic tails retain volatile process metadata for the Reviewer,
        # but do not change the identity of an otherwise identical failure.
        "errors": [
            {key: value for key, value in error.items() if key != "diagnostic_context"}
            if isinstance(error, dict) else error
            for error in state.get("error_logs") or []
        ],
        "inputs": inputs,
        "user_requirement": state.get("user_requirement", ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _reviewer_system_prompt(openfoam_target: str) -> str:
    """Use the same target contract as generation, rewriting and static review."""
    return REVIEWER_SYSTEM_PROMPT + "\n" + generation_contract(openfoam_target)
