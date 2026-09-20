import hashlib
import json
from pathlib import Path
from .case_import import snapshot_files
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from models import PlannedFileChange
from . import global_llm_service


class RewritePlan(BaseModel):
    target_files: List[PlannedFileChange] = Field(description="Files to modify and required changes")


REVIEWER_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM simulation and numerical modeling. "
    "Your task is to review the provided error logs and diagnose the underlying issues. "
    "You will be provided with a similar case reference, which is a list of similar cases that are ordered by similarity. You can use this reference to help you understand the user requirement and the error."
    "When an error indicates that a specific keyword is undefined (for example, 'div(phi,(p|rho)) is undefined'), your response must propose a solution that simply defines that exact keyword as shown in the error log. "
    "Do not reinterpret or modify the keyword (e.g., do not treat '|' as 'or'); instead, assume it is meant to be taken literally. "
    "Propose ideas on how to resolve the errors, but do not modify any files directly. "
    "Please do not propose solutions that require modifying any parameters declared in the user requirement, try other approaches instead. Do not ask the user any questions."
    "The user will supply all relevant foam files along with the error logs, and within the logs, you will find both the error content and the corresponding error command indicated by the log file name."
)


def review_error_logs(
    tutorial_reference: str,
    foamfiles: Any,
    error_logs: List[str],
    user_requirement: str,
    similar_case_advice: Optional[Any] = None,
    history_text: Optional[List[str]] = None,
    llm_service: Optional[Any] = None,
    openfoam_target: str = "",
) -> Tuple[str, List[str]]:
    """Stateless reviewer: returns (review_analysis, updated_history)."""
    advice_text = ""
    if isinstance(similar_case_advice, dict):
        advice_text = (
            f"<similar_case_advice>\n"
            f"match_level: {similar_case_advice.get('match_level')}\n"
            f"use_scope: {similar_case_advice.get('use_scope')}\n"
            f"advice: {similar_case_advice.get('advice')}\n"
            f"</similar_case_advice>\n"
        )
    elif similar_case_advice:
        advice_text = f"<similar_case_advice>{similar_case_advice}</similar_case_advice>\n"

    if history_text:
        reviewer_user_prompt = (
            f"<similar_case_reference>{tutorial_reference}</similar_case_reference>\n"
            f"{advice_text}"
            f"<foamfiles>{str(foamfiles)}</foamfiles>\n"
            f"<current_error_logs>{error_logs}</current_error_logs>\n"
            f"<history>\n{chr(10).join(history_text)}\n</history>\n\n"
            f"<user_requirement>{user_requirement}</user_requirement>\n\n"
            f"I have modified the files according to your previous suggestions. If the error persists, please provide further guidance. Make sure your suggestions adhere to user requirements and do not contradict it. Also, please consider the previous attempts and try a different approach."
        )
    else:
        reviewer_user_prompt = (
            f"<similar_case_reference>{tutorial_reference}</similar_case_reference>\n"
            f"{advice_text}"
            f"<foamfiles>{str(foamfiles)}</foamfiles>\n"
            f"<error_logs>{error_logs}</error_logs>\n"
            f"<user_requirement>{user_requirement}</user_requirement>\n"
            "Please review the error logs and provide guidance on how to resolve the reported errors. Make sure your suggestions adhere to user requirements and do not contradict it."
        )

    llm_client = llm_service if llm_service is not None else global_llm_service
    review_response = llm_client.invoke(
        reviewer_user_prompt,
        _reviewer_system_prompt(openfoam_target),
    )
    review_content = review_response

    updated_history = list(history_text) if history_text else []
    current_attempt = [
        f"<Attempt {len(updated_history)//4 + 1}>\n",
        f"<Error_Logs>\n{error_logs}\n</Error_Logs>",
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
        f"<foamfiles>{str(foamfiles)}</foamfiles>\n"
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


def error_fingerprint(state: dict[str, Any]) -> str:
    """Compare errors and case inputs, excluding changing runtime outputs."""
    files = snapshot_files(state.get("case_dir") or "")
    inputs = {
        name: digest for name, digest in files.items()
        if not _is_runtime_artifact(name)
    }
    payload = {
        "errors": state.get("error_logs") or [],
        "inputs": inputs,
        "user_requirement": state.get("user_requirement", ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _is_runtime_artifact(relative: str) -> bool:
    path = Path(relative)
    top = path.parts[0] if path.parts else ""
    try:
        float(top)
    except ValueError:
        pass
    else:
        return True
    return (
        top in {"postProcessing", "VTK", ".foamagent"}
        or top.startswith("processor") and top.removeprefix("processor").isdigit()
        or path.name.startswith("log")
        or path.name in {"Allrun.out", "Allrun.err"}
    )


def _reviewer_system_prompt(openfoam_target: str) -> str:
    """Add a native-version constraint without changing legacy review prompts."""
    if openfoam_target != "esi-v2006":
        return REVIEWER_SYSTEM_PROMPT
    return (
        REVIEWER_SYSTEM_PROMPT
        + " The target is native ESI/OpenCFD OpenFOAM v2006. Diagnose and repair "
        "only with ESI v2006 conventions from the supplied ESI tutorial reference; "
        "do not propose Foundation v10 syntax or post-generation translation."
    )
