import hashlib
from logger import log_review
from openfoam_target import generation_convention
import json
from pathlib import Path
from .case_import import snapshot_files
from typing import List, Optional, Tuple, Any, Literal
from pydantic import BaseModel, Field
from models import PlannedFileChange
from . import global_llm_service


class RewritePlan(BaseModel):
    target_files: List[PlannedFileChange] = Field(description="Files to modify and required changes")


class WorkflowRepairPlan(RewritePlan):
    repair_domain: Literal["mesh", "solver"]


class MeshRepairPlan(WorkflowRepairPlan):
    action: Literal["continue", "stop"]


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
    stage: str = "",
) -> dict:
    """Generate a minimal, explicit rewrite plan for downstream rewrite step."""
    planner_system_prompt = (
        "You are an OpenFOAM debugging planner. "
        "Given current foam files, error logs and reviewer analysis, create a minimal rewrite plan. "
        "Output MUST be strict JSON only, with this exact schema: "
        "{\"target_files\": [{\"file\": \"relative/path\", \"changes\": \"change1; change2\", \"stage\": \"solver\"}]}. "
        "Rules: "
        "1) Do not use markdown, backticks, or comments. "
        "2) Use double quotes for all strings. "
        "3) In changes, use short plain text actions separated by semicolons. "
        "4) Do not include parentheses, backticks, or quote characters inside changes text. "
        "5) Do not include run steps; only file edits. "
        "6) Set stage to mesh or solver; solver includes execution scripts."
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

    if stage:
        planner_system_prompt = (
            "Convert the diagnosis into the supplied structured repair schema. Declare each "
            "file's stage (mesh or solver, including execution scripts in solver), and concrete changes from its use, "
            "not its name. Include all required stages without dropping pending physical repairs. "
            "Set repair_domain to mesh when mesh operations are needed, even with no file edits. "
            f"Preserve explicit user requirements and target {openfoam_target}."
        )
        if stage == "mesh":
            planner_system_prompt += " Set action to stop if no justified next repair remains, otherwise continue."
        planner_user_prompt = planner_user_prompt.replace(
            "Return strict JSON now with key target_files only.",
            f"Current execution stage: {stage}. Return the full repair schema.",
        )

    llm_client = llm_service if llm_service is not None else global_llm_service
    response = llm_client.invoke(
        planner_user_prompt,
        planner_system_prompt,
        pydantic_obj=MeshRepairPlan if stage == "mesh" else WorkflowRepairPlan if stage else RewritePlan,
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
        or path.name in {"Allrun.out", "Allrun.err", "Allmesh.out", "Allmesh.err"}
    )


def review_case(state):
    """
    Reviewer node: Reviews the error logs and provides analysis and suggestions
    for fixing the errors. This node only focuses on analysis, not file modification.
    """
    mesh = bool(state.get("repairing_mesh"))
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
    if repeated and not mesh:
        print("Repair made no progress: errors and case inputs are unchanged.")
        print("</reviewer>")
        return {
            "error_fingerprints": fingerprints,
            "workflow_status": "failed",
            "termination_reason": "repair_made_no_progress",
        }

    if mesh and state.get("loop_count", 0) >= state["config"].max_loop:
        return {"workflow_status": "failed", "termination_reason": "max_review_loop_reached"}

    # Stateless review via service
    history_text = state.get("history_text") or []
    requirement = (
        state.get("user_requirement", "")
        + f"\n<actual_mesh>{state.get('mesh_info') or {}}</actual_mesh>\n"
        + f"<repeated_failure>{repeated}</repeated_failure>\n"
        + f"<existing_physics>{state.get('case_info')}</existing_physics>\n"
        + f"<pending_repairs>{state.get('rewrite_plan')}</pending_repairs>\n"
        + f"<execution>{state.get('mesh_execution')}</execution>\n"
        + f"Current execution stage: {'mesh' if mesh else 'solver'}. "
        "Explain which operations and files need repair, their stage, and whether "
        "the changes affect the existing physical plan. Use actual file contents "
        "and native errors, not filenames, to determine the stage. Mesh operations "
        "run before pending physical repairs. Decide whether there is a justified "
        "next repair or the workflow should stop; consider repeated failures."
    )
    review_content, updated_history = review_error_logs(
        tutorial_reference=state.get('tutorial_reference') or '',
        foamfiles=state.get('foamfiles'),
        error_logs=state.get('error_logs'),
        user_requirement=requirement,
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
        stage="mesh" if mesh else "solver",
    )
    log_review(str(rewrite_plan), "rewrite_plan")

    print("</reviewer>")

    next_loop_count = state.get("loop_count", 0) + 1
    result = {
        "error_fingerprints": fingerprints,
        "history_text": updated_history,
        "review_analysis": review_content,
        "rewrite_plan": {"target_files": rewrite_plan["target_files"]},
        "loop_count": next_loop_count,
        "input_writer_mode": "rewrite",
        "requires_input_writer": True,
        "repairing_mesh": mesh or rewrite_plan["repair_domain"] == "mesh"
        or any(t["stage"] == "mesh" for t in rewrite_plan["target_files"]),
    }
    if mesh and rewrite_plan["action"] == "stop":
        result.update(workflow_status="failed", termination_reason="mesh_review_stopped")
    if next_loop_count >= state["config"].max_loop:
        result["termination_reason"] = "max_review_loop_reached"
        result["workflow_status"] = "failed"
    return result


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
