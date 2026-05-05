from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from . import global_llm_service


class PlannedFileChange(BaseModel):
    file: str = Field(description="Relative file path, e.g. system/fvSchemes or 0/U")
    changes: str = Field(description="Semicolon-separated concrete changes for this file")


class RewritePlan(BaseModel):
    target_files: List[PlannedFileChange] = Field(description="Files to modify and required changes")


REVIEWER_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM simulation and numerical modeling. "
    "Your task is to review the provided error logs and diagnose the underlying issues. "
    "You will be provided with a similar case reference, which is a list of similar cases that are ordered by similarity. You can use this reference to help you understand the user requirement and the error. "
    "When an error indicates that a specific keyword is undefined (for example, 'div(phi,(p|rho)) is undefined'), your response must propose a solution that simply defines that exact keyword as shown in the error log. "
    "Do not reinterpret or modify the keyword (e.g., do not treat '|' as 'or'); instead, assume it is meant to be taken literally. "
    "Propose ideas on how to resolve the errors, but do not modify any files directly. "
    "Please do not propose solutions that require modifying any parameters declared in the user requirement, try other approaches instead. Do not ask the user any questions. "
    "The user will supply all relevant foam files along with the error logs, and within the logs, you will find both the error content and the corresponding error command indicated by the log file name."
)


def review_error_logs(
    tutorial_reference: str,
    foamfiles: Any,
    error_logs: List[str],
    user_requirement: str,
    similar_case_advice: Optional[Any] = None,
    history_text: Optional[List[str]] = None,
    loop_count: int = 0,
    pending_rewrite_plan: Optional[str] = None,
    pending_file_diff: Optional[str] = None,
    oscillation_hint: str = "",
) -> Tuple[str, List[str]]:
    """Stateless reviewer: returns (review_analysis, updated_history)."""
    advice_text = ""

    # Fix 1: patch the previous attempt with rewrite_plan + file_diff + run_outcome
    updated_history = list(history_text) if history_text else []
    if pending_rewrite_plan is not None and updated_history and updated_history[-1] == "</Attempt>\n":
        run_outcome = (
            f"Same error signature persists ({len(error_logs)} error(s))."
            if error_logs else "Solver completed without errors after rewrite."
        )
        updated_history.pop()
        updated_history.append(f"<Rewrite_Plan>\n{pending_rewrite_plan}\n</Rewrite_Plan>\n")
        updated_history.append(f"<File_Diff>\n{pending_file_diff or '(not captured)'}\n</File_Diff>\n")
        updated_history.append(f"<Run_Outcome>\n{run_outcome}\n</Run_Outcome>\n")
        updated_history.append("</Attempt>\n")

    if updated_history:
        reviewer_user_prompt = (
            f"<similar_case_reference>{tutorial_reference}</similar_case_reference>\n"
            f"{advice_text}"
            f"<foamfiles>{str(foamfiles)}</foamfiles>\n"
            f"<current_error_logs>{error_logs}</current_error_logs>\n"
            f"<history>\n{chr(10).join(updated_history)}\n</history>\n\n"
            f"<user_requirement>{user_requirement}</user_requirement>\n\n"
            f"I have modified the files according to your previous suggestions. If the error persists, please provide further guidance. Make sure your suggestions adhere to user requirements and do not contradict them. Also, please consider the previous attempts and try a different approach."
            f"{oscillation_hint}"
        )
    else:
        reviewer_user_prompt = (
            f"<similar_case_reference>{tutorial_reference}</similar_case_reference>\n"
            f"{advice_text}"
            f"<foamfiles>{str(foamfiles)}</foamfiles>\n"
            f"<error_logs>{error_logs}</error_logs>\n"
            f"<user_requirement>{user_requirement}</user_requirement>\n"
            "Please review the error logs and provide guidance on how to resolve the reported errors. Make sure your suggestions adhere to user requirements and do not contradict them."
            f"{oscillation_hint}"
        )

    review_response = global_llm_service.invoke(reviewer_user_prompt, REVIEWER_SYSTEM_PROMPT)
    review_content = review_response

    attempt_num = sum(1 for item in updated_history if item.startswith("<Attempt ")) + 1
    current_attempt = [
        f"<Attempt {attempt_num}>\n",
        f"<Error_Logs>\n{error_logs}\n</Error_Logs>",
        f"<Review_Analysis>\n{review_content}\n</Review_Analysis>",
        f"</Attempt>\n",
    ]
    updated_history.extend(current_attempt)
    return review_content, updated_history


def generate_rewrite_plan(
    foamfiles: Any,
    error_logs: List[str],
    review_analysis: str,
    user_requirement: str,
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

    planner_user_prompt = (
        f"<foamfiles>{str(foamfiles)}</foamfiles>\n"
        f"<error_logs>{error_logs}</error_logs>\n"
        f"<review_analysis>{review_analysis}</review_analysis>\n"
        f"<user_requirement>{user_requirement}</user_requirement>\n"
        "Return strict JSON now with key target_files only."
    )

    response = global_llm_service.invoke(
        planner_user_prompt,
        planner_system_prompt,
        pydantic_obj=RewritePlan,
    )
    return response.model_dump()

