import hashlib
from logger import log_review
from openfoam_target import generation_convention
import json
import re
from pathlib import Path
from .case_import import snapshot_files
from typing import List, Optional, Tuple, Any, Literal
from pydantic import BaseModel, Field, ValidationError, model_validator
from models import PlannedFileChange, StageOperation
from .stage_plan import OPERATION_CONTRACT, MESH_STAGE_CONTRACT, mesh_operation_updates, replace_pending_tasks, resolve_file_tasks, script_commands, check_stage_operations
from .case_paths import safe_case_relative_from_text, safe_case_path
from . import global_llm_service
from utils import case_file_key, resolve_case_file


class RewritePlan(BaseModel):
    target_files: List[PlannedFileChange] = Field(description="Files to modify and required changes")


class StageReviewDecision(RewritePlan):
    diagnosis: str
    evidence: List[str]
    reason: str
    operations: Optional[List[StageOperation]] = Field(description="Complete replacement of this stage's operations, or null to retain; Mesh Reviewer may also declare deferred solver operations")
    operation_request: Optional[str] = Field(description="Concrete mesh-planning or cross-stage request, or null")

    @model_validator(mode="after")
    def validate_decision(self):
        action = self.action
        if action == "stop" and (self.target_files or self.operations is not None or self.operation_request):
            raise ValueError("A stop decision cannot contain executable repair tasks")
        if action != "stop" and not (self.target_files or self.operations or self.operation_request):
            raise ValueError("A continuing decision requires a concrete repair")
        if action == "repair_mesh" and not self.operation_request:
            raise ValueError("A mesh handoff requires a concrete operation_request")
        if action == "repair_solver" and any(t.stage == "mesh" for t in self.target_files):
            raise ValueError("Mesh tasks require repair_mesh")
        if action == "repair_solver" and self.operation_request:
            raise ValueError("Solver operations must be supplied as operations, not an unexecuted request")
        keys = [(case_file_key(safe_case_relative_from_text(t.file)), t.stage) for t in self.target_files]
        if len(keys) != len(set(keys)):
            raise ValueError("Merge conflicting instructions into one task per stage/file")
        if (action != "stop" and any(path == "Allrun" for path, _ in keys)
                and self.operations is None and isinstance(self, SolverReviewDecision)):
            raise ValueError("Editing Allrun requires its updated operation plan")
        if isinstance(self, SolverReviewDecision) and self.operations is not None:
            if any(op.stage != "solver" for op in self.operations):
                raise ValueError("Solver operations must belong to solver; use operation_request for mesh work")
        return self


class MeshReviewDecision(StageReviewDecision):
    action: Literal["continue", "stop"]


class SolverReviewDecision(StageReviewDecision):
    action: Literal["repair_solver", "repair_mesh", "stop"]


REVIEWER_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM simulation and numerical modeling. "
    "Your task is to review the provided error logs and diagnose the underlying issues. "
    "Diagnose the latest current error logs and actual current file contents first. "
    "History records previous attempts, including resolved errors; do not repeat an old "
    "repair when its missing entry now exists or the current failure has changed. "
    "For target-matched tutorial files, retain applicable dictionary syntax unless the "
    "current native parser error demonstrates a correction is needed. Do not assert a "
    "different release syntax based on memory when it contradicts that evidence. "
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
    if similar_case_advice:
        advice = similar_case_advice.model_dump() if hasattr(similar_case_advice, "model_dump") else similar_case_advice
        advice_text = f"<similar_case_advice>{advice}</similar_case_advice>\n"

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
    attempt_num = sum(item.startswith("<Attempt ") for item in updated_history) + 1
    current_attempt = [
        f"<Attempt {attempt_num}>\n",
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

    llm_client = llm_service if llm_service is not None else global_llm_service
    response = llm_client.invoke(
        planner_user_prompt,
        planner_system_prompt,
        pydantic_obj=RewritePlan,
    )
    return response.model_dump()


def error_fingerprint(state: dict[str, Any]) -> str:
    """Compare errors and case inputs, excluding changing runtime outputs."""
    case = Path(state.get("case_dir") or "").resolve()
    files = snapshot_files(case, exclude_runtime=True)
    inputs = {
        name: digest for name, digest in files.items()
        if not _is_runtime_artifact(name)
    }
    if "submit_job.slurm" in inputs:
        # Keep real submission-script edits visible, but an otherwise identical
        # script gets a fresh wrapper directory on every stage attempt.
        case = Path(state["case_dir"]).resolve()
        try:
            script = (case / "submit_job.slurm").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            pass
        else:
            for stage in ("mesh", "solver"):
                jobs_root = case / ".foamagent" / (stage + "-jobs")
                for job in state.get(stage + "_jobs") or []:
                    if not job.get("script"):
                        continue
                    attempt = Path(job["script"]).parent
                    if attempt.parent == jobs_root:
                        script = script.replace(
                            attempt.relative_to(case).as_posix(),
                            f".foamagent/{stage}-jobs/<attempt>",
                        )
            inputs["submit_job.slurm"] = hashlib.sha256(script.encode("utf-8")).hexdigest()
    payload = {
        "errors": state.get("error_logs") or [],
        "inputs": inputs,
        "mesh_operations": state.get("mesh_operations"),
        "solver_operations": state.get("solver_operations"),
        "user_requirement": state.get("user_requirement", ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _is_runtime_artifact(relative: str) -> bool:
    path = Path(relative)
    top = path.parts[0] if path.parts else ""
    try:
        written_time = float(top) != 0
    except ValueError:
        pass
    else:
        if written_time:
            return True
    return (
        ".foamagent" in path.parts
        or top in {"postProcessing", "VTK", "dynamicCode"}
        or re.fullmatch(r"processor(?:\d+|s\d+(?:_\d+-\d+)?)", top) is not None
        or path.name.startswith("log")
        or path.name in {"Allrun.out", "Allrun.err", "Allmesh.out", "Allmesh.err"}
    )


def _solver_startup_paths(case_dir, operations):
    """Compare the same resolved file paths checked by solver preflight."""
    paths = set()
    for operation in operations:
        if operation['stage'] != 'solver':
            continue
        for required in operation['required_inputs']:
            relative = safe_case_relative_from_text(required['path'])
            paths.add(resolve_case_file(safe_case_path(case_dir, str(relative.parent), relative.name)))
    return paths


def review_mesh(state):
    return _review_stage(state, "mesh", MeshReviewDecision)


def review_solver(state):
    return _review_stage(state, "solver", SolverReviewDecision)


def _parse_review_response(raw, schema):
    """Accept a single outer JSON fence without changing the decision itself."""
    if isinstance(raw, str):
        raw = raw.strip().removeprefix("\ufeff").strip()
        fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", raw,
                             flags=re.DOTALL | re.IGNORECASE)
        if fence:
            raw = fence[1]
    return schema.model_validate_json(raw)


def _review_stage(state, stage, schema):
    """One schema-constrained response supplies diagnosis and executable decision."""
    errors = state["error_logs"]
    if not errors:
        return {}
    if state["loop_count"] >= state["config"].max_loop:
        return {"workflow_status": "failed", "termination_reason": "max_review_loop_reached"}
    fingerprint = error_fingerprint(state)
    history = list(state[stage + "_review_history"])
    repeated = bool(history and history[-1]["fingerprint"] == fingerprint)
    if repeated and stage == "solver":
        return {"workflow_status": "failed", "termination_reason": "repair_made_no_progress"}
    reference = state["mesh_reference"] if stage == "mesh" else {
        "content": state.get("tutorial_reference"), "script": state.get("allrun_reference")}
    user = (
        f"<stage>{stage}</stage>\n<user_requirement>{state['user_requirement']}</user_requirement>\n"
        f"<reference>{reference}</reference>\n<current_files>{state.get('foamfiles')}</current_files>\n"
        f"<current_errors>{errors}</current_errors>\n<actual_mesh>{state.get('mesh_info')}</actual_mesh>\n"
        f"<mesh_execution>{state.get('mesh_execution')}</mesh_execution>\n"
        f"<boundary_plan>{state.get('mesh_boundary_plan')}</boundary_plan>\n"
        f"<operations>{state[stage + '_operations']}</operations>\n"
        f"<execution>{state.get(stage + '_execution')}</execution>\n"
        f"<pending_repairs>{state.get('rewrite_plan')}</pending_repairs>\n"
        f"<pending_physical_tasks>{state.get('pending_solver_subtasks')}</pending_physical_tasks>\n"
        f"<pending_solver_operations>{state['pending_solver_operations']}</pending_solver_operations>\n"
        f"<cross_stage_request>{state.get('cross_stage_request')}</cross_stage_request>\n"
        f"<stage_history>{history}</stage_history>\n<repeated_failure>{repeated}</repeated_failure>"
    )
    system = _reviewer_system_prompt(generation_convention(state["config"])) + OPERATION_CONTRACT + (
        MESH_STAGE_CONTRACT + " You are the Mesh Reviewer. Diagnose mesh tools, backend execution and handoff. "
        "Return continue with concrete file changes or operation_request for Meshing Planner, "
        "or stop only when no justified repair remains. Preserve pending physical work."
        if stage == "mesh" else
        " You are the Solver Reviewer. Diagnose initialization, physical inputs, Allrun and solver "
        "completion. Choose repair_solver for solver file/operation changes; choose repair_mesh "
        "when actual mesh changes are required and describe them in operation_request. Preserve "
        "pending physical work. Do not directly edit mesh through the solver writer."
    ) + (
        " Return exactly one JSON object matching this schema, without code fences or prose. "
        "The diagnosis, evidence, decision and tasks must agree. Operations replace this review "
        "stage's plan; null preserves it. A mesh request from Solver Reviewer is planned by "
        "Meshing Planner. File writes alone do not prove an error resolved. "
        "Merge pending and new instructions for the same file into one coherent task. Schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    # Keep every raw response. One correction request is bounded within this
    # review round, and only a fully validated decision can supply repair tasks.
    log_review(json.dumps(errors, ensure_ascii=False), stage + "_review_errors")
    next_loop = state["loop_count"] + 1
    review_user = user
    for attempt in range(2):
        raw = state["llm_service"].invoke(review_user, system)
        log_review(raw, stage + "_review_raw")
        try:
            decision = _parse_review_response(raw, schema)
            break
        except ValidationError as error:
            details = json.dumps(error.errors(include_input=False, include_url=False),
                                 ensure_ascii=False, default=str)
            log_review(details, stage + "_review_validation_error")
            if attempt == 0:
                review_user = (
                    user + f"\n<invalid_review_response>{raw}</invalid_review_response>\n"
                    f"<validation_errors>{details}</validation_errors>\n"
                    "Correct the response to satisfy the supplied schema and decision constraints. "
                    "Use the same case evidence; return one complete JSON object, without prose "
                    "or code fences. Do not omit required fields or invent fallback repair tasks."
                )
                continue
            history.append({"round": next_loop, "fingerprint": fingerprint,
                            "errors": errors, "decision": None,
                            "response_error": details, "outcome": None})
            return {
                stage + "_review_history": history, "loop_count": next_loop,
                "workflow_status": "failed", "termination_reason": "invalid_review_response",
                "error_logs": [*errors, {"file": stage + "_review", "error_content":
                    "Reviewer returned an invalid decision after one correction request: " + details}],
            }
    payload = decision.model_dump()
    log_review(decision.diagnosis, "review_analysis")
    log_review(str(payload), "rewrite_plan")
    history.append({"round": next_loop, "fingerprint": fingerprint,
                    "errors": errors, "decision": payload, "outcome": None})
    targets = replace_pending_tasks((state.get("rewrite_plan") or {}).get("target_files", []), payload["target_files"])
    creates, targets = resolve_file_tasks(
        [*(state.get('mesh_subtasks') or []), *(state.get('pending_solver_subtasks') or []),
         *(state.get('subtasks') or [])], targets)
    solver_creates = [task for task in creates if task['stage'] == 'solver']
    result = {
        stage + "_review_history": history, "loop_count": next_loop,
        "review_analysis": decision.diagnosis + "\n" + decision.reason,
        "rewrite_plan": {"target_files": targets},
        "mesh_subtasks": [task for task in creates if task['stage'] == 'mesh'],
        "subtasks": solver_creates, "pending_solver_subtasks": solver_creates,
        "input_writer_mode": 'initial' if solver_creates else 'rewrite',
        "requires_input_writer": True, "repairing_mesh": stage == "mesh" or decision.action == "repair_mesh",
        "error_logs": errors,
    }
    if decision.operations is not None:
        if stage == "mesh":
            result.update(mesh_operation_updates(state, payload["operations"]))
        else:
            result["solver_operations"] = payload["operations"]
            result["pending_solver_operations"] = payload["operations"]
        if stage == "solver":
            result['solver_script_dirty'] = (state['solver_script_dirty'] or
                script_commands(payload['operations']) != script_commands(state['solver_operations']))
            if (state.get('solver_execution') or {}).get('executed') is False:
                if (check_stage_operations(state['solver_operations'], 'solver')
                        and not check_stage_operations(payload['operations'], 'solver')):
                    result['rewrite_plan']['execution_plan_repaired'] = True
                previous_inputs = _solver_startup_paths(state['case_dir'], state['solver_operations'])
                if (any(not path.is_file() for path in previous_inputs) and previous_inputs !=
                        _solver_startup_paths(state['case_dir'], payload['operations'])):
                    result['rewrite_plan']['dependencies_changed'] = True
    if decision.operation_request:
        result["cross_stage_request"] = {"source": stage, "request": decision.operation_request,
                                         "evidence": decision.evidence, "round": next_loop}
    if decision.action == "stop":
        result.update(workflow_status="failed", termination_reason=stage + "_review_stopped")
    elif next_loop >= state["config"].max_loop:
        result.update(workflow_status="failed", termination_reason="max_review_loop_reached")
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
