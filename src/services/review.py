import hashlib
from logger import log_review
from openfoam_target import generation_convention
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Any, Literal
from pydantic import BaseModel, Field, ValidationError, model_validator
from models import PlannedFileChange, StageOperation
from .stage_plan import OPERATION_CONTRACT, MESH_STAGE_CONTRACT, mesh_operation_updates, replace_pending_tasks, resolve_file_tasks, script_commands, check_stage_operations, operation_script_errors
from .case_paths import safe_case_relative_from_text, safe_case_path
from .case_inputs import case_input_fingerprint, format_case_inputs, canonical_input_protections
from openfoam_target import generation_contract
from . import get_llm_service
from utils import case_file_key, resolve_case_file
from .execution_logs import declared_startup_inputs, check_execution_outputs, diagnostic_outputs
from .run_hpc import hpc_execution_outputs


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
        if self.operations is not None:
            errors = operation_script_errors([op.model_dump() for op in self.operations])
            if errors:
                raise ValueError("\n".join(errors))
        return self


class MeshReviewDecision(StageReviewDecision):
    action: Literal["continue", "stop"]


class SolverReviewDecision(StageReviewDecision):
    action: Literal["repair_solver", "repair_mesh", "stop"]


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
    "specified by the user. A native error saying an entry is undefined remains applicable "
    "while that entry is absent from the effective input. Its absence does not prove the "
    "error is stale or resolved. Check the actual input and producing source before stopping. "
    "References demonstrate compatible syntax, not replacement physics. "
    "Separate necessary repairs from questions requiring another run and optional diagnostics. "
    "Do not assert an uncertain diagnosis as fact, add monitoring configuration "
    "as a substitute for a repair, or change a resolved setting merely to try a different approach. "
    "If the evidence does not justify a file correction, say so. Do not modify files or ask questions."
)


def _review_context_excerpt(text, omission_notice):
    """Keep the existing context allowance without replacing full evidence."""
    if len(text) > 64_000:
        return omission_notice + "\n" + text[-64_000:]
    return text


def _applied_changes_context(last_repair):
    return _review_context_excerpt(
        json.dumps(last_repair or {}, ensure_ascii=False),
        "[Earlier edit details omitted; complete diffs remain in the journal.]",
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

    applied_changes = _applied_changes_context(last_repair)
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

    llm_client = get_llm_service(llm_service)
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
        f"<foamfiles>{format_case_inputs(foamfiles)}</foamfiles>\n"
        f"<error_logs>{error_logs}</error_logs>\n"
        f"<review_analysis>{review_analysis}</review_analysis>\n"
        f"<user_requirement>{user_requirement}</user_requirement>\n"
        "Return strict JSON now with key target_files only."
    )

    llm_client = get_llm_service(llm_service)
    response = llm_client.invoke(
        planner_user_prompt,
        planner_system_prompt,
        pydantic_obj=RewritePlan,
    )
    return response.model_dump()


def error_fingerprint(state: dict[str, Any]) -> Optional[str]:
    """Compare errors and case inputs, excluding changing runtime outputs."""
    case = Path(state.get("case_dir") or "").resolve()
    if ("last_run_input_fingerprint" in state and not state.get("repairing_mesh")
            and (state.get("solver_execution") or {}).get("executed") is not False):
        inputs = state["last_run_input_fingerprint"]
    else:
        if (not state.get("repairing_mesh")
                and (state.get("solver_execution") or {}).get("executed") is False
                and state.get("case_input_source_dir")):
            case = Path(state["case_input_source_dir"]).resolve()
        inputs = case_input_fingerprint(case, protected_inputs=canonical_input_protections(
            state.get('case_input_source_dir'), declared_startup_inputs([
            *state.get('mesh_operations', []), *state.get('solver_operations', []),
            *state.get('pending_solver_operations', []),
        ])))
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
        "mesh_operations": state.get("mesh_operations"),
        "solver_operations": state.get("solver_operations"),
        "user_requirement": state.get("user_requirement", ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


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
    repeated = bool(fingerprint and history and history[-1]["fingerprint"] == fingerprint)
    if repeated and stage == "solver":
        return {"workflow_status": "failed", "termination_reason": "repair_made_no_progress"}
    from .case_inputs import read_case_inputs
    from .review_history import ReviewHistory

    initial_review_exit = {}
    if (state.get("static_review_repair") or {}).get("stage") == stage:
        from .static_review import _end_initial_review
        initial_review_exit = _end_initial_review(state, stage, "execution_error")
        state = {**state, **initial_review_exit}
    journal = ReviewHistory(state["case_dir"], state.get("case_import_report_dir"))
    protected = canonical_input_protections(
        state.get('case_input_source_dir'), declared_startup_inputs(state[stage + '_operations']))
    foamfiles, omitted = read_case_inputs(state["case_dir"], protected_inputs=protected)
    source_context = None
    if state.get("case_input_source_dir"):
        sources, source_omitted = read_case_inputs(state["case_input_source_dir"], protected_inputs=protected)
        executed = {(f.folder_name, f.file_name): f.content for f in foamfiles.list_foamfile}
        source_context = {"files": {
            "/".join(p for p in (f.folder_name, f.file_name) if p): f.content
            for f in sources.list_foamfile
            if executed.get((f.folder_name, f.file_name)) != f.content
        }, "omitted_files": source_omitted}
    reference = state["mesh_reference"] if stage == "mesh" else {
        "content": state.get("tutorial_reference"), "script": state.get("allrun_reference")}
    stage_history = _review_context_excerpt(
        str(history), "[Earlier stage history omitted; complete records remain in workflow state.]",
    )
    user = (
        f"<stage>{stage}</stage>\n<user_requirement>{state['user_requirement']}</user_requirement>\n"
        f"<reference>{reference}</reference>\n<current_files>{format_case_inputs(foamfiles)}</current_files>\n"
        f"<omitted_files>{omitted}</omitted_files>\n<source_inputs>{source_context}</source_inputs>\n"
        f"<last_repair>{_applied_changes_context(state.get('last_repair'))}</last_repair>\n"
        f"<current_errors>{errors}</current_errors>\n<actual_mesh>{state.get('mesh_info')}</actual_mesh>\n"
        f"<mesh_execution>{state.get('mesh_execution')}</mesh_execution>\n"
        f"<boundary_plan>{state.get('mesh_boundary_plan')}</boundary_plan>\n"
        f"<operations>{state[stage + '_operations']}</operations>\n"
        f"<execution>{state.get(stage + '_execution')}</execution>\n"
        f"<pending_repairs>{state.get('rewrite_plan')}</pending_repairs>\n"
        f"<pending_physical_tasks>{state.get('pending_solver_subtasks')}</pending_physical_tasks>\n"
        f"<pending_solver_operations>{state['pending_solver_operations']}</pending_solver_operations>\n"
        f"<cross_stage_request>{state.get('cross_stage_request')}</cross_stage_request>\n"
        f"<stage_history>{stage_history}</stage_history>\n<repeated_failure>{repeated}</repeated_failure>"
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
                **initial_review_exit,
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
    journal.append(
        "runtime_review", stage=stage, runtime_loop=next_loop,
        error_logs=errors, review_analysis=payload, applied_changes=state.get("last_repair"),
        omitted_files=omitted, source_inputs=source_context,
    )
    result = {**initial_review_exit, **apply_stage_decision(state, stage, decision)}
    result.update({stage + "_review_history": history, "loop_count": next_loop,
                   "repair_source": "runtime"})
    if decision.action == "stop":
        result.update(workflow_status="failed", termination_reason=stage + "_review_stopped")
    elif next_loop >= state["config"].max_loop:
        result.update(workflow_status="failed", termination_reason="max_review_loop_reached")
    return result


def apply_stage_decision(state, stage, decision):
    """Apply the existing stage task/operation contract to a validated decision."""
    payload = decision.model_dump()
    targets = replace_pending_tasks((state.get("rewrite_plan") or {}).get("target_files", []), payload["target_files"])
    creates, targets = resolve_file_tasks(
        [*(state.get('mesh_subtasks') or []), *(state.get('pending_solver_subtasks') or []),
         *(state.get('subtasks') or [])], targets)
    solver_creates = [task for task in creates if task['stage'] == 'solver']
    result = {
        "review_analysis": decision.diagnosis + "\n" + decision.reason,
        "rewrite_plan": {"target_files": targets},
        "mesh_subtasks": [task for task in creates if task['stage'] == 'mesh'],
        "subtasks": solver_creates, "pending_solver_subtasks": solver_creates,
        "input_writer_mode": 'initial' if solver_creates else 'rewrite',
        "requires_input_writer": True, "repairing_mesh": stage == "mesh" or decision.action == "repair_mesh",
        "error_logs": state.get("error_logs") or [],
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
                outputs = (hpc_execution_outputs('solver') if state.get('requires_hpc') else ['Allrun.out', 'Allrun.err']) + diagnostic_outputs(state['case_dir'])
                previous_conflicts = check_execution_outputs(outputs, declared_startup_inputs(state['solver_operations']))
                if ((any(not path.is_file() for path in previous_inputs) or previous_conflicts) and previous_inputs !=
                        _solver_startup_paths(state['case_dir'], payload['operations'])):
                    result['rewrite_plan']['dependencies_changed'] = True
    if decision.operation_request:
        result["cross_stage_request"] = {"source": stage, "request": decision.operation_request,
                                         "evidence": decision.evidence, "round": state.get("static_review_retries", 0) if state.get("repair_source") == "static_review" else state["loop_count"] + 1}
    return result



def _reviewer_system_prompt(openfoam_target: str) -> str:
    """Use the same target contract as generation, rewriting and static review."""
    return REVIEWER_SYSTEM_PROMPT + "\n" + generation_contract(openfoam_target)
