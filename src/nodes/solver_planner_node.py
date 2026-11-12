"""Plan physical and solver files after Meshing has prepared the actual mesh."""
from logger import print

from pathlib import Path

from services.plan import _build_advice, decompose_to_subtasks, plan_imported_case, prepare_solver_context
from openfoam_target import generation_convention
from services.stage_plan import replace_pending_tasks, resolve_file_tasks, script_commands
from router_func import llm_requires_hpc, llm_requires_visualization


def solver_planner_node(state):
    print("<solver_planner>")
    pending = list((state.get("rewrite_plan") or {}).get("target_files", []))
    deferred = list(state.get("pending_solver_subtasks") or [])
    planning_requirement = (
        state["user_requirement"]
        + f"\n<pending_file_tasks>{deferred}</pending_file_tasks>"
        + f"\n<deferred_physical_operations>{state['pending_solver_operations']}</deferred_physical_operations>"
        + "\nReconcile these deferred physical operations with the user requirement, actual mesh "
          "and solver reference into the final ordered solver plan. Include applicable physical "
          "initialization and solver setup; do not execute them as mesh work. All returned "
          "operations belong to solver; request additional mesh work through mesh file tasks."
        + f"\n<pending_repairs>{pending}</pending_repairs>"
        + f"\n<review_analysis>{state.get('review_analysis') or ''}</review_analysis>"
        + f"\n<cross_stage_request>{state.get('cross_stage_request')}</cross_stage_request>"
        + f"\n<mesh_boundary_plan>{state.get('mesh_boundary_plan') or []}</mesh_boundary_plan>"
        + f"\n<mesh_reference_source>{state['mesh_reference']['case']}</mesh_reference_source>"
    )
    preparation = prepare_solver_context({**state, "user_requirement": planning_requirement})
    dir_counts_str = preparation.pop("dir_counts_str")
    state = {**state, **preparation}
    if state.get("case_origin") == "imported":
        result = plan_imported_case({**state, "user_requirement": planning_requirement})
        if result.get("workflow_status") == "running":
            changes = replace_pending_tasks(pending, (result.get("rewrite_plan") or {}).get("target_files", []))
            creates, changes = resolve_file_tasks(
                [*(state.get("mesh_subtasks") or []), *deferred], changes)
    else:
        plan = decompose_to_subtasks(
            planning_requirement + f"\n<actual_mesh>{state.get('mesh_info') or {}}</actual_mesh>\n"
            f"<mesh_execution>{state.get('mesh_execution')}</mesh_execution>\n"
            f"<current_directory>{state['dir_structure']}</current_directory>\n"
            f"<current_files>{state['foamfiles']}</current_files>\n"
            f"<omitted_files>{state.get('input_omissions') or {}}</omitted_files>\n"
            f"<solver_reference>{state['tutorial_reference']}</solver_reference>\n"
            f"<solver_script_reference>{state['allrun_reference']}</solver_script_reference>\n"
            "Check that the actual mesh supports the requested physics and intended boundary "
            "roles. Actual patch_bounds are measured from the generated mesh; use them "
            "rather than assuming the planned patch positions were achieved. "
            "Distinguish user requirements from tutorial and inferred assumptions. "
            "If incompatible, explicitly plan mesh-stage changes rather than adapting the "
            "physics to an unsuitable mesh. The mesh is already prepared. Plan physical fields, physical models, solver "
            "settings and execution scripts using its actual patch names and types. "
            "State the physical boundary conditions in the relevant file tasks, using "
            "the request and reference contents as evidence rather than patch names alone. "
            "Explain changes needed when the reference physics is incompatible. "
            "Adapt existing mesh-tool startup dictionaries with explicit modify tasks "
            "where needed. Mesh dictionaries, geometry, conversion and mesh scripts are "
            "planned separately by meshing_planner. If the physical problem needs further "
            "mesh changes, list those tasks with stage mesh instead of omitting them. "
            "Label other tasks solver and specify create or modify with concrete changes.",
            state["dir_structure_reference"],
            dir_counts_str,
            llm_service=state.get("llm_service"),
            openfoam_target=generation_convention(state["config"]),
        )
        subtasks = [task.model_dump() for task in plan.subtasks]
        creates, changes = resolve_file_tasks(
            [*(state.get("mesh_subtasks") or []), *deferred], pending, subtasks)
        # Replanning may retain physical files and only reconcile execution.
        if not creates and not changes and (
            not state.get("solver_planned") or not plan.operations
        ):
            raise ValueError("Failed to generate subtasks.")
        tasks = [*creates, *[
            dict(folder_name='' if Path(t['file']).parent == Path('.') else str(Path(t['file']).parent),
                 file_name=Path(t['file']).name, stage=t['stage'], action='modify', changes=t['changes'])
            for t in changes]]
        advice = _build_advice(
            planning_requirement,
            f"{state['case_info']}; OpenFOAM target: {generation_convention(state['config'])}",
            state["tutorial_reference"],
            [task for task in tasks if task["stage"] == "solver"],
            state.get("llm_service"),
        )
        result = {
            "similar_case_advice": advice.model_dump(),
            "solver_operations": [operation.model_dump() for operation in plan.operations],
            "workflow_status": "running",
            "requires_input_writer": True, "rewrite_plan": {"target_files": changes},
            "review_analysis": "Apply the solver planner's explicit file changes.",
        }
    if result.get("workflow_status") == "running":
        solver_creates = [t for t in creates if t['stage'] == 'solver']
        mesh_creates = [t for t in creates if t['stage'] == 'mesh']
        needs_mesh = bool(result.get("requires_meshing")) or bool(mesh_creates) or any(t["stage"] == "mesh" for t in changes)
        result.update(
            rewrite_plan={"target_files": changes} if changes else None,
            subtasks=solver_creates, mesh_subtasks=mesh_creates,
            input_writer_mode='initial' if solver_creates else 'rewrite',
            repairing_mesh=needs_mesh, requires_meshing=needs_mesh,
            replan_solver=needs_mesh,
            pending_solver_subtasks=solver_creates if needs_mesh else [],
        )
        if changes or creates:
            result["requires_input_writer"] = True
        if needs_mesh:
            # Planner-requested mesh corrections consume the same budget as reviews.
            result["loop_count"] = state["loop_count"] + 1
            if result["loop_count"] >= state["config"].max_loop:
                result.update(workflow_status="failed", termination_reason="max_review_loop_reached")
    if result.get("workflow_status") == "running":
        context = {**state, **result}
        if context.get("requires_hpc") is None:
            result["requires_hpc"] = llm_requires_hpc(context)
        if context.get("requires_visualization") is None:
            result["requires_visualization"] = llm_requires_visualization(context)
        result["solver_planned"] = True
        # This plan has not reached the solver writer yet. Mesh detours must
        # retain its current instructions for the next solver planning pass.
        result["pending_solver_operations"] = result["solver_operations"]
        result['solver_script_dirty'] = (state['solver_script_dirty'] or
                                         script_commands(result['solver_operations']) != script_commands(state['solver_operations']))
    print("</solver_planner>")
    return {**preparation, **result}
