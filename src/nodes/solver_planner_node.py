"""Plan physical and solver files after Meshing has prepared the actual mesh."""

from pathlib import Path

from services.plan import decompose_to_subtasks, plan_imported_case, prepare_solver_context
from openfoam_target import generation_convention
from router_func import llm_requires_hpc, llm_requires_visualization


def solver_planner_node(state):
    print("<solver_planner>")
    pending = list((state.get("rewrite_plan") or {}).get("target_files", []))
    deferred = list(state.get("pending_solver_subtasks") or [])
    planning_requirement = (
        state["user_requirement"]
        + f"\n<pending_file_tasks>{deferred}</pending_file_tasks>"
        + f"\n<pending_repairs>{pending}</pending_repairs>"
        + f"\n<review_analysis>{state.get('review_analysis') or ''}</review_analysis>"
    )
    preparation = prepare_solver_context({**state, "user_requirement": planning_requirement})
    dir_counts_str = preparation.pop("dir_counts_str")
    state = {**state, **preparation}
    if state.get("case_origin") == "imported":
        result = plan_imported_case({**state, "user_requirement": planning_requirement})
        if deferred and result.get("workflow_status") == "running":
            changes = list((result.get("rewrite_plan") or {}).get("target_files", []))
            changes.extend({"file": str(Path(t["folder_name"]) / t["file_name"]),
                            "stage": t["stage"],
                            "changes": t["changes"]} for t in deferred)
            result["rewrite_plan"] = {"target_files": changes}
    else:
        subtasks = decompose_to_subtasks(
            planning_requirement + f"\n<actual_mesh>{state.get('mesh_info') or {}}</actual_mesh>\n"
            "The mesh is already prepared. Plan physical fields, physical models, solver "
            "settings and execution scripts using its actual patch names and types. "
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
        if not subtasks:
            raise ValueError("Failed to generate subtasks.")
        creates, changes = [], []
        # Keep deferred tasks, allowing the new plan to replace a task for the same path.
        tasks = {str(Path(t["folder_name"]) / t["file_name"]): t for t in deferred}
        tasks.update({str(Path(t["folder_name"]) / t["file_name"]): t for t in subtasks})
        for path, task in tasks.items():
            if task["stage"] == "mesh" or task["action"] == "modify":
                changes.append({"file": path, "stage": task["stage"],
                                "changes": task["changes"]})
            else:
                creates.append(task)
        result = {
            "subtasks": creates, "workflow_status": "running",
            "input_writer_mode": "initial" if creates else "rewrite",
            "requires_input_writer": True, "rewrite_plan": {"target_files": changes},
            "review_analysis": "Apply the solver planner's explicit file changes.",
        }
    if result.get("workflow_status") == "running":
        changes = list((result.get("rewrite_plan") or {}).get("target_files", []))
        # Existing physical repair instructions remain available after replanning.
        changes = pending + changes
        needs_mesh = bool(result.get("requires_meshing")) or any(t["stage"] == "mesh" for t in changes)
        result.update(
            rewrite_plan={"target_files": changes} if changes else None,
            repairing_mesh=needs_mesh, requires_meshing=needs_mesh,
            replan_solver=needs_mesh,
            pending_solver_subtasks=result.get("subtasks", []) if needs_mesh else [],
        )
        if changes:
            result["requires_input_writer"] = True
    if result.get("workflow_status") == "running":
        context = {**state, **result}
        if context.get("requires_hpc") is None:
            result["requires_hpc"] = llm_requires_hpc(context)
        if context.get("requires_visualization") is None:
            result["requires_visualization"] = llm_requires_visualization(context)
        result["solver_planned"] = True
    print("</solver_planner>")
    return {**preparation, **result}
