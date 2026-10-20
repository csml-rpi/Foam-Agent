"""Plan physical and solver files after Meshing has prepared the actual mesh."""

from services.plan import decompose_to_subtasks, plan_imported_case
from openfoam_target import generation_convention
from router_func import llm_requires_hpc, llm_requires_visualization


def solver_planner_node(state):
    print("<solver_planner>")
    if state.get("case_origin") == "imported":
        result = plan_imported_case(state, phase="solver")
    else:
        subtasks = decompose_to_subtasks(
            state["user_requirement"], state["dir_structure_reference"],
            state.get("dir_counts_reference") or "",
            llm_service=state.get("llm_service"),
            openfoam_target=generation_convention(state["config"]),
            phase="solver", mesh_info=state.get("mesh_info"),
        )
        if not subtasks:
            raise ValueError("Failed to generate subtasks.")
        result = {
            "subtasks": subtasks, "workflow_status": "running",
            "input_writer_mode": "initial", "requires_input_writer": True,
        }
        print(f"<subtask_count>{len(subtasks)} subtasks generated.</subtask_count>")
    if result.get("workflow_status") == "running":
        context = {**state, **result}
        if context.get("requires_hpc") is None:
            result["requires_hpc"] = llm_requires_hpc(context)
        if context.get("requires_visualization") is None:
            result["requires_visualization"] = llm_requires_visualization(context)
        result["solver_planned"] = True
    print("</solver_planner>")
    return result
