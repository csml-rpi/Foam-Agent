from langgraph.graph import END

from utils import GraphState


def llm_requires_hpc(state: GraphState) -> bool:
    """
    Use LLM to determine if user requires HPC/cluster execution based on their requirement.
    
    Args:
        state: Current graph state containing user requirement and LLM service
        
    Returns:
        bool: True if HPC execution is required, False otherwise
    """
    user_requirement = state["user_requirement"]
    
    system_prompt = (
        "You are an expert in OpenFOAM workflow analysis. "
        "Analyze the user requirement to determine if they want to run the simulation on HPC (High Performance Computing) or locally. "
        "Look for keywords like: HPC, cluster, supercomputer, SLURM, PBS, job queue, "
        "parallel computing, distributed computing, or any mention of running on remote systems. "
        "If the user explicitly mentions or implies they want to run on HPC/cluster, return 'hpc_run'. "
        "If they want to run locally or don't specify, return 'local_run'. "
        "Be conservative - if unsure, assume local run unless clearly specified otherwise."
        "Only return 'hpc_run' or 'local_run'. Don't return anything else."
    )
    
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "return 'hpc_run' or 'local_run'"
    )
    
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    return "hpc_run" in response.lower()


def llm_requires_visualization(state: GraphState) -> bool:
    """Use LLM to decide whether to run the visualization node.

    Policy: ONLY visualize when the user explicitly asks for it.
    If uncertain, default to NO visualization (avoid expensive/flaky post-processing).
    """
    user_requirement = state["user_requirement"]

    system_prompt = (
        "You are an expert in OpenFOAM workflow analysis. "
        "Analyze the user requirement to determine if they explicitly want visualization/post-processing of results. "
        "Signals include requests to: visualize/plot/render results, create images/figures, contours, vectors, streamlines, "
        "Paraview/PyVista, post-processing, screenshots, or animations. "
        "Return 'yes_visualization' ONLY if the user explicitly requests visualization. "
        "If they do not mention visualization, or you are unsure, return 'no_visualization'. "
        "Only return 'yes_visualization' or 'no_visualization'."
    )

    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "Return exactly: 'yes_visualization' or 'no_visualization'."
    )

    response = state["llm_service"].invoke(user_prompt, system_prompt)
    return "yes_visualization" in response.lower()


def route_after_solver_planner(state: GraphState):
    """Successful solver planning hands physical files to the writer."""
    if state.get("workflow_status") == "failed":
        return END
    return "meshing_planner" if state.get("repairing_mesh") else "input_writer"


def route_after_meshing(state: GraphState):
    """Plan the solver after initial meshing; preserve pending repairs on retries."""
    if state.get("termination_reason"):
        return END
    if state.get("error_logs"):
        return "reviewer"
    return "solver_planner" if not state.get("solver_planned") or state.get("replan_solver") else "input_writer"


def route_after_input_writer(state: GraphState):
    """
    Route after input_writer node based on whether user wants to run on HPC.
    Prefer solver-planner-cached decisions to avoid repeated LLM routing jitter.
    """
    if state.get("error_logs"):
        return "reviewer"

    if state.get("repairing_mesh"):
        return "meshing_planner"

    return _route_runner(state)


def _route_runner(state: GraphState):
    """Select the shared local or HPC runner."""
    requires_hpc = state.get("requires_hpc")
    if requires_hpc is None:
        requires_hpc = llm_requires_hpc(state)
        state["requires_hpc"] = requires_hpc

    if requires_hpc:
        print("<router>HPC run requested. Routing to hpc_runner node.</router>")
        return "hpc_runner"
    else:
        print("<router>Local run requested. Routing to local_runner node.</router>")
        return "local_runner"

def route_after_runner(state: GraphState):
    if state.get("error_logs") and len(state["error_logs"]) > 0:
        return "reviewer"

    requires_visualization = state.get("requires_visualization")
    if requires_visualization is None:
        requires_visualization = llm_requires_visualization(state)
        state["requires_visualization"] = requires_visualization

    if requires_visualization:
        return "visualization"
    return END

def route_after_preparation(state: GraphState):
    """Prepared inputs enter mesh planning before solver selection."""
    if state.get("workflow_status") == "failed":
        return END
    return "meshing_planner"


def route_after_reviewer(state: GraphState):
    """Use the same file-rewrite loop for generated and imported cases."""
    if state.get("termination_reason") in {"repair_made_no_progress", "mesh_review_stopped"}:
        return END
    loop_count = state.get("loop_count", 0)
    if loop_count >= state["config"].max_loop:
        print(f"<router>Maximum loop count ({state['config'].max_loop}) reached. Ending workflow.</router>")
        if state.get("repairing_mesh"):
            return END
        requires_visualization = state.get("requires_visualization")
        if requires_visualization is None:
            requires_visualization = llm_requires_visualization(state)
            state["requires_visualization"] = requires_visualization
        return "visualization" if requires_visualization else END

    print(f"<router>Loop {loop_count}: Continuing to fix errors.</router>")
    if state.get("repairing_mesh"):
        return "meshing_planner"
    return "input_writer"
