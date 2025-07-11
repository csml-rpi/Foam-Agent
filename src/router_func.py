from typing import TypedDict, List, Optional
from config import Config
from utils import LLMService
from langgraph.graph import StateGraph, START, END

class GraphState(TypedDict):
    user_requirement: str
    config: Config
    case_dir: str
    tutorial: str
    case_name: str
    subtasks: List[dict]
    current_subtask_index: int
    error_command: Optional[str]
    error_content: Optional[str]
    loop_count: int
    # Additional state fields that will be added during execution
    llm_service: Optional[LLMService]
    case_stats: Optional[dict]
    tutorial_reference: Optional[str]
    case_path_reference: Optional[str]
    dir_structure_reference: Optional[str]
    case_info: Optional[str]
    allrun_reference: Optional[str]
    dir_structure: Optional[dict]
    commands: Optional[List[str]]
    foamfiles: Optional[dict]
    error_logs: Optional[List[str]]
    history_text: Optional[List[str]]
    case_domain: Optional[str]
    case_category: Optional[str]
    case_solver: Optional[str]
    # Mesh-related state fields
    mesh_info: Optional[dict]
    mesh_commands: Optional[List[str]]
    custom_mesh_used: Optional[bool]
    mesh_type: Optional[str]
    custom_mesh_path: Optional[str]


def llm_requires_custom_mesh(state: GraphState) -> int:
    """
    Use LLM to determine if user requires custom mesh based on their requirement.
    
    Args:
        state: Current graph state containing user requirement and LLM service
        
    Returns:
        int: 1 if custom mesh is required, 2 if gmsh mesh is required, 0 otherwise
    """
    user_requirement = state["user_requirement"]
    
    system_prompt = (
        "You are an expert in OpenFOAM workflow analysis. "
        "Analyze the user requirement to determine if they want to use a custom mesh file. "
        "Look for keywords like: custom mesh, mesh file, .msh, .stl, .obj, gmsh, snappyHexMesh, "
        "or any mention of importing/using external mesh files. "
        "If the user explicitly mentions or implies they want to use a custom mesh file, return 'custom_mesh'. "
        "If they want to use standard OpenFOAM mesh generation (blockMesh, snappyHexMesh with STL, etc.), return 'standard_mesh'. "
        "Look for keywords like gmsh and determine if they want to create mesh using gmsh. If they want to create mesh using gmsh, return 'gmsh_mesh'. "
        "Be conservative - if unsure, assume standard mesh unless clearly specified otherwise."
    )
    
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "Determine if the user wants to use a custom mesh file. "
        "Return exactly 'custom_mesh' if they want to use a custom mesh file, "
        "'standard_mesh' if they want standard OpenFOAM mesh generation or 'gmsh_mesh' if they want to create mesh using gmsh."
    )
    
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    if "custom_mesh" in response.lower():
        return 1
    elif "gmsh_mesh" in response.lower():
        return 2
    else:
        return 0


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
    )
    
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "Determine if the user wants to run the simulation on HPC/cluster. "
        "Return exactly 'hpc_run' if they want to use HPC/cluster, "
        "or 'local_run' if they want to run locally."
    )
    
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    return "hpc_run" in response.lower()


def llm_requires_visualization(state: GraphState) -> bool:
    """
    Use LLM to determine if user requires visualization based on their requirement.
    
    Args:
        state: Current graph state containing user requirement and LLM service
        
    Returns:
        bool: True if visualization is required, False otherwise
    """
    user_requirement = state["user_requirement"]
    
    system_prompt = (
        "You are an expert in OpenFOAM workflow analysis. "
        "Analyze the user requirement to determine if they want visualization of results. "
        "Look for keywords like: plot, visualize, graph, chart, contour, streamlines, "
        "paraview, post-processing, results analysis, or any mention of viewing/displaying results. "
        "If the user explicitly mentions or implies they want visualization, return 'visualization'. "
        "If they don't mention visualization or only want to run the simulation, return 'no_visualization'. "
        "Be conservative - if unsure, assume visualization is wanted unless clearly specified otherwise."
    )
    
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "Determine if the user wants visualization of simulation results. "
        "Return exactly 'visualization' if they want to visualize results, "
        "or 'no_visualization' if they don't need visualization."
    )
    
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    return "visualization" in response.lower()


def route_after_architect(state: GraphState):
    """
    Route after architect node based on whether user wants custom mesh.
    """
    mesh_type = state.get("mesh_type", "standard_mesh")
    if mesh_type == "custom_mesh":
        print("Router: Custom mesh requested. Routing to meshing node.")
        return "meshing"
    elif mesh_type == "gmsh_mesh":
        print("Router: GMSH mesh requested. Routing to meshing node.")
        return "meshing"
    else:
        print("Router: Standard mesh generation. Routing to input_writer node.")
        return "input_writer"


def route_after_input_writer(state: GraphState):
    """
    Route after input_writer node based on whether user wants to run on HPC.
    """
    if llm_requires_hpc(state):
        print("LLM determined: HPC run requested. Routing to hpc_runner node.")
        return "hpc_runner"
    else:
        print("LLM determined: Local run requested. Routing to local_runner node.")
        return "local_runner"

def route_after_runner(state: GraphState):
    if state.get("error_logs") and len(state["error_logs"]) > 0:
        return "reviewer"
    elif llm_requires_visualization(state):
        return "visualization"
    else:
        return END

def route_after_reviewer(state: GraphState):
    loop_count = state.get("loop_count", 0)
    max_loop = state["config"].max_loop
    if loop_count >= max_loop:
        print(f"Maximum loop count ({max_loop}) reached. Ending workflow.")

        if llm_requires_visualization(state):
            return "visualization"
        else:
            return END
    
    # state["loop_count"] = loop_count + 1
    print(f"Loop {loop_count}: Continuing to fix errors.")
    
    return route_after_input_writer(state)
