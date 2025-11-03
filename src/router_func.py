from typing import TypedDict, List, Optional
from config import Config
from utils import LLMService, GraphState
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


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
        "Be conservative - if unsure, assume 'standard_mesh' unless clearly specified otherwise."
        "Only return 'custom_mesh' or 'standard_mesh' or 'gmsh_mesh'. Don't return anything else."
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
        "Only return 'hpc_run' or 'local_run'. Don't return anything else."
    )
    
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "return 'hpc_run' or 'local_run'"
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
        "Look for keywords like: plot, visualize, graph, chart, contour, streamlines, paraview, post-processing."
        "Only if the user explicitly mentions they want visualization, return 'yes_visualization'. "
        "If they don't mention visualization or only want to run the simulation, return 'no_visualization'. "
        "Be conservative - if unsure, assume visualization is wanted unless clearly specified otherwise."
        "Only return 'yes_visualization' or 'no_visualization'. Don't return anything else."
    )
    
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "return 'yes_visualization' or 'no_visualization'"
    )
    
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    return "yes_visualization" in response.lower()


def route_after_planner(state: GraphState):
    """
    Route after planner node based on whether user wants custom mesh.
    For current version, if user wants custom mesh, user should be able to provide a path to the mesh file.
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
    print(f"Loop {loop_count}: Continuing to fix errors.")

    return "input_writer"
