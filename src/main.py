from dataclasses import dataclass, field
from typing import List, Optional, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import argparse
from pathlib import Path
from utils import LLMService, GraphState

from config import Config
from nodes.architect_node import architect_node
from nodes.meshing_node import meshing_node
from nodes.input_writer_node import input_writer_node
from nodes.local_runner_node import local_runner_node
from nodes.reviewer_node import reviewer_node
from nodes.visualization_node import visualization_node
from nodes.hpc_runner_node import hpc_runner_node
from router_func import (
    route_after_architect, 
    route_after_input_writer, 
    route_after_runner, 
    route_after_reviewer
)
import json

def create_foam_agent_graph() -> StateGraph:
    """Create the OpenFOAM agent workflow graph."""
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("architect", architect_node)
    workflow.add_node("meshing", meshing_node)
    workflow.add_node("input_writer", input_writer_node)
    workflow.add_node("local_runner", local_runner_node)
    workflow.add_node("hpc_runner", hpc_runner_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("visualization", visualization_node)
    
    # Add edges
    workflow.add_edge(START, "architect")
    workflow.add_conditional_edges("architect", route_after_architect)
    workflow.add_edge("meshing", "input_writer")
    workflow.add_conditional_edges("input_writer", route_after_input_writer)
    workflow.add_conditional_edges("hpc_runner", route_after_runner)
    workflow.add_conditional_edges("local_runner", route_after_runner)
    workflow.add_conditional_edges("reviewer", route_after_reviewer)
    workflow.add_edge("visualization", END)
    
    return workflow

def initialize_state(user_requirement: str, config: Config, custom_mesh_path: Optional[str] = None) -> GraphState:
    case_stats = json.load(open(f"{config.database_path}/raw/openfoam_case_stats.json", "r"))
    mesh_type = "custom_mesh" if custom_mesh_path else "standard_mesh"
    state = GraphState(
        user_requirement=user_requirement,
        config=config,
        case_dir="",
        tutorial="",
        case_name="",
        subtasks=[],
        current_subtask_index=0,
        error_command=None,
        error_content=None,
        loop_count=0,
        llm_service=LLMService(config),
        case_stats=case_stats,
        tutorial_reference=None,
        case_path_reference=None,
        dir_structure_reference=None,
        case_info=None,
        allrun_reference=None,
        dir_structure=None,
        commands=None,
        foamfiles=None,
        error_logs=None,
        history_text=None,
        case_domain=None,
        case_category=None,
        case_solver=None,
        mesh_info=None,
        mesh_commands=None,
        custom_mesh_used=None,
        mesh_type=mesh_type,
        custom_mesh_path=custom_mesh_path,
        review_analysis=None,
        input_writer_mode="initial"
    )
    if custom_mesh_path:
        print(f"Custom mesh path: {custom_mesh_path}")
    else:
        print("No custom mesh path provided.")
    return state

def main(user_requirement: str, config: Config, custom_mesh_path: Optional[str] = None):
    """Main function to run the OpenFOAM workflow."""
    
    # Create and compile the graph
    workflow = create_foam_agent_graph()
    app = workflow.compile()
    
    # Initialize the state
    initial_state = initialize_state(user_requirement, config, custom_mesh_path)
    
    print("Starting Foam-Agent...")
    
    # Invoke the graph
    try:
        result = app.invoke(initial_state)
        print("Workflow completed successfully!")
        
        # Print final statistics
        if result.get("llm_service"):
            result["llm_service"].print_statistics()
        
        # print(f"Final state: {result}")
        
    except Exception as e:
        print(f"Workflow failed with error: {e}")
        raise

if __name__ == "__main__":
    # python main.py
    parser = argparse.ArgumentParser(
        description="Run the OpenFOAM workflow"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=f"{Path(__file__).parent.parent}/user_requirement.txt",
        help="User requirement file path for the workflow.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for the workflow.",
    )
    parser.add_argument(
        "--custom_mesh_path",
        type=str,
        default=None,
        help="Path to custom mesh file (e.g., .msh, .stl, .obj). If not provided, no custom mesh will be used.",
    )
    
    args = parser.parse_args()
    print(args)
    
    # Initialize configuration.
    config = Config()
    if args.output_dir != "":
        config.case_dir = args.output_dir
    
    with open(args.prompt_path, 'r') as f:
        user_requirement = f.read()
    
    main(user_requirement, config, args.custom_mesh_path)
