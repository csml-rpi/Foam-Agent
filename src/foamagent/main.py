from dataclasses import dataclass, field
from typing import List, Optional, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import argparse
from pathlib import Path
from foamagent import paths
from foamagent.services import get_llm_service
from foamagent.utils import GraphState

from foamagent.config import Config
from foamagent.nodes.planner_node import planner_node
from foamagent.nodes.meshing_node import meshing_node
from foamagent.nodes.input_writer_node import input_writer_node
from foamagent.nodes.local_runner_node import local_runner_node
from foamagent.nodes.reviewer_node import reviewer_node
from foamagent.nodes.visualization_node import visualization_node
from foamagent.nodes.hpc_runner_node import hpc_runner_node
from foamagent.router_func import (
    route_after_planner,
    route_after_input_writer,
    route_after_runner,
    route_after_reviewer
)
from foamagent.logger import close_logging, get_logger

logger = get_logger(__name__)
import json

def create_foam_agent_graph() -> StateGraph:
    """Create the OpenFOAM agent workflow graph."""
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("meshing", meshing_node)
    workflow.add_node("input_writer", input_writer_node)
    workflow.add_node("local_runner", local_runner_node)
    workflow.add_node("hpc_runner", hpc_runner_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("visualization", visualization_node)
    
    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges("planner", route_after_planner)
    workflow.add_edge("meshing", "input_writer")
    workflow.add_conditional_edges("input_writer", route_after_input_writer)
    workflow.add_conditional_edges("hpc_runner", route_after_runner)
    workflow.add_conditional_edges("local_runner", route_after_runner)
    workflow.add_conditional_edges("reviewer", route_after_reviewer)
    workflow.add_edge("visualization", END)
    
    return workflow

def initialize_state(user_requirement: str, config: Config, custom_mesh_path: Optional[str] = None) -> GraphState:
    case_stats = json.load(open(f"{config.database_path}/raw/openfoam_case_stats.json", "r"))
    # mesh_type = "custom_mesh" if custom_mesh_path else "standard_mesh"
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
        llm_service=get_llm_service(config),
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
        mesh_type=None,
        custom_mesh_path=custom_mesh_path,
        review_analysis=None,
        rewrite_plan=None,
        input_writer_mode="initial",
        requires_hpc=None,
        requires_visualization=None,
        job_id=None,
        cluster_info=None,
        slurm_script_path=None,
        termination_reason=None
    )
    if custom_mesh_path:
        print(f"<custom_mesh_path>{custom_mesh_path}</custom_mesh_path>")  # noqa: T201
    else:
        print("<custom_mesh_path>None</custom_mesh_path>")  # noqa: T201
    return state

def main(user_requirement: str, config: Config, custom_mesh_path: Optional[str] = None):
    """Main function to run the OpenFOAM workflow."""
    
    # Create and compile the graph
    workflow = create_foam_agent_graph()
    app = workflow.compile()
    
    # Initialize the state
    initial_state = initialize_state(user_requirement, config, custom_mesh_path)
    
    print("<workflow_start>Starting Foam-Agent...</workflow_start>")  # noqa: T201

    # Invoke the graph
    try:
        result = app.invoke(initial_state, config={"recursion_limit": config.recursion_limit})

        termination_reason = result.get("termination_reason")
        if termination_reason == "max_review_loop_reached":
            print("<workflow_end>Workflow finished after reaching the maximum review loop limit.</workflow_end>")  # noqa: T201
        else:
            print("<workflow_end>Workflow completed successfully!</workflow_end>")  # noqa: T201

        # Print final statistics
        if result.get("llm_service"):
            result["llm_service"].print_statistics()

    except Exception as e:
        print(f"<workflow_error>{e}</workflow_error>")  # noqa: T201
        raise
    finally:
        close_logging()

if __name__ == "__main__":
    # python main.py
    parser = argparse.ArgumentParser(
        description="Run the OpenFOAM workflow"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=str(paths.repo_root() / "user_requirement.txt"),
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
    parser.add_argument(
        "--reuse_generated_dir",
        type=str,
        default="",
        help=(
            "Path to a directory containing previously generated OpenFOAM files. "
            "If a file exists at <reuse_generated_dir>/<folder>/<file>, Foam-Agent will copy it into the current output and skip generation for that file."
        ),
    )
    
    args = parser.parse_args()
    logger.info("args: %s", args)
    
    # Initialize configuration.
    config = Config()

    logger.info("config: %s", config)

    if args.output_dir != "":
        config.case_dir = args.output_dir

    if args.reuse_generated_dir:
        config.reuse_generated_dir = args.reuse_generated_dir
    
    with open(args.prompt_path, 'r') as f:
        user_requirement = f.read()
    
    main(user_requirement, config, args.custom_mesh_path)
