from typing import Optional
from langgraph.graph import StateGraph, START, END
import argparse
from pathlib import Path
from utils import LLMService, GraphState

from config import Config
from nodes.planner_node import planner_node
from nodes.meshing_node import meshing_node
from nodes.input_writer_node import input_writer_node
from nodes.local_runner_node import local_runner_node
from nodes.reviewer_node import reviewer_node
from nodes.visualization_node import visualization_node
from nodes.hpc_runner_node import hpc_runner_node
from nodes.imported_case_node import case_import_node
from router_func import (
    route_after_case_import,
    route_after_meshing,
    route_after_planner,
    route_after_input_writer,
    route_after_runner,
    route_after_reviewer,
    route_workflow_entry,
)
from logger import close_logging
import json


def workflow_entry_node(_state: GraphState) -> dict:
    """Provide one graph entry point before routing by input mode."""
    return {}


def create_foam_agent_graph() -> StateGraph:
    """Create the OpenFOAM agent workflow graph."""
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("entry", workflow_entry_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("meshing", meshing_node)
    workflow.add_node("input_writer", input_writer_node)
    workflow.add_node("local_runner", local_runner_node)
    workflow.add_node("hpc_runner", hpc_runner_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("visualization", visualization_node)
    workflow.add_node("case_import", case_import_node)
    
    # Add edges
    workflow.add_edge(START, "entry")
    workflow.add_conditional_edges("entry", route_workflow_entry)
    workflow.add_conditional_edges("planner", route_after_planner)
    workflow.add_conditional_edges("meshing", route_after_meshing)
    workflow.add_conditional_edges("input_writer", route_after_input_writer)
    workflow.add_conditional_edges("hpc_runner", route_after_runner)
    workflow.add_conditional_edges("local_runner", route_after_runner)
    workflow.add_conditional_edges("reviewer", route_after_reviewer)
    workflow.add_conditional_edges("case_import", route_after_case_import)
    workflow.add_edge("visualization", END)
    
    return workflow

def initialize_state(
    user_requirement: str,
    config: Config,
    custom_mesh_path: Optional[str] = None,
    *,
    workflow_mode: str = "prompt",
    case_import_path: Optional[str] = None,
    case_import_subdir: Optional[str] = None,
    requires_visualization: Optional[bool] = None,
) -> GraphState:
    """Build the state for either prompt generation or controlled case import.

    Import mode intentionally avoids loading RAG metadata or constructing an
    LLM client: neither is needed to execute a user-provided case safely.
    """
    case_stats = None
    llm_service = None
    if workflow_mode == "prompt":
        case_stats_path = Path(config.database_path) / "raw" / "openfoam_case_stats.json"
        with case_stats_path.open(encoding="utf-8") as case_stats_file:
            case_stats = json.load(case_stats_file)
        llm_service = LLMService(config)
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
        llm_service=llm_service,
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
        requires_visualization=requires_visualization,
        job_id=None,
        cluster_info=None,
        slurm_script_path=None,
        termination_reason=None,
        workflow_mode=workflow_mode,
        # These policies make the common execution/review nodes explicit about
        # what they may do.  ``workflow_mode`` remains the entry router's
        # concern; downstream nodes use policy, not the input transport.
        execution_policy=(
            "generated_allrun" if workflow_mode == "prompt" else "controlled_import"
        ),
        repair_policy=(
            "llm_rewrite" if workflow_mode == "prompt" else "numeric_invariant_only"
        ),
        case_import_path=case_import_path,
        case_import_subdir=case_import_subdir,
        case_import_manifest=None,
        case_import_original_dir=None,
        case_import_report_dir=None,
        case_import_attempts=[],
        case_import_overrides={},
        case_import_error_fingerprints=[],
        case_import_status=None,
    )
    if custom_mesh_path:
        print(f"<custom_mesh_path>{custom_mesh_path}</custom_mesh_path>")
    else:
        print("<custom_mesh_path>None</custom_mesh_path>")
    return state

def main(user_requirement: str, config: Config, custom_mesh_path: Optional[str] = None) -> GraphState:
    """Main function to run the OpenFOAM workflow."""
    
    # Create and compile the graph
    workflow = create_foam_agent_graph()
    app = workflow.compile()
    
    # Initialize the state
    initial_state = initialize_state(user_requirement, config, custom_mesh_path)
    
    print("<workflow_start>Starting Foam-Agent...</workflow_start>")

    # Invoke the graph
    try:
        # Every graph node passes this invocation's LLM service explicitly to
        # its service-layer operation.  This keeps concurrent configurations
        # isolated instead of relying on mutable process-wide state.
        result = app.invoke(initial_state, config={"recursion_limit": config.recursion_limit})

        termination_reason = result.get("termination_reason")
        if termination_reason == "max_review_loop_reached":
            print("<workflow_end>Workflow finished after reaching the maximum review loop limit.</workflow_end>")
        else:
            print("<workflow_end>Workflow completed successfully!</workflow_end>")

        # Print final statistics
        if result.get("llm_service"):
            result["llm_service"].print_statistics()

        return result

    except Exception as e:
        print(f"<workflow_error>{e}</workflow_error>")
        raise
    finally:
        close_logging()


def main_imported_case(
    case_path: str,
    config: Config,
    *,
    case_subdir: Optional[str] = None,
    visualize: bool = False,
) -> dict:
    """Run an existing case through the protected branch of the StateGraph.

    The branch omits Planner, Meshing, Input Writer, and the LLM reviewer so
    user dictionaries cannot be regenerated or numerically modified.
    """

    if not config.case_dir:
        raise ValueError("--output_dir is required when --case_path is used.")
    try:
        workflow = create_foam_agent_graph()
        app = workflow.compile()
        initial_state = initialize_state(
            "Run the supplied OpenFOAM case without changing user-provided numeric inputs.",
            config,
            workflow_mode="imported_case",
            case_import_path=case_path,
            case_import_subdir=case_subdir,
            requires_visualization=visualize,
        )
        print("<workflow_start>Starting imported-case Foam-Agent workflow...</workflow_start>")
        final_state = app.invoke(
            initial_state,
            config={"recursion_limit": config.recursion_limit},
        )
        manifest = final_state.get("case_import_manifest")
        errors = final_state.get("error_logs") or []
        result = {
            "status": final_state.get("case_import_status", "blocked"),
            "original_dir": final_state.get("case_import_original_dir"),
            "work_dir": final_state.get("case_dir"),
            "report_dir": final_state.get("case_import_report_dir"),
            "manifest": manifest.to_dict() if manifest is not None else None,
            "attempts": final_state.get("case_import_attempts") or [],
            "errors": errors,
        }
        summary = {
            "status": result["status"],
            "work_dir": result["work_dir"],
            "report_dir": result["report_dir"],
            "attempt_count": len(result["attempts"]),
        }
        print(f"<case_import_result>{json.dumps(summary)}</case_import_result>")
        return result
    finally:
        close_logging()

if __name__ == "__main__":
    # python main.py
    parser = argparse.ArgumentParser(
        description="Run the OpenFOAM workflow"
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="User requirement file path for the workflow.",
    )
    input_group.add_argument(
        "--case_path",
        type=str,
        default=None,
        help=(
            "Existing Foundation OpenFOAM v10 case directory or ZIP archive. "
            "This bypasses prompt generation and runs only validated case commands."
        ),
    )
    parser.add_argument(
        "--case_subdir",
        type=str,
        default=None,
        help=(
            "Relative case directory inside --case_path when an archive or "
            "directory contains more than one system/controlDict."
        ),
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
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Explicitly allow replacing an existing non-empty output directory.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate a PyVista visualization after a successful imported case run.",
    )
    
    args = parser.parse_args()
    print(f"args: {args}")
    
    # Initialize configuration.
    config = Config()

    print(f"config: {config}")

    if args.output_dir != "":
        config.case_dir = args.output_dir

    if args.reuse_generated_dir:
        config.reuse_generated_dir = args.reuse_generated_dir

    config.overwrite_case_dir = args.overwrite_output
    
    if args.case_path:
        if args.custom_mesh_path:
            parser.error("--custom_mesh_path is not available with --case_path.")
        if args.reuse_generated_dir:
            parser.error("--reuse_generated_dir is not available with --case_path.")
        if not args.output_dir:
            parser.error("--output_dir is required with --case_path.")
        imported_result = main_imported_case(
            args.case_path,
            config,
            case_subdir=args.case_subdir,
            visualize=args.visualize,
        )
        if imported_result.get("status") != "success":
            raise SystemExit(2)
    else:
        if args.case_subdir:
            parser.error("--case_subdir requires --case_path.")
        if args.visualize:
            parser.error("--visualize requires --case_path.")
        prompt_path = args.prompt_path or f"{Path(__file__).parent.parent}/user_requirement.txt"
        with open(prompt_path, 'r') as f:
            user_requirement = f.read()

        final_state = main(user_requirement, config, args.custom_mesh_path)
        if final_state.get("termination_reason") == "max_review_loop_reached":
            raise SystemExit(2)
