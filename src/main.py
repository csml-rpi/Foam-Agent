from typing import Optional
from langgraph.graph import StateGraph, START, END
import argparse
from pathlib import Path
from utils import GraphState, load_agent_resources

from config import Config
from openfoam_target import (
    normalise_openfoam_target,
)
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
from services.plan import DEFAULT_IMPORTED_REQUIREMENT


def workflow_exit_code(state: GraphState) -> int:
    """Return a non-zero process status for any terminal workflow failure."""
    status = state.get("workflow_status")
    return 2 if status in {"failed", "partial_success"} or state.get("termination_reason") else 0


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
    workflow.add_node("case_import", case_import_node)
    
    # Add edges
    workflow.add_conditional_edges(START, route_workflow_entry)
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
    user_requirement_explicit: bool = True,
) -> GraphState:
    """Build state for generated or imported agent workflows."""
    case_stats = None
    llm_service = None
    if workflow_mode == "prompt":
        llm_service, case_stats = load_agent_resources(config)
    # mesh_type = "custom_mesh" if custom_mesh_path else "standard_mesh"
    state = GraphState(
        user_requirement=user_requirement,
        config=config,
        case_dir="",
        case_name="",
        subtasks=[],
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
        case_origin="generated" if workflow_mode == "prompt" else "imported",
        user_requirement_explicit=user_requirement_explicit,
        case_import_path=case_import_path,
        case_import_subdir=case_import_subdir,
        case_import_manifest=None,
        case_import_report_dir=None,
        case_context=None,
        workflow_status="planning",
        error_fingerprints=[],
        target_mismatch=False,
        requires_meshing=False,
        requires_input_writer=False,
    )
    if custom_mesh_path:
        print(f"<custom_mesh_path>{custom_mesh_path}</custom_mesh_path>")
    else:
        print("<custom_mesh_path>None</custom_mesh_path>")
    return state

def main(
    user_requirement: str,
    config: Config,
    custom_mesh_path: Optional[str] = None,
    *,
    case_path: Optional[str] = None,
    case_subdir: Optional[str] = None,
) -> GraphState:
    """Run the shared workflow for a prompt or an existing case."""
    if case_path and not config.case_dir:
        raise ValueError("--output_dir is required when --case_path is used.")
    try:
        # Create and compile the graph
        workflow = create_foam_agent_graph()
        app = workflow.compile()

        # Initialize the state
        initial_state = initialize_state(
            (user_requirement.strip() or DEFAULT_IMPORTED_REQUIREMENT) if case_path else user_requirement,
            config,
            custom_mesh_path,
            workflow_mode="imported_case" if case_path else "prompt",
            case_import_path=case_path,
            case_import_subdir=case_subdir,
            user_requirement_explicit=bool(user_requirement.strip() or custom_mesh_path) if case_path else True,
        )

        print("<workflow_start>Starting Foam-Agent...</workflow_start>")

        # Invoke the graph
        result = app.invoke(initial_state, config={"recursion_limit": config.recursion_limit})

        termination_reason = result.get("termination_reason")
        if result.get("workflow_status") == "partial_success":
            print("<workflow_end>Simulation completed successfully, but visualization failed.</workflow_end>")
        elif termination_reason:
            print(
                "<workflow_end>Workflow stopped without completing successfully: "
                f"{termination_reason}</workflow_end>"
            )
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


if __name__ == "__main__":
    # python main.py
    parser = argparse.ArgumentParser(
        description="Run the OpenFOAM workflow"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="User requirement file path for the workflow.",
    )
    parser.add_argument(
        "--case_path",
        type=str,
        default=None,
        help=(
            "Existing OpenFOAM case directory or ZIP archive. Uses the explicitly "
            "configured target; otherwise attempts to detect Foundation v10 or ESI "
            "v2006 from case headers. Specify --openfoam_target if detection is inconclusive."
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
        "--openfoam_target",
        type=str,
        default=None,
        help=(
            "Explicit native OpenFOAM target. Use esi-v2006 for ESI/OpenCFD "
            "v2006; omit it to preserve the existing Foundation/generic-ESI path."
        ),
    )
    parser.add_argument(
        "--custom_mesh_path",
        type=str,
        default=None,
        help="Path to a Gmsh .msh file (ASCII 2.2 format). If not provided, no custom mesh will be used.",
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
    
    args = parser.parse_args()
    print(f"args: {args}")
    
    # Initialize configuration.
    config = Config()

    if args.openfoam_target is not None:
        try:
            config.openfoam_target = normalise_openfoam_target(args.openfoam_target)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"<config>openfoam_target={config.openfoam_target} (cli)</config>")

    print(f"config: {config}")

    if args.output_dir != "":
        config.case_dir = args.output_dir

    if args.reuse_generated_dir:
        config.reuse_generated_dir = args.reuse_generated_dir
    
    config.overwrite_case_dir = args.overwrite_output

    if args.case_path:
        if args.reuse_generated_dir:
            parser.error("--reuse_generated_dir is not available with --case_path.")
        if not args.output_dir:
            parser.error("--output_dir is required with --case_path.")
        user_requirement = (
            Path(args.prompt_path).read_text(encoding="utf-8")
            if args.prompt_path
            else ""
        )
    else:
        if args.case_subdir:
            parser.error("--case_subdir requires --case_path.")
        prompt_path = args.prompt_path or f"{Path(__file__).parent.parent}/user_requirement.txt"
        with open(prompt_path, 'r', encoding="utf-8") as f:
            user_requirement = f.read()
    
    final_state = main(
        user_requirement,
        config,
        args.custom_mesh_path,
        case_path=args.case_path,
        case_subdir=args.case_subdir,
    )
    if exit_code := workflow_exit_code(final_state):
        raise SystemExit(exit_code)
