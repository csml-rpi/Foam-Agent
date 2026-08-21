"""Thin LangGraph adapter for planning and output-directory preparation."""

from utils import save_file
from services.plan import generate_simulation_plan
from services.output_safety import (
    OutputDirectorySafetyError,
    prepare_output_directory,
    validate_output_preparation,
)
from router_func import llm_requires_custom_mesh, llm_requires_hpc, llm_requires_visualization
from logger import setup_logging


def planner_node(state):
    """
    Planner node: Parse the user requirement to a standard case description,
    finds a similar reference case from the FAISS databases, and splits the work into subtasks.
    Updates state with:
      - case_dir, tutorial, case_name, subtasks.
    """
    config = state["config"]
    user_requirement = state["user_requirement"]

    # Fail before planning/RAG/LLM calls when the CLI points at a protected
    # non-empty output directory. The later check remains necessary for the
    # default case-name-derived output path.
    requested_case_dir = getattr(config, "case_dir", "")
    if requested_case_dir:
        try:
            validate_output_preparation(
                requested_case_dir,
                overwrite=getattr(config, "overwrite_case_dir", False),
            )
        except OutputDirectorySafetyError as exc:
            raise ValueError(str(exc)) from exc

    # Generate simulation plan using the core planning logic
    plan_data = generate_simulation_plan(
        user_requirement=user_requirement,
        case_stats=state["case_stats"],
        case_dir=getattr(config, "case_dir", ""),
        searchdocs=getattr(config, "searchdocs", 2),
        run_directory=getattr(config, "run_directory", None),
        run_times=getattr(config, "run_times", 1),
        config=config,
        llm_service=state.get("llm_service"),
    )
    
    # Extract plan data
    case_name = plan_data["case_name"]
    case_domain = plan_data["case_domain"]
    case_category = plan_data["case_category"]
    case_solver = plan_data["case_solver"]
    case_dir = plan_data["case_dir"]
    faiss_detailed = plan_data["tutorial_reference"]
    case_path_reference = plan_data["case_path_reference"]
    dir_structure_reference = plan_data["dir_structure_reference"]
    allrun_reference = plan_data["allrun_reference"]
    subtasks = plan_data["subtasks"]
    similar_case_advice = plan_data.get("similar_case_advice")
    
    # Prepare the known Foam-Agent-owned case directory only after planning.
    # The preflight above avoids expensive LLM/RAG work when an explicit target
    # is already unsafe; this call also covers a planner-derived default path.
    try:
        case_dir = str(
            prepare_output_directory(
                case_dir,
                overwrite=getattr(config, "overwrite_case_dir", False),
            )
        )
    except OutputDirectorySafetyError as exc:
        raise ValueError(str(exc)) from exc

    # Initialize logging now that case_dir exists
    setup_logging(case_dir)

    print("<planner>")
    print(f"<case_name>{case_name}</case_name>")
    print(f"<case_domain>{case_domain}</case_domain>")
    print(f"<case_category>{case_category}</case_category>")
    print(f"<case_solver>{case_solver}</case_solver>")
    print(f"<case_dir>{case_dir}</case_dir>")
    print(f"<similar_case_structure>{dir_structure_reference}</similar_case_structure>")
    print(f"<subtask_count>{len(subtasks)} subtasks generated.</subtask_count>")
    if similar_case_advice:
        print(f"<similar_case_advice>{similar_case_advice}</similar_case_advice>")

    # Save reference file
    save_file(case_path_reference, f"{faiss_detailed}\n\n\n{allrun_reference}")

    # Determine mesh type
    mesh_type = llm_requires_custom_mesh(state)
    if mesh_type == 1:
        mesh_type_value = "custom_mesh"
        print("<mesh_type>custom_mesh - Custom mesh requested.</mesh_type>")
    elif mesh_type == 2:
        mesh_type_value = "gmsh_mesh"
        print("<mesh_type>gmsh_mesh - GMSH mesh requested.</mesh_type>")
    else:
        mesh_type_value = "standard_mesh"
        print("<mesh_type>standard_mesh - Standard mesh generation.</mesh_type>")

    # Cache routing decisions to avoid repeated LLM calls in routing.
    requires_hpc = llm_requires_hpc(state)
    requires_visualization = llm_requires_visualization(state)
    print(f"<routing_decisions>requires_hpc={requires_hpc}, requires_visualization={requires_visualization}</routing_decisions>")
    print("</planner>")

    # Return updated state
    case_info = f"case name: {case_name}\ncase domain: {case_domain}\ncase category: {case_category}\ncase solver: {case_solver}"
    return {
        "case_name": case_name,
        "case_domain": case_domain,
        "case_category": case_category,
        "case_solver": case_solver,
        "case_dir": case_dir,
        "tutorial_reference": faiss_detailed,
        "case_path_reference": case_path_reference,
        "dir_structure_reference": dir_structure_reference,
        "case_info": case_info,
        "allrun_reference": allrun_reference,
        "subtasks": subtasks,
        "mesh_type": mesh_type_value,
        "requires_hpc": requires_hpc,
        "requires_visualization": requires_visualization,
        "similar_case_advice": similar_case_advice,
    }
