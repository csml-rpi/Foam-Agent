"""Thin LangGraph adapter for OpenFOAM input generation and rewriting."""

from utils import read_case_foamfiles, scan_case_directory
from services.input_writer import initial_write, build_allrun, rewrite_files
from translation.esi_translator import convert_case_to_esi_if_needed

def input_writer_node(state):
    """
    InputWriter node: Generate the complete OpenFOAM foamfile.
    
    Args:
        state: The current state containing all necessary information
    """

    mode = state["input_writer_mode"]
    
    if mode == "rewrite":
        return _rewrite_mode(state)
    else:
        return _initial_write_mode(state)

def _rewrite_mode(state):
    """Rewrite mode: delegate to service to modify files based on review analysis."""
    print("<input_writer mode=\"rewrite\">")
    if not state.get("review_analysis"):
        print("No review analysis available for rewrite mode.")
        print("</input_writer>")
        return state
    out = rewrite_files(
        case_dir=state["case_dir"],
        error_logs=state.get("error_logs", []),
        review_analysis=state.get("review_analysis", ""),
        rewrite_plan=state.get("rewrite_plan"),
        user_requirement=state.get("user_requirement", ""),
        foamfiles=state.get("foamfiles"),
        dir_structure=state.get("dir_structure", {}),
        openfoam_fork=getattr(state["config"], "openfoam_fork", "foundation"),
        case_solver=state.get("case_solver", ""),
        llm_service=state.get("llm_service"),
    )
    print("</input_writer>")
    
    convert_case_to_esi_if_needed(state["case_dir"], state["config"])
    
    # Rescan the directory and foam files to reflect any translations
    out["dir_structure"] = scan_case_directory(state["case_dir"])
    out["foamfiles"] = read_case_foamfiles(state["case_dir"], out["dir_structure"])

    return out

def _initial_write_mode(state):
    """
    Initial write mode: Generate files from scratch
    """
    print("<input_writer mode=\"initial\">")
    
    config = state["config"]
    write_out = initial_write(
        case_dir=state["case_dir"],
        subtasks=state["subtasks"],
        user_requirement=state["user_requirement"],
        tutorial_reference=state["tutorial_reference"],
        case_solver=state["case_solver"],
        openfoam_fork=getattr(config, "openfoam_fork", "foundation"),
        generation_mode=getattr(config, "input_writer_generation_mode", "sequential_dependency"),
        similar_case_advice=state.get("similar_case_advice"),
        reuse_generated_dir=getattr(config, "reuse_generated_dir", ""),
        llm_service=state.get("llm_service"),
        config=config,
    )

    dir_structure = write_out["dir_structure"]
    foamfiles = write_out["foamfiles"]

    # Build Allrun via service
    mesh_type = state.get("mesh_type")
    mesh_commands = state.get("mesh_commands") or []
    allrun_out = build_allrun(
        case_dir=state["case_dir"],
        database_path=config.database_path,
        searchdocs=config.searchdocs,
        dir_structure=dir_structure,
        case_info=state["case_info"],
        allrun_reference=state["allrun_reference"],
        mesh_type=mesh_type,
        mesh_commands=mesh_commands,
        user_requirement=state["user_requirement"],
        llm_service=state.get("llm_service"),
        config=config,
    )

    print("</input_writer>")

    convert_case_to_esi_if_needed(state["case_dir"], config)
    
    # Rescan the directory and foam files to reflect any translations
    dir_structure = scan_case_directory(state["case_dir"])
    foamfiles = read_case_foamfiles(state["case_dir"], dir_structure)

    return {
        "dir_structure": dir_structure,
        "commands": allrun_out["commands"],
        "foamfiles": foamfiles,
    }
