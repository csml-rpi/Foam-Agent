"""Thin LangGraph adapter for OpenFOAM input generation and rewriting."""

from pathlib import Path
from utils import read_case_foamfiles, scan_case_directory, load_case_files, save_file
from services.input_writer import initial_write, build_allrun, rewrite_files
from services.case_paths import safe_case_relative_from_text
from translation.esi_translator import convert_case_to_esi_if_needed
from openfoam_target import database_path_for_config, generation_convention, uses_legacy_esi_translation

def input_writer_node(state):
    """
    InputWriter node: Generate the complete OpenFOAM foamfile.
    
    Args:
        state: The current state containing all necessary information
    """

    state = dict(state)
    state["user_requirement"] += (
        f"\n<actual_mesh>{state.get('mesh_info') or {}}</actual_mesh>\n"
        "Mesh operations belong to Allmesh and Meshing. When writing Allrun, "
        "preserve its solver-only stage; do not add mesh generation, conversion, "
        "checks, modifications or cleanup. Physical field preparation and "
        "solver-parallel decomposition remain in Allrun."
    )
    script = state.get("solver_allrun")
    if script:
        save_file(str(Path(state["case_dir"]) / "Allrun"), script)
        state.update(load_case_files(state))
    mode = state["input_writer_mode"]
    created_allrun = False
    
    if state.get("case_origin") == "imported" and state.get("requires_input_writer") is False:
        out = {}
    elif mode == "rewrite":
        out = _rewrite_mode(state) if (state.get("rewrite_plan") or {}).get("target_files") else {}
    else:
        out = _initial_write_mode(state)
        if not out.get("error_logs"):
            # Preserve explicit creates even after pending tasks are consumed.
            created_allrun = any(
                str(Path(t["folder_name"]) / t["file_name"]) == "Allrun"
                for t in state["subtasks"]
            )
            if state.get("pending_solver_subtasks"):
                # Consume successful creates even if the following edits fail.
                out.update(pending_solver_subtasks=[], subtasks=[], input_writer_mode="rewrite")
            if (state.get("rewrite_plan") or {}).get("target_files"):
                out.update(_rewrite_mode({**state, **out}))
    if out.get("error_logs"):
        return out
    if script is not None:
        out["solver_allrun"] = None
    if (not (Path(state["case_dir"]) / "Allrun").is_file()
            or (state.get("case_origin") == "imported" and script == ""
                and not created_allrun
                and not any(
                    safe_case_relative_from_text(t["file"]).as_posix() == "Allrun"
                    for t in (state.get("rewrite_plan") or {}).get("target_files", [])
                ))):
        _build_solver_allrun({**state, **out})
    out.update(load_case_files({**state, **out}))
    return out


def _build_solver_allrun(state):
    config = state["config"]
    return build_allrun(
        case_dir=state["case_dir"], database_path=str(database_path_for_config(config)),
        searchdocs=config.searchdocs, dir_structure=state.get("dir_structure") or {},
        case_info=state.get("case_info") or f"solver: {state.get('case_solver') or ''}",
        allrun_reference=state.get("allrun_reference") or "",
        mesh_type=state.get("mesh_type"), mesh_commands=[],
        user_requirement=state["user_requirement"], llm_service=state.get("llm_service"),
        config=config, openfoam_fork=generation_convention(config), phase="solver",
    )

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
        openfoam_fork=generation_convention(state["config"]),
        case_solver=state.get("case_solver", ""),
        llm_service=state.get("llm_service"),
    )
    print("</input_writer>")
    
    if uses_legacy_esi_translation(state["config"]):
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
        openfoam_fork=generation_convention(config),
        generation_mode=getattr(config, "input_writer_generation_mode", "sequential_dependency"),
        similar_case_advice=state.get("similar_case_advice"),
        reuse_generated_dir=getattr(config, "reuse_generated_dir", ""),
        llm_service=state.get("llm_service"),
        config=config,
    )

    dir_structure = write_out["dir_structure"]
    foamfiles = write_out["foamfiles"]

    # Imported scripts have already been split by solver_planner.
    explicit_allrun = any(
        str(Path(t["folder_name"]) / t["file_name"]) == "Allrun" for t in state["subtasks"]
    ) or any(
        safe_case_relative_from_text(t["file"]).as_posix() == "Allrun"
        for t in (state.get("rewrite_plan") or {}).get("target_files", [])
    )
    if not state.get("solver_allrun") and not explicit_allrun:
        _build_solver_allrun({**state, **write_out})

    print("</input_writer>")

    if uses_legacy_esi_translation(config):
        convert_case_to_esi_if_needed(state["case_dir"], config)
    
    # Rescan the directory and foam files to reflect any translations
    dir_structure = scan_case_directory(state["case_dir"])
    foamfiles = read_case_foamfiles(state["case_dir"], dir_structure)

    return {
        "dir_structure": dir_structure,
        "commands": [],
        "foamfiles": foamfiles,
    }

