"""Thin LangGraph adapter for OpenFOAM input generation and rewriting."""

from utils import scan_case_directory
from services.input_writer import initial_write, build_allrun, rewrite_files
from services.review_history import ReviewHistory
from services.case_inputs import read_case_inputs
from translation.esi_translator import convert_case_to_esi_if_needed
from openfoam_target import database_path_for_config, generation_convention, uses_legacy_esi_translation

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
    history = ReviewHistory(state["case_dir"], state.get("case_import_report_dir"))
    static_repair = state.get("repair_source") == "static_review"
    write_dir = state.get("case_input_source_dir") or state["case_dir"]
    if state.get("case_input_source_dir"):
        foamfiles, omitted = read_case_inputs(write_dir)
        directory = scan_case_directory(write_dir)
    else:
        foamfiles, omitted = state.get("foamfiles"), state.get("input_omissions")
        directory = state.get("dir_structure", {})
    out = rewrite_files(
        case_dir=write_dir,
        error_logs=state.get("error_logs", []),
        review_analysis=state.get("review_analysis", ""),
        rewrite_plan=state.get("rewrite_plan"),
        user_requirement=state.get("user_requirement", ""),
        foamfiles=foamfiles,
        dir_structure=directory,
        openfoam_fork=generation_convention(state["config"]),
        case_solver=state.get("case_solver", ""),
        llm_service=state.get("llm_service"),
        review_context=history.context(events={"file_repair", "static_review_exit"}) if static_repair else "",
        tutorial_reference=state.get("tutorial_reference") or "",
        allrun_reference=state.get("allrun_reference") or "",
        omitted_files=omitted,
    )
    print("</input_writer>")
    
    if uses_legacy_esi_translation(state["config"]):
        convert_case_to_esi_if_needed(write_dir, state["config"])
    
    # Rescan the directory and foam files to reflect any translations
    out["dir_structure"] = scan_case_directory(write_dir)
    out["foamfiles"], out["input_omissions"] = read_case_inputs(write_dir)
    out["last_repair"] = {
        "updated_files": out.get("updated_files", []),
        "file_diffs": out.get("file_diffs", []),
    }

    history.append(
        "file_repair", source=state.get("repair_source"), runtime_loop=state.get("loop_count", 0),
        static_iteration=state.get("static_review_retries", 0) if static_repair else None,
        review_analysis=state.get("review_analysis"), updated_files=out.get("updated_files", []),
        file_diffs=out.get("file_diffs", []), input_source_dir=state.get("case_input_source_dir"),
    )
    if static_repair:
        reason = None
        if not out.get("updated_files"):
            reason = "no_changes"
        elif state.get("static_review_retries", 0) >= state["config"].max_static_review_retries:
            reason = "iteration_limit"
        if reason:
            history.append("static_review_exit", reason=reason, runtime_loop=state.get("loop_count", 0))
            out.update({"static_review_complete": True, "repair_source": None})
    elif state.get("error_logs") and not out.get("updated_files") and not state.get("repairing_mesh"):
        print("Repair made no file changes. Stopping without rerunning the case.")
        out.update({
            "error_logs": state.get("error_logs") or [],
            "workflow_status": "failed",
            "termination_reason": "repair_made_no_progress",
            "workflow_message": "Repair made no file changes; execution was not started.",
        })

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
        case_info=state.get("case_info", ""),
        openfoam_fork=generation_convention(config),
        generation_mode=getattr(config, "generation_mode", "sequential_dependency"),
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
    advice = state.get("similar_case_advice") or {}
    advice = advice.model_dump() if hasattr(advice, "model_dump") else advice
    build_allrun(
        pre_solver_steps=advice.get("pre_solver_steps", []) if isinstance(advice, dict) else [],
        case_dir=state["case_dir"],
        database_path=str(database_path_for_config(config)),
        searchdocs=config.searchdocs,
        dir_structure=dir_structure,
        case_info=state["case_info"],
        allrun_reference=state["allrun_reference"],
        mesh_type=mesh_type,
        mesh_commands=mesh_commands,
        user_requirement=state["user_requirement"],
        llm_service=state.get("llm_service"),
        config=config,
        openfoam_fork=generation_convention(config),
    )

    print("</input_writer>")

    if uses_legacy_esi_translation(config):
        convert_case_to_esi_if_needed(state["case_dir"], config)
    
    # Rescan the directory and foam files to reflect any translations
    dir_structure = scan_case_directory(state["case_dir"])
    foamfiles, omitted = read_case_inputs(state["case_dir"])

    return {
        "dir_structure": dir_structure,
        "commands": [],
        "foamfiles": foamfiles,
        "input_omissions": omitted,
    }

