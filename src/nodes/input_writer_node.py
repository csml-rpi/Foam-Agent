"""Thin LangGraph adapter for OpenFOAM input generation and rewriting."""

from pathlib import Path
from utils import read_case_foamfiles, scan_case_directory, load_case_files
from services.input_writer import initial_write, build_allrun, rewrite_files
from services.case_paths import safe_case_relative_path
from services.stage_plan import check_stage_operations, check_stage_inputs, check_stage_script, stage_outcome, pending_file_creates, unwritten_file_errors
from translation.esi_translator import convert_case_to_esi_if_needed
from openfoam_target import database_path_for_config, generation_convention, uses_legacy_esi_translation

def input_writer_node(state):
    """
    InputWriter node: Generate the complete OpenFOAM foamfile.
    
    Args:
        state: The current state containing all necessary information
    """

    errors = check_stage_operations(state["solver_operations"], "solver")
    if errors:
        outcome = {"executed": False, "error_logs": errors}
        return {"error_logs": errors, "solver_execution": outcome,
                **stage_outcome(state, "solver", outcome)}
    state = dict(state)
    state['solver_script_dirty'] = state['solver_script_dirty'] or bool(
        check_stage_script(state['case_dir'], state['solver_operations'], 'solver'))
    state["user_requirement"] += (
        f"\n<actual_mesh>{state.get('mesh_info') or {}}</actual_mesh>\n"
        f"<mesh_execution>{state.get('mesh_execution')}</mesh_execution>\n"
        f"<mesh_boundary_plan>{state.get('mesh_boundary_plan') or []}</mesh_boundary_plan>\n"
        "Mesh operations belong to Allmesh and Meshing. When writing Allrun, "
        "preserve its solver-only stage; do not add mesh generation, conversion, "
        "checks, modifications or cleanup. Physical field preparation and "
        "solver-parallel decomposition remain in Allrun."
    )
    mode = state["input_writer_mode"]
    created_allrun = False
    
    if mode == "rewrite":
        out = _rewrite_mode(state) if (state.get("rewrite_plan") or {}).get("target_files") else {}
        if (not (state.get('rewrite_plan') or {}).get('target_files') and state.get('error_logs')
                and not state.get('repairing_mesh') and not state['solver_script_dirty']
                and not _preflight_plan_repaired(state)):
            out.update(error_logs=state['error_logs'], workflow_status='failed',
                       termination_reason='repair_made_no_progress')
    else:
        out = _initial_write_mode(state)
        receipts = out['written_files']
        out.update(
            subtasks=pending_file_creates(state['subtasks'], receipts),
            pending_solver_subtasks=pending_file_creates(state.get('pending_solver_subtasks') or [], receipts),
        )
        if not out.get("error_logs"):
            created_allrun = 'Allrun' in receipts
            out['input_writer_mode'] = 'rewrite' if not out['subtasks'] else 'initial'
            if (state.get("rewrite_plan") or {}).get("target_files"):
                out.update(_rewrite_mode({**state, **out}))
    if out.get("error_logs"):
        return out
    script_written = created_allrun or 'Allrun' in out.get('written_files', [])
    if (not (Path(state["case_dir"]) / "Allrun").is_file()
            or (state['solver_script_dirty'] and not script_written)):
        _build_solver_allrun({**state, **out})
    out.update(load_case_files({**state, **out}))
    out.update(error_logs=[], rewrite_plan=None, solver_script_dirty=False, cross_stage_request=None,
               pending_solver_operations=[])
    return out


def _preflight_plan_repaired(state):
    """A proven operation/dependency correction can resolve preflight without file edits."""
    plan = state.get('rewrite_plan') or {}
    if plan.get('execution_plan_repaired') and not check_stage_operations(state['solver_operations'], 'solver'):
        return True
    return bool(plan.get('dependencies_changed')) and not check_stage_inputs(
        state['case_dir'], state['solver_operations'], 'solver')


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
        operations=state["solver_operations"],
    )

def _rewrite_mode(state):
    """Rewrite mode: delegate to service to modify files based on review analysis."""
    print("<input_writer mode=\"rewrite\">")
    out = rewrite_files(
        case_dir=state["case_dir"],
        error_logs=state.get("error_logs", []),
        review_analysis=state.get("review_analysis", ""),
        rewrite_plan=state.get("rewrite_plan"),
        user_requirement=state.get("user_requirement", "")
        + f"\n<solver_reference>{state.get('tutorial_reference') or ''}</solver_reference>",
        foamfiles=state.get("foamfiles"),
        dir_structure=state.get("dir_structure", {}),
        openfoam_fork=generation_convention(state["config"]),
        case_solver=state.get("case_solver", ""),
        allrun_reference=state.get("allrun_reference") or "",
        operations=state["solver_operations"],
        llm_service=state.get("llm_service"),
    )
    print("</input_writer>")
    
    if uses_legacy_esi_translation(state["config"]):
        convert_case_to_esi_if_needed(state["case_dir"], state["config"])
    
    # Rescan the directory and foam files to reflect any translations
    out["dir_structure"] = scan_case_directory(state["case_dir"])
    out["foamfiles"] = read_case_foamfiles(state["case_dir"], out["dir_structure"])

    if (state.get("error_logs") and not out.get("updated_files") and not state.get("repairing_mesh")
            and not state["solver_script_dirty"] and not _preflight_plan_repaired(state)):
        print("Repair made no file changes. Stopping without rerunning the case.")
        out.update({
            "error_logs": state["error_logs"] + out.get('error_logs', []),
            "workflow_status": "failed",
            "termination_reason": "repair_made_no_progress",
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
        allrun_reference=state.get("allrun_reference") or "",
        allrun_phase="solver",
        operations=state["solver_operations"],
        openfoam_fork=generation_convention(config),
        generation_mode=getattr(config, "input_writer_generation_mode", "sequential_dependency"),
        similar_case_advice=state.get("similar_case_advice"),
        reuse_generated_dir=getattr(config, "reuse_generated_dir", ""),
        llm_service=state.get("llm_service"),
        config=config,
    )
    pending = pending_file_creates(state['subtasks'], write_out['written_files'])
    if pending and not write_out.get('error_logs'):
        write_out['error_logs'] = unwritten_file_errors(
            safe_case_relative_path(t['folder_name'], t['file_name']).as_posix() for t in pending)
    if write_out.get('error_logs'):
        return {**write_out, **load_case_files(state)}

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
        "written_files": write_out['written_files'],
    }

