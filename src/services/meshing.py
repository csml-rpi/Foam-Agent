"""Shared mesh planning and single-pass backend execution services."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from openfoam_target import database_path_for_config, generation_convention, runtime_openfoam_target
from utils import load_case_files, run_command, _EXPLICIT_FOAM_ERROR_RE
from .input_writer import initial_write, build_allrun, rewrite_files
from .mesh import copy_custom_mesh, handle_gmsh_mesh
from .plan import SubtaskModel


class MeshingPlan(BaseModel):
    mesh_type: Literal["standard_mesh", "gmsh_mesh", "custom_mesh"]
    mesh_source: str = Field(default="", description="Selected Gmsh .msh file, relative to the case or absolute; empty on custom_mesh uses the case's current polyMesh")
    mesh_subtasks: list[SubtaskModel] = Field(default_factory=list)
    mesh_commands: list[str] = Field(default_factory=list)


def plan_meshing(state):
    case = Path(state["case_dir"])
    original = case / "Allrun"
    script = original.read_text(encoding="utf-8") if original.is_file() else ""
    prompt = (
        f"<user_requirement>{state['user_requirement']}</user_requirement>\n"
        f"<custom_mesh_path>{state.get('custom_mesh_path') or ''}</custom_mesh_path>\n"
        f"<case_origin>{state.get('case_origin')}</case_origin>\n"
        f"<explicit_requirement>{state.get('user_requirement_explicit', True)}</explicit_requirement>\n"
        f"<case_files>{state.get('dir_structure') or {}}</case_files>\n"
        f"<case_contents>{state.get('foamfiles') or ''}</case_contents>\n"
        f"<reference>{state.get('tutorial_reference') or ''}</reference>\n"
        f"<allrun_reference>{state.get('allrun_reference') or ''}</allrun_reference>\n"
        f"<existing_allrun>{script}</existing_allrun>\n"
        f"<repair_analysis>{state.get('review_analysis') or ''}</repair_analysis>\n"
        f"<error_logs>{state.get('error_logs') or []}</error_logs>\n"
        f"<repair_plan>{state.get('rewrite_plan') or {}}</repair_plan>\n"
        f"<openfoam_target>{generation_convention(state['config'])}</openfoam_target>"
    )
    plan = state["llm_service"].invoke(
        prompt,
        "Plan only the mesh stage from the user requirement and actual supplied files. "
        "Do not select a simulation solver or plan physical fields. Choose standard_mesh "
        "for OpenFOAM utilities, gmsh_mesh for Gmsh generation, or custom_mesh for "
        "working with the case's existing polyMesh or converting a Gmsh .msh file. The input "
        "source does not force a backend or a reuse/regeneration decision. Select the "
        "operations that meet this request. "
        "Use mesh_source on custom_mesh to explicitly select an available Gmsh .msh "
        "file. An empty source on custom_mesh means the current "
        "case polyMesh. List the files to create or change and ordered native utility "
        "invocations with arguments; Allmesh provides shell setup. Do not automatically "
        "add blockMesh. Gmsh generates and converts its mesh; custom_mesh converts "
        "Gmsh .msh files. These backends check the converted mesh themselves; list "
        "only additional requested operations for them. Using the case's polyMesh "
        "does not run checks; include any checks you decide are needed in mesh_commands. "
        "For an existing mesh, decide checks and changes from the request and evidence. "
        "Use existing Allrun as evidence of mesh operations, preserving mesh-parallel "
        "setup as needed. Do not output the solver script; solver_planner owns it. "
        "Keep physical field preparation and solver-parallel setup out of Allmesh. "
        "Plan the startup dictionaries required by the selected utilities, including backend "
        "gmshToFoam/checkMesh calls, as stage mesh file tasks before execution. Inspect existing "
        "files and preserve them unless changes are needed. Create only missing dictionaries "
        "the selected tools require, such as controlDict or tool-dependent fvSchemes/fvSolution; "
        "the executor does not supply default dictionaries. Specify their required contents "
        "in changes. These files remain for Solver Planner to adapt to the physical problem. "
        "Requested scaling belongs in ordered mesh_commands. During repair, "
        "use feedback to reconsider the backend and commands. In mesh_subtasks declare "
        "each file's stage, action (create or modify), and concrete changes. "
        "Include modifications to existing files even on the first plan. Mesh files and "
        "Allmesh use stage mesh; any deferred physical or solver-script tasks use solver. "
        "Do not infer stages from filenames. Backend checks are already "
        "performed; additional commands are executed literally, including repetitions.",
        pydantic_obj=MeshingPlan,
    )
    source = Path(plan.mesh_source).expanduser() if plan.mesh_source else None
    if source is not None and not source.is_absolute():
        source = case / source
    changes = list((state.get("rewrite_plan") or {}).get("target_files", []))
    creates = []
    pending = list(state.get("pending_solver_subtasks") or [])
    for task in plan.mesh_subtasks:
        if task.action == "modify":
            changes.append({"file": str(Path(task.folder_name) / task.file_name),
                            "stage": task.stage,
                            "changes": task.changes})
        elif task.stage == "mesh":
            creates.append(task.model_dump())
        else:
            pending.append(task.model_dump())
    return {
        "rewrite_plan": {"target_files": changes} if changes else None,
        "pending_solver_subtasks": pending,
        "review_analysis": state.get("review_analysis") or
                           "Apply the mesh planner's explicit file changes: " + str(changes),
        "mesh_type": plan.mesh_type,
        "mesh_source": str(source) if source else None,
        "mesh_subtasks": creates,
        "mesh_commands": plan.mesh_commands,
        "allrun_reference": script or state.get("allrun_reference"),
    }


def _write_mesh_files(state):
    tasks = list(state.get("mesh_subtasks") or [])
    case = Path(state["case_dir"])
    if not tasks:
        return {}
    config = state["config"]
    return initial_write(
        case_dir=str(case), subtasks=tasks,
        user_requirement=state["user_requirement"] + "\nWrite only the mesh-stage "
        "files. Any controlDict here is the minimal bootstrap for mesh utilities.",
        tutorial_reference=(state.get("tutorial_reference") or "")
        + f"\nExisting case files; preserve settings not requested to change:\n{state.get('foamfiles') or ''}",
        case_solver=state.get("case_solver") or "",
        openfoam_fork=generation_convention(config),
        generation_mode=config.input_writer_generation_mode,
        reuse_generated_dir=config.reuse_generated_dir,
        llm_service=state["llm_service"], config=config,
    )


def _run_mesh_commands(state, commands):
    config = state["config"]
    case = Path(state["case_dir"])
    if any(t["file"] == "Allmesh" for t in (state.get("rewrite_plan") or {}).get("target_files", [])):
        script = {"allrun_path": str(case / "Allmesh")}
    elif not commands:
        return {"error_logs": []}
    else:
        script = build_allrun(
            case_dir=state["case_dir"],
            database_path=str(database_path_for_config(config)), searchdocs=config.searchdocs,
            dir_structure=state.get("dir_structure") or {},
            case_info=state.get("case_info") or "",
            allrun_reference=state.get("allrun_reference") or "",
            mesh_type=state["mesh_type"], mesh_commands=commands,
            user_requirement=state["user_requirement"], llm_service=state["llm_service"],
            config=config, openfoam_fork=generation_convention(config), phase="mesh",
        )
    before, observations = _mesh_log_snapshot(case)
    result = run_command(
        script["allrun_path"], str(case / "Allmesh.out"), str(case / "Allmesh.err"),
        str(case), config.max_time_limit, openfoam_target=runtime_openfoam_target(config),
    )
    after, failures = _mesh_log_snapshot(case)
    observations.extend(failures)
    names = {name for name, signature in after.items() if before.get(name) != signature}
    names.update({"Allmesh.out", "Allmesh.err"})
    output, errors = {}, []
    for name in sorted(names):
        try:
            output[name] = (case / name).read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            observations.append(f"Could not read {name}: {error}")
            continue
        match = _EXPLICIT_FOAM_ERROR_RE.search(output[name])
        if match:
            errors.append({"file": name, "error_content": match.group(1).strip()})
    evidence = {"command_result": result, "logs": output, "observations": observations}
    if result.get("returncode") or result.get("timed_out"):
        errors.append({"file": "Allmesh", "error_content": str(evidence)})
    return {"error_logs": errors, "mesh_execution": evidence}


def _mesh_log_snapshot(case):
    files, observations = {}, []
    try:
        for path in case.rglob("*"):
            if path.name.startswith("log") or path.suffix.lower() in {".log", ".out", ".err"}:
                try:
                    if path.is_file():
                        stat = path.stat()
                        files[path.relative_to(case).as_posix()] = (stat.st_size, stat.st_mtime_ns)
                except OSError as error:
                    observations.append(f"Could not inspect {path}: {error}")
    except OSError as error:
        observations.append(f"Could not scan mesh logs: {error}")
    return files, observations


def standard_meshing(state):
    return {**_run_mesh_commands(state, state.get("mesh_commands") or []), "mesh_info": None}


def gmsh_meshing(state):
    result = handle_gmsh_mesh(
        state["user_requirement"], state["case_dir"], state["config"].max_loop,
        llm_service=state["llm_service"],
        openfoam_target=runtime_openfoam_target(state["config"]),
        repair_feedback=state.get("review_analysis") or "",
        retry=bool(state.get("repairing_mesh")),
    )
    commands = state.get("mesh_commands") or []
    if not result.get("error_logs"):
        result.update(_run_mesh_commands(state, commands))
        if commands and result.get("mesh_info"):
            result["mesh_info"]["check_mesh_passed"] = None
    result["mesh_commands"] = state.get("mesh_commands") or []
    result["mesh_subtasks"] = []
    return result


def imported_meshing(state):
    source = state.get("mesh_source")
    if not source:
        return _run_mesh_commands(state, state.get("mesh_commands") or [])
    result = copy_custom_mesh(
        source, state["case_dir"],
        openfoam_target=runtime_openfoam_target(state["config"]),
        retry=bool(state.get("repairing_mesh")),
    )
    if not result.get("error_logs"):
        commands = (result.get("mesh_commands") or []) + (state.get("mesh_commands") or [])
        result.update(_run_mesh_commands(state, commands))
        if commands and result.get("mesh_info"):
            result["mesh_info"]["check_mesh_passed"] = None
    result["mesh_commands"] = state.get("mesh_commands") or []
    return result


def _rewrite_mesh(state):
    if not (state.get("rewrite_plan") or {}).get("target_files"):
        return {}
    return rewrite_files(
        case_dir=state["case_dir"], error_logs=state.get("error_logs") or [],
        review_analysis=state.get("review_analysis") or "", rewrite_plan=state["rewrite_plan"],
        user_requirement=state["user_requirement"]
        + f"\n<actual_mesh>{state.get('mesh_info') or {}}</actual_mesh>",
        foamfiles=state.get("foamfiles"),
        dir_structure=state.get("dir_structure") or {},
        openfoam_fork=generation_convention(state["config"]),
        case_solver=state.get("case_solver") or "", llm_service=state["llm_service"],
    )


class MeshRepairImpact(BaseModel):
    replan_solver: bool
    reason: str


def execute_meshing(state, backend):
    current = dict(state)
    result = {}
    resume = state.get("mesh_resume_state") or {
        key: state.get(key) for key in
        ("input_writer_mode", "review_analysis", "requires_input_writer")
    }
    previous_mesh = resume.get("previous_mesh", state.get("mesh_info"))
    pending = list(resume.get("pending", []))
    targets = (current.get("rewrite_plan") or {}).get("target_files", [])
    pending.extend(t for t in targets if t["stage"] != "mesh")
    current["rewrite_plan"] = {"target_files": [t for t in targets if t["stage"] == "mesh"]}
    try:
        written = _write_mesh_files(current)
        result.update(written)
        current.update(written)
        current["mesh_subtasks"] = []
        result["mesh_subtasks"] = []
        if not written.get("error_logs"):
            rewritten = _rewrite_mesh(current)
            result.update(rewritten)
            current.update(rewritten)
            if not rewritten.get("error_logs"):
                result.update(backend(current))
    except Exception as error:
        native_output = [str(error), str(getattr(error, "stdout", "") or ""),
                         str(getattr(error, "stderr", "") or "")]
        result["error_logs"] = [{"file": "Allmesh", "error_content": "\n".join(native_output)}]
    current.update(result)
    result.update(load_case_files(current))
    current.update(result)
    if not current.get("error_logs"):
        mesh_dir = Path(current["case_dir"]) / "constant/polyMesh"
        boundary = mesh_dir / "boundary"
        info = {
            **(current.get("mesh_info") or {}),
            "poly_mesh_path": str(mesh_dir),
            "boundary_text": boundary.read_text(encoding="utf-8", errors="replace")
            if boundary.is_file() else "",
        }
        result["mesh_info"] = info
        impact_reason = ""
        if current.get("solver_planned"):
            impact = current["llm_service"].invoke(
                f"<user_requirement>{current['user_requirement']}</user_requirement>\n"
                f"<previous_mesh>{previous_mesh}</previous_mesh>\n"
                f"<actual_mesh>{info}</actual_mesh>\n"
                f"<files>{current.get('foamfiles')}</files>\n"
                f"<execution>{current.get('mesh_execution')}</execution>\n"
                f"<review_analysis>{current.get('review_analysis')}</review_analysis>\n"
                f"<pending_repairs>{pending}</pending_repairs>",
                "As the mesh planner, confirm the impact of the completed mesh work on the "
                "existing physical plan. Decide whether Solver Planner must reconsider physics, "
                "solver settings or boundary conditions. Base the decision on actual results "
                "and the request, not filenames or a fixed change checklist. Pending physical "
                "file repairs remain available in either route.",
                pydantic_obj=MeshRepairImpact,
            )
            result["replan_solver"] = impact.replan_solver
            impact_reason = impact.reason
        result.update({key: resume.get(key) for key in
                        ("input_writer_mode", "review_analysis", "requires_input_writer")})
        if impact_reason:
            result["review_analysis"] = (result.get("review_analysis") or "") + "\n" + impact_reason
        if current.get("pending_solver_subtasks"):
            result.update(subtasks=current["pending_solver_subtasks"],
                           input_writer_mode="initial", requires_input_writer=True)
        result.update(rewrite_plan={"target_files": pending} if pending else None,
                       repairing_mesh=False, mesh_resume_state=None, error_logs=[])
        return result
    result.update(
        repairing_mesh=True,
        rewrite_plan={"target_files": pending + current["rewrite_plan"]["target_files"]},
        mesh_resume_state={**resume, "pending": pending, "previous_mesh": previous_mesh},
    )
    return result
