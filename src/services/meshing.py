"""Mesh planning and execution behind the Meshing child graph."""

from pathlib import Path
import shlex
from typing import Literal

from pydantic import BaseModel, Field

from openfoam_target import database_path_for_config, generation_convention, runtime_openfoam_target
from utils import load_case_files, run_command, check_foam_errors
from .input_writer import initial_write, build_allrun, rewrite_files
from .mesh import copy_custom_mesh, handle_gmsh_mesh
from .mesh_context import mesh_state, mesh_info, is_mesh_file
from .plan import SubtaskModel, setup_generated_case
from .review import review_case


class MeshingPlan(BaseModel):
    mesh_type: Literal["standard_mesh", "gmsh_mesh", "custom_mesh"]
    reuse_mesh: bool = False
    mesh_subtasks: list[SubtaskModel] = Field(default_factory=list)
    mesh_commands: list[str] = Field(default_factory=list)
    solver_allrun: str = ""


def plan_meshing(state):
    preparation = setup_generated_case(state) if not state.get("case_dir") else {}
    state = {**state, **preparation}
    case = Path(state["case_dir"])
    original = case / "Allrun"
    script = original.read_text(encoding="utf-8") if original.is_file() else ""
    status = mesh_state(case)
    prompt = (
        f"<user_requirement>{state['user_requirement']}</user_requirement>\n"
        f"<custom_mesh_path>{state.get('custom_mesh_path') or ''}</custom_mesh_path>\n"
        f"<mesh_state>{status}</mesh_state>\n"
        f"<case_origin>{state.get('case_origin')}</case_origin>\n"
        f"<explicit_requirement>{state.get('user_requirement_explicit', True)}</explicit_requirement>\n"
        f"<case_files>{state.get('dir_structure') or {}}</case_files>\n"
        f"<case_contents>{state.get('foamfiles') or ''}</case_contents>\n"
        f"<reference>{state.get('tutorial_reference') or ''}</reference>\n"
        f"<allrun_reference>{state.get('allrun_reference') or ''}</allrun_reference>\n"
        f"<existing_allrun>{script}</existing_allrun>\n"
        f"<repair_analysis>{state.get('review_analysis') or ''}</repair_analysis>\n"
        f"<repair_plan>{state.get('rewrite_plan') or {}}</repair_plan>\n"
        f"<openfoam_target>{generation_convention(state['config'])}</openfoam_target>"
    )
    plan = state["llm_service"].invoke(
        prompt,
        "Plan only the mesh stage. Select standard_mesh for OpenFOAM mesh generation, "
        "gmsh_mesh for Gmsh generation, or custom_mesh for an uploaded Gmsh ASCII 2.2 "
        "mesh or an existing case polyMesh. An explicit custom_mesh_path selects custom_mesh. "
        "Reuse existing polyMesh unless the user requests replacement or remeshing. "
        "Do not regenerate it merely because the original Allrun contains mesh generation. "
        "For standard_mesh list only mesh dictionaries, geometry files and ordered mesh "
        "commands as utility invocations, including their arguments; shell wrappers "
        "belong in Allmesh. Reuse existing dictionaries; list missing "
        "files or files requiring changes, not physical solver files. "
        "For gmsh_mesh the backend generates, converts and checks its mesh itself; "
        "for uploaded custom_mesh the backend converts and checks its mesh itself. "
        "List only additional requested mesh operations for those backends. "
        "For reused polyMesh retain existing requested checks or mesh operations, but "
        "omit generation and conversion commands. Do not add new checks. "
        "When existing_allrun is present, return its solver-only remainder in solver_allrun. "
        "Preserve its solver commands, arguments, setup, functions and necessary field "
        "preparation. Remove all mesh operations and mesh cleanup, including operations "
        "inside helper functions. Mesh-parallel preparation belongs to the mesh stage; "
        "solver-parallel preparation stays in solver_allrun. Return complete runnable "
        "shell code, without markdown. When no existing Allrun exists, use an empty string. "
        "During repair, use the review and rewrite plan to select the backend and retry "
        "commands, including regeneration if required. List only missing files in "
        "mesh_subtasks; existing file changes are applied by the rewrite plan.",
        pydantic_obj=MeshingPlan,
    )
    if state.get("custom_mesh_path"):
        plan.mesh_type, plan.reuse_mesh = "custom_mesh", False
    elif (status == "existing-polyMesh" and not state.get("user_requirement_explicit", True)
          and not state.get("repairing_mesh")):
        plan.mesh_type, plan.reuse_mesh = "custom_mesh", True
    manifest = state.get("case_import_manifest")
    if manifest and manifest.mesh_state == "unprocessed":
        manifest.mesh_state = status
        from .case_import import _write_json
        _write_json(Path(state["case_import_report_dir"]) / "case_manifest.json", manifest.to_dict())
    return {
        **preparation,
        "mesh_type": plan.mesh_type,
        "mesh_subtasks": [task.model_dump() for task in plan.mesh_subtasks],
        "mesh_commands": plan.mesh_commands,
        "requires_meshing": not plan.reuse_mesh,
        "solver_allrun": state.get("solver_allrun") if state.get("repairing_mesh") else plan.solver_allrun or None,
        "allrun_reference": state.get("allrun_reference") if state.get("repairing_mesh") else script or state.get("allrun_reference"),
        "case_context": {**(state.get("case_context") or {}), "mesh_state": status},
    }


def _write_mesh_files(state):
    tasks = list(state.get("mesh_subtasks") or [])
    case = Path(state["case_dir"])
    if not (case / "system" / "controlDict").is_file():
        if not any(t["file_name"] == "controlDict" for t in tasks):
            tasks.append({"folder_name": "system", "file_name": "controlDict"})
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
    if not commands:
        return {"error_logs": []}
    config = state["config"]
    case = Path(state["case_dir"])
    if (state.get("repairing_mesh") and (case / "Allmesh").is_file()
            and any(t["file"] == "Allmesh" for t in (state.get("rewrite_plan") or {}).get("target_files", []))):
        script = {"allrun_path": str(case / "Allmesh")}
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
    logs = [f"log.{shlex.split(command)[0]}" for command in commands]
    for name in logs:
        (case / name).unlink(missing_ok=True)
    result = run_command(
        script["allrun_path"], str(case / "Allmesh.out"), str(case / "Allmesh.err"),
        str(case), config.max_time_limit, openfoam_target=runtime_openfoam_target(config),
    )
    errors = check_foam_errors(str(case), [name for name in logs if (case / name).is_file()])
    if result.get("returncode") or result.get("timed_out"):
        output = "\n".join(
            (case / name).read_text(encoding="utf-8", errors="replace")
            for name in ("Allmesh.out", "Allmesh.err") if (case / name).is_file()
        )
        errors.append({"file": "Allmesh", "error_content": output or str(result)})
    return {"error_logs": errors}


def standard_meshing(state):
    if state.get("mesh_subtasks") or not state.get("repairing_mesh"):
        state = {**state, **_write_mesh_files(state)}
    return {**_run_mesh_commands(state, state.get("mesh_commands") or []), "mesh_subtasks": []}


def gmsh_meshing(state):
    result = handle_gmsh_mesh(
        state["user_requirement"], state["case_dir"], state["config"].max_loop,
        llm_service=state["llm_service"],
        openfoam_target=runtime_openfoam_target(state["config"]),
        repair_feedback=state.get("review_analysis") or "",
        retry=bool(state.get("repairing_mesh")),
    )
    if not result.get("error_logs"):
        result.update(_run_mesh_commands(state, state.get("mesh_commands") or []))
    result["mesh_commands"] = state.get("mesh_commands") or []
    return result


def imported_meshing(state):
    if not state.get("custom_mesh_path") and mesh_state(state["case_dir"]) == "existing-polyMesh":
        return _run_mesh_commands(state, state.get("mesh_commands") or [])
    result = copy_custom_mesh(
        state["custom_mesh_path"], state["user_requirement"], state["case_dir"],
        llm_service=state["llm_service"],
        openfoam_target=runtime_openfoam_target(state["config"]),
        repair_feedback=state.get("review_analysis") or "",
        retry=bool(state.get("repairing_mesh")),
    )
    if not result.get("error_logs"):
        commands = list(dict.fromkeys((result.get("mesh_commands") or []) + (state.get("mesh_commands") or [])))
        result.update(_run_mesh_commands(state, commands))
    result["mesh_commands"] = state.get("mesh_commands") or []
    return result


def _rewrite_mesh(state):
    if not (state.get("rewrite_plan") or {}).get("target_files"):
        return {}
    return rewrite_files(
        case_dir=state["case_dir"], error_logs=state.get("error_logs") or [],
        review_analysis=state.get("review_analysis") or "", rewrite_plan=state["rewrite_plan"],
        user_requirement=state["user_requirement"], foamfiles=state.get("foamfiles"),
        dir_structure=state.get("dir_structure") or {},
        openfoam_fork=generation_convention(state["config"]),
        case_solver=state.get("case_solver") or "", llm_service=state["llm_service"],
        mesh_info=state.get("mesh_info"),
    )


def execute_meshing(state, backend):
    current = dict(state)
    resume = {
        key: state.get(key) for key in
        ("input_writer_mode", "rewrite_plan", "review_analysis", "requires_input_writer")
    }
    if current.get("repairing_mesh"):
        targets = (current.get("rewrite_plan") or {}).get("target_files", [])
        mesh_targets = [t for t in targets if is_mesh_file(t["file"]) or (
            t["file"] == "system/controlDict"
            and not (Path(current["case_dir"]) / t["file"]).is_file()
        )]
        solver_targets = [t for t in targets if t not in mesh_targets]
        resume["rewrite_plan"] = {"target_files": solver_targets} if solver_targets else None
        current["rewrite_plan"] = {"target_files": mesh_targets}
        current.update(_rewrite_mesh(current))
    while True:
        try:
            current.update(backend(current))
        except Exception as error:
            current["error_logs"] = [{"file": "Allmesh", "error_content": str(error)}]
        current.update(load_case_files(current))
        if not current.get("error_logs"):
            current["mesh_info"] = mesh_info(current["case_dir"], current.get("mesh_info"))
            current["case_context"] = {
                **(current.get("case_context") or {}), "mesh_state": mesh_state(current["case_dir"]),
                "issues": [i for i in (current.get("case_context") or {}).get("issues", [])
                           if i != "No existing polyMesh or system/blockMeshDict was found."],
            }
            if current.get("case_import_report_dir"):
                from .case_import import _write_json
                _write_json(Path(current["case_import_report_dir"]) / "case_context.json", current["case_context"])
            current.update(resume)
            current.update(repairing_mesh=False, mesh_resume_state=None, error_logs=[])
            return current
        current["repairing_mesh"] = True
        current.update(review_case(current, mesh=True))
        if current.get("termination_reason"):
            return current
        current.update(plan_meshing(current))
        backend = {
            "standard_mesh": standard_meshing,
            "gmsh_mesh": gmsh_meshing,
            "custom_mesh": imported_meshing,
        }[current["mesh_type"]]
        current.update(_rewrite_mesh(current))
