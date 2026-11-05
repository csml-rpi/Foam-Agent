"""Shared mesh planning and single-pass backend execution services."""

from pathlib import Path
import shutil
import tempfile
import shlex
import subprocess
import re
import os
from typing import Literal

from pydantic import BaseModel, Field

from openfoam_target import database_path_for_config, generation_convention, runtime_openfoam_target
from utils import load_case_files, run_command, run_openfoam_utility, _EXPLICIT_FOAM_ERROR_RE
from .input_writer import initial_write, build_allrun, rewrite_files
from .case_paths import safe_case_relative_from_text, safe_case_relative_path
from .mesh import copy_custom_mesh, handle_gmsh_mesh, _FAILED_MESH_CHECKS_RE
from .plan import SubtaskModel, retrieve_mesh_reference
from models import StageOperation
from .stage_plan import OPERATION_CONTRACT, MESH_STAGE_CONTRACT, mesh_operation_updates, check_mesh_plan, check_stage_inputs, check_stage_script, script_commands, replace_pending_tasks, resolve_file_tasks, pending_file_creates, unwritten_file_errors, stage_outcome
from .run_hpc import HpcWorkflowStop


_MESH_FILES = ("points", "faces", "owner", "neighbour", "boundary")
_MESH_SOURCE_SUFFIXES = {
    ".msh", ".stl", ".obj", ".ftr", ".nas", ".vtk", ".vtp", ".ply", ".off", ".geo", ".gz", ".emesh",
}


class FinalReconstruction(BaseModel):
    sources: list[str] = Field(min_length=1, description="Exact case-relative partition constant/polyMesh directories, including processorN, processorsN or processorsN_start-end layouts; not glob patterns")
    argv: list[str] = Field(min_length=1, description="Native reconstruction argv producing constant/polyMesh from the declared partition sources, e.g. reconstructParMesh -constant")
    target: Literal["constant/polyMesh"] = "constant/polyMesh"


class MeshBoundaryPlan(BaseModel):
    patch: str
    location: str | None
    role: str | None
    basis: Literal["user", "input", "reference", "inferred"]
    evidence: str


class MeshingPlan(BaseModel):
    mesh_type: Literal["standard_mesh", "gmsh_mesh", "custom_mesh"]
    mesh_script_action: Literal["none", "run_existing", "build_and_run"]
    boundary_plan: list[MeshBoundaryPlan]
    mesh_source: str = Field(default="", description="Selected Gmsh .msh file, relative to the case or absolute; empty on custom_mesh uses the case's current polyMesh")
    mesh_subtasks: list[SubtaskModel] = Field(default_factory=list)
    operations: list[StageOperation]
    final_reconstruction: FinalReconstruction | None = None


def _unique_pending_repairs(tasks):
    """Remove stored copies of a repair, retaining distinct instructions per file."""
    return replace_pending_tasks([], tasks)


def plan_meshing(state):
    reference = state.get("mesh_reference")
    if reference is None:
        reference = retrieve_mesh_reference(state)
        print(f"<mesh_reference>{reference['case']}: {reference['reason']}</mesh_reference>")
    case = Path(state["case_dir"])
    original = case / "Allrun"
    script = original.read_text(encoding="utf-8") if original.is_file() else ""
    # Include root-level mesh sources and nested geometry, using current files
    # rather than the import snapshot. Archived attempts are not input sources.
    files = sorted(
        path.relative_to(case).as_posix() for path in case.rglob("*")
        if path.is_file() and ".foamagent" not in path.relative_to(case).parts
    )
    prompt = (
        f"<user_requirement>{state['user_requirement']}</user_requirement>\n"
        f"<custom_mesh_path>{state.get('custom_mesh_path') or ''}</custom_mesh_path>\n"
        f"<case_origin>{state.get('case_origin')}</case_origin>\n"
        f"<explicit_requirement>{state.get('user_requirement_explicit', True)}</explicit_requirement>\n"
        f"<case_files>{files}</case_files>\n"
        f"<case_contents>{state.get('foamfiles') or ''}</case_contents>\n"
        f"<mesh_candidates>{mesh_candidates(case)}</mesh_candidates>\n"
        f"<actual_mesh>{state.get('mesh_info') or {}}</actual_mesh>\n"
        f"<mesh_reference>{reference}</mesh_reference>\n"
        f"<previous_boundary_plan>{state.get('mesh_boundary_plan') or []}</previous_boundary_plan>\n"
        f"<previous_mesh_execution>{state.get('mesh_execution') or {}}</previous_mesh_execution>\n"
        f"<existing_allmesh>{(case / 'Allmesh').read_text() if (case / 'Allmesh').is_file() else ''}</existing_allmesh>\n"
        f"<existing_allrun>{script}</existing_allrun>\n"
        f"<repair_analysis>{state.get('review_analysis') or ''}</repair_analysis>\n"
        f"<cross_stage_request>{state.get('cross_stage_request')}</cross_stage_request>\n"
        f"<previous_operations>{state.get('mesh_operations')}</previous_operations>\n"
        f"<pending_solver_operations>{state['pending_solver_operations']}</pending_solver_operations>\n"
        f"<error_logs>{state.get('error_logs') or []}</error_logs>\n"
        f"<repair_plan>{state.get('rewrite_plan') or {}}</repair_plan>\n"
        f"<openfoam_target>{generation_convention(state['config'])}</openfoam_target>"
    )
    plan = state["llm_service"].invoke(
        prompt,
        OPERATION_CONTRACT + MESH_STAGE_CONTRACT + " Prepare the final mesh in constant/polyMesh for a single-region case, or constant/<region>/polyMesh for every active region declared by native constant/regionProperties. "
        "An unused assembly mesh at constant/polyMesh does not replace active regional meshes. "
        "Time-directory or unreconstructed parallel meshes require explicit overwrite/reconstruction operations. "
        "For final parallel reconstruction of a single-region root mesh, declare final_reconstruction with exact partition source paths "
        "and native argv that reconstructs them into constant/polyMesh. This runs separately AFTER all "
        "other mesh operations. Do not also put that final reconstruction in script operations or Allmesh; "
        "use normal file tasks to adjust an existing script if needed. Before this operation the executor "
        "requires complete declared sources, removes the old serial target, and checks the newly written "
        "target. Do not use final_reconstruction for commands that depend on the old serial target. "
        "Leave it null for serial reuse or modifications. Other time/parallel candidates must be explicitly "
        "resolved by the plan; they are not automatically deleted or selected by timestamp. "
        "For active regional meshes, leave final_reconstruction null and declare native reconstruction "
        "as ordered script operations with explicit region arguments, followed by explicit resolution "
        "of the partition outputs before handoff. The root-only final_reconstruction cannot target regions. "
        "Plan only the mesh stage from the user requirement and actual supplied files. "
        "Declare mesh_script_action every time: none means no additional Allmesh execution "
        "after the backend; run_existing executes the existing or explicitly written Allmesh; "
        "build_and_run builds Allmesh from ordered script operations and executes it. File edits alone "
        "do not rerun commands. When fixing a utility input, explicitly plan its rerun, "
        "including run_existing when the existing Allmesh already has the right operations. "
        "Record boundary_plan with patch, location, physical role, basis and evidence. "
        "Use basis user for explicit requirements, input for supplied file facts, reference "
        "for tutorial assumptions, and inferred for your own assumptions. Unknown locations "
        "or roles are null. Distinguish proposed layout from the existing actual boundary. "
        "Evaluate the mesh reference for relevance; do not copy incompatible geometry, "
        "dimensions or boundary assumptions, or infer a role from a patch name alone. "
        "Do not select a simulation solver or generate physical fields in this stage. "
        "Record deferred physical requirements for Solver Planner instead. Choose standard_mesh "
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
        "does not run checks; include any checks you decide are needed in script operations. "
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
        "Requested scaling belongs in ordered script operations. During repair, "
        "use feedback to reconsider the backend and commands. In mesh_subtasks declare "
        "each file's stage, action (create or modify), and concrete changes. "
        "Include modifications to existing files even on the first plan. Mesh files and "
        "Allmesh use stage mesh; any deferred physical or solver-script tasks use solver. "
        "Do not infer stages from filenames. Native Gmsh and .msh converter backends perform "
        "their checks; Standard Meshing executes only declared script operations. Additional "
        "commands execute literally, including repetitions. For run_existing, declare the "
        "complete actual Allmesh shell program as a script operation, including its retained "
        "setup and helpers. For build_and_run, declare literal ordered shell fragments, "
        "including any setup, helper definitions and unrelated content that must be retained.",
        pydantic_obj=MeshingPlan,
    )
    operation_update = mesh_operation_updates(state, [op.model_dump() for op in plan.operations])
    source = Path(plan.mesh_source).expanduser() if plan.mesh_source else None
    if source is not None and not source.is_absolute():
        source = case / source
    old_changes = replace_pending_tasks(
        (state.get('mesh_resume_state') or {}).get('pending', []),
        (state.get("rewrite_plan") or {}).get("target_files", []))
    creates, changes = resolve_file_tasks(
        [*(state.get("mesh_subtasks") or []), *(state.get("pending_solver_subtasks") or [])],
        old_changes, [task.model_dump() for task in plan.mesh_subtasks])
    return {
        "rewrite_plan": {"target_files": changes} if changes else None,
        "pending_solver_subtasks": [t for t in creates if t['stage'] == 'solver'],
        "review_analysis": state.get("review_analysis") or
                           "Apply the mesh planner's explicit file changes: " + str(changes),
        "mesh_type": plan.mesh_type,
        "mesh_script_action": plan.mesh_script_action,
        "mesh_reference": reference,
        "mesh_boundary_plan": [item.model_dump() for item in plan.boundary_plan],
        "mesh_source": str(source) if source else None,
        "mesh_subtasks": [t for t in creates if t['stage'] == 'mesh'],
        **({'mesh_resume_state': {
            **state['mesh_resume_state'],
            'pending': [t for t in changes if t['stage'] == 'solver'],
            'input_writer_mode': 'initial' if any(t['stage'] == 'solver' for t in creates) else 'rewrite',
        }} if state.get('mesh_resume_state') else {}),
        **operation_update,
        "mesh_commands": script_commands(operation_update["mesh_operations"]),
        "final_reconstruction": plan.final_reconstruction.model_dump() if plan.final_reconstruction else None,
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
        "files. Any controlDict here is the minimal bootstrap for mesh utilities."
        + f"\n<mesh_boundary_plan>{state['mesh_boundary_plan']}</mesh_boundary_plan>"
        + f"\n<approved_operations>{state['mesh_operations']}</approved_operations>\nFollow the approved operations; do not independently select commands.",
        tutorial_reference=state["mesh_reference"]["content"]
        + f"\n<mesh_script_reference>{state['mesh_reference']['script']}</mesh_script_reference>"
        + f"\nExisting case files; preserve settings not requested to change:\n{state.get('foamfiles') or ''}",
        case_solver=state.get("case_solver") or "",
        openfoam_fork=generation_convention(config),
        generation_mode=config.input_writer_generation_mode,
        reuse_generated_dir=config.reuse_generated_dir,
        llm_service=state["llm_service"], config=config,
        allrun_phase='mesh',
        # File-only edits with explicit none intent are not executed. A later
        # execution plan must approve their actual contents before running them.
        operations=state['mesh_operations'] if state['mesh_script_action'] != 'none' else None,
    )


def _run_mesh_commands(state):
    config = state["config"]
    case = Path(state["case_dir"])
    action = state["mesh_script_action"]
    commands = script_commands(state['mesh_operations'])
    evidence = {"action": action, "executed": False, "commands": commands}
    state["mesh_execution"] = evidence
    if action == "none":
        return {"error_logs": [], "mesh_execution": evidence}
    if action == "run_existing":
        if not (case / "Allmesh").is_file():
            return {"error_logs": [{"file": "Allmesh", "error_content": "run_existing requires an Allmesh script."}],
                    "mesh_execution": evidence}
        script = {"allrun_path": str(case / "Allmesh")}
    elif action == "build_and_run":
        if not commands:
            return {"error_logs": [{"file": "mesh_plan", "error_content": "build_and_run requires mesh_commands."}],
                    "mesh_execution": evidence}
        script = build_allrun(
            case_dir=state["case_dir"],
            database_path=str(database_path_for_config(config)), searchdocs=config.searchdocs,
            dir_structure=state.get("dir_structure") or {},
            case_info=state.get("case_info") or "",
            allrun_reference=state["mesh_reference"]["script"],
            mesh_type=state["mesh_type"], mesh_commands=commands,
            user_requirement=state["user_requirement"], llm_service=state["llm_service"],
            config=config, openfoam_fork=generation_convention(config), phase="mesh",
            operations=state['mesh_operations'],
        )
    else:
        raise ValueError(f"Unknown mesh_script_action: {action}")
    script_errors = check_stage_script(str(case), state['mesh_operations'], 'mesh')
    if script_errors:
        return {'error_logs': script_errors, 'mesh_execution': evidence}
    old, observations = _mesh_log_snapshot(case)
    # RunFunctions skips a utility when log.<application>[.<suffix>] exists.
    # Rotate only conventional case-root logs. Nested files and mesh sources
    # can be utility inputs even when their names start with log.
    sources = {(case / Path(source).expanduser()).resolve() for source in
               (state.get("mesh_source"), state.get("custom_mesh_path")) if source}
    old = {name: signature for name, signature in old.items()
           if name in {"Allmesh.out", "Allmesh.err"} or (
               Path(name).parent == Path(".") and name.startswith("log.")
               and Path(name).suffix.lower() not in _MESH_SOURCE_SUFFIXES
               and (case / name).resolve() not in sources
           )}
    if old:
        history = Path(tempfile.mkdtemp(prefix="attempt-", dir=_mesh_history(case)))
        for name in old:
            destination = history / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(case / name), destination)
    before, failures = _mesh_log_snapshot(case)
    observations.extend(failures)
    runner = _mesh_script_runner(state)
    evidence.update(executed=True, script=script["allrun_path"])
    result = runner(
        script["allrun_path"], str(case / "Allmesh.out"), str(case / "Allmesh.err"),
        str(case), config.max_time_limit, openfoam_target=runtime_openfoam_target(config),
        fail_fast=True,
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
        if "already run on" in output[name] and "re-run" in output[name]:
            errors.append({"file": name, "error_content": "Mesh command skipped due to an existing log: " + output[name]})
        match = _EXPLICIT_FOAM_ERROR_RE.search(output[name])
        if match:
            errors.append({"file": name, "error_content": match.group(1).strip()})
        if _FAILED_MESH_CHECKS_RE.search(output[name]):
            errors.append({"file": name, "error_content": "checkMesh reported failed mesh checks.\n" + output[name]})
    evidence.update(command_result=result, logs=output, observations=observations)
    if result.get("returncode") or result.get("timed_out"):
        errors.append({"file": "Allmesh", "error_content": str(evidence)})
    return {"error_logs": errors, "mesh_execution": evidence}


def _mesh_log_snapshot(case):
    files, observations = {}, []
    try:
        for path in case.rglob("*"):
            relative = path.relative_to(case)
            if ".foamagent" in relative.parts or relative.as_posix() in {"workflow.log", "review.log"}:
                continue
            if path.name.startswith("log.") or path.suffix.lower() in {".log", ".out", ".err"}:
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
    return {**_run_mesh_commands(state), "mesh_info": None}


def gmsh_meshing(state):
    (Path(state["case_dir"]) / ".foamagent/mesh-source.json").unlink(missing_ok=True)
    result = handle_gmsh_mesh(
        state["user_requirement"]
        + f"\n<mesh_reference>{state['mesh_reference']}</mesh_reference>"
        + f"\n<mesh_boundary_plan>{state['mesh_boundary_plan']}</mesh_boundary_plan>"
        + f"\n<approved_operations>{state['mesh_operations']}</approved_operations>\nFollow the approved operations; do not independently select commands.",
        state["case_dir"], timeout=state['config'].max_time_limit,
        script_prepared='generate_mesh.py' in state['written_files'],
        llm_service=state["llm_service"],
        openfoam_target=runtime_openfoam_target(state["config"]),
        repair_feedback=state.get("review_analysis") or "",
        **_native_mesh_runners(state),
    )
    commands = script_commands(state['mesh_operations'])
    if not result.get("error_logs"):
        result.update(_run_mesh_commands(state))
        if commands and result.get("mesh_info"):
            result["mesh_info"]["check_mesh_passed"] = None
    result["mesh_commands"] = commands
    result['mesh_execution'] = {**state['mesh_execution'], 'backend': result.pop('native_execution')}
    result["mesh_subtasks"] = []
    return result


def custom_meshing(state):
    """Use an existing polyMesh or convert a selected custom .msh input."""
    source = state.get("mesh_source")
    if not source:
        return _run_mesh_commands(state)
    source_path = Path(source).expanduser().resolve()
    result = copy_custom_mesh(
        source, state["case_dir"],
        timeout=state['config'].max_time_limit,
        openfoam_target=runtime_openfoam_target(state["config"]),
        retry=bool(state.get("repairing_mesh")),
        source_written=any((Path(state["case_dir"]) / name).resolve() == source_path
                           for name in state.get("written_files") or []),
        **{key: value for key, value in _native_mesh_runners(state).items() if key == "utility_runner"},
    )
    if not result.get("error_logs"):
        commands = script_commands(state['mesh_operations'])
        result.update(_run_mesh_commands(state))
        if commands and result.get("mesh_info"):
            result["mesh_info"]["check_mesh_passed"] = None
    result["mesh_commands"] = script_commands(state['mesh_operations'])
    result['mesh_execution'] = {**state['mesh_execution'], 'backend': result.pop('native_execution')}
    return result


def _rewrite_mesh(state):
    if not (state.get("rewrite_plan") or {}).get("target_files"):
        return {}
    return rewrite_files(
        case_dir=state["case_dir"], error_logs=state.get("error_logs") or [],
        review_analysis=state.get("review_analysis") or "", rewrite_plan=state["rewrite_plan"],
        user_requirement=state["user_requirement"]
        + f"\n<actual_mesh>{state.get('mesh_info') or {}}</actual_mesh>"
        + f"\n<mesh_reference>{state['mesh_reference']}</mesh_reference>"
        + f"\n<mesh_boundary_plan>{state['mesh_boundary_plan']}</mesh_boundary_plan>"
        + f"\n<approved_operations>{state['mesh_operations']}</approved_operations>\nFollow the approved operations; do not independently select commands.",
        foamfiles=state.get("foamfiles"),
        dir_structure=state.get("dir_structure") or {},
        openfoam_fork=generation_convention(state["config"]),
        case_solver=state.get("case_solver") or "", llm_service=state["llm_service"],
        script_phase='mesh', allrun_reference=state['mesh_reference']['script'],
        operations=state['mesh_operations'] if state['mesh_script_action'] != 'none' else None,
    )


class MeshRepairImpact(BaseModel):
    replan_solver: bool
    reason: str


def _active_mesh_directories(case, config=None):
    properties = case / "constant/regionProperties"
    if not (properties.is_file() or properties.with_suffix(".gz").is_file()):
        return {"": case / "constant/polyMesh"}
    target = runtime_openfoam_target(config) if config is not None else ""
    expand = target == "foundation-v10" or (not target and os.getenv("WM_PROJECT_VERSION") == "10")
    result = run_openfoam_utility(
        ["foamDictionary", *(["-expand"] if expand else []), "constant/regionProperties", "-entry", "regions", "-value"],
        working_dir=str(case), timeout=config.max_time_limit if config is not None else 300,
        openfoam_target=target,
    )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)
    # Native regionProperties is a list of category/wordList pairs. Categories
    # identify the declared inventory; they do not assign boundary roles.
    groups = re.compile(r'([^()\s]+)\s+(?:(\d+)\s*)?\(([^()]*)\)')
    matches = list(groups.finditer(result.stdout))
    remainder = groups.sub('', result.stdout)
    if not matches or not re.fullmatch(r'\s*(?:\d+\s*)?\(\s*\)\s*', remainder):
        raise ValueError("Cannot resolve native regionProperties region lists: " + result.stdout)
    regions = []
    for match in matches:
        names = shlex.split(match[3])
        if match[2] is not None and int(match[2]) != len(names):
            raise ValueError("regionProperties has an inconsistent region-list size")
        for name in names:
            relative = safe_case_relative_from_text(name)
            if len(relative.parts) != 1 or relative.as_posix() != name or name in regions:
                raise ValueError(f"Invalid or duplicate active mesh region: {name}")
            regions.append(name)
    if not regions:
        raise ValueError("regionProperties declares no active mesh regions")
    return {name: case / "constant" / name / "polyMesh" for name in regions}


def read_patch_bounds(case, regions=None):
    """Read boundary positions from the produced mesh, independently of the plan."""
    import pyvista as pv
    with tempfile.NamedTemporaryFile(dir=case, suffix=".foam") as marker:
        reader = pv.OpenFOAMReader(marker.name)
        reader.skip_zero_time = False
        reader.disable_all_cell_arrays()
        if regions is None:
            reader.enable_all_patch_arrays()
        else:
            reader.disable_all_patch_arrays()
            for name in reader.patch_array_names:
                if any(name.lstrip("/").startswith(f"{region}/") for region in regions):
                    reader.enable_patch_array(name)
        mesh = reader.read()
        bounds, found_boundary = {}, False
        def visit(block, prefix=()):
            nonlocal found_boundary
            for name in block.keys():
                child = block[name]
                if name == "boundary":
                    found_boundary = True
                    for patch in child.keys():
                        bounds["/".join((*prefix, patch))] = list(child[patch].bounds)
                elif hasattr(child, "keys"):
                    visit(child, (*prefix, name))
        visit(mesh)
        if not found_boundary:
            raise ValueError(f"Expected mesh boundary blocks; actual blocks: {list(mesh.keys())}")
        return bounds


def _mesh_boundary_info(case, directories):
    import gzip
    bounds = read_patch_bounds(str(case), regions=list(directories)) if "" not in directories else read_patch_bounds(str(case))
    regions = {}
    for name, mesh_dir in directories.items():
        boundary = mesh_dir / "boundary"
        if boundary.is_file():
            text = boundary.read_text(encoding="utf-8", errors="replace")
        else:
            with gzip.open(str(boundary) + ".gz", "rt", errors="replace") as stream:
                text = stream.read()
        if name:
            patch_bounds = {key[len(name) + 1:]: value for key, value in bounds.items() if key.startswith(name + "/")}
            if not patch_bounds:
                raise ValueError(f"Cannot read actual mesh boundary for active region {name}")
        else:
            patch_bounds = {key: value for key, value in bounds.items() if "/" not in key}
            if not patch_bounds:
                patch_bounds = {key[len("defaultRegion/"):]: value for key, value in bounds.items()
                                if key.startswith("defaultRegion/")}
        regions[name] = {"poly_mesh_path": str(mesh_dir), "boundary_text": text, "patch_bounds": patch_bounds}
    if "" in regions:
        return regions[""]
    return {"regions": regions, "patch_bounds": {
        f"{region}/{patch}": value for region, info in regions.items() for patch, value in info["patch_bounds"].items()}}


def execute_meshing(state, backend):
    current = dict(state)
    current.setdefault("mesh_jobs", [])
    current["mesh_execution"] = {"action": current["mesh_script_action"], "executed": False}
    current['written_files'] = []
    result = {"mesh_jobs": current["mesh_jobs"], "mesh_execution": current["mesh_execution"]}
    plan_errors = check_mesh_plan(current)
    if plan_errors:
        return {**result, "error_logs": plan_errors, "repairing_mesh": True,
                **stage_outcome(current, "mesh", {"error_logs": plan_errors,
                                                 "execution": current["mesh_execution"]})}
    resume = state.get("mesh_resume_state") or {
        key: state.get(key) for key in
        ("input_writer_mode", "review_analysis", "requires_input_writer")
    }
    previous_mesh = resume.get("previous_mesh", state.get("mesh_info"))
    targets = (current.get("rewrite_plan") or {}).get("target_files", [])
    pending = _unique_pending_repairs(
        list(resume.get("pending", [])) + [t for t in targets if t["stage"] != "mesh"]
    )
    current["rewrite_plan"] = {"target_files": [t for t in targets if t["stage"] == "mesh"]}
    reconstructed_sources = []
    try:
        written = _write_mesh_files(current)
        result.update(written)
        current.update(written)
        receipts = written['written_files'] if current.get('mesh_subtasks') else []
        current['mesh_subtasks'] = pending_file_creates(current.get('mesh_subtasks') or [], receipts)
        result['mesh_subtasks'] = current['mesh_subtasks']
        if current['mesh_subtasks'] and not written.get('error_logs'):
            written['error_logs'] = unwritten_file_errors(
                safe_case_relative_path(t['folder_name'], t['file_name']).as_posix()
                for t in current['mesh_subtasks'])
            result['error_logs'] = written['error_logs']
        if not written.get("error_logs"):
            if written:
                # Creation returns only new files; edits also need existing case contents.
                current.update(load_case_files(current))
            rewritten = _rewrite_mesh(current)
            result.update(rewritten)
            current.update(rewritten)
            current['written_files'] = list(dict.fromkeys(
                [*written.get('written_files', []), *rewritten.get('written_files', [])]))
            result['written_files'] = current['written_files']
            if not rewritten.get("error_logs"):
                dependency_errors = check_stage_inputs(current["case_dir"], current["mesh_operations"], "mesh")
                result.update({"error_logs": dependency_errors} if dependency_errors else backend(current))
                current.update(result)
                if not current.get("error_logs") and current.get("final_reconstruction"):
                    reconstructed_sources = _run_final_reconstruction(current)
                    result["mesh_execution"] = current["mesh_execution"]
    except HpcWorkflowStop as stop:
        result.update(stop.state_update())
        result["mesh_execution"] = current["mesh_execution"]
    except Exception as error:
        native_output = [str(error), str(getattr(error, "stdout", "") or ""),
                         str(getattr(error, "stderr", "") or "")]
        result["error_logs"] = [{"file": "Allmesh", "error_content": "\n".join(native_output)}]
        result["mesh_execution"] = current["mesh_execution"]
    finally:
        result["loop_count"] = current.get("loop_count", 0)
        if "cluster_info" in current:
            result["cluster_info"] = current["cluster_info"]
    if result.get("termination_reason"):
        return result
    current.update(result)
    if not current.get("error_logs"):
        try:
            mesh_directories = _active_mesh_directories(Path(current["case_dir"]), current["config"])
            result["error_logs"] = validate_mesh_handoff(Path(current["case_dir"]), reconstructed_sources,
                                                        mesh_directories=mesh_directories)
        except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
            result["error_logs"] = [{"file": "mesh_handoff", "error_content": str(error)}]
        current.update(result)
    if not current.get("error_logs"):
        try:
            prior = {key: value for key, value in (current.get("mesh_info") or {}).items()
                     if key not in {"regions", "poly_mesh_path", "boundary_text", "patch_bounds"}}
            result["mesh_info"] = {
                **prior, **_mesh_boundary_info(Path(current["case_dir"]), mesh_directories),
            }
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            result.update(mesh_info=None, error_logs=[{
                "file": "mesh_handoff", "error_content":
                    f"Cannot read actual mesh boundary from {mesh_directories}: {type(error).__name__}: {error}",
            }])
        current.update(result)
    _finish_reconstruction(current)
    result.update(load_case_files(current))
    if (current.get("mesh_execution") or {}).get("final_reconstruction", {}).get("status") == "complete":
        result["final_reconstruction"] = None
    result.update(stage_outcome(current, "mesh", {"error_logs": current.get("error_logs", []), "execution": current.get("mesh_execution")}))
    current.update(result)
    if not current.get("error_logs"):
        info = result["mesh_info"]
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
            result["replan_solver"] = impact.replan_solver or bool(current["pending_solver_operations"])
            impact_reason = impact.reason
        result.update({key: resume.get(key) for key in
                        ("input_writer_mode", "review_analysis")})
        result["requires_input_writer"] = bool(pending or current.get("pending_solver_subtasks"))
        if impact_reason:
            result["review_analysis"] = (result.get("review_analysis") or "") + "\n" + impact_reason
        if current.get("pending_solver_subtasks"):
            result.update(subtasks=current["pending_solver_subtasks"],
                           input_writer_mode="initial", requires_input_writer=True)
        elif pending:
            result["input_writer_mode"] = "rewrite"
        result.update(rewrite_plan={"target_files": pending} if pending else None,
                       repairing_mesh=False, mesh_resume_state=None, error_logs=[])
        return result
    result.update(
        repairing_mesh=True,
        rewrite_plan={"target_files": pending + current["rewrite_plan"]["target_files"]},
        mesh_resume_state={**resume, "pending": pending, "previous_mesh": previous_mesh},
    )
    return result


def _mesh_history(case):
    path = case / ".foamagent/mesh-logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _missing_mesh_files(directory):
    return [name for name in _MESH_FILES
            if not ((directory / name).is_file() or (directory / (name + ".gz")).is_file())]


def mesh_candidates(case):
    return [{"path": path.relative_to(case).as_posix(), "missing_files": _missing_mesh_files(path)}
            for path in sorted(case.rglob("polyMesh"))
            if path.is_dir() and ".foamagent" not in path.relative_to(case).parts]


def _reconstruction_sources(case, declared):
    """Check every native partition, including directories missing their mesh."""
    partitions, ranks, groups = [], [], []
    for path in case.iterdir():
        match = re.fullmatch(r"processor(?:(\d+)|s(\d+)(?:_(\d+)-(\d+))?)", path.name)
        if not match:
            continue
        partitions.append(path / "constant/polyMesh")
        if match[1] is not None:
            ranks.append(int(match[1]))
        else:
            count = int(match[2])
            first, last = (int(match[3]), int(match[4])) if match[3] is not None else (0, count - 1)
            groups.append((count, first, last))
    if ranks and groups:
        raise ValueError("Ambiguous reconstruction sources: uncollated and collated partitions coexist")
    if ranks and sorted(ranks) != list(range(len(ranks))):
        raise ValueError("Reconstruction processor ranks must cover a contiguous range starting at zero")
    if groups:
        count = groups[0][0]
        next_rank = 0
        for total, first, last in sorted(groups, key=lambda group: group[1]):
            if total != count or count < 1 or first != next_rank or last < first or last >= count:
                raise ValueError("Collated reconstruction groups must cover one processor count without gaps or overlaps")
            next_rank = last + 1
        if next_rank != count:
            raise ValueError("Collated reconstruction groups do not cover all processor ranks")
    actual = {path.relative_to(case).as_posix() for path in partitions}
    requested = {safe_case_relative_from_text(source).as_posix() for source in declared}
    if not actual or requested != actual:
        raise ValueError(f"Reconstruction sources differ from actual partitions: declared {sorted(requested)}; actual {sorted(actual)}")
    for path in partitions:
        if not path.resolve().is_relative_to(case):
            raise ValueError(f"Reconstruction source escapes the working case: {path}")
        missing = _missing_mesh_files(path)
        if missing:
            raise ValueError(f"Incomplete reconstruction source {path.relative_to(case)}: missing {missing}")
    return sorted(actual)


def _restore_reconstruction(state):
    evidence = state["mesh_execution"]["final_reconstruction"]
    target = Path(state["case_dir"]) / "constant/polyMesh"
    attempt = Path(evidence["attempt_path"])
    if target.exists() or target.is_symlink():
        failed = attempt / "failed-polyMesh"
        target.rename(failed)
        evidence["failed_target_path"] = str(failed)
    if evidence["backup_path"]:
        Path(evidence["backup_path"]).rename(target)
    evidence["status"] = "rolled_back"


def _finish_reconstruction(state):
    evidence = (state.get("mesh_execution") or {}).get("final_reconstruction")
    if not evidence or evidence.get("status") != "awaiting_handoff":
        return
    if state.get("error_logs"):
        _restore_reconstruction(state)
        return
    if evidence["backup_path"]:
        backup = Path(evidence["backup_path"])
        if backup.is_symlink() or backup.is_file():
            backup.unlink()
        else:
            shutil.rmtree(backup)
    Path(evidence["attempt_path"]).rmdir()
    evidence["status"] = "complete"


def _run_final_reconstruction(state):
    plan = FinalReconstruction.model_validate(state["final_reconstruction"])
    case = Path(state["case_dir"]).resolve()
    target = case / plan.target
    if not target.parent.resolve().is_relative_to(case):
        raise ValueError("Final reconstruction target escapes the working case")
    sources = _reconstruction_sources(case, plan.sources)
    if any((case / source).resolve() == target.resolve() for source in sources):
        raise ValueError("Reconstruction sources must be independent of the serial target")
    root = target.parent / ".foamagent/reconstruction"
    root.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
    backup = attempt / "previous-polyMesh" if target.exists() or target.is_symlink() else None
    if backup:
        target.rename(backup)
    evidence = {**plan.model_dump(), "sources": sources, "status": "running",
                "attempt_path": str(attempt), "backup_path": str(backup) if backup else None}
    state["mesh_execution"] = {
        **(state.get("mesh_execution") or {}),
        "final_reconstruction": evidence,
    }
    try:
        runner = _native_mesh_runners(state).get("utility_runner", run_openfoam_utility)
        completed = runner(plan.argv, working_dir=str(case), timeout=state["config"].max_time_limit,
                           openfoam_target=runtime_openfoam_target(state["config"]))
        evidence.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, plan.argv,
                                                output=completed.stdout, stderr=completed.stderr)
        missing = _missing_mesh_files(target)
        if missing:
            raise ValueError(f"Final reconstruction did not produce complete {plan.target}: missing {missing}")
    except HpcWorkflowStop as stop:
        evidence.update(error=str(stop), termination_reason=stop.reason)
        if stop.reason in {"hpc_job_state_unconfirmed", "hpc_submission_unconfirmed"}:
            evidence["status"] = "state_unconfirmed"
            stop.record["reconstruction_backup"] = evidence["backup_path"]
        else:
            _restore_reconstruction(state)
        raise
    except Exception as error:
        evidence.update(error=str(error), stdout=str(getattr(error, "stdout", "") or ""),
                        stderr=str(getattr(error, "stderr", "") or ""))
        _restore_reconstruction(state)
        raise
    evidence["status"] = "awaiting_handoff"
    return sources


def validate_mesh_handoff(case, reconstructed_sources=(), *, mesh_directories=None):
    mesh_directories = _active_mesh_directories(case) if mesh_directories is None else mesh_directories
    active = {directory.relative_to(case) for directory in mesh_directories.values()}
    missing = [str(directory / name) for directory in mesh_directories.values()
               for name in _missing_mesh_files(directory)]
    pending = []
    for path in case.rglob("polyMesh"):
        relative = path.relative_to(case)
        if ".foamagent" in relative.parts:
            continue
        if relative.as_posix() in reconstructed_sources:
            continue
        if relative in active or relative == Path("constant/polyMesh"):
            continue
        if (any(directory in relative.parents for directory in [*active, Path("constant/polyMesh")])
                or relative.parts[0].startswith("processor")):
            pending.append(str(path))
        else:
            try:
                # Time-zero mesh components override constant too. A cellMap-only
                # directory from refineMesh -overwrite is auxiliary output.
                if (float(relative.parts[0]) != 0
                        or len(_missing_mesh_files(path)) < len(_MESH_FILES)):
                    pending.append(str(path))
            except ValueError:
                pass
    if missing or pending:
        return [{"file": "mesh_handoff", "error_content":
                 f"Prepare final active meshes {[str(path) for path in mesh_directories.values()]} before solver planning. Missing files: {missing}; "
                 f"unresolved time/parallel or nested mesh outputs: {pending}. Use native overwrite or reconstruction operations; do not use the old mesh."}]
    return []


def _mesh_script_runner(state):
    if state.get("requires_hpc"):
        from .run_hpc import run_stage_hpc
        def run(script, out, err, case, timeout, **kwargs):
            result = run_stage_hpc(state, script, "mesh", timeout)
            return result
        return run
    return run_command


def _native_mesh_runners(state):
    if not state.get("requires_hpc"):
        return {}
    def run(args, **kwargs):
        case = Path(state["case_dir"])
        root = case / ".foamagent/mesh-commands"
        root.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="command-", dir=root))
        script = folder / "command.sh"
        script.write_text("#!/bin/bash\nset -e\n" + shlex.join(args) + "\n")
        from .run_hpc import run_stage_hpc
        # Slurm monitoring includes queue time; use the configured stage deadline,
        # not the backend's shorter local process timeout.
        result = run_stage_hpc(state, str(script), "mesh", state["config"].max_time_limit)
        out = Path(result["output"]).read_text(errors="replace") if Path(result["output"]).exists() else ""
        err = Path(result["error"]).read_text(errors="replace") if Path(result["error"]).exists() else str(result)
        if result.get("returncode"):
            raise subprocess.CalledProcessError(result["returncode"], args, output=out, stderr=err)
        return subprocess.CompletedProcess(args, 0, out, err)
    return {"utility_runner": run, "python_runner": run}
