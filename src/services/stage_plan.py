"""Declared stage dependencies; no inferred commands or generated defaults."""
from models import StageOperation
from utils import case_file_key, resolve_case_file
from .case_paths import safe_case_relative_from_text, safe_case_relative_path, safe_case_path


OPERATION_CONTRACT = (
    "Return ordered operations together with file tasks. Each operation declares its command, "
    "stage, purpose, executor (script or backend), required_inputs (case-relative file path and evidence), and "
    "generated_inputs (inputs produced by earlier operations, described with their producer). "
    "Only list inputs needed BEFORE stage execution in required_inputs. Do not require an "
    "intermediate output to exist before its producer runs. Declare bootstrap dictionaries "
    "needed by native mesh backends too. Plan file creates/edits for missing required inputs. "
    "Solver operations always use executor script: there is no separate solver backend executor. "
    "Standard mesh utilities and operations on an existing polyMesh also use script. "
    "Executor backend only describes the native generation/conversion/checking already performed "
    "by gmsh_mesh or custom_mesh with a selected .msh source; it cannot run arbitrary commands. "
    "Select build_and_run or run_existing for declared mesh script operations; none declares "
    "that there are no script operations to execute. "
    "Preserve imported scripts unless an explicit change is required. Resolve old and new "
    "instructions for each file into one final task; retain all still-pending work. "
    "For executor script, command is literal executable Bash, including exact arguments, "
    "required shell setup, helper definitions and retained script content. Put explanations "
    "only in purpose; do not put action labels or prose instructions in command. "
    "The writer renders these commands in order without another command-selection call. "
    "During workflow execution the stage runner already activates the selected native "
    "OpenFOAM runtime and runs Bash from the case directory with errexit and pipefail. "
    "Use these actual execution facts when planning required shell setup. "
    "To retain an existing execution script, copy its complete approved shell body into "
    "the script operations, preserving unrelated content. Allrun edits require the updated "
    "solver operation plan, including all retained shell content. "
    "Do not invent default physical conditions."
)


MESH_STAGE_CONTRACT = (
    "Mesh operations construct, transform or inspect the geometric mesh. Physical field "
    "initialization, physical boundary conditions, solver-parallel preparation and solving "
    "belong to stage solver, even when a tutorial runs them next to mesh commands. Classify "
    "each operation by its purpose and effect, never by command, filename or patch name. "
    "A missing physical input is a reason to reconsider operation ownership, not to create "
    "physical fields in the mesh stage. Defer physical operations with stage solver and "
    "physical file tasks with stage solver for Solver Planner to reconcile after handoff. "
    "Keep mesh startup dictionaries distinct from the eventual physical configuration. "
    "When an existing Allmesh contains deferred operations, explicitly edit it or select "
    "build_and_run for the mesh-only operations; run_existing must not retain those operations. "
)


def mesh_operation_updates(state, operations):
    """Only declared mesh operations execute now; physical requests await solver planning."""
    deferred = list(state["pending_solver_operations"])
    for operation in operations:
        if operation["stage"] == "solver" and operation not in deferred:
            deferred.append(operation)
    return {"mesh_operations": [op for op in operations if op["stage"] == "mesh"],
            "pending_solver_operations": deferred}


def check_stage_operations(operations, stage, *, allow_backend=False):
    """Validate execution ownership without guessing from commands or filenames."""
    errors = []
    if stage == "solver" and not operations:
        errors.append({"file": "stage_plan", "error_content": "solver requires an ordered script operation plan."})
    for raw in operations:
        operation = StageOperation.model_validate(raw)
        if operation.stage != stage:
            errors.append({"file": "stage_plan", "error_content":
                f"{stage} cannot execute a declared {operation.stage} operation: "
                f"{operation.command}; purpose: {operation.purpose}"})
            continue
        if operation.executor == "backend" and not allow_backend:
            errors.append({"file": "stage_plan", "error_content":
                f"{stage} has no backend executor for this plan: {operation.command}. "
                "Declare a script operation and its execution intent."})
    return errors


def check_mesh_plan(state):
    backend = state["mesh_type"]
    action = state["mesh_script_action"]
    if backend not in ("standard_mesh", "gmsh_mesh", "custom_mesh"):
        raise ValueError(f"Unknown mesh_type: {backend}")
    if action not in ("none", "run_existing", "build_and_run"):
        raise ValueError(f"Unknown mesh_script_action: {action}")
    operations = state["mesh_operations"]
    errors = check_stage_operations(operations, "mesh", allow_backend=(
        backend == "gmsh_mesh" or backend == "custom_mesh" and bool(state["mesh_source"])
    ))
    commands = script_commands(operations)
    if action == "none" and commands:
        errors.append({"file": "stage_plan", "error_content":
            "mesh_script_action none cannot execute the declared script operations. "
            "Select run_existing or build_and_run, or remove operations not requested to run."})
    if action in ("build_and_run", "run_existing") and not commands:
        errors.append({"file": "stage_plan", "error_content":
            f"mesh_script_action {action} requires declared script operations."})
    return errors


def check_stage_inputs(case_dir, operations, stage):
    errors = check_stage_operations(operations, stage, allow_backend=stage == "mesh")
    for raw in operations:
        operation = StageOperation.model_validate(raw)
        if operation.stage != stage:
            continue
        for required in operation.required_inputs:
            relative = safe_case_relative_from_text(required.path)
            path = safe_case_path(case_dir, str(relative.parent), relative.name)
            if not resolve_case_file(path).is_file():
                errors.append({"file": relative.as_posix(), "error_content":
                    f"{stage} dependency missing before execution: {relative}; "
                    f"operation: {operation.command}; evidence: {required.evidence}"})
    return errors


def script_commands(operations):
    return [operation['command'] for operation in operations if operation['executor'] == 'script']


def render_stage_script(operations):
    """Render the approved shell fragments without interpreting their meaning."""
    commands = script_commands(operations)
    if not commands or any(not command.strip() for command in commands):
        raise ValueError('An execution script requires nonempty literal script operations.')
    if len(commands) == 1 and commands[0].lstrip().startswith('#!'):
        return commands[0]
    return '#!/bin/bash\nset -e\nset -o pipefail\n' + '\n'.join(commands) + '\n'


def check_stage_script(case_dir, operations, stage):
    """Require the executed file to contain the approved program or its wrapper."""
    name = {'mesh': 'Allmesh', 'solver': 'Allrun'}[stage]
    path = safe_case_path(case_dir, '', name)
    if not path.is_file():
        return [{'file': name, 'error_content': f'The declared {stage} execution script is missing.'}]
    expected = render_stage_script(operations)
    body = '\n'.join(script_commands(operations))
    if path.read_text() not in (expected, body):
        return [{'file': 'stage_plan', 'error_content':
                 f'{name} differs from the approved {stage} operations. Reconcile the complete '
                 'literal script plan, including retained setup and content, before execution.'}]
    return []


def stage_outcome(state, stage, result):
    """Attach the native outcome to this stage's latest diagnostic attempt."""
    history = list(state[stage + '_review_history'])
    if history and history[-1]['outcome'] is None:
        history[-1] = {**history[-1], 'outcome': result}
    return {stage + '_review_history': history}


def replace_pending_tasks(pending, replacements):
    """A new planner instruction replaces the old instruction for that stage/path."""
    resolved = {(case_file_key(safe_case_relative_from_text(t['file'])), t['stage']): t for t in pending}
    for task in replacements:
        key = (case_file_key(safe_case_relative_from_text(task['file'])), task['stage'])
        resolved[key] = task
    return list(resolved.values())


def resolve_file_tasks(creates, changes, replacements=()):
    """Reconcile declared actions before separating creates from modifications.

    Pending edits include the latest Reviewer instructions. A new planner task
    replaces either action for its explicit stage/path, without joining prose.
    A shared startup file may still have separate mesh and later solver tasks.
    """
    resolved = {}
    for task in creates:
        path = safe_case_relative_path(task['folder_name'], task['file_name'])
        resolved[(case_file_key(path), task['stage'])] = dict(task)
    for task in changes:
        path = safe_case_relative_from_text(task['file'])
        resolved[(case_file_key(path), task['stage'])] = dict(
            folder_name='' if path.parent.as_posix() == '.' else path.parent.as_posix(),
            file_name=path.name, stage=task['stage'], action='modify', changes=task['changes'])
    seen = set()
    for task in replacements:
        path = safe_case_relative_path(task['folder_name'], task['file_name'])
        key = (case_file_key(path), task['stage'])
        if key in seen:
            raise ValueError(f'Duplicate planned file task: {task["stage"]} {path}')
        seen.add(key)
        resolved[key] = dict(task)
    creates, changes = [], []
    for (_, stage), task in resolved.items():
        path = safe_case_relative_path(task['folder_name'], task['file_name']).as_posix()
        if task['action'] == 'create':
            creates.append(task)
        elif task['action'] == 'modify':
            changes.append(dict(file=path, stage=stage, changes=task['changes']))
        else:
            raise ValueError(f'Unknown file task action: {task["action"]}')
    return creates, changes


def check_file_targets(case_dir, targets):
    """File tasks cannot target an actual directory or another target's parent."""
    paths = [safe_case_relative_from_text(target) for target in targets]
    errors = []
    for path in paths:
        target = safe_case_path(case_dir, str(path.parent), path.name)
        if target.is_dir() or any(path in other.parents for other in paths):
            errors.append({'file': path.as_posix(), 'error_content':
                           'Declared file task targets a directory. Declare individual file paths.'})
    return errors


def pending_file_creates(tasks, written_files):
    written = {case_file_key(path) for path in written_files}
    return [task for task in tasks if
            case_file_key(safe_case_relative_path(task['folder_name'], task['file_name'])) not in written]


def unwritten_file_errors(paths):
    return [{'file': path, 'error_content':
             'Requested file output was not written; its file task remains pending.'} for path in paths]
