import os
import re
from typing import Dict, List, Any, Optional, Callable
import shutil
from utils import save_file, parse_context, retrieve_faiss, FoamPydantic, FoamfilePydantic, scan_case_directory, read_case_foamfiles, read_file
from config import Config
from pydantic import BaseModel, Field
from . import global_llm_service
from .case_paths import (
    CasePathSafetyError,
    safe_case_path,
    safe_case_relative_from_text,
    safe_case_relative_path,
)
from .allrun_commands import (
    allrun_uses_runfunctions,
    invoked_command,
    is_run_wrapper,
    line_invokes_command,
    line_invokes_solver,
    shell_tokens,
)
from .openfoam_commands import MESH_MUTATING_COMMANDS


def compute_priority(subtask):
    if subtask["folder_name"] == "system":
        return 0
    elif subtask["folder_name"] == "constant":
        return 1
    elif subtask["folder_name"] == "0":
        return 2
    else:
        return 3


def _report_progress(
    callback: Optional[Callable[[int, int, str], None]],
    current: int,
    total: int,
    message: str,
) -> None:
    """Notify an optional observer without letting UI updates stop generation."""
    if callback is None:
        return
    try:
        callback(current, total, message)
    except Exception as exc:  # noqa: BLE001 - callback implementation is external
        print(f"<progress_callback_error>{exc}</progress_callback_error>")


def _generation_contract(openfoam_fork: str) -> str:
    """Return version/fork constraints shared by initial writes and rewrites.

    The contract is intentionally solver-level rather than case-level: it contains
    no geometry, material, boundary, or operating values from a particular case.
    """
    fork = (openfoam_fork or "foundation").strip().lower()
    if fork == "foundation":
        constraints = [
            "The target runtime is Foundation OpenFOAM v10 from openfoam.org.",
            "Use only Foundation OpenFOAM v10 solver, file, field, dictionary, and boundary-condition conventions.",
            "Do not use or mix in ESI/OpenCFD openfoam.com conventions from versioned releases such as v20xx, v21xx, v22xx, v23xx, or v24xx.",
            "If a reference case conflicts with Foundation OpenFOAM v10, ignore the conflicting syntax and follow Foundation v10.",
            "For blockMesh boundary dictionaries, every symmetryPlane patch must contain only coplanar faces. For a pair of parallel but spatially separate symmetry faces, either use the generic symmetry patch type or create one symmetryPlane patch per plane.",
            "When a Foundation momentumTransport or momentumTransport.<phase> dictionary is generated, it must declare simulationType (for example, laminar when the prompt requests laminar flow) before its model coefficients.",
        ]
    elif fork == "esi":
        constraints = [
            "The configured target runtime is ESI/OpenCFD OpenFOAM from openfoam.com.",
            "Use ESI-compatible solver, file, field, dictionary, and boundary-condition conventions consistently.",
            "Do not mix Foundation-only conventions into the generated files.",
        ]
    else:
        constraints = [
            f"The configured OpenFOAM fork is {fork!r}.",
            "Keep all generated files internally consistent with that configured fork and do not mix conventions from another fork.",
        ]

    return "\n".join(f"- {constraint}" for constraint in constraints)


_INITIAL_WRITE_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM simulation and numerical modeling."
    "Your task is to generate a complete and functional file named: <file_name>{file_name}</file_name> within the <folder_name>{folder_name}</folder_name> directory. "
    "Ensure all required values are present and match with the files content already generated."
    "Before finalizing the output, ensure:\n"
    "- All necessary fields exist (e.g., if `nu` is defined in `constant/transportProperties`, it must be used correctly in `0/U`).\n"
    "- Cross-check field names between different files to avoid mismatches.\n"
    "- Ensure units and dimensions are correct** for all physical variables.\n"
    "- Ensure case solver settings are consistent with the user's requirements. The selected solver is: {case_solver}.\n"
    "OpenFOAM compatibility contract:\n"
    "{generation_contract}\n"
    "Provide only the code—no explanations, comments, or additional text."
)


def _normalise_initial_subtask(subtask: Dict[str, str]) -> Dict[str, str]:
    """Validate one generation target and store it as a case-relative path."""
    if not isinstance(subtask, dict):
        raise ValueError(f"Invalid subtask format: {subtask!r}")
    try:
        relative = safe_case_relative_path(
            subtask.get("folder_name", ""),
            subtask.get("file_name", ""),
        )
    except CasePathSafetyError as exc:
        raise ValueError(f"Unsafe generated subtask path: {subtask!r}") from exc
    normalized = dict(subtask)
    normalized["folder_name"] = "" if relative.parent.as_posix() == "." else relative.parent.as_posix()
    normalized["file_name"] = relative.name
    return normalized


def _initial_file_prompts(
    *,
    file_name: str,
    folder_name: str,
    case_solver: str,
    generation_contract: str,
    user_requirement: str,
    tutorial_reference: str,
    similar_case_advice: Optional[Any],
    generation_mode: str,
    written_files: List[FoamfilePydantic],
) -> tuple[str, str]:
    """Build one file-generation prompt with only the needed prior context."""
    system_prompt = _INITIAL_WRITE_SYSTEM_PROMPT.format(
        file_name=file_name,
        folder_name=folder_name,
        case_solver=case_solver,
        generation_contract=generation_contract,
    )
    if isinstance(similar_case_advice, dict):
        advice_text = (
            f"Similar case match level: {similar_case_advice.get('match_level')}\n"
            f"Use scope: {similar_case_advice.get('use_scope')}\n"
            f"Advice: {similar_case_advice.get('advice')}\n"
        )
    else:
        advice_text = str(similar_case_advice or "")
    similar_reference = (
        f"Refer to the following similar case file content if helpful:\n<similar_case_reference>{tutorial_reference}</similar_case_reference>\n"
        if tutorial_reference
        else "No suitable similar case was found for this domain.\n"
    )
    user_prompt = (
        f"User requirement: {user_requirement}\n"
        f"{similar_reference}{advice_text}\n"
        "If the similar case is a weak match, do not copy it blindly. Use it only where it is consistent with the user requirement. "
        "The OpenFOAM compatibility contract in the system prompt takes precedence over incompatible syntax in a reference case. "
        "Just modify the necessary parts to make the file complete and functional."
        "Please ensure that the generated file is complete, functional, and logically sound."
        "Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence."
        "When generating controlDict, do not include anything to preform post processing. Just include the necessary settings to run the simulation."
    )
    if file_name == "fvSolution":
        user_prompt += (
            "\n\nCRITICAL for transient pressure-velocity coupling solvers using PISO/PIMPLE: "
            "the solvers dictionary must include matching Final solver entries for fields used on the final correction. "
            "For example, if p is defined, include pFinal { $p; relTol 0; }; "
            "if U is defined, include UFinal { $U; relTol 0; }. "
            "For grouped regex entries, use the matching grouped Final entry, e.g. "
            "\"(U|k|epsilon)Final\" { $U; relTol 0; }. "
            "Do not emit placeholder text such as $<field>; in the generated file. "
            "Also ensure the PIMPLE/PISO sub-dictionary matches the selected solver."
        )
    if generation_mode == "sequential_dependency" and written_files:
        user_prompt += (
            f"The following are files content already generated: {written_files}\n\n\n"
            "You should ensure that the new file is consistent with the previous files. Such as boundary conditions, mesh settings, etc."
        )
    return user_prompt, system_prompt


def _generate_initial_file(
    subtask: Dict[str, str],
    *,
    case_dir: str,
    reuse_generated_dir: str,
    llm_client: Any,
    case_solver: str,
    generation_contract: str,
    user_requirement: str,
    tutorial_reference: str,
    similar_case_advice: Optional[Any],
    generation_mode: str,
    written_files: List[FoamfilePydantic],
) -> FoamfilePydantic:
    """Generate or safely reuse a single OpenFOAM dictionary."""
    file_name = subtask["file_name"]
    folder_name = subtask["folder_name"]
    try:
        file_path = safe_case_path(case_dir, folder_name, file_name)
    except CasePathSafetyError as exc:
        raise ValueError(f"Unsafe generated subtask path: {subtask!r}") from exc
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if reuse_generated_dir:
        reuse_source = safe_case_path(reuse_generated_dir, folder_name, file_name)
        if reuse_source.exists():
            print(f"Reusing generated file: {reuse_source}")
            shutil.copy2(reuse_source, file_path)
            return FoamfilePydantic(
                file_name=file_name,
                folder_name=folder_name,
                content=read_file(str(reuse_source)),
            )
    user_prompt, system_prompt = _initial_file_prompts(
        file_name=file_name,
        folder_name=folder_name,
        case_solver=case_solver,
        generation_contract=generation_contract,
        user_requirement=user_requirement,
        tutorial_reference=tutorial_reference,
        similar_case_advice=similar_case_advice,
        generation_mode=generation_mode,
        written_files=written_files,
    )
    content = parse_context(llm_client.invoke(user_prompt, system_prompt))
    save_file(str(file_path), content)
    return FoamfilePydantic(file_name=file_name, folder_name=folder_name, content=content)


def initial_write(
    case_dir: str,
    subtasks: List[Dict[str, str]],
    user_requirement: str,
    tutorial_reference: str,
    case_solver: str,
    generation_mode: str = "sequential_dependency",
    case_info: str = "",
    allrun_reference: str = "",
    mesh_type: str = "blockMesh",
    mesh_commands: List[str] = None,
    database_path: str = "",
    searchdocs: int = 2,
    similar_case_advice: Optional[Any] = None,
    reuse_generated_dir: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    openfoam_fork: str = "foundation",
    llm_service: Optional[Any] = None,
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """
    Generate OpenFOAM files from scratch based on user requirements and subtasks.
    
    This function creates OpenFOAM input files by analyzing user requirements,
    using similar case references, and generating files in the correct order
    (system -> constant -> 0 -> others). It also generates an Allrun script
    for automated execution.
    
    Args:
        case_dir (str): Directory path where the case files will be created
        subtasks (List[Dict[str, str]]): List of subtasks, each containing:
            - file_name: Name of the OpenFOAM file to create
            - folder_name: Directory where the file should be placed
        user_requirement (str): Natural language description of simulation requirements
        tutorial_reference (str): Reference content from similar tutorial cases
        case_solver (str): OpenFOAM solver to use (e.g., "simpleFoam", "pimpleFoam")
        case_info (str, optional): Additional case information. Defaults to "".
        allrun_reference (str, optional): Reference Allrun scripts from similar cases. Defaults to "".
        mesh_type (str, optional): Type of mesh to use. Defaults to "blockMesh".
        mesh_commands (List[str], optional): Custom mesh commands. Defaults to None.
        database_path (str, optional): Path to FAISS database for command lookup. Defaults to "".
        searchdocs (int, optional): Number of documents to search for commands. Defaults to 2.
        openfoam_fork (str, optional): Target OpenFOAM fork. Defaults to "foundation".
    
    Returns:
        Dict[str, Any]: Contains:
            - dir_structure (Dict[str, List[str]]): Directory structure with files
            - foamfiles (FoamPydantic): Generated OpenFOAM files with metadata
    
    Raises:
        ValueError: If subtask format is invalid or file generation fails
        FileNotFoundError: If database files cannot be found
        RuntimeError: If LLM service fails to generate files
    
    Example:
        >>> subtasks = [
        ...     {"file_name": "controlDict", "folder_name": "system"},
        ...     {"file_name": "transportProperties", "folder_name": "constant"},
        ...     {"file_name": "U", "folder_name": "0"}
        ... ]
        >>> result = initial_write(
        ...     case_dir="/path/to/case",
        ...     subtasks=subtasks,
        ...     user_requirement="Simple fluid flow simulation",
        ...     tutorial_reference="Reference case content...",
        ...     case_solver="simpleFoam",
        ... )
        >>> print(f"Generated {len(result['dir_structure'])} directories")
    """
    print("<initial_write_service>")
    llm_client = llm_service if llm_service is not None else global_llm_service

    if generation_mode not in {"sequential_dependency", "parallel_no_context"}:
        raise ValueError(
            f"Unsupported generation_mode: {generation_mode}. "
            "Expected one of: sequential_dependency, parallel_no_context"
        )

    subtasks = sorted(
        (_normalise_initial_subtask(item) for item in subtasks),
        key=compute_priority,
    )
    seen_targets: set[tuple[str, str]] = set()
    for subtask in subtasks:
        target = (subtask["folder_name"], subtask["file_name"])
        if target in seen_targets:
            relative = "/".join(part for part in target if part)
            raise ValueError(f"Duplicate generated subtask target: {relative}")
        seen_targets.add(target)
    written_files = []
    dir_structure = {}
    generation_contract = _generation_contract(openfoam_fork)
    file_generation_options = {
        "case_dir": case_dir,
        "reuse_generated_dir": reuse_generated_dir,
        "llm_client": llm_client,
        "case_solver": case_solver,
        "generation_contract": generation_contract,
        "user_requirement": user_requirement,
        "tutorial_reference": tutorial_reference,
        "similar_case_advice": similar_case_advice,
        "generation_mode": generation_mode,
    }

    # Build dir_structure upfront (deterministic ordering) and generate files
    for subtask in subtasks:
        folder_name = subtask.get("folder_name")
        file_name = subtask.get("file_name")
        if folder_name not in dir_structure:
            dir_structure[folder_name] = []
        dir_structure[folder_name].append(file_name)

    total_steps = len(subtasks) + (2 if database_path else 0)
    _report_progress(
        progress_callback,
        0,
        total_steps,
        f"Starting file generation for {len(subtasks)} files",
    )

    if generation_mode == "parallel_no_context":
        print("<generation_mode>parallel_no_context (no cross-file context)</generation_mode>")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        # Parallelize all file generations; keep output order consistent with sorted subtasks.
        results: List[Optional[FoamfilePydantic]] = [None] * len(subtasks)
        completed_count = 0
        count_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=min(32, max(4, len(subtasks)))) as ex:
            future_map = {
                ex.submit(
                    _generate_initial_file,
                    subtasks[i],
                    written_files=[],
                    **file_generation_options,
                ): i
                for i in range(len(subtasks))
            }
            for fut in as_completed(future_map):
                i = future_map[fut]
                results[i] = fut.result()
                with count_lock:
                    completed_count += 1
                    _report_progress(
                        progress_callback,
                        completed_count, total_steps,
                        f"Generated {subtasks[i]['file_name']} in {subtasks[i]['folder_name']} (parallel)"
                    )

        written_files.extend([r for r in results if r is not None])

    else:
        print("<generation_mode>sequential_dependency</generation_mode>")
        for idx, subtask in enumerate(subtasks):
            file_name = subtask["file_name"]
            folder_name = subtask["folder_name"]
            print(f"<generating_file>{file_name} in folder: {folder_name}</generating_file>")
            foamfile = _generate_initial_file(
                subtask,
                written_files=written_files,
                **file_generation_options,
            )
            written_files.append(foamfile)
            _report_progress(
                progress_callback,
                idx + 1,
                total_steps,
                f"Generated {file_name} in {folder_name}",
            )
    
    # Generate Allrun script if database_path is provided
    if database_path:
        allrun_result = build_allrun(
            case_dir, database_path, searchdocs, dir_structure, case_info,
            allrun_reference, mesh_type, mesh_commands or [], user_requirement,
            progress_callback=progress_callback,
            progress_offset=len(subtasks),
            total_steps=total_steps,
            llm_service=llm_client,
            config=config,
        )
        # ``FoamfilePydantic`` stores paths relative to ``case_dir``.  Keeping
        # Allrun at the case root makes this entry safe to reuse in a later
        # rewrite pass instead of embedding an absolute output directory.
        written_files.append(
            FoamfilePydantic(
                file_name="Allrun",
                folder_name="",
                content=allrun_result["allrun_script"],
            )
        )
    
    foamfiles = FoamPydantic(list_foamfile=written_files)
    print("</initial_write_service>")
    return {"dir_structure": dir_structure, "foamfiles": foamfiles}


_MESH_DICTIONARY_COMMANDS = (
    ("blockMeshDict", "blockMesh"),
    ("snappyHexMeshDict", "snappyHexMesh"),
    ("extrudeMeshDict", "extrudeMesh"),
)
_MESH_GENERATION_COMMANDS = set(MESH_MUTATING_COMMANDS)


def _extract_case_solver(case_info: str) -> str:
    text = str(case_info or "")
    patterns = (
        r"case\s+solver\s*:\s*['\"]?([A-Za-z_][A-Za-z0-9_.+-]*)",
        r"['\"]case_solver['\"]\s*:\s*['\"]([A-Za-z_][A-Za-z0-9_.+-]*)['\"]",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _dir_structure_contains(dir_structure: Dict[str, List[str]], file_name: str) -> bool:
    return any(file_name in (files or []) for files in (dir_structure or {}).values())


def _format_allrun_command(command: str, use_run_application: bool) -> str:
    command = command.strip()
    if not command:
        return ""
    if command.startswith(("runApplication ", "runParallel ")):
        return command
    if use_run_application:
        return f"runApplication {command}"
    return command


def _ensure_runfunctions_logging(lines: List[str], solver: str) -> List[str]:
    """Wrap direct OpenFOAM commands so their per-command logs are retained."""
    logged_commands = set(_MESH_GENERATION_COMMANDS)
    logged_commands.add("checkMesh")
    if solver:
        logged_commands.add(solver)

    normalized: List[str] = []
    for line in lines:
        tokens = shell_tokens(line)
        if tokens and tokens[0] in logged_commands:
            leading = line[: len(line) - len(line.lstrip())]
            line = f"{leading}runApplication {line.lstrip()}"
        normalized.append(line)

    uses_run_functions = any(is_run_wrapper(line) for line in normalized)
    has_run_functions_source = allrun_uses_runfunctions(normalized)
    if uses_run_functions and not has_run_functions_source:
        insert_at = 1 if normalized and normalized[0].lstrip().startswith("#!") else 0
        normalized.insert(insert_at, ". $WM_PROJECT_DIR/bin/tools/RunFunctions")

    return normalized


def _allrun_mesh_plan(
    dir_structure: Dict[str, List[str]],
    mesh_type: str,
    mesh_commands: List[str],
) -> tuple[bool, List[str], List[str]]:
    """Classify expected and externally supplied mesh commands."""
    normalized_mesh_type = (mesh_type or "").strip().lower()
    external_mesh = normalized_mesh_type in {"custom_mesh", "gmsh_mesh"}
    expected_mesh_commands: List[str] = []
    if not external_mesh:
        for dictionary_name, command in _MESH_DICTIONARY_COMMANDS:
            if _dir_structure_contains(dir_structure, dictionary_name):
                expected_mesh_commands.append(command)
        if normalized_mesh_type in {"blockmesh", "block_mesh"} and "blockMesh" not in expected_mesh_commands:
            expected_mesh_commands.append("blockMesh")
    supplied_mesh_commands = [str(command).strip() for command in (mesh_commands or []) if str(command).strip()]
    supplied_non_checkmesh = [
        command for command in supplied_mesh_commands
        if not line_invokes_command(command, "checkMesh") and command != "checkMesh"
    ]
    return external_mesh, expected_mesh_commands, supplied_non_checkmesh


def _remove_external_mesh_generators(
    lines: List[str],
    supplied_commands: List[str],
) -> List[str]:
    """Prevent an LLM script from overwriting an already prepared mesh."""
    allowed_generator_names = {
        generator
        for generator in _MESH_GENERATION_COMMANDS
        if any(line_invokes_command(command, generator) for command in supplied_commands)
    }
    return [
        line
        for line in lines
        if not any(
            line_invokes_command(line, generator) and generator not in allowed_generator_names
            for generator in _MESH_GENERATION_COMMANDS
        )
    ]


def _mesh_commands_in_lines(lines: List[str], expected_commands: List[str]) -> List[str]:
    """Return all mesh generators that must be ordered before checkMesh."""
    commands = list(expected_commands)
    for generator in _MESH_GENERATION_COMMANDS:
        if generator not in commands and any(line_invokes_command(line, generator) for line in lines):
            commands.append(generator)
    return commands


def _requires_checkmesh(
    *,
    external_mesh: bool,
    expected_commands: List[str],
    supplied_commands: List[str],
    had_checkmesh: bool,
    discovered_commands: List[str],
) -> bool:
    return bool(
        external_mesh
        or expected_commands
        or supplied_commands
        or had_checkmesh
        or discovered_commands
    )


def _solver_index(lines: List[str], solver: str) -> int:
    for index, line in enumerate(lines):
        if line_invokes_solver(line, solver):
            return index
    solver_label = solver or "the selected solver"
    raise ValueError(
        f"Generated Allrun does not invoke {solver_label}; cannot place checkMesh before the solver."
    )


def _validate_mesh_command_order(
    lines: List[str],
    mesh_commands: List[str],
    solver_index: int,
    solver: str,
) -> None:
    """Reject scripts that run a mesh generator after the selected solver."""
    for command in mesh_commands:
        if any(
            index > solver_index
            for index, line in enumerate(lines)
            if line_invokes_command(line, command)
        ):
            raise ValueError(
                f"Generated Allrun invokes mesh command {command} after solver {solver or '<dynamic>'}."
            )


def _missing_mesh_commands(
    lines: List[str],
    solver_index: Optional[int],
    expected_commands: List[str],
    supplied_commands: List[str],
    use_run_application: bool,
) -> List[str]:
    """Format mesh commands absent from the selected execution scope.

    ``solver_index`` is ``None`` for mesh-only applications such as
    ``blockMesh``.  In that case the complete script is the scope and the
    caller appends the missing commands before the final ``checkMesh`` gate.
    """
    scope = lines if solver_index is None else lines[:solver_index]
    commands: List[str] = []
    for command in expected_commands:
        if not any(line_invokes_command(line, command) for line in scope):
            commands.append(_format_allrun_command(command, use_run_application))
    for command in supplied_commands:
        command_name = invoked_command(command)
        if command_name and not any(
            line_invokes_command(line, command_name) for line in scope
        ):
            commands.append(_format_allrun_command(command, use_run_application))
    return [command for command in commands if command]


def _assert_mesh_gate(
    lines: List[str],
    solver: str,
    mesh_commands: List[str],
) -> None:
    """Verify the normalizer's mesh -> checkMesh -> solver postcondition."""
    check_indexes = [
        index for index, line in enumerate(lines) if line_invokes_command(line, "checkMesh")
    ]
    solver_indexes = [
        index for index, line in enumerate(lines) if line_invokes_solver(line, solver)
    ]
    if len(check_indexes) != 1:
        raise ValueError("Could not enforce exactly one checkMesh gate in generated Allrun.")
    if solver in _MESH_GENERATION_COMMANDS:
        if any(
            index >= check_indexes[0]
            for index, line in enumerate(lines)
            for command in mesh_commands
            if line_invokes_command(line, command)
        ):
            raise ValueError("Could not enforce generated mesh command -> checkMesh ordering.")
        return
    if not solver_indexes or check_indexes[0] >= solver_indexes[0]:
        raise ValueError("Could not enforce checkMesh before the solver in generated Allrun.")
    if any(
        index >= check_indexes[0]
        for index, line in enumerate(lines)
        for command in mesh_commands
        if line_invokes_command(line, command)
    ):
        raise ValueError("Could not enforce generated mesh command -> checkMesh -> solver ordering.")


def _ensure_checkmesh_before_solver(
    allrun_script: str,
    *,
    case_info: str,
    dir_structure: Dict[str, List[str]],
    mesh_type: str,
    mesh_commands: List[str],
) -> str:
    """Normalize a generated Allrun into mesh -> checkMesh -> solver order."""
    solver = _extract_case_solver(case_info)
    lines = _ensure_runfunctions_logging((allrun_script or "").splitlines(), solver)
    external_mesh, expected_commands, supplied_commands = _allrun_mesh_plan(
        dir_structure,
        mesh_type,
        mesh_commands,
    )

    # A custom mesh has already been converted by the meshing service. Remove an
    # LLM-invented generator so it cannot overwrite the uploaded/prepared mesh.
    if external_mesh:
        lines = _remove_external_mesh_generators(lines, supplied_commands)

    existing_checkmesh_lines = [
        line for line in lines if line_invokes_command(line, "checkMesh")
    ]
    had_checkmesh = bool(existing_checkmesh_lines)
    lines = [line for line in lines if not line_invokes_command(line, "checkMesh")]

    discovered_commands = _mesh_commands_in_lines(lines, expected_commands)
    if not _requires_checkmesh(
        external_mesh=external_mesh,
        expected_commands=expected_commands,
        supplied_commands=supplied_commands,
        had_checkmesh=had_checkmesh,
        discovered_commands=discovered_commands,
    ):
        return allrun_script

    use_run_application = any(is_run_wrapper(line) for line in lines)
    if solver in _MESH_GENERATION_COMMANDS:
        # Mesh tutorials may legitimately declare blockMesh (or another mesh
        # utility) as ``application``.  That command is the mesh-generation
        # stage itself, not a solver that must run *after* checkMesh.  Treating
        # it as both used to duplicate blockMesh and fail the gate assertion.
        lines.extend(
            _missing_mesh_commands(
                lines,
                None,
                expected_commands,
                supplied_commands,
                use_run_application,
            )
        )
        checkmesh_line = (
            existing_checkmesh_lines[0].strip()
            if existing_checkmesh_lines
            else _format_allrun_command("checkMesh", use_run_application)
        )
        lines.append(checkmesh_line)
        _assert_mesh_gate(
            lines,
            solver,
            _mesh_commands_in_lines(lines, expected_commands),
        )
        normalized = "\n".join(lines)
        if allrun_script.endswith("\n"):
            normalized += "\n"
        return normalized

    solver_index = _solver_index(lines, solver)
    _validate_mesh_command_order(lines, discovered_commands, solver_index, solver)
    for command in _missing_mesh_commands(
        lines,
        solver_index,
        expected_commands,
        supplied_commands,
        use_run_application,
    ):
        lines.insert(solver_index, command)
        solver_index += 1

    checkmesh_line = (
        existing_checkmesh_lines[0].strip()
        if existing_checkmesh_lines
        else _format_allrun_command("checkMesh", use_run_application)
    )
    lines.insert(solver_index, checkmesh_line)

    _assert_mesh_gate(lines, solver, discovered_commands)

    normalized = "\n".join(lines)
    if allrun_script.endswith("\n"):
        normalized += "\n"
    return normalized


class CommandsPydantic(BaseModel):
    """Structured command list used to build an Allrun script."""

    commands: List[str] = Field(description="List of commands")


def _parse_allrun_response(text: str) -> str:
    """Extract shell content from an optional Markdown fence."""
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if not match:
        return text.strip()
    return re.sub(
        r"^\s*(?:sh|bash|shell|zsh|ksh)\s*\r?\n",
        "",
        match.group(1),
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def _read_available_commands(database_path: str) -> str:
    command_path = os.path.join(database_path, "raw", "openfoam_commands.txt")
    try:
        with open(command_path, encoding="utf-8") as command_file:
            commands = [line.strip() for line in command_file]
    except OSError as exc:
        raise ValueError(f"Could not read commands file {command_path}: {exc}") from exc
    return f"[{', '.join(commands)}]"


def _command_help_context(
    commands: List[str],
    searchdocs: int,
    config: Optional[Config],
) -> str:
    """Retrieve one concise reference block for each LLM-selected command."""
    retrieval_kwargs: Dict[str, Any] = {"topk": searchdocs}
    if config is not None:
        retrieval_kwargs["config"] = config
    help_text = [
        retrieve_faiss("openfoam_command_help", command, **retrieval_kwargs)[0]["full_content"]
        for command in commands
    ]
    return "\n".join(help_text)


def _custom_mesh_prompt_info(mesh_type: str, mesh_commands: List[str]) -> str:
    if mesh_type != "custom_mesh" or not mesh_commands:
        return ""
    return f"\nCustom mesh commands to include: {mesh_commands}"


def _command_selection_prompts(
    *,
    commands: str,
    dir_structure: Dict[str, List[str]],
    case_info: str,
    allrun_reference: str,
    mesh_type: str,
    mesh_commands: List[str],
    user_requirement: str,
) -> tuple[str, str]:
    """Build the command-selection prompts for a generated Allrun script."""
    custom_mesh = mesh_type == "custom_mesh"
    mesh_commands_info = _custom_mesh_prompt_info(mesh_type, mesh_commands)
    command_system = (
        "You are an expert in OpenFOAM. The user will provide a list of available commands. "
        "Generate only the necessary commands for an Allrun script based on the provided directory structure. "
        "Return only the command list with no explanation."
    )
    if custom_mesh:
        command_system += " Include custom mesh commands in their appropriate order."
    command_user = (
        f"Available OpenFOAM commands for the Allrun script: {commands}\n"
        f"Case directory structure: {dir_structure}\n"
        f"User case information: {case_info}\n"
        f"User requirement: {user_requirement}\n"
        f"Reference Allrun scripts from similar cases: {allrun_reference}\n"
        f"{mesh_commands_info}\n"
        "Generate only the required OpenFOAM command list."
    )
    return command_user, command_system


def _allrun_script_prompts(
    *,
    command_help: str,
    dir_structure: Dict[str, List[str]],
    case_info: str,
    allrun_reference: str,
    mesh_type: str,
    mesh_commands: List[str],
    user_requirement: str,
) -> tuple[str, str]:
    """Build the final script prompts after command references are available."""
    custom_mesh = mesh_type == "custom_mesh"
    mesh_commands_info = _custom_mesh_prompt_info(mesh_type, mesh_commands)
    script_system = (
        "You are an expert in OpenFOAM. Generate an Allrun script based on the provided details."
        f"Available commands with descriptions: {command_help}\n\n"
        f"Reference Allrun scripts from similar cases: {allrun_reference}\n\n"
        "Do not include post-processing commands, mesh conversion commands, or commands that run Gmsh."
    )
    script_user = (
        f"User requirement: {user_requirement}\n"
        f"Case directory structure: {dir_structure}\n"
        f"User case information: {case_info}\n"
        f"{mesh_commands_info}\n"
        "Reference scripts are informative only; follow the case structure and user requirement. "
        "Do not include post-processing, mesh conversion, or Gmsh commands. "
        "Generate only the Allrun script in ``` tags."
    )
    if custom_mesh:
        rule = " Do not include mesh commands other than the supplied custom mesh commands."
        script_system += rule
        script_user += rule
    return script_user, script_system


def build_allrun(
    case_dir: str,
    database_path: str,
    searchdocs: int,
    dir_structure: Dict[str, List[str]],
    case_info: str,
    allrun_reference: str,
    mesh_type: str,
    mesh_commands: List[str],
    user_requirement: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    progress_offset: int = 0,
    total_steps: int = 0,
    llm_service: Optional[Any] = None,
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """
    Build an Allrun script for automated OpenFOAM simulation execution.
    
    This function generates a complete Allrun script by analyzing the case structure,
    retrieving appropriate OpenFOAM commands from the database, and creating
    a shell script that automates the simulation workflow.
    
    Args:
        case_dir (str): Directory path where the Allrun script will be created
        database_path (str): Path to the FAISS database containing OpenFOAM commands
        searchdocs (int): Number of documents to search for command help
        dir_structure (Dict[str, List[str]]): Directory structure with file lists
        case_info (str): Case information including name, solver, domain, category
        allrun_reference (str): Reference Allrun scripts from similar cases
        mesh_type (str): Type of mesh ("blockMesh", "snappyHexMesh", "custom_mesh")
        mesh_commands (List[str]): Custom mesh commands to include
        user_requirement (str, optional): User requirements for context. Defaults to "".
    
    Returns:
        Dict[str, Any]: Contains:
            - allrun_path (str): Path to the created Allrun script
            - allrun_script (str): Content of the Allrun script
            - commands (List[str]): List of OpenFOAM commands used
    
    Raises:
        ValueError: If commands file cannot be read or no commands are generated
        FileNotFoundError: If database files are not found
        RuntimeError: If LLM service fails to generate script
    
    Example:
        >>> result = build_allrun(
        ...     case_dir="/path/to/case",
        ...     database_path="/path/to/database",
        ...     searchdocs=2,
        ...     dir_structure={"system": ["controlDict"], "0": ["U"]},
        ...     case_info="case name: test\ncase solver: simpleFoam",
        ...     allrun_reference="Reference scripts...",
        ...     mesh_type="blockMesh",
        ...     mesh_commands=[]
        ... )
        >>> print(f"Generated script with {len(result['commands'])} commands")
    """
    llm_client = llm_service if llm_service is not None else global_llm_service
    commands = _read_available_commands(database_path)
    command_user_prompt, command_system_prompt = _command_selection_prompts(
        commands=commands,
        dir_structure=dir_structure,
        case_info=case_info,
        allrun_reference=allrun_reference,
        mesh_type=mesh_type,
        mesh_commands=mesh_commands,
        user_requirement=user_requirement,
    )
    command_response = llm_client.invoke(
        command_user_prompt,
        command_system_prompt,
        pydantic_obj=CommandsPydantic,
    )

    _report_progress(
        progress_callback,
        progress_offset + 1,
        total_steps,
        "Generated Allrun commands",
    )

    if not command_response.commands:
        raise ValueError("Failed to generate commands.")
    command_help = _command_help_context(command_response.commands, searchdocs, config)
    allrun_user_prompt, allrun_system_prompt = _allrun_script_prompts(
        command_help=command_help,
        dir_structure=dir_structure,
        case_info=case_info,
        allrun_reference=allrun_reference,
        mesh_type=mesh_type,
        mesh_commands=mesh_commands,
        user_requirement=user_requirement,
    )
    allrun_response = llm_client.invoke(allrun_user_prompt, allrun_system_prompt)

    _report_progress(
        progress_callback,
        progress_offset + 2,
        total_steps,
        "Generated Allrun script",
    )

    allrun_script = _parse_allrun_response(allrun_response)
    allrun_script = _ensure_checkmesh_before_solver(
        allrun_script,
        case_info=case_info,
        dir_structure=dir_structure,
        mesh_type=mesh_type,
        mesh_commands=mesh_commands,
    )
    allrun_file_path = os.path.join(case_dir, "Allrun")
    save_file(allrun_file_path, allrun_script)
    
    return {
        "allrun_path": allrun_file_path,
        "allrun_script": allrun_script,
        "commands": command_response.commands,
    }


def _load_rewrite_context(
    case_dir: str,
    dir_structure: Optional[Dict[str, List[str]]],
    foamfiles: Optional[Any],
) -> tuple[Dict[str, List[str]], FoamPydantic]:
    """Load the current files only when the workflow did not provide them."""
    if dir_structure is None:
        print(f"Scanning directory structure from: {case_dir}")
        dir_structure = scan_case_directory(case_dir)
    if foamfiles is None:
        print(f"Reading OpenFOAM files from: {case_dir}")
        foamfiles = read_case_foamfiles(case_dir, dir_structure)
    return dir_structure, foamfiles


def _rewrite_prompts(
    *,
    foamfiles: FoamPydantic,
    error_logs: List[str],
    review_analysis: str,
    rewrite_plan: Optional[Dict[str, Any]],
    user_requirement: str,
    openfoam_fork: str,
    case_solver: str,
) -> tuple[str, str]:
    """Build the constrained rewrite request and its compatibility contract."""
    generation_contract = _generation_contract(openfoam_fork)
    if rewrite_plan is None:
        scope_instruction = (
            "No rewrite plan was supplied. Modify only the files necessary to "
            "resolve the reported error. "
        )
        scope_request = "Update only the files necessary to resolve the reported error."
    else:
        scope_instruction = (
            "Follow rewrite_plan strictly; do not modify files outside "
            "rewrite_plan.target_files. "
        )
        scope_request = "Update only the files listed in rewrite_plan.target_files."
    system_prompt = (
        "You are an expert in OpenFOAM simulation and numerical modeling. "
        "Modify files only to resolve the reported error without changing user-specified parameters. "
        f"{scope_instruction}"
        f"The selected solver is {case_solver or 'not specified'}. "
        "Apply this OpenFOAM compatibility contract to every rewritten file:\n"
        f"{generation_contract}\n"
        "Return complete corrected files as JSON: "
        "list of foamfile: [{file_name, folder_name, content}]."
    )
    user_prompt = (
        f"<foamfiles>{foamfiles}</foamfiles>\n"
        f"<error_logs>{error_logs}</error_logs>\n"
        f"<reviewer_analysis>{review_analysis}</reviewer_analysis>\n"
        f"<rewrite_plan>{rewrite_plan}</rewrite_plan>\n"
        f"<user_requirement>{user_requirement}</user_requirement>\n"
        f"<openfoam_fork>{openfoam_fork}</openfoam_fork>\n"
        f"<case_solver>{case_solver}</case_solver>\n"
        f"{scope_request}"
    )
    return user_prompt, system_prompt


def _allowed_rewrite_files(
    rewrite_plan: Optional[Dict[str, Any]],
) -> Optional[set[str]]:
    """Validate supplied review targets while preserving legacy unscoped rewrites.

    The MCP ``apply_fixes`` tool predates rewrite plans and intentionally calls
    this path with ``None``.  In that case, retain the original behavior of
    applying all safe paths returned by the LLM.  Graph-driven rewrites that
    supply a plan remain constrained to its validated target files.
    """
    if rewrite_plan is None:
        return None

    allowed_files: set[str] = set()
    if isinstance(rewrite_plan, dict):
        for item in rewrite_plan.get("target_files", []):
            file_path = item.get("file") if isinstance(item, dict) else None
            if not file_path:
                continue
            try:
                allowed_files.add(safe_case_relative_from_text(file_path).as_posix())
            except CasePathSafetyError as exc:
                raise ValueError(f"Unsafe rewrite target in plan: {file_path!r}") from exc
    if not allowed_files:
        raise ValueError("rewrite_plan.target_files must contain at least one safe file path")
    return allowed_files


def _apply_rewrite_response(
    *,
    case_dir: str,
    dir_structure: Dict[str, List[str]],
    foamfiles: FoamPydantic,
    response: FoamPydantic,
    allowed_files: Optional[set[str]],
) -> tuple[Dict[str, List[str]], FoamPydantic, List[str]]:
    """Apply safe LLM files, constrained when an allow-list is supplied."""
    updated_dir = {folder: list(files) for folder, files in dir_structure.items()}
    by_relative_path = {
        safe_case_relative_path(item.folder_name, item.file_name).as_posix(): item
        for item in foamfiles.list_foamfile
    }
    updated_files: List[str] = []
    for foamfile in response.list_foamfile:
        try:
            relative = safe_case_relative_path(foamfile.folder_name, foamfile.file_name)
            file_path = safe_case_path(case_dir, foamfile.folder_name, foamfile.file_name)
        except CasePathSafetyError as exc:
            raise ValueError(
                "LLM returned an unsafe rewrite path: "
                f"{foamfile.folder_name!r}/{foamfile.file_name!r}"
            ) from exc
        relative_path = relative.as_posix()
        if allowed_files is not None and relative_path not in allowed_files:
            print(f"Warning: Skipping unplanned rewrite file: {relative_path}")
            continue
        folder_name = "" if relative.parent.as_posix() == "." else relative.parent.as_posix()
        file_name = relative.name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(str(file_path), foamfile.content)
        updated_dir.setdefault(folder_name, [])
        if file_name not in updated_dir[folder_name]:
            updated_dir[folder_name].append(file_name)
        by_relative_path[relative_path] = FoamfilePydantic(
            file_name=file_name,
            folder_name=folder_name,
            content=foamfile.content,
        )
        updated_files.append(relative_path)
    return updated_dir, FoamPydantic(list_foamfile=list(by_relative_path.values())), updated_files



def rewrite_files(
    case_dir: str,
    error_logs: List[str],
    review_analysis: str,
    rewrite_plan: Optional[Dict[str, Any]],
    user_requirement: str,
    foamfiles: Optional[Any] = None,
    dir_structure: Optional[Dict[str, List[str]]] = None,
    openfoam_fork: str = "foundation",
    case_solver: str = "",
    llm_service: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Rewrite OpenFOAM files based on error analysis and reviewer suggestions.
    
    This function analyzes error logs and reviewer suggestions to identify
    problematic files, then uses LLM to generate corrected versions of
    the files that need modification.
    
    The function automatically reads foamfiles and directory structure from
    case_dir if they are not provided.
    
    Args:
        case_dir (str): Directory path where the case files are located
        error_logs (List[str]): List of error messages from simulation runs
        review_analysis (str): Analysis and suggestions from the reviewer (required)
        user_requirement (str): Original user requirements for context
        foamfiles (Optional[Any]): FoamPydantic object containing current file contents.
                                   If None, will be read from case_dir.
        dir_structure (Optional[Dict[str, List[str]]]): Current directory structure.
                                                        If None, will be scanned from case_dir.
        openfoam_fork (str, optional): Target OpenFOAM fork. Defaults to "foundation".
        case_solver (str, optional): Selected solver for compatibility checks.
    
    Returns:
        Dict[str, Any]: Contains:
            - dir_structure (Dict[str, List[str]]): Updated directory structure
            - foamfiles (FoamPydantic): Updated file contents with corrections
            - error_logs (List[str]): Cleared error logs (empty on success)
    
    Raises:
        FileNotFoundError: If case directory does not exist
        ValueError: If review_analysis is empty or foamfiles format is invalid
        RuntimeError: If LLM service fails to generate corrections
    
    Example:
        >>> result = rewrite_files(
        ...     case_dir="/path/to/case",
        ...     error_logs=["Error: undefined reference"],
        ...     review_analysis="Add missing boundary condition",
        ...     user_requirement="Simple flow simulation"
        ...     # foamfiles and dir_structure will be read automatically
        ... )
        >>> print(f"Updated {len(result['foamfiles'].list_foamfile)} files")
    """
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    if not review_analysis or review_analysis.strip() == "":
        raise ValueError("review_analysis is required and cannot be empty")
    dir_structure, foamfiles = _load_rewrite_context(case_dir, dir_structure, foamfiles)
    rewrite_user_prompt, rewrite_system_prompt = _rewrite_prompts(
        foamfiles=foamfiles,
        error_logs=error_logs,
        review_analysis=review_analysis,
        rewrite_plan=rewrite_plan,
        user_requirement=user_requirement,
        openfoam_fork=openfoam_fork,
        case_solver=case_solver,
    )
    allowed_files = _allowed_rewrite_files(rewrite_plan)
    llm_client = llm_service if llm_service is not None else global_llm_service
    response = llm_client.invoke(
        rewrite_user_prompt,
        rewrite_system_prompt,
        pydantic_obj=FoamPydantic,
    )

    updated_dir, updated_foamfiles, updated_files = _apply_rewrite_response(
        case_dir=case_dir,
        dir_structure=dir_structure,
        foamfiles=foamfiles,
        response=response,
        allowed_files=allowed_files,
    )
    return {
        "dir_structure": updated_dir,
        "foamfiles": updated_foamfiles,
        "error_logs": [],
        "updated_files": updated_files,
    }
