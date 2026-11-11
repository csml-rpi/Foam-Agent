from logger import print
import os
import re
import gzip
import zlib
import subprocess
from threading import Lock
from difflib import unified_diff
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from utils import save_file, parse_context, retrieve_faiss, FoamPydantic, FoamfilePydantic, scan_case_directory, case_file_key, resolve_case_file, read_case_text, save_case_text
import json
from config import Config
from openfoam_target import database_path_for_config, generation_contract as _generation_contract, runtime_openfoam_target
from . import get_llm_service
from .stage_plan import script_commands, render_stage_script, check_file_targets, unwritten_file_errors
from .case_inputs import format_case_inputs, read_case_inputs
from .execution_logs import declared_startup_inputs
from .solver_execution import read_case_application
from .case_paths import (
    CasePathSafetyError,
    safe_case_path,
    safe_case_relative_from_text,
    safe_case_relative_path,
)


def parse_shell_script(text: str) -> str:
    """Remove an outer code fence without interpreting the shell body."""
    stripped = text.strip()
    lines = stripped.splitlines(keepends=True)
    if (len(lines) >= 2
            and re.fullmatch(r"```(?:bash|sh|shell|zsh|ksh)?", lines[0].strip(), re.I)
            and lines[-1].strip() == "```"):
        return "".join(lines[1:-1])
    return text


def normalise_initial_subtasks(subtasks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Validate generation targets before preparing or writing their output."""
    normalized = []
    seen = set()
    for subtask in subtasks:
        if not isinstance(subtask, dict):
            raise ValueError(f"Invalid subtask format: {subtask!r}")
        try:
            relative = safe_case_relative_path(
                subtask.get("folder_name", ""), subtask.get("file_name", ""),
            )
        except CasePathSafetyError as exc:
            raise ValueError(f"Unsafe generated subtask path: {subtask!r}") from exc
        folder = "" if relative.parent.as_posix() == "." else relative.parent.as_posix()
        target = (folder, case_file_key(relative.name))
        if target in seen:
            raise ValueError(f"Duplicate generated subtask target: {relative}")
        seen.add(target)
        normalized.append({**subtask, "folder_name": folder, "file_name": relative.name})
    return normalized


def generate_allrun_script(
    user_prompt: str, *, allrun_reference: str, openfoam_fork: str,
    phase: str, llm_client: Any, command_context: str = "", operations=None,
) -> str:
    """Shared renderer for automatic, explicitly requested and repaired Allrun."""
    if operations is not None:
        return render_stage_script(operations)
    script_name = 'Allmesh' if phase == 'mesh' else 'Allrun'
    system = (
        f"You are an expert in OpenFOAM. Generate the complete {script_name} shell script. "
        "Return only shell code, optionally inside one outer shell code fence. "
        "For edits, preserve the existing script except for the requested corrections. "
        "Propagate failures of every required preparation and solver command. Use set -e; "
        "if using pipelines, use bash with set -o pipefail. Explicitly handle any expected "
        "nonzero status. Do not mask errors with a final successful command. "
        "RunFunctions owns its log files: never redirect runApplication or runParallel "
        "to the same log it creates. Use distinct log suffixes for repeated commands. "
        "Retain the solver's native output, including its Exec header and normal End, "
        "in root-level log files or standard output. Do not fabricate completion markers. "
        "Do not add post-processing commands.\n"
        + _generation_contract(openfoam_fork)
    )
    if phase == "solver":
        system += (
            "\nThe script is Allrun. The mesh has already been prepared. Only physical "
            "field preparation, solver-parallel setup and solving belong here. Do not "
            "generate, convert, check, modify or clean up the mesh, or call Allmesh."
        )
    elif phase == 'mesh':
        system += '\nAllmesh contains mesh work only; preserve unrelated shell content and do not prepare physical fields or run a solver.'
    system += f"\nCommand guidance: {command_context}\nReference Allrun: {allrun_reference}"
    script = parse_shell_script(llm_client.invoke(user_prompt, system))
    if not script.strip():
        raise ValueError("The LLM returned an empty Allrun script.")
    return script


def initial_write(
    case_dir: str,
    subtasks: List[Dict[str, str]],
    user_requirement: str,
    tutorial_reference: str,
    case_solver: str,
    generation_mode: Optional[str] = None,
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
    allrun_phase: str = "all",
    operations: Optional[List[dict]] = None,
    file_written_callback: Optional[Callable[[str, bool], None]] = None,
) -> Dict[str, Any]:
    """
    Generate OpenFOAM files from scratch based on user requirements and subtasks.
    
    This function creates OpenFOAM input files by analyzing user requirements,
    using similar case references and preserving the Planner's dependency order
    when sharing generated context. It also generates an Allrun script
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
            - written_files / updated_files: Successful receipts / actual content or representation changes
    
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
    llm_client = llm_service
    client_lock = Lock()
    receipt_lock = Lock()
    changed_files = set()

    def _file_written(path: str, changed: bool) -> None:
        with receipt_lock:
            if changed:
                changed_files.add(path)
            if file_written_callback is not None:
                file_written_callback(path, changed)

    def _before_write(path):
        previous = resolve_case_file(path)
        try:
            content = read_case_text(previous, errors='replace') if previous.is_file() else None
        except (EOFError, gzip.BadGzipFile, zlib.error):
            content = None  # Explicit generation can replace an unreadable gzip input.
        return previous, content

    def _client():
        nonlocal llm_client
        with client_lock:
            llm_client = get_llm_service(llm_client, config)
        return llm_client

    def _report_progress(current: int, total: int, message: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(current, total, message)
        except Exception as exc:  # noqa: BLE001 - progress_callback implementation is external
            print(f"<progress_callback_error>{exc}</progress_callback_error>")

    if generation_mode is None:
        generation_mode = getattr(config, "generation_mode", "sequential_dependency")
    if generation_mode not in {"sequential_dependency", "parallel_no_context"}:
        raise ValueError(
            f"Unsupported generation_mode: {generation_mode}. "
            "Expected one of: sequential_dependency, parallel_no_context"
        )

    subtasks = normalise_initial_subtasks(subtasks)
    seen_targets = {(task["folder_name"], case_file_key(task["file_name"])) for task in subtasks}
    targets = [safe_case_relative_path(t['folder_name'], t['file_name']).as_posix() for t in subtasks]
    if database_path and ("", "Allrun") not in seen_targets:
        targets.append("Allrun")
    target_errors = check_file_targets(case_dir, targets)
    if target_errors:
        return {'dir_structure': {}, 'foamfiles': FoamPydantic(list_foamfile=[]),
                'written_files': [], 'updated_files': [], 'error_logs': target_errors}
    written_files = []
    dir_structure = {}
    generation_contract = _generation_contract(openfoam_fork)
    advice = similar_case_advice.model_dump() if hasattr(similar_case_advice, "model_dump") else similar_case_advice
    pre_solver_steps = advice.get("pre_solver_steps", []) if operations is None and isinstance(advice, dict) else []
    INITIAL_WRITE_SYSTEM_PROMPT = (
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

    def _build_prompts(file_name: str, folder_name: str, written_files_ctx: List[FoamfilePydantic]) -> tuple[str, str]:
        logical_name = case_file_key(file_name)
        code_system_prompt = INITIAL_WRITE_SYSTEM_PROMPT.format(
            file_name=logical_name,
            folder_name=folder_name,
            case_solver=case_solver,
            generation_contract=generation_contract,
        )
        code_system_prompt += (
            "\nThe per-file planning advice describes how to use the reference. "
            "For reuse, retain its structure and settings except changes required by the user, "
            "the target runtime, or consistency with actual generated dependencies. "
            "For modify, apply the described delta and any other changes necessary for those same constraints. "
            "Do not redesign compatible reference settings or add speculative numerical tuning. "
            "The advice is fallible: explicit user requirements and actual compatibility take precedence; "
            "a reference default is never an additional user requirement. "
        )
        advice_text = ""
        file_reference = tutorial_reference
        if isinstance(advice, dict) and "per_file" in advice:
            file_advice = next((item for item in advice["per_file"]
                                if item.get("folder_name", "").strip("/.") == folder_name.strip("/.")
                                and case_file_key(item.get("file_name", "")) == logical_name), None)
            if file_advice:
                advice_text = f"<file_advice>{file_advice}</file_advice>\n"
                if file_advice.get("classification") in {"present_but_unused", "write_fresh"}:
                    file_reference = ""
        elif advice:
            advice_text = f"<similar_case_advice>{advice}</similar_case_advice>\n"
        similar_ref_block = (
            f"<similar_case_reference>{file_reference}</similar_case_reference>\n"
            if file_reference else "Generate this file from the user requirement and compatibility contract.\n"
        )

        code_user_prompt = (
            f"User requirement: {user_requirement}\n"
            f"Case plan: {case_info}\n"
            f"Planned inputs in generation order: {json.dumps(subtasks, ensure_ascii=False)}\n"
            f"{similar_ref_block}{advice_text}\n"
            "If the similar case is a weak match, do not copy it blindly. Use it only where it is consistent with the user requirement. "
            "The OpenFOAM compatibility contract in the system prompt takes precedence over incompatible syntax in a reference case. "
            "Just modify the necessary parts to make the file complete and functional."
            "Please ensure that the generated file is complete, functional, and logically sound."
            "Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence."
            "When generating controlDict, do not include anything to preform post processing. Just include the necessary settings to run the simulation."
        )

        if logical_name == "fvSolution":
            code_user_prompt += (
                "\n\nCRITICAL for transient pressure-velocity coupling solvers using PISO/PIMPLE: "
                "the solvers dictionary must include matching Final solver entries for fields used on the final correction. "
                "For example, if p is defined, include pFinal { $p; relTol 0; }; "
                "if U is defined, include UFinal { $U; relTol 0; }. "
                "For grouped regex entries, use the matching grouped Final entry, e.g. "
                "\"(U|k|epsilon)Final\" { $U; relTol 0; }. "
                "Do not emit placeholder text such as $<field>; in the generated file. "
                "Also ensure the PIMPLE/PISO sub-dictionary matches the selected solver."
            )

        if generation_mode == "sequential_dependency" and written_files_ctx:
            generated_paths = [Path(file.folder_name) / file.file_name for file in written_files_ctx]
            generated, omitted = read_case_inputs(
                case_dir, protected_inputs=generated_paths, selected_paths=generated_paths)
            code_user_prompt += (
                "The following are files content already generated:\n"
                f"{format_case_inputs(generated)}\n"
                f"<omitted_files>{json.dumps(omitted, ensure_ascii=False)}</omitted_files>\n"
                "You should ensure that the new file is consistent with the previous files. Such as boundary conditions, mesh settings, etc."
            )

        return code_user_prompt, code_system_prompt

    def _generate_one_file(subtask: Dict[str, str], written_files_ctx: List[FoamfilePydantic], file_path: Path) -> FoamfilePydantic:
        file_name = subtask["file_name"]
        folder_name = subtask["folder_name"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        script_name = 'Allmesh' if allrun_phase == 'mesh' else 'Allrun'
        if folder_name in ('', '.') and file_name == script_name and operations is not None:
            content = render_stage_script(operations)
            save_file(str(file_path), content)
            return FoamfilePydantic(file_name=file_name, folder_name=folder_name, content=content)
        if reuse_generated_dir:
            reuse_src = resolve_case_file(safe_case_path(reuse_generated_dir, folder_name, file_name))
            if reuse_src.is_file():
                print(f"Reusing generated file: {reuse_src}")
                content = read_case_text(reuse_src)
                actual = save_case_text(file_path, content)
                return FoamfilePydantic(
                    file_name=actual.name,
                    folder_name=folder_name,
                    content=content,
                )
        code_user_prompt, code_system_prompt = _build_prompts(file_name, folder_name, written_files_ctx)
        code_user_prompt += f"\n<file_task>{subtask.get('changes', '')}</file_task>"
        if operations is not None:
            code_user_prompt += f"\n<approved_operations>{operations}</approved_operations>\nFollow these operations without adding or independently selecting commands."
        if folder_name == "" and file_name == "Allrun":
            current_inputs, omitted = read_case_inputs(case_dir)
            code_context = generate_allrun_script(
                code_user_prompt + f"\nSelected solver: {case_solver}\n<existing_files>{format_case_inputs(current_inputs)}</existing_files>"
                + f"\n<omitted_files>{json.dumps(omitted, ensure_ascii=False)}</omitted_files>"
                + f"\nRequired pre-solver steps: {pre_solver_steps}",
                allrun_reference=allrun_reference, openfoam_fork=openfoam_fork,
                phase=allrun_phase, llm_client=_client(),
            )
            save_file(str(file_path), code_context)
            return FoamfilePydantic(file_name=file_name, folder_name=folder_name, content=code_context)
        generation_response = _client().invoke(code_user_prompt, code_system_prompt)
        shell = folder_name in ("", ".") and file_name in {"Allrun", "Allmesh"}
        code_context = parse_shell_script(generation_response) if shell else parse_context(generation_response)
        actual = save_case_text(file_path, code_context)
        return FoamfilePydantic(file_name=actual.name, folder_name=folder_name, content=code_context)

    def _generate_one(subtask: Dict[str, str], written_files_ctx: List[FoamfilePydantic]) -> FoamfilePydantic:
        file_path = safe_case_path(case_dir, subtask['folder_name'], subtask['file_name'])
        previous, before = _before_write(file_path)
        generated = _generate_one_file(subtask, written_files_ctx, file_path)
        relative = safe_case_relative_path(generated.folder_name, generated.file_name).as_posix()
        _file_written(relative, before != generated.content or previous != file_path.with_name(generated.file_name))
        return generated

    # Build dir_structure upfront (deterministic ordering) and generate files
    for subtask in subtasks:
        folder_name = subtask.get("folder_name")
        file_name = subtask.get("file_name")
        if folder_name not in dir_structure:
            dir_structure[folder_name] = []
        dir_structure[folder_name].append(file_name)

    script_tasks = [t for t in subtasks if (t["folder_name"], t["file_name"]) == ("", "Allrun")]
    subtasks = [t for t in subtasks if (t["folder_name"], t["file_name"]) != ("", "Allrun")]
    generate_allrun = bool(database_path) and ("", "Allrun") not in seen_targets
    total_steps = len(subtasks) + len(script_tasks) + (2 if generate_allrun else 0)
    _report_progress(0, total_steps, f"Starting file generation for {len(subtasks)} files")

    if generation_mode == "parallel_no_context":
        print("<generation_mode>parallel_no_context (no cross-file context)</generation_mode>")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from contextvars import copy_context
        import threading

        # Parallelize generation; keep returned files in the provided plan order.
        results: List[Optional[FoamfilePydantic]] = [None] * len(subtasks)
        completed_count = 0
        count_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=min(32, max(4, len(subtasks)))) as ex:
            future_map = {
                ex.submit(copy_context().run, _generate_one, subtasks[i], []): i
                for i in range(len(subtasks))
            }
            for fut in as_completed(future_map):
                i = future_map[fut]
                results[i] = fut.result()
                with count_lock:
                    completed_count += 1
                    _report_progress(
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
            foamfile = _generate_one(subtask, written_files)
            written_files.append(foamfile)
            _report_progress(idx + 1, total_steps, f"Generated {file_name} in {folder_name}")
    
    for task in script_tasks:
        written_files.append(_generate_one(task, written_files))
        _report_progress(len(subtasks) + 1, total_steps, "Generated Allrun")

    # Generate a default Allrun only when no explicit root-level task supplies it.
    if generate_allrun:
        script_path = safe_case_path(case_dir, '', 'Allmesh' if allrun_phase == 'mesh' else 'Allrun')
        previous, before = _before_write(script_path)
        allrun_result = build_allrun(
            case_dir, database_path, searchdocs, dir_structure, case_info,
            allrun_reference, mesh_type, mesh_commands or [], user_requirement,
            progress_callback=progress_callback,
            progress_offset=len(subtasks),
            total_steps=total_steps,
            llm_service=llm_client,
            config=config,
            openfoam_fork=openfoam_fork,
            pre_solver_steps=pre_solver_steps,
            phase=allrun_phase, operations=operations,
        )
        actual = Path(allrun_result['allrun_path'])
        _file_written(actual.relative_to(Path(case_dir).expanduser().resolve()).as_posix(),
                      before != allrun_result['allrun_script'] or previous != actual)
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
    for foamfile in written_files:
        folder = dir_structure.setdefault(foamfile.folder_name, [])
        folder[:] = [name for name in folder if case_file_key(name) != case_file_key(foamfile.file_name)]
        folder.append(foamfile.file_name)
    print("</initial_write_service>")
    return {"dir_structure": dir_structure, "foamfiles": foamfiles,
            "written_files": [safe_case_relative_path(f.folder_name, f.file_name).as_posix()
                              for f in written_files],
            "updated_files": [safe_case_relative_path(f.folder_name, f.file_name).as_posix()
                              for f in written_files
                              if safe_case_relative_path(f.folder_name, f.file_name).as_posix() in changed_files]}


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
    openfoam_fork: str = "foundation",
    phase: str = "all",
    pre_solver_steps: Optional[List[str]] = None,
    operations: Optional[List[dict]] = None,
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
        llm_service: Task-specific LLM; defaults to the global service.
        config: Selects the target corpus and embedding model for retrieval.
        openfoam_fork: OpenFOAM generation conventions for the selected target.
    
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
    if operations is not None:
        script = render_stage_script(operations)
        path = str(safe_case_path(case_dir, '', 'Allmesh' if phase == 'mesh' else 'Allrun'))
        save_file(path, script)
        return {'allrun_path': path, 'allrun_script': script, 'commands': script_commands(operations)}
    from pydantic import BaseModel, Field
    from typing import List
    
    llm_client = get_llm_service(llm_service, config)
    retrieval_config = config if config is not None else Config()
    if config is None and openfoam_fork == "esi-v2006":
        retrieval_config.openfoam_target = "esi-v2006"
    if config is not None or openfoam_fork == "esi-v2006":
        database_path = str(database_path_for_config(retrieval_config))

    input_context = ""
    if os.path.isdir(case_dir):
        current_inputs, omitted = read_case_inputs(case_dir)
        input_context = (
            f"<current_case_inputs>{format_case_inputs(current_inputs)}</current_case_inputs>\n"
            f"<omitted_files>{json.dumps(omitted, ensure_ascii=False)}</omitted_files>\n"
        )

    # CommandsPydantic class for structured response
    class CommandsPydantic(BaseModel):
        commands: List[str] = Field(description="List of commands")
    
    # Retrieve commands from file
    command_path = f"{database_path}/raw/openfoam_commands.txt"
    try:
        with open(command_path, 'r') as file:
            commands = file.readlines()
        commands = f"[{', '.join([c.strip() for c in commands])}]"
    except (FileNotFoundError, IOError) as e:
        raise ValueError(f"Could not read commands file {command_path}: {e}")

    # Handle mesh commands info
    mesh_commands_info = ""
    if mesh_type == "custom_mesh" and mesh_commands:
        mesh_commands_info = f"\nCustom mesh commands to include: {mesh_commands}"
        print(f"Including custom mesh commands: {mesh_commands}")

    # Command generation system prompt
    command_system_prompt = (
        "You are an expert in OpenFOAM. The user will provide a list of available commands. "
        "Your task is to generate only the necessary OpenFOAM commands required to create an Allrun script for the given user case, based on its current inputs and case plan. "
        "Return only the list of commands—no explanations, comments, or additional text."
    )

    stage_prompt = ""
    if phase == "mesh":
        stage_prompt = (
            "\nGenerate only the mesh stage, following the supplied ordered commands "
            "and their arguments exactly. Do not run a solver or prepare physical fields. "
            "Retain shell setup and helpers necessary for those commands. "
            "The script is named Allmesh. Use set -e and propagate every command failure. Use distinct runApplication log suffixes for repeated utilities. Final mesh must be in constant/polyMesh; use the appropriate overwrite or reconstruction operations.\n"
        )
    elif phase == "solver":
        stage_prompt = (
            "\nThe actual mesh is already prepared by Meshing. Generate only the solver "
            "stage. Do not generate, convert, check, modify or clean up the mesh, and "
            "do not call Allmesh or other mesh scripts. Preserve necessary physical "
            "field preparation and solver-parallel decomposition. The script is Allrun.\n"
        )

    if mesh_type == "custom_mesh" and phase == "all":
        command_system_prompt += "If custom mesh commands are provided, include them in the appropriate order (typically after blockMesh or instead of blockMesh if custom mesh is used). "
    
    command_user_prompt = (
        f"Available OpenFOAM commands for the Allrun script: {commands}\n"
        f"User requirement: {user_requirement}\n"
        f"Case directory structure: {dir_structure}\n"
        f"{input_context}"
        f"User case information: {case_info}\n"
        f"Reference Allrun scripts from similar cases: {allrun_reference}\n"
        "Generate only the required OpenFOAM command list—no extra text."
    )

    if mesh_type == "custom_mesh":
        command_user_prompt += f"{mesh_commands_info}\n"
    
    command_system_prompt += stage_prompt + "\n" + _generation_contract(openfoam_fork)

    if pre_solver_steps:
        command_user_prompt += f"\nRequired pre-solver steps in order: {pre_solver_steps}\n"
    command_response = (
        CommandsPydantic(commands=mesh_commands) if phase == "mesh" else
        llm_client.invoke(command_user_prompt, command_system_prompt, pydantic_obj=CommandsPydantic)
    )

    if progress_callback:
        try:
            progress_callback(progress_offset + 1, total_steps, "Generated Allrun commands")
        except Exception:
            pass

    if len(command_response.commands) == 0:
        print("Failed to generate commands.")
        raise ValueError("Failed to generate commands.")

    print(f"Need {len(command_response.commands)} commands.")
    
    # Get command help from FAISS
    commands_help = []
    for command in command_response.commands:
        command_help = retrieve_faiss("openfoam_command_help", command, topk=searchdocs, config=retrieval_config)
        commands_help.append(command_help[0]['full_content'])
    commands_help = "\n".join(commands_help)

    # Allrun generation system prompt
    allrun_system_prompt = (
        "You are an expert in OpenFOAM. Generate an Allrun script based on the provided details."
        "The listed case inputs have already been generated. Allrun orchestrates tools; "
        "do not recreate those inputs with duplicate inline definitions. It must remain "
        "safe to rerun after repairs: preserve preprocessing templates and do not overwrite "
        "a repaired dictionary with an embedded copy. "
        f"Available commands with descriptions: {commands_help}\n\n"
        f"Reference Allrun scripts from similar cases: {allrun_reference}\n\n"
        "If custom mesh commands are provided, make sure to include them in the appropriate order in the Allrun script. "
        "CRITICAL: Do not include any post processing commands in the Allrun script."
        + (
            "CRITICAL: Do not include any commands to convert mesh to foam format like gmshToFoam or others."
            if phase != "mesh" else ""
        )
    )

    if mesh_type == "custom_mesh" and phase == "all":
        allrun_system_prompt += "CRITICAL: Do not include any other mesh commands other than the custom mesh commands.\n"
        allrun_system_prompt += "CRITICAL: Do not include any gmshToFoam commands in the Allrun script."
    
    allrun_user_prompt = (
        f"User requirement: {user_requirement}\n"
        f"Case directory structure: {dir_structure}\n"
        f"{input_context}"
        f"User case infomation: {case_info}\n"
        f"{mesh_commands_info}\n"
        "All run scripts for these similar cases are for reference only and may not be correct, as you might be a different case solver or have a different directory structure. " 
        "You need to rely on your OpenFOAM and physics knowledge to discern this, and pay more attention to user requirements, " 
        "as your ultimate goal is to fulfill the user's requirements and generate an allrun script that meets those requirements."
        "CRITICAL: Do not include any post processing commands in the Allrun script."
        + (
            "CRITICAL: Do not include any commands to convert mesh to foam format like gmshToFoam or others."
            "CRITICAL: Do not include any commands that run gmsh to create the mesh."
            if phase != "mesh" else ""
        )
        + "Generate the Allrun script strictly based on the above information. Do not include explanations, comments, or additional text. Put the code in ``` tags."
    )

    if mesh_type == "custom_mesh" and phase == "all":
        allrun_user_prompt += "CRITICAL: Do not include any other mesh commands other than the custom mesh commands.\n"
        allrun_user_prompt += "CRITICAL: Do not include any gmshToFoam commands in the Allrun script."

    allrun_system_prompt += stage_prompt + "\n" + _generation_contract(openfoam_fork)
    if phase == "mesh":
        allrun_user_prompt += f"\nExecute these planned commands in order: {mesh_commands}\n"

    if pre_solver_steps:
        allrun_user_prompt += f"\nRequired pre-solver steps in order: {pre_solver_steps}\n"
    allrun_response = (
        llm_client.invoke(allrun_user_prompt, allrun_system_prompt)
        if phase == "mesh" else generate_allrun_script(
            allrun_user_prompt,
            allrun_reference=allrun_reference, openfoam_fork=openfoam_fork,
            phase=phase, llm_client=llm_client,
            command_context=f"Planned commands: {command_response.commands}\n{commands_help}",
        )
    )

    if progress_callback:
        try:
            progress_callback(progress_offset + 2, total_steps, "Generated Allrun script")
        except Exception:
            pass

    allrun_script = parse_shell_script(allrun_response)
    if not allrun_script.strip():
        raise ValueError("The LLM returned an empty Allrun script.")
    allrun_file_path = str(safe_case_path(case_dir, '', 'Allmesh' if phase == 'mesh' else 'Allrun'))
    save_file(allrun_file_path, allrun_script)
    
    return {"allrun_path": allrun_file_path, "allrun_script": allrun_script, "commands": command_response.commands}



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
    allrun_reference: str = "",
    operations: Optional[List[dict]] = None,
    script_phase: str = 'solver',
    review_context: str = "",
    tutorial_reference: str = "",
    omitted_files: Optional[Dict[str, str]] = None,
    static_repair: bool = False,
    config: Optional[Config] = None,
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
    # Validate case directory exists
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    
    # Validate review_analysis is provided
    if not review_analysis or review_analysis.strip() == "":
        raise ValueError("review_analysis is required and cannot be empty")
    
    # Read directory structure if not provided
    if dir_structure is None:
        print(f"Scanning directory structure from: {case_dir}")
        dir_structure = scan_case_directory(case_dir)
    
    # Read foamfiles if not provided
    if foamfiles is None:
        print(f"Reading OpenFOAM files from: {case_dir}")
        foamfiles, omitted_files = read_case_inputs(
            case_dir, protected_inputs=declared_startup_inputs(operations or []))
    
    # Resolve the actual selected dictionary, independently of bounded prompt
    # contents or a caller's stale plain/gzip context.
    control_dict = Path(case_dir) / 'system/controlDict'
    try:
        if resolve_case_file(control_dict).is_file():
            target = runtime_openfoam_target(config) if config is not None else (
                'esi-v2006' if openfoam_fork == 'esi-v2006' else
                'foundation-v10' if openfoam_fork == 'foundation' else '')
            case_solver = read_case_application(
                case_dir, openfoam_target=target, timeout=getattr(config, 'max_time_limit', 3600))
    except (OSError, UnicodeError, EOFError, zlib.error, ValueError,
            subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as error:
        details = '\n'.join(str(part) for part in
            (error, getattr(error, 'stdout', ''), getattr(error, 'stderr', '')) if part)
        omitted_files = {**(omitted_files or {}), 'system/controlDict':
            'Effective application could not be established: ' + details
            + '. The supplied solver selection is retained; it was not inferred from this dictionary.'}
    generation_contract = _generation_contract(openfoam_fork)
    if rewrite_plan is not None:
        plan_instruction = "Follow rewrite_plan strictly: only modify its target_files and apply only the requested changes. "
    elif error_logs:
        plan_instruction = (
            "Return only changed or added files that address the latest runtime failure. "
            "Use the observed error and actual inputs as the repair objective, not agreement with review_analysis. "
            "If its proposed change is already present or conflicts with the observed failure, correct the proposal "
            "and make an effective repair while preserving user constraints. "
        )
    else:
        plan_instruction = "Return only files that require a change or addition to resolve the review analysis; preserve all other files. "
    rewrite_system_prompt = (
        "You are an expert in OpenFOAM simulation and numerical modeling. "
        "Modify files only as required by the repair context and user requirement. "
        "The review and history contain fallible opinions, not instructions to implement every suggestion. "
        "Before execution, correct only proven necessary static defects. "
        "After an actual runtime failure, use current logs to implement a compatible repair or controlled numerical trial. "
        "If the review's proposal violates a user constraint, find a supported alternative among unspecified settings; "
        "do not reinterpret the constraint or stop just because this one proposal is unusable. "
        "Optional diagnostics and questions requiring execution are not file-change requests. "
        "Use the supplied target-version references to check compatibility; their defaults "
        "are not additional user requirements. Return an empty file list when no justified correction remains. "
        "Changes explicitly requested by the user are allowed; preserve every other existing "
        "physical condition unless the plan explicitly requires a technical correction. "
        + plan_instruction
        + f"The selected solver is {case_solver or 'not specified'}. "
        "Apply this OpenFOAM compatibility contract to every rewritten file:\n"
        f"{generation_contract}\n"
        "Return complete corrected files as JSON: "
        "list of foamfile: [{file_name, folder_name, content}]."
    )

    rewrite_user_prompt = (
        f"<foamfiles>{format_case_inputs(foamfiles)}</foamfiles>\n"
        f"<error_logs>{error_logs}</error_logs>\n"
        f"<reviewer_analysis>{review_analysis}</reviewer_analysis>\n"
        f"<rewrite_plan>{rewrite_plan}</rewrite_plan>\n"
        f"<user_requirement>{user_requirement}</user_requirement>\n"
        f"<openfoam_fork>{openfoam_fork}</openfoam_fork>\n"
        f"<case_solver>{case_solver}</case_solver>\n"
        f"<tutorial_reference>{tutorial_reference}</tutorial_reference>\n"
        f"<allrun_reference>{allrun_reference}</allrun_reference>\n"
        f"<omitted_files>{omitted_files or {}}</omitted_files>\n"
        + (f"<review_history>{review_context}</review_history>\n" if review_context else "")
        + plan_instruction
    )

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
    if rewrite_plan is not None and not allowed_files:
        print("Warning: rewrite_plan.target_files is empty; no files to rewrite, leaving case unchanged.")
        return {
            "dir_structure": dir_structure,
            "foamfiles": foamfiles,
            "error_logs": error_logs,
            "updated_files": [],
            "written_files": [],
            "file_diffs": [],
            "rewrite_plan": {'target_files': []},
        }
    planned_paths = {case_file_key(path): path for path in allowed_files}
    if len(planned_paths) != len(allowed_files):
        raise ValueError("Duplicate plain/gzip representations of a rewrite target")

    target_errors = check_file_targets(case_dir, allowed_files)
    if target_errors:
        return {'dir_structure': dir_structure, 'foamfiles': foamfiles,
                'error_logs': target_errors, 'written_files': [], 'updated_files': [],
                'file_diffs': [], 'rewrite_plan': rewrite_plan}

    llm_client = get_llm_service(llm_service, config)
    script_name = 'Allmesh' if script_phase == 'mesh' else 'Allrun'
    script_requested = script_name in allowed_files
    other_targets = allowed_files - {script_name}
    response = (
        llm_client.invoke(
            rewrite_user_prompt,
            rewrite_system_prompt + (f" {script_name} is handled separately: do not return it." if script_requested else ""),
            pydantic_obj=FoamPydantic,
        ) if not script_requested or other_targets else FoamPydantic(list_foamfile=[])
    )
    script_requested |= any(
        f.file_name == script_name and f.folder_name in ("", ".")
        and (rewrite_plan is None or script_name in allowed_files)
        for f in response.list_foamfile
    )
    if script_requested:
        # Generate after the other proposed repairs are available, without changing
        # the on-disk script before the normal diff/allowed-target handling below.
        repaired_inputs = [f for f in response.list_foamfile if f.file_name != script_name or f.folder_name not in ("", ".")]
        script = generate_allrun_script(
            rewrite_user_prompt
            + f"\n<proposed_file_repairs>{repaired_inputs}</proposed_file_repairs>"
            + f"\n<approved_operations>{operations}</approved_operations>\nWhen provided, preserve this operation plan exactly.",
            allrun_reference=allrun_reference, openfoam_fork=openfoam_fork,
            phase=script_phase, llm_client=llm_client, operations=operations,
        )
        response = FoamPydantic(list_foamfile=[*repaired_inputs,
            FoamfilePydantic(file_name=script_name, folder_name="", content=script)])

    updated_dir = {folder: list(files) for folder, files in dir_structure.items()}
    by_relative_path = {
        safe_case_relative_path(item.folder_name, item.file_name).as_posix(): item
        for item in foamfiles.list_foamfile
    }
    updated_files: List[str] = []
    written_files: List[str] = []
    file_diffs: List[str] = []
    # Check every proposed target before writing the first approved file.
    target_errors = check_file_targets(case_dir, [
        safe_case_relative_path(f.folder_name, f.file_name).as_posix()
        for f in response.list_foamfile
        if rewrite_plan is None or case_file_key(
            safe_case_relative_path(f.folder_name, f.file_name).as_posix()) in planned_paths
    ])
    if target_errors:
        return {'dir_structure': dir_structure, 'foamfiles': foamfiles,
                'error_logs': target_errors, 'written_files': [], 'updated_files': [],
                'file_diffs': [], 'rewrite_plan': rewrite_plan}
    for foamfile in response.list_foamfile:
        try:
            relative = safe_case_relative_path(foamfile.folder_name, foamfile.file_name)
        except CasePathSafetyError as exc:
            raise ValueError(
                "LLM returned an unsafe rewrite path: "
                f"{foamfile.folder_name!r}/{foamfile.file_name!r}"
            ) from exc
        relative_path = relative.as_posix()
        logical_path = case_file_key(relative_path)
        if rewrite_plan is not None and logical_path not in planned_paths:
            print(f"Warning: Skipping unplanned rewrite file: {relative_path}")
            continue
        file_path = safe_case_path(case_dir, foamfile.folder_name, foamfile.file_name)
        if rewrite_plan is not None:
            # A logical task may name U while current context names U.gz. The
            # task selects the representation; the model supplies its contents.
            relative = safe_case_relative_from_text(planned_paths[logical_path])
            file_path = safe_case_path(case_dir, str(relative.parent), relative.name)
        folder_name = "" if relative.parent.as_posix() == "." else relative.parent.as_posix()
        previous = resolve_case_file(file_path)
        try:
            before = read_case_text(previous, errors="replace") if previous.is_file() else None
        except (EOFError, gzip.BadGzipFile, zlib.error) as error:
            if previous.suffix != '.gz' or rewrite_plan is None:
                raise
            # An explicit repair can replace corrupt gzip content even when
            # its previous text cannot be recovered for a diff.
            print(f"Warning: Replacing explicitly targeted unreadable gzip file {previous}: {error}")
            before = None
        actual = save_case_text(file_path, foamfile.content)
        relative_path = actual.relative_to(Path(case_dir).expanduser().resolve()).as_posix()
        file_name = actual.name
        written_files.append(relative_path)
        updated_dir.setdefault(folder_name, [])
        updated_dir[folder_name] = [name for name in updated_dir[folder_name]
                                    if case_file_key(name) != case_file_key(file_name)] + [file_name]
        by_relative_path = {path: item for path, item in by_relative_path.items()
                            if case_file_key(path) != logical_path}
        by_relative_path[relative_path] = FoamfilePydantic(
            file_name=file_name,
            folder_name=folder_name,
            content=foamfile.content,
        )
        if before != foamfile.content or previous != actual:
            updated_files.append(relative_path)
            file_diffs.append(
                "".join(
                    line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                    for line in unified_diff(
                        (before or "").splitlines(keepends=True),
                        foamfile.content.splitlines(keepends=True),
                        fromfile=f"before/{relative_path}",
                        tofile=f"after/{relative_path}",
                    )
                )
            )
    updated_foamfiles = FoamPydantic(list_foamfile=list(by_relative_path.values()))
    written_keys = {case_file_key(path) for path in written_files}
    pending = [task for task in rewrite_plan['target_files']
               if case_file_key(safe_case_relative_from_text(task['file'])) not in written_keys] if rewrite_plan is not None else []
    return {
        "dir_structure": updated_dir,
        "foamfiles": updated_foamfiles,
        "error_logs": [] if static_repair else unwritten_file_errors(task['file'] for task in pending),
        "updated_files": updated_files,
        "written_files": written_files,
        "file_diffs": file_diffs,
        "rewrite_plan": {'target_files': pending},
    }
