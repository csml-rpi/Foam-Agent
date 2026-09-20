import os
import re
from difflib import unified_diff
from typing import Dict, List, Any, Optional, Callable
import shutil
from utils import save_file, parse_context, retrieve_faiss, FoamPydantic, FoamfilePydantic, scan_case_directory, read_case_foamfiles, read_file
from config import Config
from openfoam_target import database_path_for_config
from . import global_llm_service
from .case_paths import (
    CasePathSafetyError,
    safe_case_path,
    safe_case_relative_from_text,
    safe_case_relative_path,
)


def compute_priority(subtask):
    if subtask["folder_name"] == "system":
        return 0
    elif subtask["folder_name"] == "constant":
        return 1
    elif subtask["folder_name"] == "0":
        return 2
    else:
        return 3


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

    def _report_progress(current: int, total: int, message: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(current, total, message)
        except Exception as exc:  # noqa: BLE001 - progress_callback implementation is external
            print(f"<progress_callback_error>{exc}</progress_callback_error>")

    if generation_mode not in {"sequential_dependency", "parallel_no_context"}:
        raise ValueError(
            f"Unsupported generation_mode: {generation_mode}. "
            "Expected one of: sequential_dependency, parallel_no_context"
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
        code_system_prompt = INITIAL_WRITE_SYSTEM_PROMPT.format(
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
        similar_ref_block = (
            f"Refer to the following similar case file content if helpful:\n<similar_case_reference>{tutorial_reference}</similar_case_reference>\n"
            if tutorial_reference else "No suitable similar case was found for this domain.\n"
        )

        code_user_prompt = (
            f"User requirement: {user_requirement}\n"
            f"{similar_ref_block}{advice_text}\n"
            "If the similar case is a weak match, do not copy it blindly. Use it only where it is consistent with the user requirement. "
            "The OpenFOAM compatibility contract in the system prompt takes precedence over incompatible syntax in a reference case. "
            "Just modify the necessary parts to make the file complete and functional."
            "Please ensure that the generated file is complete, functional, and logically sound."
            "Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence."
            "When generating controlDict, do not include anything to preform post processing. Just include the necessary settings to run the simulation."
        )

        if file_name == "fvSolution":
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
            code_user_prompt += (
                f"The following are files content already generated: {written_files_ctx}\n\n\n"
                "You should ensure that the new file is consistent with the previous files. Such as boundary conditions, mesh settings, etc."
            )

        return code_user_prompt, code_system_prompt

    def _generate_one(subtask: Dict[str, str], written_files_ctx: List[FoamfilePydantic]) -> FoamfilePydantic:
        file_name = subtask["file_name"]
        folder_name = subtask["folder_name"]
        try:
            file_path = safe_case_path(case_dir, folder_name, file_name)
        except CasePathSafetyError as exc:
            raise ValueError(f"Unsafe generated subtask path: {subtask!r}") from exc
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if reuse_generated_dir:
            reuse_src = safe_case_path(reuse_generated_dir, folder_name, file_name)
            if reuse_src.exists():
                print(f"Reusing generated file: {reuse_src}")
                shutil.copy2(reuse_src, file_path)
                return FoamfilePydantic(
                    file_name=file_name,
                    folder_name=folder_name,
                    content=read_file(str(reuse_src)),
                )
        code_user_prompt, code_system_prompt = _build_prompts(file_name, folder_name, written_files_ctx)
        generation_response = llm_client.invoke(code_user_prompt, code_system_prompt)
        code_context = parse_context(generation_response)
        save_file(str(file_path), code_context)
        return FoamfilePydantic(file_name=file_name, folder_name=folder_name, content=code_context)

    # Build dir_structure upfront (deterministic ordering) and generate files
    for subtask in subtasks:
        folder_name = subtask.get("folder_name")
        file_name = subtask.get("file_name")
        if folder_name not in dir_structure:
            dir_structure[folder_name] = []
        dir_structure[folder_name].append(file_name)

    total_steps = len(subtasks) + (2 if database_path else 0)
    _report_progress(0, total_steps, f"Starting file generation for {len(subtasks)} files")

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
                ex.submit(_generate_one, subtasks[i], []): i
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
            openfoam_fork=openfoam_fork,
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
    from pydantic import BaseModel, Field
    from typing import List
    
    llm_client = llm_service if llm_service is not None else global_llm_service
    retrieval_config = config if config is not None else Config()
    if config is None and openfoam_fork == "esi-v2006":
        retrieval_config.openfoam_target = "esi-v2006"
    if config is not None or openfoam_fork == "esi-v2006":
        database_path = str(database_path_for_config(retrieval_config))

    # Parse allrun helper function
    def parse_allrun(text: str) -> str:
        match = re.search(r'```(.*?)```', text, re.DOTALL)
        if not match:
            return text.strip()
        return re.sub(
            r"^\s*(?:sh|bash|shell|zsh|ksh)\s*\r?\n",
            "",
            match.group(1),
            count=1,
            flags=re.IGNORECASE,
        ).strip()
    
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
        "Your task is to generate only the necessary OpenFOAM commands required to create an Allrun script for the given user case, based on the provided directory structure. "
        "Return only the list of commands—no explanations, comments, or additional text."
    )

    if mesh_type == "custom_mesh":
        command_system_prompt += "If custom mesh commands are provided, include them in the appropriate order (typically after blockMesh or instead of blockMesh if custom mesh is used). "
    
    command_user_prompt = (
        f"Available OpenFOAM commands for the Allrun script: {commands}\n"
        f"Case directory structure: {dir_structure}\n"
        f"User case information: {case_info}\n"
        f"Reference Allrun scripts from similar cases: {allrun_reference}\n"
        "Generate only the required OpenFOAM command list—no extra text."
    )

    if mesh_type == "custom_mesh":
        command_user_prompt += f"{mesh_commands_info}\n"
    
    command_system_prompt += "\n" + _generation_contract(openfoam_fork)

    command_response = llm_client.invoke(command_user_prompt, command_system_prompt, pydantic_obj=CommandsPydantic)

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
        f"Available commands with descriptions: {commands_help}\n\n"
        f"Reference Allrun scripts from similar cases: {allrun_reference}\n\n"
        "If custom mesh commands are provided, make sure to include them in the appropriate order in the Allrun script. "
        "CRITICAL: Do not include any post processing commands in the Allrun script."
        "CRITICAL: Do not include any commands to convert mesh to foam format like gmshToFoam or others."
    )

    if mesh_type == "custom_mesh":
        allrun_system_prompt += "CRITICAL: Do not include any other mesh commands other than the custom mesh commands.\n"
        allrun_system_prompt += "CRITICAL: Do not include any gmshToFoam commands in the Allrun script."
    
    allrun_user_prompt = (
        f"User requirement: {user_requirement}\n"
        f"Case directory structure: {dir_structure}\n"
        f"User case infomation: {case_info}\n"
        f"{mesh_commands_info}\n"
        "All run scripts for these similar cases are for reference only and may not be correct, as you might be a different case solver or have a different directory structure. " 
        "You need to rely on your OpenFOAM and physics knowledge to discern this, and pay more attention to user requirements, " 
        "as your ultimate goal is to fulfill the user's requirements and generate an allrun script that meets those requirements."
        "CRITICAL: Do not include any post processing commands in the Allrun script."
        "CRITICAL: Do not include any commands to convert mesh to foam format like gmshToFoam or others."
        "CRITICAL: Do not include any commands that run gmsh to create the mesh."
        "Generate the Allrun script strictly based on the above information. Do not include explanations, comments, or additional text. Put the code in ``` tags."
    )

    if mesh_type == "custom_mesh":
        allrun_user_prompt += "CRITICAL: Do not include any other mesh commands other than the custom mesh commands.\n"
        allrun_user_prompt += "CRITICAL: Do not include any gmshToFoam commands in the Allrun script."

    allrun_system_prompt += "\n" + _generation_contract(openfoam_fork)

    allrun_response = llm_client.invoke(allrun_user_prompt, allrun_system_prompt)

    if progress_callback:
        try:
            progress_callback(progress_offset + 2, total_steps, "Generated Allrun script")
        except Exception:
            pass

    allrun_script = parse_allrun(allrun_response)
    if not allrun_script:
        raise ValueError("The LLM returned an empty Allrun script.")
    allrun_file_path = os.path.join(case_dir, "Allrun")
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
        foamfiles = read_case_foamfiles(case_dir, dir_structure)
    
    # Prefer the current file over the solver cached during initial planning.
    files = (
        foamfiles.list_foamfile if hasattr(foamfiles, "list_foamfile")
        else foamfiles.get("list_foamfile", [])
    )
    for entry in files:
        entry = entry.model_dump() if hasattr(entry, "model_dump") else entry
        if entry.get("folder_name") == "system" and entry.get("file_name") == "controlDict":
            content = re.sub(
                r"/\*.*?\*/|//[^\n]*", "", entry.get("content", ""), flags=re.DOTALL
            )
            application = re.search(r"\bapplication\s+([^;\s]+)\s*;", content)
            case_solver = application.group(1) if application else ""
            break
    generation_contract = _generation_contract(openfoam_fork)
    rewrite_system_prompt = (
        "You are an expert in OpenFOAM simulation and numerical modeling. "
        "Modify files only as required by the repair context and user requirement. "
        "Changes explicitly requested by the user are allowed; preserve every other existing "
        "physical condition unless the plan explicitly requires a technical correction. "
        "You will receive a rewrite_plan. Follow it strictly: only modify files listed in rewrite_plan.target_files and apply only the requested changes. "
        "Do not modify files outside the plan. "
        f"The selected solver is {case_solver or 'not specified'}. "
        "Apply this OpenFOAM compatibility contract to every rewritten file:\n"
        f"{generation_contract}\n"
        "Return complete corrected files as JSON: "
        "list of foamfile: [{file_name, folder_name, content}]."
    )

    rewrite_user_prompt = (
        f"<foamfiles>{foamfiles}</foamfiles>\n"
        f"<error_logs>{error_logs}</error_logs>\n"
        f"<reviewer_analysis>{review_analysis}</reviewer_analysis>\n"
        f"<rewrite_plan>{rewrite_plan}</rewrite_plan>\n"
        f"<user_requirement>{user_requirement}</user_requirement>\n"
        f"<openfoam_fork>{openfoam_fork}</openfoam_fork>\n"
        f"<case_solver>{case_solver}</case_solver>\n"
        "Please update OpenFOAM files according to rewrite_plan only. "
        "Only include files from rewrite_plan.target_files in your output."
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
    if not allowed_files:
        print("Warning: rewrite_plan.target_files is empty; no files to rewrite, leaving case unchanged.")
        return {
            "dir_structure": dir_structure,
            "foamfiles": foamfiles,
            "error_logs": error_logs,
            "updated_files": [],
            "file_diffs": [],
        }

    llm_client = llm_service if llm_service is not None else global_llm_service
    response = llm_client.invoke(
        rewrite_user_prompt,
        rewrite_system_prompt,
        pydantic_obj=FoamPydantic,
    )

    updated_dir = {folder: list(files) for folder, files in dir_structure.items()}
    by_relative_path = {
        safe_case_relative_path(item.folder_name, item.file_name).as_posix(): item
        for item in foamfiles.list_foamfile
    }
    updated_files: List[str] = []
    file_diffs: List[str] = []
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
        if relative_path not in allowed_files:
            print(f"Warning: Skipping unplanned rewrite file: {relative_path}")
            continue
        folder_name = "" if relative.parent.as_posix() == "." else relative.parent.as_posix()
        file_name = relative.name
        before = file_path.read_text(encoding="utf-8", errors="replace") if file_path.is_file() else None
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
        if before != foamfile.content:
            updated_files.append(relative_path)
            file_diffs.append(
                "".join(
                    unified_diff(
                        (before or "").splitlines(keepends=True),
                        foamfile.content.splitlines(keepends=True),
                        fromfile=f"before/{relative_path}",
                        tofile=f"after/{relative_path}",
                    )
                )
            )
    updated_foamfiles = FoamPydantic(list_foamfile=list(by_relative_path.values()))
    return {
        "dir_structure": updated_dir,
        "foamfiles": updated_foamfiles,
        "error_logs": [],
        "updated_files": updated_files,
        "file_diffs": file_diffs,
    }


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
        ]
    elif fork == "esi-v2006":
        constraints = [
            "The target runtime is ESI/OpenCFD OpenFOAM v2006 from openfoam.com.",
            "Generate a native ESI v2006 case directly from the supplied ESI v2006 tutorial references.",
            "Use only ESI v2006 solver, file, dictionary, function-object, and boundary-condition conventions.",
            "Do not use Foundation OpenFOAM v10 conventions and do not rely on any post-generation translator.",
            "Keep all generated files internally consistent with the ESI v2006 solver selected by the planner.",
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
