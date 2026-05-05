import os
import re
from typing import Dict, List, Any, Optional, Callable
import shutil
from utils import save_file, parse_context, retrieve_faiss, FoamPydantic, FoamfilePydantic, scan_case_directory, read_case_foamfiles, read_file
from . import global_llm_service


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

    def _report_progress(current: int, total: int, message: str) -> None:
        if progress_callback:
            try:
                progress_callback(current, total, message)
            except Exception:
                pass

    if generation_mode not in {"sequential_dependency", "parallel_no_context"}:
        raise ValueError(
            f"Unsupported generation_mode: {generation_mode}. "
            "Expected one of: sequential_dependency, parallel_no_context"
        )

    subtasks = sorted(subtasks, key=compute_priority)
    written_files = []
    dir_structure = {}

    # System prompt for file generation
    INITIAL_WRITE_SYSTEM_PROMPT = (
        "You are an expert in OpenFOAM simulation and numerical modeling. "
        "Your task is to generate a complete and functional file named: <file_name>{file_name}</file_name> within the <folder_name>{folder_name}</folder_name> directory. "
        "Ensure all required values are present and match with the files content already generated. "
        "Before finalizing the output, ensure:\n"
        "- All necessary fields exist (e.g., if `nu` is defined in `constant/transportProperties`, it must be used correctly in `0/U`).\n"
        "- Cross-check field names between different files to avoid mismatches.\n"
        "- Ensure units and dimensions are correct for all physical variables.\n"
        "- Ensure case solver settings are consistent with the user's requirements. Available solvers are: {case_solver}.\n"
        "Provide only the code—no explanations, comments, or additional text."
    )

    def _build_prompts(file_name: str, folder_name: str, written_files_ctx: List[FoamfilePydantic]) -> tuple[str, str]:
        code_system_prompt = INITIAL_WRITE_SYSTEM_PROMPT.format(
            file_name=file_name,
            folder_name=folder_name,
            case_solver=case_solver,
        )

        similar_ref_block = (
            f"Refer to the following similar case file content to ensure the generated file aligns with the user requirement:\n"
            f"<similar_case_reference>{tutorial_reference}</similar_case_reference>\n"
            "Similar case reference is always correct. If you find the user requirement is very consistent with the similar case reference, "
            "you should use the similar case reference as the template to generate the file. "
            "Just modify the necessary parts to make the file complete and functional.\n"
            if tutorial_reference else "No suitable similar case was found for this domain.\n"
        )

        code_user_prompt = (
            f"User requirement: {user_requirement}\n"
            f"{similar_ref_block}"
            "Please ensure that the generated file is complete, functional, and logically sound. "
            "Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence. "
            "When generating controlDict, do not include anything to perform post processing. Just include the necessary settings to run the simulation."
        )

        if generation_mode == "sequential_dependency" and written_files_ctx:
            code_user_prompt += (
                f"The following are files content already generated: {str(written_files_ctx)}\n\n\n"
                "You should ensure that the new file is consistent with the previous files. Such as boundary conditions, mesh settings, etc."
            )

        return code_user_prompt, code_system_prompt

    def _generate_one(subtask: Dict[str, str], written_files_ctx: List[FoamfilePydantic]) -> FoamfilePydantic:
        file_name = subtask["file_name"]
        folder_name = subtask["folder_name"]
        if not file_name or not folder_name:
            raise ValueError(f"Invalid subtask format: {subtask}")

        # Target output path (this run)
        file_path = os.path.join(case_dir, folder_name, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Optional reuse: if a pre-generated file exists, copy it into output and
        # treat it as a written file (also included in context for subsequent generations).
        if reuse_generated_dir:
            reuse_src = os.path.join(reuse_generated_dir, folder_name, file_name)
            if os.path.exists(reuse_src):
                print(f"Reusing generated file: {reuse_src}")
                shutil.copy2(reuse_src, file_path)
                reused_content = read_file(reuse_src)
                return FoamfilePydantic(file_name=file_name, folder_name=folder_name, content=reused_content)

        code_user_prompt, code_system_prompt = _build_prompts(file_name, folder_name, written_files_ctx)

        if generation_mode == "parallel_no_context":
            # Avoid shared global LLM instance in parallel mode.
            from utils import LLMService
            from config import Config
            llm = LLMService(Config())
            generation_response = llm.invoke(code_user_prompt, code_system_prompt)
        else:
            generation_response = global_llm_service.invoke(code_user_prompt, code_system_prompt)

        code_context = parse_context(generation_response)
        save_file(file_path, code_context)
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
        )
        written_files.append(FoamfilePydantic(file_name="Allrun", folder_name=case_dir, content=allrun_result["allrun_script"]))
    
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
    pre_solver_steps: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    progress_offset: int = 0,
    total_steps: int = 0,
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
    from pydantic import BaseModel, Field
    from typing import List
    
    # Parse allrun helper function
    def parse_allrun(text: str) -> str:
        match = re.search(r'```(.*?)```', text, re.DOTALL)
        return match.group(1).strip() if match else text
    
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
        + (f"Required pre-solver steps (must be included in this order before the solver): {pre_solver_steps}\n" if pre_solver_steps else "")
        + "Generate only the required OpenFOAM command list—no extra text."
    )

    if mesh_type == "custom_mesh":
        command_user_prompt += f"{mesh_commands_info}\n"
    
    command_response = global_llm_service.invoke(command_user_prompt, command_system_prompt, pydantic_obj=CommandsPydantic)

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
        command_help = retrieve_faiss("openfoam_command_help", command, topk=searchdocs)
        commands_help.append(command_help[0]['full_content'])
    commands_help = "\n".join(commands_help)

    # Allrun generation system prompt
    allrun_system_prompt = (
        "You are an expert in OpenFOAM. Generate an Allrun script based on the provided details. "
        f"Available commands with descriptions: {commands_help}\n\n"
        f"Reference Allrun scripts from similar cases: {allrun_reference}\n\n"
        "If custom mesh commands are provided, make sure to include them in the appropriate order in the Allrun script. "
        "CRITICAL: Do not include any post processing commands in the Allrun script. "
        "CRITICAL: Do not include any commands to convert mesh to foam format like gmshToFoam or others. "
    )

    if mesh_type == "custom_mesh":
        allrun_system_prompt += "CRITICAL: Do not include any other mesh commands other than the custom mesh commands.\n"
        allrun_system_prompt += "CRITICAL: Do not include any gmshToFoam commands in the Allrun script."
    
    allrun_user_prompt = (
        f"User requirement: {user_requirement}\n"
        f"Case directory structure: {dir_structure}\n"
        f"User case information: {case_info}\n"
        f"{mesh_commands_info}\n"
        "All run scripts for these similar cases are for reference only and may not be correct, as you might be a different case solver or have a different directory structure. " 
        "You need to rely on your OpenFOAM and physics knowledge to discern this, and pay more attention to user requirements, " 
        "as your ultimate goal is to fulfill the user's requirements and generate an allrun script that meets those requirements. "
        "CRITICAL: Do not include any post processing commands in the Allrun script. "
        "CRITICAL: Do not include any commands to convert mesh to foam format like gmshToFoam or others. "
        "CRITICAL: Do not include any commands that run gmsh to create the mesh. "
        "Generate the Allrun script strictly based on the above information. Do not include explanations, comments, or additional text. Put the code in ``` tags."
    )

    if mesh_type == "custom_mesh":
        allrun_user_prompt += "CRITICAL: Do not include any other mesh commands other than the custom mesh commands.\n"
        allrun_user_prompt += "CRITICAL: Do not include any gmshToFoam commands in the Allrun script."

    allrun_response = global_llm_service.invoke(allrun_user_prompt, allrun_system_prompt)

    if progress_callback:
        try:
            progress_callback(progress_offset + 2, total_steps, "Generated Allrun script")
        except Exception:
            pass

    allrun_script = parse_allrun(allrun_response)
    allrun_file_path = os.path.join(case_dir, "Allrun")
    save_file(allrun_file_path, allrun_script)

    return {"allrun_path": allrun_file_path, "allrun_script": allrun_script, "commands": command_response.commands}



def rewrite_files(
    case_dir: str,
    error_logs: List[str],
    review_analysis: str,
    user_requirement: str,
    rewrite_plan: Optional[Dict[str, Any]] = None,  # unused; kept for backwards-compat
    foamfiles: Optional[Any] = None,
    dir_structure: Optional[Dict[str, List[str]]] = None,
    loop_count: int = 0
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
    
    from utils import FoamPydantic, FoamfilePydantic  # local import to avoid cycles
    import re

    rewrite_system_prompt = (
        "You are an expert in OpenFOAM simulation and numerical modeling. "
        "Your task is to modify and rewrite the necessary OpenFOAM files to fix the reported error. "
        "Please do not propose solutions that require modifying any parameters declared in the user requirement, try other approaches instead. "
        "The user will provide the error content, error command, reviewer's suggestions, and all relevant foam files. "
        "Only return files that require rewriting, modification, or addition; do not include files that remain unchanged. "
        "Return the complete, corrected file contents in the following JSON format: "
        "list of foamfile: [{file_name: 'file_name', folder_name: 'folder_name', content: 'content'}]. "
        "Ensure your response includes only the modified file content with no extra text, as it will be parsed using Pydantic."
    )

    rewrite_user_prompt = (
        f"<foamfiles>{str(foamfiles)}</foamfiles>\n"
        f"<error_logs>{error_logs}</error_logs>\n"
        f"<reviewer_analysis>{review_analysis}</reviewer_analysis>\n\n"
        f"<user_requirement>{user_requirement}</user_requirement>\n\n"
        "Please update the relevant OpenFOAM files to resolve the reported errors, ensuring that all modifications strictly adhere to the specified formats. Ensure all modifications adhere to user requirement."
    )

    response = global_llm_service.invoke(rewrite_user_prompt, rewrite_system_prompt, pydantic_obj=FoamPydantic)

    # Prepare updated structures
    updated_dir = dict(dir_structure) if dir_structure else {}
    foamfiles_list = []
    if foamfiles and hasattr(foamfiles, "list_foamfile") and foamfiles.list_foamfile:
        foamfiles_list = list(foamfiles.list_foamfile)

    for foamfile in response.list_foamfile:
        file_path = os.path.join(case_dir, foamfile.folder_name, foamfile.file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_file(file_path, foamfile.content)

        if foamfile.folder_name not in updated_dir:
            updated_dir[foamfile.folder_name] = []
        if foamfile.file_name not in updated_dir[foamfile.folder_name]:
            updated_dir[foamfile.folder_name].append(foamfile.file_name)

        foamfiles_list = [
            f for f in foamfiles_list
            if not (f.folder_name == foamfile.folder_name and f.file_name == foamfile.file_name)
        ]
        foamfiles_list.append(foamfile)

    updated_foamfiles = FoamPydantic(list_foamfile=foamfiles_list)
    return {
        "dir_structure": updated_dir,
        "foamfiles": updated_foamfiles,
        "error_logs": [],
    }
