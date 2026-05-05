import os
import re
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from utils import LLMService, retrieve_faiss, parse_directory_structure
from . import global_llm_service


class CaseSummaryModel(BaseModel):
    case_name: str = Field(description="name of the case")
    case_domain: str = Field(description="domain of the case")
    case_category: str = Field(description="category of the case")
    case_solver: str = Field(description="solver of the case")


class SubtaskModel(BaseModel):
    file_name: str
    folder_name: str


class OpenFOAMPlanModel(BaseModel):
    subtasks: List[SubtaskModel]


def parse_requirement_to_case_info(user_requirement: str, case_stats: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Parse user requirements into structured case information using LLM.
    
    This function uses LLM to analyze natural language user requirements
    and extract structured case information including name, domain, category,
    and solver. The extracted values are validated against available options.
    
    Args:
        user_requirement (str): Natural language description of simulation requirements
        case_stats (Dict[str, List[str]]): Available case statistics with keys:
            - case_domain: List of available domains (e.g., ["fluid", "solid"])
            - case_category: List of available categories (e.g., ["tutorial", "advanced"])
            - case_solver: List of available solvers (e.g., ["simpleFoam", "pimpleFoam"])
    
    Returns:
        Dict[str, str]: Structured case information containing:
            - case_name (str): Parsed case name with spaces replaced by underscores
            - case_domain (str): Selected domain from available options
            - case_category (str): Selected category from available options
            - case_solver (str): Selected solver from available options
    
    Raises:
        ValueError: If LLM fails to parse requirements or returns invalid values
        RuntimeError: If LLM service is unavailable
    
    Example:
        >>> case_stats = {
        ...     "case_domain": ["fluid", "solid"],
        ...     "case_category": ["tutorial", "advanced"],
        ...     "case_solver": ["simpleFoam", "pimpleFoam"]
        ... }
        >>> result = parse_requirement_to_case_info(
        ...     "Create a simple fluid flow tutorial",
        ...     case_stats
        ... )
        >>> print(f"Case: {result['case_name']}, Solver: {result['case_solver']}")
    """
    parse_system_prompt = (
        "Please transform the following user requirement into a standard case description using a structured format. "
        "The key elements should include case name, case domain, case category, and case solver. "
        f"Note: case domain must be one of {case_stats.get('case_domain', [])}."
        f"Note: case category must be one of {case_stats.get('case_category', [])}."
        f"Note: case solver must be one of {case_stats.get('case_solver', [])}."
    )
    parse_user_prompt = f"User requirement: {user_requirement}."
    res = global_llm_service.invoke(parse_user_prompt, parse_system_prompt, pydantic_obj=CaseSummaryModel)
    return {
        "case_name": res.case_name.replace(" ", "_"),
        "case_domain": res.case_domain,
        "case_category": res.case_category,
        "case_solver": res.case_solver,
    }


def resolve_case_dir(
    case_name: str,
    case_dir: str = "",
    run_times: int = 1,
    run_directory: str = None
) -> str:
    """
    Resolve the case directory path based on case name and run configuration.
    
    This function determines the appropriate directory path for a case,
    handling both custom paths and default run directories with
    optional run numbering for multiple executions.
    
    Args:
        case_name (str): Name of the case (used for directory naming)
        case_dir (str, optional): Custom case directory path. If provided, this is returned directly.
        run_times (int, optional): Number of runs for this case. Defaults to 1.
        run_directory (str, optional): Base directory for runs. If None, uses default runs directory.
    
    Returns:
        str: Resolved case directory path
    
    Example:
        >>> # Custom directory
        >>> path = resolve_case_dir("test_case", case_dir="/custom/path")
        >>> print(path)  # "/custom/path"
        
        >>> # Default directory with run numbering
        >>> path = resolve_case_dir("test_case", run_times=3)
        >>> print(path)  # "/path/to/runs/test_case_3"
        
        >>> # Single run in default directory
        >>> path = resolve_case_dir("test_case")
        >>> print(path)  # "/path/to/runs/test_case"
    """
    if case_dir:
        return case_dir
    if run_directory is None:
        run_directory = str(Path(__file__).resolve().parent.parent / "runs")
    base_dir = str(run_directory)
    if run_times > 1:
        return os.path.join(base_dir, f"{case_name}_{run_times}")
    return os.path.join(base_dir, case_name)


class PerFileAdvice(BaseModel):
    folder_name: str = Field(description="folder containing the file, e.g. system")
    file_name: str = Field(description="file name, e.g. fvSolution")
    classification: str = Field(description="one of: reuse, modify, present_but_unused, write_fresh")
    delta: Optional[str] = Field(default=None, description="for 'modify' only: specific changes to apply")


class SimilarCaseAdviceModel(BaseModel):
    pre_solver_steps: List[str] = Field(description="commands to run before the solver, e.g. blockMesh, potentialFoam")
    solver: str = Field(description="OpenFOAM solver, e.g. simpleFoam")
    per_file: List[PerFileAdvice] = Field(description="per-file classification for each subtask")


def _log_top3(label: str, items: List[Dict[str, Any]]) -> None:
    print(f"{label} (top-3):")
    for i, it in enumerate(items[:3], 1):
        print(
            f"  {i}. {it.get('case_name')} | {it.get('case_domain')} | {it.get('case_category')} | {it.get('case_solver')} | score={it.get('score')}"
        )


def _rerank_candidates(
    candidates: List[Dict[str, Any]],
    case_solver: str,
    case_domain: str,
) -> List[Dict[str, Any]]:
    def key(item: Dict[str, Any]) -> tuple:
        solver_match = 1 if item.get("case_solver") == case_solver else 0
        domain_match = 1 if item.get("case_domain") == case_domain else 0
        score = item.get("score")
        score_val = 0.0 if score is None else float(score)
        return (-domain_match, -solver_match, score_val)

    return sorted(candidates, key=key)


def _build_advice(
    user_requirement: str,
    case_info: str,
    selected: Optional[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    faiss_detailed: str,
    subtasks: List[Dict[str, str]],
) -> SimilarCaseAdviceModel:
    if not faiss_detailed or not subtasks:
        return SimilarCaseAdviceModel(
            pre_solver_steps=[],
            solver="",
            per_file=[
                PerFileAdvice(folder_name=s["folder_name"], file_name=s["file_name"], classification="write_fresh")
                for s in subtasks
            ],
        )

    subtask_lines = "\n".join(f"  {s['folder_name']}/{s['file_name']}" for s in subtasks)

    sys_prompt = (
        "You are a CFD expert. Analyze the reference tutorial and provide per-file guidance for generating a new OpenFOAM simulation. "
        "For each file in the subtask list, examine the tutorial content and classify how it should be used:\n"
        "  reuse: tutorial file is a close match — use it directly with only user-requirement-driven changes\n"
        "  modify: tutorial file has the right structure but needs targeted changes — describe those changes in delta\n"
        "  present_but_unused: tutorial has this file but it does not apply to the user's case — generate from scratch\n"
        "  write_fresh: tutorial has no useful version of this file — generate entirely from scratch\n"
        "Also identify pre_solver_steps (e.g. blockMesh, potentialFoam) and solver from the tutorial Allrun and user requirement. "
        "Output strict JSON only."
    )
    user_prompt = (
        f"User requirement:\n{user_requirement}\n\n"
        f"Case info:\n{case_info}\n\n"
        f"Reference tutorial:\n{faiss_detailed}\n\n"
        f"Files to generate:\n{subtask_lines}\n\n"
        "Return JSON with keys: pre_solver_steps (list of strings), solver (string), "
        "per_file (list of {folder_name, file_name, classification, delta})."
    )

    return global_llm_service.invoke(user_prompt, sys_prompt, pydantic_obj=SimilarCaseAdviceModel)


def retrieve_references(case_name: str,
                        case_solver: str,
                        case_domain: str,
                        case_category: str,
                        searchdocs: int = 2,
                        user_requirement: str = "") -> Tuple[str, str, str, str, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    # Build case_info
    case_info = f"case name: {case_name}\ncase domain: {case_domain}\ncase category: {case_category}\ncase solver: {case_solver}"
    print("Retrieval query:\n" + case_info)

    recall_k = int(searchdocs)
    faiss_structure_all = retrieve_faiss("openfoam_tutorials_structure", case_info, topk=recall_k)
    print(f"Retrieved {len(faiss_structure_all)} candidates from FAISS.")

    # Stash top-3 tutorial-structure similarity scores + case names on the
    # LLM service so every downstream invoke() logs them into the JSONL.
    try:
        top3 = faiss_structure_all[:3]
        global_llm_service.retrieval_top3_scores = [float(c.get("score")) for c in top3]
        global_llm_service.retrieval_top3_cases = [str(c.get("case_name", "")) for c in top3]
    except Exception as _e:
        print(f"Warning: failed to stash top-3 retrieval metadata: {_e}")

    if not faiss_structure_all:
        print("No similar cases returned by FAISS retrieval.")
        return "", "", "", "", None, []

    ranked = faiss_structure_all
    _log_top3("Structure candidates (FAISS order)", ranked)
    selected = ranked[0]

    # Restore FA 1.1.0's two-step retrieval (structure -> details). The
    # structure entry's full_content is used as the query text for the
    # details lookup so both queries anchor on the same case. The
    # details-index full_content contains the <tutorials> block with
    # every file's body inside <file_content>, which is what the writer
    # prompt's <similar_case_reference> is meant to carry.
    faiss_structure = selected.get("full_content", "")
    faiss_structure = re.sub(r"\n{3}", "\n", faiss_structure)
    faiss_detailed = retrieve_faiss(
        "openfoam_tutorials_details", faiss_structure, topk=searchdocs
    )[0]["full_content"]
    faiss_detailed = re.sub(r"\n{3}", "\n", faiss_detailed)

    m = re.search(r"<directory_structure>(.*?)</directory_structure>", faiss_detailed, re.DOTALL)
    if not m:
        print("Warning: No directory_structure found in selected similar case details.")
        return "", "", "", "", selected, ranked
    dir_structure = m.group(1).strip()
    dir_counts = parse_directory_structure(dir_structure)
    dir_counts_str = ',\n'.join([f"There are {count} files in Directory: {directory}" for directory, count in dir_counts.items()])

    # Build allrun reference
    index_content = f"<index>\ncase name: {selected.get('case_name')}\ncase solver: {selected.get('case_solver')}\n</index>\n<directory_structure>\n{dir_structure}\n</directory_structure>"
    faiss_allrun = retrieve_faiss("openfoam_allrun_scripts", index_content, topk=searchdocs)
    allrun_reference = "Similar cases are ordered, with smaller numbers indicating greater similarity. For example, similar_case_1 is more similar than similar_case_2, and similar_case_2 is more similar than similar_case_3.\n"
    for idx, item in enumerate(faiss_allrun):
        allrun_reference += f"<similar_case_{idx + 1}>{item['full_content']}</similar_case_{idx + 1}>\n\n\n"

    return faiss_detailed, dir_structure, dir_counts_str, allrun_reference, selected, ranked


def decompose_to_subtasks(user_requirement: str, dir_structure: str, dir_counts_str: str) -> List[Dict]:
    decompose_system_prompt = (
        "You are an experienced Planner specializing in OpenFOAM projects. "
        "Your task is to break down the following user requirement into a series of smaller, manageable subtasks. "
        "For each subtask, identify the file name of the OpenFOAM input file (foamfile) and the corresponding folder name where it should be stored. "
        "Your final output must strictly follow the JSON schema below and include no additional keys or information:\n\n"
        "```\n{\n  \"subtasks\": [\n    {\n      \"file_name\": \"<string>\",\n      \"folder_name\": \"<string>\"\n    }\n    // ... more subtasks\n  ]\n}\n```\n\n"
        "Make sure that your output is valid JSON and strictly adheres to the provided schema. "
    )

    decompose_user_prompt = (
        f"User Requirement: {user_requirement}\n\n"
        f"Reference Directory Structure (similar case): {dir_structure}\n\n{dir_counts_str}\n\n"
        "Make sure you generate all the necessary files for the user's requirements. "
        "Do not include any gmsh files like .geo etc. in the subtasks. "
        "Only include blockMesh or snappyHexMesh if the user hasn't requested for gmsh mesh or user isn't using an external uploaded custom mesh. "
        "Please generate the output as structured JSON."
    )

    res = global_llm_service.invoke(decompose_user_prompt, decompose_system_prompt, pydantic_obj=OpenFOAMPlanModel)
    return [{"file_name": s.file_name, "folder_name": s.folder_name} for s in res.subtasks]


def generate_simulation_plan(
    user_requirement: str,
    case_stats: Dict[str, List[str]],
    case_dir: str = "",
    searchdocs: int = 2
) -> Dict[str, Any]:
    """
    Generate a complete simulation plan by parsing requirements and creating subtasks.
    
    This function orchestrates the entire planning process:
    1. Parse user requirements into structured case information
    2. Resolve case directory
    3. Retrieve similar case references from FAISS database
    4. Decompose requirements into manageable subtasks
    
    Args:
        user_requirement (str): Natural language description of simulation requirements
        case_stats (Dict[str, List[str]]): Available case statistics
        case_dir (str, optional): Custom case directory path
        searchdocs (int, optional): Number of similar documents to retrieve
    
    Returns:
        Dict[str, Any]: Complete plan containing:
            - case_name, case_domain, case_category, case_solver
            - case_dir: Resolved case directory
            - tutorial_reference: FAISS detailed reference
            - case_path_reference: Path to reference file
            - dir_structure_reference: Directory structure
            - allrun_reference: Allrun script references
            - subtasks: List of subtasks with file and folder names
    
    Raises:
        ValueError: If subtasks cannot be generated
        RuntimeError: If any step in the planning process fails
    """
    # Step 1: Parse user requirement to case info
    case_info = parse_requirement_to_case_info(user_requirement, case_stats)
    case_name = case_info["case_name"]
    case_domain = case_info["case_domain"]
    case_category = case_info["case_category"]
    case_solver = case_info["case_solver"]
    
    # Step 2: Resolve case directory
    resolved_case_dir = resolve_case_dir(
        case_name=case_name,
        case_dir=case_dir,
        run_times=1,
        run_directory=str(Path(__file__).resolve().parent.parent / "runs")
    )
    
    # Step 3: Retrieve references
    faiss_detailed, dir_structure, dir_counts_str, allrun_reference, selected, ranked = retrieve_references(
        case_name=case_name,
        case_solver=case_solver,
        case_domain=case_domain,
        case_category=case_category,
        searchdocs=searchdocs,
        user_requirement=user_requirement,
    )

    # Step 4: Decompose to subtasks
    subtasks = decompose_to_subtasks(user_requirement, dir_structure, dir_counts_str)

    # Step 5: Build per-file advice (needs both tutorial bodies and the actual subtask list)
    case_info_str = f"case name: {case_name}\ncase domain: {case_domain}\ncase category: {case_category}\ncase solver: {case_solver}"
    advice = _build_advice(user_requirement, case_info_str, selected, ranked, faiss_detailed, subtasks)
    
    if len(subtasks) == 0:
        raise ValueError("Failed to generate subtasks.")
    
    # Prepare reference file path
    case_path_reference = os.path.join(resolved_case_dir, "similar_case.txt")
    
    return {
        "case_name": case_name,
        "case_domain": case_domain,
        "case_category": case_category,
        "case_solver": case_solver,
        "case_dir": resolved_case_dir,
        "tutorial_reference": faiss_detailed,
        "case_path_reference": case_path_reference,
        "dir_structure_reference": dir_structure,
        "allrun_reference": allrun_reference,
        "subtasks": subtasks,
        "similar_case_advice": advice,
    }


