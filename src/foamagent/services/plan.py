import os
import re
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from foamagent import paths
from foamagent.utils import LLMService, retrieve_faiss, parse_directory_structure
from foamagent.services import get_llm_service

from foamagent.logger import get_logger

logger = get_logger(__name__)


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
        >>> logger.info(f"Case: {result['case_name']}, Solver: {result['case_solver']}")
    """
    parse_system_prompt = (
        "Please transform the following user requirement into a standard case description using a structured format."
        "The key elements should include case name, case domain, case category, and case solver."
        f"Note: case domain must be one of {case_stats.get('case_domain', [])}."
        f"Note: case category must be one of {case_stats.get('case_category', [])}."
        f"Note: case solver must be one of {case_stats.get('case_solver', [])}."
    )
    parse_user_prompt = f"User requirement: {user_requirement}."
    res = get_llm_service().invoke(parse_user_prompt, parse_system_prompt, pydantic_obj=CaseSummaryModel)
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
        >>> logger.info(path)  # "/custom/path"
        
        >>> # Default directory with run numbering
        >>> path = resolve_case_dir("test_case", run_times=3)
        >>> logger.info(path)  # "/path/to/runs/test_case_3"
        
        >>> # Single run in default directory
        >>> path = resolve_case_dir("test_case")
        >>> logger.info(path)  # "/path/to/runs/test_case"
    """
    if case_dir:
        return case_dir
    if run_directory is None:
        run_directory = str(paths.runs_dir())
    base_dir = str(run_directory)
    if run_times > 1:
        return os.path.join(base_dir, f"{case_name}_{run_times}")
    return os.path.join(base_dir, case_name)


class SimilarCaseAdviceModel(BaseModel):
    match_level: str = Field(description="high/medium/low/none")
    use_scope: str = Field(description="short guidance about which files can use the reference")
    advice: str = Field(description="one-sentence advice to include in prompts")


def _log_top3(label: str, items: List[Dict[str, Any]]) -> None:
    logger.info(f"{label} (top-3):")
    for i, it in enumerate(items[:3], 1):
        logger.info(
            f"  {i}. {it.get('case_name')} | {it.get('case_domain')} | {it.get('case_category')} | {it.get('case_solver')} | score={it.get('score')}"
        )


def _rerank_candidates(
    candidates: List[Dict[str, Any]],
    case_solver: str,
) -> List[Dict[str, Any]]:
    def key(item: Dict[str, Any]) -> tuple:
        solver_match = 1 if item.get("case_solver") == case_solver else 0
        score = item.get("score")
        score_val = 0.0 if score is None else float(score)
        return (-solver_match, score_val)

    return sorted(candidates, key=key)


def _build_advice(
    user_requirement: str,
    case_info: str,
    selected: Optional[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> SimilarCaseAdviceModel:
    cand_lines = [
        f"- {c.get('case_name')} | {c.get('case_domain')} | {c.get('case_category')} | {c.get('case_solver')} | score={c.get('score')}"
        for c in candidates[:5]
    ]
    cand_block = "\n".join(cand_lines) if cand_lines else "(none)"

    selected_line = (
        f"{selected.get('case_name')} | {selected.get('case_domain')} | {selected.get('case_category')} | {selected.get('case_solver')} | score={selected.get('score')}"
        if selected else "(none)"
    )

    sys_prompt = (
        "You are a CFD expert. Based on the user requirement and the retrieved similar cases, "
        "produce a concise usage guidance. If no suitable case is available, set match_level to 'none' "
        "and advise not to rely on similar case templates."
    )
    user_prompt = (
        f"User requirement:\n{user_requirement}\n\n"
        f"Case info:\n{case_info}\n\n"
        f"Selected similar case:\n{selected_line}\n\n"
        f"Top candidates:\n{cand_block}\n\n"
        "Return JSON with keys: match_level (high/medium/low/none), use_scope, advice."
    )

    return get_llm_service().invoke(user_prompt, sys_prompt, pydantic_obj=SimilarCaseAdviceModel)


def retrieve_references(case_name: str,
                        case_solver: str,
                        case_domain: str,
                        case_category: str,
                        searchdocs: int = 2,
                        user_requirement: str = "") -> Tuple[str, str, str, str, SimilarCaseAdviceModel]:
    # Build case_info
    case_info = f"case name: {case_name}\ncase domain: {case_domain}\ncase category: {case_category}\ncase solver: {case_solver}"
    logger.info("Retrieval query:\n" + case_info)

    recall_k = max(10, int(searchdocs))
    faiss_structure_all = retrieve_faiss("openfoam_tutorials_structure", case_info, topk=recall_k)
    logger.info(f"Retrieved {len(faiss_structure_all)} candidates from FAISS.")

    # Hard constraint: domain must match
    domain_matched = [c for c in faiss_structure_all if c.get("case_domain") == case_domain]
    _log_top3("Domain-matched structure candidates", domain_matched)

    if not domain_matched:
        logger.info(f"No suitable similar case found under domain={case_domain}.")
        advice = _build_advice(user_requirement, case_info, None, faiss_structure_all)
        return "", "", "", "", advice

    # Rerank by solver match, then semantic score
    ranked = _rerank_candidates(domain_matched, case_solver)
    selected = ranked[0]

    # Use details from the same candidate (no re-query on structure text)
    faiss_detailed = selected.get("full_content", "")
    faiss_detailed = re.sub(r"\n{3}", "\n", faiss_detailed)

    m = re.search(r"<directory_structure>(.*?)</directory_structure>", faiss_detailed, re.DOTALL)
    if not m:
        logger.warning("Warning: No directory_structure found in selected similar case details.")
        advice = _build_advice(user_requirement, case_info, selected, ranked)
        return "", "", "", "", advice
    dir_structure = m.group(1).strip()
    dir_counts = parse_directory_structure(dir_structure)
    dir_counts_str = ',\n'.join([f"There are {count} files in Directory: {directory}" for directory, count in dir_counts.items()])

    # Build allrun reference
    index_content = f"<index>\ncase name: {selected.get('case_name')}\ncase solver: {selected.get('case_solver')}\n</index>\n<directory_structure>\n{dir_structure}\n</directory_structure>"
    faiss_allrun = retrieve_faiss("openfoam_allrun_scripts", index_content, topk=searchdocs)
    allrun_reference = "Similar cases are ordered, with smaller numbers indicating greater similarity. For example, similar_case_1 is more similar than similar_case_2, and similar_case_2 is more similar than similar_case_3.\n"
    for idx, item in enumerate(faiss_allrun):
        allrun_reference += f"<similar_case_{idx + 1}>{item['full_content']}</similar_case_{idx + 1}>\n\n\n"

    advice = _build_advice(user_requirement, case_info, selected, ranked)
    return faiss_detailed, dir_structure, dir_counts_str, allrun_reference, advice


def decompose_to_subtasks(user_requirement: str, dir_structure: str, dir_counts_str: str) -> List[Dict]:
    decompose_system_prompt = (
        "You are an experienced Planner specializing in OpenFOAM projects. "
        "Your task is to break down the following user requirement into a series of smaller, manageable subtasks. "
        "For each subtask, identify the file name of the OpenFOAM input file (foamfile) and the corresponding folder name where it should be stored. "
        "Your final output must strictly follow the JSON schema below and include no additional keys or information:\n\n"
        "```\n{\n  \"subtasks\": [\n    {\n      \"file_name\": \"<string>\",\n      \"folder_name\": \"<string>\"\n    }\n    // ... more subtasks\n  ]\n}\n```\n\n"
        "Make sure that your output is valid JSON and strictly adheres to the provided schema."
        "Make sure you generate all the necessary files for the user's requirements."
    )

    decompose_user_prompt = (
        f"User Requirement: {user_requirement}\n\n"
        f"Reference Directory Structure (similar case): {dir_structure}\n\n{dir_counts_str}\n\n"
        "Make sure you generate all the necessary files for the user's requirements."
        "Do not include any gmsh files like .geo etc. in the subtasks."
        "Only include blockMesh or snappyHexMesh if the user hasnt requested for gmsh mesh or user isnt using an external uploaded custom mesh"
        "Please generate the output as structured JSON."
    )

    res = get_llm_service().invoke(decompose_user_prompt, decompose_system_prompt, pydantic_obj=OpenFOAMPlanModel)
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
        run_directory=str(paths.runs_dir())
    )
    
    # Step 3: Retrieve references
    faiss_detailed, dir_structure, dir_counts_str, allrun_reference, advice = retrieve_references(
        case_name=case_name,
        case_solver=case_solver,
        case_domain=case_domain,
        case_category=case_category,
        searchdocs=searchdocs,
        user_requirement=user_requirement,
    )
    
    # Step 4: Decompose to subtasks
    subtasks = decompose_to_subtasks(user_requirement, dir_structure, dir_counts_str)
    
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


