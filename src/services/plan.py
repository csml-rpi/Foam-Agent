from typing import Dict, Any, List
from models import PlanIn, PlanOut, Subtask
from utils import LLMService
from nodes.architect_node import architect_node
from services import global_llm_service


def plan_simulation_structure(
    case_id: str,
    user_requirement: str,
    case_stats: Dict[str, List[str]],
    case_dir: str = "",
    database_path: str = "",
    searchdocs: int = 2,
    max_loop: int = 3
) -> PlanOut:
    """
    Plan the simulation structure by analyzing user requirements and generating subtasks.
    
    This function uses LLM to parse user requirements, retrieve similar case references,
    and decompose the requirements into manageable subtasks for OpenFOAM file generation.
    
    Args:
        case_id (str): Unique identifier for the case
        user_requirement (str): Natural language description of the simulation requirements
        case_stats (Dict[str, List[str]]): Available case statistics with keys:
            - case_domain: List of available domains (e.g., ["fluid", "solid"])
            - case_category: List of available categories (e.g., ["tutorial", "advanced"])
            - case_solver: List of available solvers (e.g., ["simpleFoam", "pimpleFoam"])
        case_dir (str, optional): Directory path for the case. Defaults to "".
        database_path (str, optional): Path to the FAISS database. Defaults to "".
        searchdocs (int, optional): Number of similar documents to retrieve. Defaults to 2.
        max_loop (int, optional): Maximum number of retry loops. Defaults to 3.
    
    Returns:
        PlanOut: Contains:
            - plan (List[Subtask]): List of subtasks with file and folder information
            - case_info (Dict): Case metadata including name, solver, domain, category
    
    Raises:
        ValueError: If user requirement cannot be parsed or case stats are invalid
        RuntimeError: If LLM service fails to generate plan
    
    Example:
        >>> case_stats = {
        ...     "case_domain": ["fluid", "solid"],
        ...     "case_category": ["tutorial", "advanced"],
        ...     "case_solver": ["simpleFoam", "pimpleFoam"]
        ... }
        >>> result = plan_simulation_structure(
        ...     case_id="test_case",
        ...     user_requirement="Create a simple fluid flow simulation",
        ...     case_stats=case_stats
        ... )
        >>> print(f"Generated {len(result.plan)} subtasks")
    """
    # Build minimal state for architect_node
    mock_config = type('MockConfig', (), {
        'user_requirement': user_requirement,
        'case_stats': case_stats,
        'case_dir': case_dir,
        'database_path': database_path,
        'searchdocs': searchdocs,
        'max_loop': max_loop
    })()
    
    state = {
        "config": mock_config,
        "user_requirement": user_requirement,
        "llm_service": global_llm_service,
        "case_stats": case_stats,
        "case_dir": case_dir,
    }
    
    try:
        out = architect_node(state)
        plan = [Subtask(file=s["file_name"], folder=s["folder_name"]) for s in out.get("subtasks", [])]
        case_info = {
            "case_name": out.get("case_name"),
            "solver": out.get("case_solver"),
            "domain": out.get("case_domain"),
            "category": out.get("case_category"),
        }
        return PlanOut(plan=plan, case_info=case_info)
    except Exception as e:
        raise RuntimeError(f"Failed to generate simulation plan: {str(e)}")


