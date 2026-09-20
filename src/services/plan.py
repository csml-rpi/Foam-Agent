import os
import re
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from utils import retrieve_faiss, parse_directory_structure
from config import Config
from openfoam_target import generation_convention, configured_openfoam_target
from models import ExistingCasePlan
from . import global_llm_service
from .case_paths import CasePathSafetyError, safe_case_relative_path


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


def parse_requirement_to_case_info(
    user_requirement: str,
    case_stats: Dict[str, List[str]],
    *,
    llm_service: Optional[Any] = None,
    openfoam_target: str = "",
) -> Dict[str, str]:
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
        "Please transform the following user requirement into a standard case description using a structured format."
        "The key elements should include case name, case domain, case category, and case solver."
        f"Note: case domain must be one of {case_stats.get('case_domain', [])}."
        f"Note: case category must be one of {case_stats.get('case_category', [])}."
        f"Note: case solver must be one of {case_stats.get('case_solver', [])}."
        + (
            " The selected native runtime is ESI/OpenCFD OpenFOAM v2006. "
            "Choose only a solver present in the supplied v2006 case statistics."
            if openfoam_target == "esi-v2006"
            else ""
        )
    )
    parse_user_prompt = f"User requirement: {user_requirement}."
    llm_client = llm_service if llm_service is not None else global_llm_service
    res = llm_client.invoke(parse_user_prompt, parse_system_prompt, pydantic_obj=CaseSummaryModel)
    raw_case_name = res.case_name.replace(" ", "_")
    try:
        case_name = safe_case_relative_path("", raw_case_name).name
    except CasePathSafetyError as exc:
        raise ValueError(f"Unsafe generated case name: {raw_case_name!r}") from exc
    return {
        "case_name": case_name,
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
    try:
        case_name = safe_case_relative_path("", case_name).name
    except CasePathSafetyError as exc:
        raise ValueError(f"Unsafe generated case name: {case_name!r}") from exc
    if run_directory is None:
        run_directory = str(Path(__file__).resolve().parent.parent / "runs")
    base_dir = str(run_directory)
    if run_times > 1:
        return os.path.join(base_dir, f"{case_name}_{run_times}")
    return os.path.join(base_dir, case_name)


class SimilarCaseAdviceModel(BaseModel):
    match_level: str = Field(description="high/medium/low/none")
    use_scope: str = Field(description="short guidance about which files can use the reference")
    advice: str = Field(description="one-sentence advice to include in prompts")


_DETAIL_RECALL_FLOOR = 50
_DETAIL_RECALL_MULTIPLIER = 5


def _log_top3(label: str, items: List[Dict[str, Any]]) -> None:
    print(f"{label} (top-3):")
    for i, it in enumerate(items[:3], 1):
        print(
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
    llm_service: Optional[Any] = None,
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

    llm_client = llm_service if llm_service is not None else global_llm_service
    return llm_client.invoke(user_prompt, sys_prompt, pydantic_obj=SimilarCaseAdviceModel)


def retrieve_references(case_name: str,
                        case_solver: str,
                        case_domain: str,
                        case_category: str,
                        searchdocs: int = 2,
                        user_requirement: str = "",
                        config: Optional[Config] = None,
                        llm_service: Optional[Any] = None) -> Tuple[str, str, str, str, SimilarCaseAdviceModel]:
    # Build case_info
    case_info = f"case name: {case_name}\ncase domain: {case_domain}\ncase category: {case_category}\ncase solver: {case_solver}"
    print("Retrieval query:\n" + case_info)

    requested_docs = max(1, int(searchdocs))
    recall_k = max(10, int(searchdocs))
    detail_recall_k = max(
        _DETAIL_RECALL_FLOOR,
        requested_docs * _DETAIL_RECALL_MULTIPLIER,
    )
    retrieval_query = case_info

    structure_kwargs: Dict[str, Any] = {"topk": recall_k}
    if config is not None:
        structure_kwargs["config"] = config
    faiss_structure_all = retrieve_faiss(
        "openfoam_tutorials_structure",
        retrieval_query,
        **structure_kwargs,
    )
    print(f"Retrieved {len(faiss_structure_all)} candidates from FAISS.")

    # Hard constraint: domain must match; solver match is a ranking preference.
    domain_matched = [
        candidate for candidate in faiss_structure_all
        if _normalise_metadata(candidate.get("case_domain")) == _normalise_metadata(case_domain)
    ]
    _log_top3("Domain-matched structure candidates", domain_matched)

    if not domain_matched:
        print(f"No suitable similar case found under domain={case_domain}.")
        advice = _build_advice(user_requirement, case_info, None, faiss_structure_all, llm_service)
        return "", "", "", "", advice

    # Rerank by solver match, then semantic score.
    ranked = _rerank_candidates(domain_matched, case_solver)
    selected = ranked[0]
    dir_structure = _extract_directory_structure(selected)
    if not dir_structure:
        print("Warning: No directory_structure found in selected similar case.")
        advice = _build_advice(user_requirement, case_info, selected, ranked, llm_service)
        return "", "", "", "", advice

    # The structure index only contains the file tree.  Fetch the exact same
    # metadata identity from the details index before exposing file contents.
    detail = _retrieve_matching_details(
        selected,
        dir_structure,
        detail_recall_k,
        config=config,
    )
    faiss_detailed = ""
    if detail is not None:
        faiss_detailed = str(detail.get("full_content") or "")
        if _normalise_metadata(faiss_detailed) == "unknown":
            faiss_detailed = ""
        else:
            faiss_detailed = re.sub(r"\n{3,}", "\n\n", faiss_detailed)

    dir_counts = parse_directory_structure(dir_structure)
    dir_counts_str = ',\n'.join([f"There are {count} files in Directory: {directory}" for directory, count in dir_counts.items()])

    # Build allrun reference
    index_content = (
        "<index>\n"
        f"case name: {selected.get('case_name')}\n"
        f"case domain: {selected.get('case_domain')}\n"
        f"case category: {selected.get('case_category')}\n"
        f"case solver: {selected.get('case_solver')}\n"
        "</index>\n"
        f"<directory_structure>\n{dir_structure}\n</directory_structure>"
    )
    allrun_kwargs: Dict[str, Any] = {"topk": requested_docs}
    if config is not None:
        allrun_kwargs["config"] = config
    faiss_allrun = retrieve_faiss(
        "openfoam_allrun_scripts",
        index_content,
        **allrun_kwargs,
    )
    allrun_reference = "Similar cases are ordered, with smaller numbers indicating greater similarity. For example, similar_case_1 is more similar than similar_case_2, and similar_case_2 is more similar than similar_case_3.\n"
    for idx, item in enumerate(faiss_allrun):
        allrun_reference += f"<similar_case_{idx + 1}>{item['full_content']}</similar_case_{idx + 1}>\n\n\n"

    advice = _build_advice(user_requirement, case_info, selected, ranked, llm_service)
    return faiss_detailed, dir_structure, dir_counts_str, allrun_reference, advice


def decompose_to_subtasks(
    user_requirement: str,
    dir_structure: str,
    dir_counts_str: str,
    *,
    llm_service: Optional[Any] = None,
    openfoam_target: str = "",
) -> List[Dict]:
    decompose_system_prompt = (
        "You are an experienced Planner specializing in OpenFOAM projects. "
        "Your task is to break down the following user requirement into a series of smaller, manageable subtasks. "
        "For each subtask, identify the file name of the OpenFOAM input file (foamfile) and the corresponding folder name where it should be stored. "
        "Your final output must strictly follow the JSON schema below and include no additional keys or information:\n\n"
        "```\n{\n  \"subtasks\": [\n    {\n      \"file_name\": \"<string>\",\n      \"folder_name\": \"<string>\"\n    }\n    // ... more subtasks\n  ]\n}\n```\n\n"
        "Make sure that your output is valid JSON and strictly adheres to the provided schema."
        "Make sure you generate all the necessary files for the user's requirements."
        + (
            " The target is native ESI/OpenCFD OpenFOAM v2006. Select only "
            "the file and dictionary layout present in the supplied v2006 reference."
            if openfoam_target == "esi-v2006"
            else ""
        )
    )

    decompose_user_prompt = (
        f"User Requirement: {user_requirement}\n\n"
        f"Reference Directory Structure (similar case): {dir_structure}\n\n{dir_counts_str}\n\n"
        "Make sure you generate all the necessary files for the user's requirements."
        "Do not include any gmsh files like .geo etc. in the subtasks."
        "Only include blockMesh or snappyHexMesh if the user hasnt requested for gmsh mesh or user isnt using an external uploaded custom mesh"
        "Please generate the output as structured JSON."
    )

    llm_client = llm_service if llm_service is not None else global_llm_service
    res = llm_client.invoke(decompose_user_prompt, decompose_system_prompt, pydantic_obj=OpenFOAMPlanModel)
    return [{"file_name": s.file_name, "folder_name": s.folder_name} for s in res.subtasks]


def generate_simulation_plan(
    user_requirement: str,
    case_stats: Dict[str, List[str]],
    case_dir: str = "",
    searchdocs: int = 2,
    run_directory: str | Path | None = None,
    run_times: int = 1,
    config: Optional[Config] = None,
    llm_service: Optional[Any] = None,
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
    target_convention = generation_convention(config) if config is not None else ""
    case_info = parse_requirement_to_case_info(
        user_requirement,
        case_stats,
        llm_service=llm_service,
        openfoam_target=target_convention,
    )
    case_name = case_info["case_name"]
    case_domain = case_info["case_domain"]
    case_category = case_info["case_category"]
    case_solver = case_info["case_solver"]
    
    # Step 2: Resolve case directory
    resolved_case_dir = resolve_case_dir(
        case_name=case_name,
        case_dir=case_dir,
        run_times=run_times,
        run_directory=(
            str(run_directory)
            if run_directory is not None
            else str(Path(__file__).resolve().parent.parent / "runs")
        ),
    )
    
    # Step 3: Retrieve references
    faiss_detailed, dir_structure, dir_counts_str, allrun_reference, advice = retrieve_references(
        case_name=case_name,
        case_solver=case_solver,
        case_domain=case_domain,
        case_category=case_category,
        searchdocs=searchdocs,
        user_requirement=user_requirement,
        config=config,
        llm_service=llm_service,
    )
    
    # Step 4: Decompose to subtasks.
    subtasks = decompose_to_subtasks(
        user_requirement,
        dir_structure,
        dir_counts_str,
        llm_service=llm_service,
        openfoam_target=target_convention,
    )
    
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


DEFAULT_IMPORTED_REQUIREMENT = (
    "Run the supplied OpenFOAM case while preserving its existing physical definition. "
    "Diagnose and repair technical problems only when execution requires it."
)



def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected a structured model response, got {type(value).__name__}")


def plan_imported_case(state: dict[str, Any]) -> dict[str, Any]:
    """Choose graph branches without regenerating an existing case."""
    # The import snapshot is used only for source selection and initial checks.
    # Subsequent planning reads current files, not the snapshot's solver/issues.
    context = dict(state.get("case_context") or {})
    issues = list(context.get("issues") or [])
    target = configured_openfoam_target(state.get("config")) or context.get("platform")
    if target not in {"foundation-v10", "esi-v2006"}:
        return {
            "workflow_status": "failed",
            "requires_meshing": False,
            "requires_input_writer": False,
            "termination_reason": "Specify --openfoam_target foundation-v10 or esi-v2006.",
        }

    manifest = state.get("case_import_manifest")
    if getattr(manifest, "target_mismatch", False):
        mismatch_errors = [
            issue for issue in issues
            if issue.startswith("Configured target ") and "differs from detected" in issue
        ]
        if not mismatch_errors:
            mismatch_errors = [
                "Configured OpenFOAM target does not match the detected case platform."
            ]
        return {
            "workflow_status": "failed",
            "target_mismatch": True,
            "requires_meshing": False,
            "requires_input_writer": False,
            "error_logs": [
                {"file": "case_import", "error_content": issue}
                for issue in mismatch_errors
            ],
            "termination_reason": "case_target_mismatch",
        }

    explicit_requirement = bool(state.get("user_requirement_explicit"))
    if (
        not explicit_requirement
        and not state.get("loop_count")
        and context.get("has_allrun")
        and not issues
    ):
        return {
            "workflow_status": "running",
            "requires_meshing": False,
            "requires_input_writer": False,
            # A complete, untouched case with no explicit user requirement has
            # nothing to infer from: keep the same conservative defaults
            # llm_requires_hpc/llm_requires_visualization use when unsure, and
            # skip the LLM calls entirely for this no-LLM-needed fast path.
            "requires_hpc": False,
            "requires_visualization": False,
            "termination_reason": None,
        }

    prompt = (
        "You are planning work on an existing OpenFOAM case. Return structured routing decisions. "
        "Set requires_input_writer only when files must be created or changed, and requires_meshing "
        "only when an external or Gmsh mesh operation is needed. The graph prepares the mesh first, "
        "then modifies files if needed, then runs the case. A complete case with no requested changes "
        "must set both flags to false. Preserve existing physical conditions unless explicitly changed. "
        "If a required physical condition cannot be inferred, return failed and explain the missing information in reason. "
        "When requires_input_writer is true, target_files must list the files to create or modify and the required changes. "
        "Explain the requested modifications in reason.\n"
        f"<user_requirement>{state.get('user_requirement', '')}</user_requirement>\n"
        f"<dir_structure>{state.get('dir_structure') or {}}</dir_structure>\n"
        f"<foamfiles>{state.get('foamfiles')}</foamfiles>\n"
        f"<error_logs>{state.get('error_logs') or []}</error_logs>\n"
        f"<review_analysis>{state.get('review_analysis') or ''}</review_analysis>\n"
        f"<custom_mesh_path>{state.get('custom_mesh_path') or ''}</custom_mesh_path>\n"
        f"<openfoam_target>{target}</openfoam_target>"
    )
    llm = state.get("llm_service")
    response = llm.invoke(
        prompt,
        "Plan the minimum work required for this existing case.",
        pydantic_obj=ExistingCasePlan,
    )
    plan = _model_dump(response)
    status = plan.get("status", "failed")
    targets = plan.get("target_files") or []
    if status == "ready" and plan.get("requires_input_writer") and not targets:
        return {
            "workflow_status": "failed",
            "requires_meshing": False,
            "requires_input_writer": False,
            "termination_reason": "Planner must specify target_files before requesting file changes.",
        }
    return {
        "workflow_status": "running" if status == "ready" else status,
        "requires_meshing": bool(plan.get("requires_meshing")),
        "requires_input_writer": bool(plan.get("requires_input_writer")),
        "input_writer_mode": "rewrite",
        "review_analysis": plan.get("reason") or "Apply the requested changes to the existing case.",
        "rewrite_plan": {"target_files": targets} if targets else None,
        "termination_reason": None if status == "ready" else plan.get("reason") or status,
    }


def _normalise_metadata(value: Any) -> str:
    return str(value or "").strip().casefold()


def _normalise_structure(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()






def _score_value(item: Dict[str, Any]) -> float:
    score = item.get("score")
    if score is None:
        return float("inf")
    try:
        return float(score)
    except (TypeError, ValueError):
        return float("inf")


def _metadata_identity(item: Dict[str, Any]) -> tuple[str, str, str, str]:
    return tuple(
        _normalise_metadata(item.get(key))
        for key in ("case_name", "case_domain", "case_category", "case_solver")
    )


def _extract_directory_structure(item: Dict[str, Any]) -> str:
    structure = item.get("dir_structure")
    if structure and _normalise_metadata(structure) != "unknown":
        return str(structure).strip()

    full_content = str(item.get("full_content") or "")
    match = re.search(
        r"<directory_structure>(.*?)</directory_structure>",
        full_content,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _retrieve_matching_details(
    selected: Dict[str, Any],
    dir_structure: str,
    recall_k: int,
    config: Optional[Config] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch details for the exact case selected from the structure index."""
    detail_query = (
        "<index>\n"
        f"case name: {selected.get('case_name')}\n"
        f"case domain: {selected.get('case_domain')}\n"
        f"case category: {selected.get('case_category')}\n"
        f"case solver: {selected.get('case_solver')}\n"
        "</index>\n"
        f"<directory_structure>\n{dir_structure}\n</directory_structure>"
    )

    try:
        kwargs: Dict[str, Any] = {"topk": recall_k}
        if config is not None:
            kwargs["config"] = config
        candidates = retrieve_faiss(
            "openfoam_tutorials_details",
            detail_query,
            **kwargs,
        )
    except ValueError as exc:
        print(f"Warning: Could not retrieve tutorial details: {exc}")
        return None

    selected_identity = _metadata_identity(selected)
    exact_matches = [
        candidate
        for candidate in candidates
        if _metadata_identity(candidate) == selected_identity
    ]
    if not exact_matches:
        print(
            "Warning: Details index did not return the exact case selected from "
            "the structure index; skipping tutorial contents."
        )
        return None

    target_structure = _normalise_structure(dir_structure)
    exact_structure_matches = [
        candidate
        for candidate in exact_matches
        if _normalise_structure(_extract_directory_structure(candidate))
        == target_structure
    ]
    if not exact_structure_matches:
        print(
            "Warning: Details index returned matching case metadata but not the "
            "same directory structure; skipping tutorial contents."
        )
        return None

    return min(exact_structure_matches, key=_score_value)
