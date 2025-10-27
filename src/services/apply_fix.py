from typing import Dict, List, Optional
from models import ApplyFixIn, ApplyFixOut
from services.input_writer import rewrite_files
import os


def apply_fix(inp: ApplyFixIn, case_dir: str) -> ApplyFixOut:
    """
    Apply fixes to OpenFOAM files based on review analysis and error logs.
    
    This function serves as the MCP adapter layer that takes review suggestions
    and applies them to the OpenFOAM files. It calls the core rewrite logic
    from input_writer to generate corrected file contents.
    
    Args:
        inp (ApplyFixIn): Input containing:
            - case_id (str): Unique identifier for the case
            - foamfiles (Optional[Any]): Current FoamPydantic object with file contents
            - error_logs (List[str]): List of error messages from simulation
            - review_analysis (str): Analysis and suggestions from reviewer
            - user_requirement (str): Original user requirements
            - dir_structure (Optional[Dict]): Current directory structure
        case_dir (str): Directory path where the case files are located
    
    Returns:
        ApplyFixOut: Contains:
            - status (str): Fix status ("ok" if files were written, "no_changes" otherwise)
            - written (List[str]): List of file paths that were written/updated
            - updated_dir_structure (Optional[Dict]): Updated directory structure
            - updated_foamfiles (Optional[Any]): Updated FoamPydantic object
            - cleared_error_logs (List[str]): Cleared error logs (empty on success)
    
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If file rewriting fails
        FileNotFoundError: If case directory does not exist
    
    Example:
        >>> inp = ApplyFixIn(
        ...     case_id="test_case",
        ...     error_logs=["Error: undefined reference"],
        ...     review_analysis="Add missing boundary condition",
        ...     user_requirement="Simple flow simulation"
        ... )
        >>> result = apply_fix(inp, "/path/to/case")
        >>> print(f"Fix status: {result.status}")
    """
    # Call the core rewrite logic from input_writer
    result = rewrite_files(
        case_dir=case_dir,
        foamfiles=inp.foamfiles,
        error_logs=inp.error_logs,
        review_analysis=inp.review_analysis,
        user_requirement=inp.user_requirement,
        dir_structure=inp.dir_structure or {}
    )
    
    # Extract written files from the result
    written_files = []
    foamfiles_obj = result.get("foamfiles")
    if foamfiles_obj and hasattr(foamfiles_obj, "list_foamfile"):
        for foamfile in foamfiles_obj.list_foamfile:
            file_path = os.path.join(case_dir, foamfile.folder_name, foamfile.file_name)
            written_files.append(file_path)
    
    return ApplyFixOut(
        status="ok" if written_files else "no_changes",
        written=written_files,
        updated_dir_structure=result.get("dir_structure", {}),
        updated_foamfiles=foamfiles_obj,
        cleared_error_logs=result.get("error_logs", [])
    )


