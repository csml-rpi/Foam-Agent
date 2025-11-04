"""FastMCP-based OpenFOAM Agent Server.

This module provides a modern MCP server implementation using FastMCP,
exposing OpenFOAM simulation capabilities through clean, well-typed interfaces.
"""

import os
import json
from typing import Dict, List, Optional, Any

from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

# Import existing services
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.plan import (
    resolve_case_dir,
    retrieve_references,
    generate_simulation_plan
)
from services.input_writer import initial_write
from services.run_local import run_allrun_and_collect_errors

from utils import FoamPydantic
from services.review import review_error_logs
from services.visualization import (
    ensure_foam_file,
    generate_pyvista_script,
    run_pyvista_script,
    fix_pyvista_script
)
from config import Config


# Global configuration
global_config = Config()


# Create FastMCP server
mcp = FastMCP(
    name="Foam-Agent",
    version="1.1.0",
    instructions="""
Foam-Agent is a multi-agent framework that automates the entire OpenFOAM-based CFD simulation workflow from a single natural language prompt. 
By managing the full pipeline—from meshing and case setup to execution and post-processing—Foam-Agent dramatically lowers the expertise barrier for Computational Fluid Dynamics.
"""
)


# ============================================================================
# Tool: plan
# ============================================================================

class PlanRequest(BaseModel):
    """Request to plan simulation structure."""
    user_requirement: str = Field(description="User requirements for the simulation")


class PlanResponse(BaseModel):
    """Response from simulation planning."""
    subtasks: List[Dict[str, str]] = Field(description="List of subtasks with file and folder information")
    case_name: str = Field(description="Generated case name")
    case_solver: str = Field(description="OpenFOAM solver to use")
    case_domain: str = Field(description="Simulation domain (e.g., 'fluid', 'solid')")
    case_category: str = Field(description="Case category (e.g., 'tutorial', 'advanced')")


@mcp.tool(name="plan")
async def plan(
    request: PlanRequest,
    ctx: Context
) -> PlanResponse:
    """Plan the simulation structure by analyzing requirements and generating subtasks.
    
    This function uses AI to break down user requirements into manageable subtasks
    for OpenFOAM file generation.
    """
    try:
        await ctx.info("Planning simulation structure from user requirements")
        
        # Load case statistics, available domains, categories, and solvers
        case_stats_path = os.path.join(global_config.database_path, "raw", "openfoam_case_stats.json")
        with open(case_stats_path, 'r') as f:
            case_stats = json.load(f)
        
        # Generate simulation plan
        plan_data = generate_simulation_plan(
            user_requirement=request.user_requirement,
            case_stats=case_stats,
            case_dir="",  # Will be resolved later
            searchdocs=global_config.searchdocs,
            file_dependency_threshold=global_config.file_dependency_threshold
        )
        
        await ctx.info(f"Generated {len(plan_data['subtasks'])} subtasks")
        
        # Convert subtasks to PlanResponse format
        subtasks = [{"file": s["file_name"], "folder": s["folder_name"]} for s in plan_data["subtasks"]]
        
        return PlanResponse(
            subtasks=subtasks,
            case_name=plan_data["case_name"],
            case_solver=plan_data["case_solver"],
            case_domain=plan_data["case_domain"],
            case_category=plan_data["case_category"]
        )
        
    except Exception as e:
        await ctx.error(f"Failed to plan simulation: {str(e)}")
        raise


# ============================================================================
# Tool: input_writer
# ============================================================================

class GenerateFilesRequest(BaseModel):
    """Request to generate OpenFOAM files."""
    case_name: str = Field(description="Case name (from plan response)")
    subtasks: List[Dict[str, str]] = Field(description="List of subtasks to generate files for")
    user_requirement: str = Field(description="User requirements")
    case_solver: str = Field(description="OpenFOAM solver to use")
    case_domain: str = Field(description="Simulation domain")
    case_category: str = Field(description="Case category")


class GenerateFilesResponse(BaseModel):
    """Response from file generation."""
    case_dir: str = Field(description="Path to the case directory")
    foamfiles: FoamPydantic = Field(description="Generated OpenFOAM files with metadata")
    allrun_script: str = Field(description="Path to the generated Allrun script")


@mcp.tool(name="input_writer")
async def input_writer(
    request: GenerateFilesRequest,
    ctx: Context
) -> GenerateFilesResponse:
    """Generate OpenFOAM input files based on subtasks and requirements.
    
    This function creates all necessary OpenFOAM input files including
    system, constant, and initial condition files.
    """
    try:
        await ctx.info(f"Generating OpenFOAM files for case: {request.case_name}")
        
        # Resolve case directory
        case_dir = resolve_case_dir(
            case_name=request.case_name,
            case_dir="",
            run_times=global_config.run_times
        )

        ctx.info(f"Case directory: {case_dir}")
        
        # Load case statistics and retrieve references
        case_stats_path = os.path.join(global_config.database_path, "raw", "openfoam_case_stats.json")
        with open(case_stats_path, 'r') as f:
            case_stats = json.load(f)
        
        # Build case info from request
        case_info = {
            "case_name": request.case_name,
            "case_solver": request.case_solver,
            "case_domain": request.case_domain,
            "case_category": request.case_category
        }

        ctx.info(f"Case info: {case_info}")
        
        # Retrieve references
        tutorial_reference, dir_structure, dir_counts_str, allrun_reference, file_dependency_flag = retrieve_references(
            case_name=case_info["case_name"],
            case_solver=case_info["case_solver"],
            case_domain=case_info["case_domain"],
            case_category=case_info["case_category"],
            searchdocs=global_config.searchdocs,
            file_dependency_threshold=global_config.file_dependency_threshold
        )
        
        # Convert subtasks format from {file, folder} to {file_name, folder_name}
        converted_subtasks = []
        for st in request.subtasks:
            if isinstance(st, dict):
                # Handle both formats: {file, folder} or {file_name, folder_name}
                file_name = st.get("file_name") or st.get("file")
                folder_name = st.get("folder_name") or st.get("folder")
                if file_name and folder_name:
                    converted_subtasks.append({
                        "file_name": file_name,
                        "folder_name": folder_name
                    })
                else:
                    raise ValueError(f"Invalid subtask format: {st}. Must have 'file'/'file_name' and 'folder'/'folder_name'")
            else:
                raise ValueError(f"Invalid subtask type: {type(st)}. Expected dict, got {st}")
        
        ctx.info(f"converted_subtasks: {converted_subtasks}")
        
        # Generate files
        result = initial_write(
            case_dir=case_dir,
            subtasks=converted_subtasks,
            user_requirement=request.user_requirement,
            tutorial_reference=tutorial_reference,
            case_solver=request.case_solver,
            file_dependency_flag=file_dependency_flag,
            case_info=str(case_info),
            allrun_reference=allrun_reference,
            database_path=str(global_config.database_path),
            searchdocs=global_config.searchdocs
        )

        ctx.info(f"result: {result}")
        
        # Get foamfiles from result
        foamfiles = result.get("foamfiles")
        if not foamfiles:
            raise ValueError("No foamfiles returned from initial_write")
        
        allrun_script = os.path.join(case_dir, "Allrun")
        
        num_files = len(foamfiles.list_foamfile) if hasattr(foamfiles, "list_foamfile") else 0
        await ctx.info(f"Generated {num_files} OpenFOAM files in {case_dir}")
        
        return GenerateFilesResponse(
            case_dir=case_dir,
            foamfiles=foamfiles,
            allrun_script=allrun_script
        )
        
    except Exception as e:
        await ctx.error(f"Failed to generate OpenFOAM files: {str(e)}")
        raise


# ============================================================================
# Tool: run
# ============================================================================

class RunSimulationRequest(BaseModel):
    """Request to run local simulation."""
    case_dir: str = Field(description="Path to the case directory")
    timeout: int = Field(default=3600, description="Timeout in seconds")


class RunSimulationResponse(BaseModel):
    """Response from simulation run."""
    status: str = Field(description="Run status: 'success' or 'failed'")
    errors: List[str] = Field(description="List of errors found")
    log_files: Dict[str, str] = Field(description="Paths to log files")


@mcp.tool(name="run")
async def run(
    request: RunSimulationRequest,
    ctx: Context
) -> RunSimulationResponse:
    """Run the OpenFOAM simulation locally.
    
    This function executes the Allrun script and collects any errors.
    """
    try:
        await ctx.info(f"Running simulation in directory: {request.case_dir}")
        
        # Validate case directory exists
        if not os.path.exists(request.case_dir):
            raise ValueError(f"Case directory does not exist: {request.case_dir}")
        
        # Run locally
        error_logs = run_allrun_and_collect_errors(
            case_dir=request.case_dir,
            timeout=request.timeout,
            max_retries=3
        )
        
        # Convert error logs to strings if they're dictionaries
        errors = []
        for err in error_logs:
            if isinstance(err, dict):
                # Format: "file: error_content"
                file_name = err.get("file", "unknown")
                error_content = err.get("error_content", str(err))
                errors.append(f"{file_name}: {error_content}")
            else:
                errors.append(str(err))
        
        # Prepare log file paths (not content)
        log_files = {}
        out_path = os.path.join(request.case_dir, 'Allrun.out')
        err_path = os.path.join(request.case_dir, 'Allrun.err')
        
        if os.path.exists(out_path):
            log_files['Allrun.out'] = out_path
        
        if os.path.exists(err_path):
            log_files['Allrun.err'] = err_path
        
        status = "success" if not errors else "failed"
        
        await ctx.info(f"Simulation {status} with {len(errors)} error(s)")
        
        return RunSimulationResponse(
            status=status,
            errors=errors,
            log_files=log_files
        )
            
    except Exception as e:
        await ctx.error(f"Failed to run simulation: {str(e)}")
        raise


# ============================================================================
# Tool: review
# ============================================================================

class ReviewRequest(BaseModel):
    """Request to review simulation errors."""
    case_dir: str = Field(description="Path to the case directory")
    errors: List[str] = Field(description="List of error messages from simulation")
    user_requirement: str = Field(description="Original user requirements")


class ReviewResponse(BaseModel):
    """Response from simulation review."""
    analysis: str = Field(description="Analysis of simulation errors")


@mcp.tool(name="review")
async def review(
    request: ReviewRequest,
    ctx: Context
) -> ReviewResponse:
    """Review simulation errors and suggest improvements.
    
    This function analyzes simulation errors and provides suggestions for fixes.
    """
    try:
        await ctx.info(f"Reviewing errors for case directory: {request.case_dir}")
        
        # Validate case directory exists
        if not os.path.exists(request.case_dir):
            raise ValueError(f"Case directory does not exist: {request.case_dir}")
        
        # Load case statistics
        case_stats_path = os.path.join(global_config.database_path, "raw", "openfoam_case_stats.json")
        with open(case_stats_path, 'r') as f:
            case_stats = json.load(f)
        
        # Extract case name from case_dir for reference lookup
        case_name = os.path.basename(request.case_dir)
        
        # Get tutorial reference
        case_info = {
            "case_name": case_name,
            "case_solver": "simpleFoam",  # Default
            "case_domain": "fluid",
            "case_category": "tutorial"
        }
        
        tutorial_reference, _, _, _, _ = retrieve_references(
            case_name=case_info["case_name"],
            case_solver=case_info["case_solver"],
            case_domain=case_info["case_domain"],
            case_category=case_info["case_category"],
            searchdocs=global_config.searchdocs,
            file_dependency_threshold=global_config.file_dependency_threshold
        )
        
        # Read current foamfiles from case directory for review context
        await ctx.info("Reading OpenFOAM files for review context...")
        from utils import read_case_foamfiles
        foamfiles = read_case_foamfiles(request.case_dir)
        await ctx.info(f"Read {len(foamfiles.list_foamfile)} file(s) for review")
        
        # Review results - directly call review_error_logs
        review_content, _ = review_error_logs(
            tutorial_reference=tutorial_reference,
            foamfiles=foamfiles,
            error_logs=request.errors,
            user_requirement=request.user_requirement,
            history_text=None
        )
        
        await ctx.info(f"Review completed, found {len(request.errors)} error(s)")
        
        # Format response (suggestions and issues are empty as review_error_logs only returns analysis)
        return ReviewResponse(
            analysis=review_content
        )
        
    except Exception as e:
        await ctx.error(f"Failed to review results: {str(e)}")
        raise


# ============================================================================
# Tool: apply_fixes
# ============================================================================

class ApplyFixesRequest(BaseModel):
    """Request to apply fixes to an OpenFOAM case based on review analysis."""
    case_dir: str = Field(description="Path to the OpenFOAM case directory")
    error_logs: List[str] = Field(description="List of error log messages from simulation")
    review_analysis: str = Field(description="Review analysis with fix suggestions from the review tool. Must be provided.")
    user_requirement: str = Field(description="Original user requirements or simulation description for context")


class ApplyFixesResponse(BaseModel):
    """Response from applying fixes."""
    updated_files: List[str] = Field(description="List of file paths that were updated")
    status: str = Field(description="Fix application status ('ok' or 'no_changes')")


@mcp.tool(name="apply_fixes")
async def apply_fixes(
    request: ApplyFixesRequest,
    ctx: Context
) -> ApplyFixesResponse:
    """Apply fixes to the OpenFOAM case files based on review analysis.
    
    This tool rewrites OpenFOAM files to fix errors identified during review.
    It must be called after the 'review' tool has provided analysis.
    
    The tool directly calls rewrite_files which handles:
    - Reading current foamfiles and directory structure from case_dir
    - Using LLM to generate corrected file contents based on review_analysis
    - Writing updated files back to the case directory
    
    Workflow:
    1. First call 'review' tool to get review_analysis
    2. Then call 'apply_fixes' with the review_analysis to rewrite files
    
    Args:
        request: ApplyFixesRequest containing:
            - case_dir: Path to the OpenFOAM case directory
            - error_logs: List of error messages from simulation
            - review_analysis: Analysis and fix suggestions from review tool (required)
            - user_requirement: Original user requirements (optional)
    
    Returns:
        ApplyFixesResponse with list of updated files and status
    
    Raises:
        ValueError: If case directory does not exist or review_analysis is empty
        RuntimeError: If fix application fails
    
    Example:
        # Two-step workflow (review first, then fix)
        review_resp = await review(case_dir, errors, user_requirement)
        fix_resp = await apply_fixes(case_dir, errors, review_resp.analysis, user_requirement)
    """
    try:
        await ctx.info(f"Applying fixes for case directory: {request.case_dir}")
        
        # Validate case directory exists
        if not os.path.exists(request.case_dir):
            raise ValueError(f"Case directory does not exist: {request.case_dir}")
        
        # Validate review_analysis is provided
        if not request.review_analysis or request.review_analysis.strip() == "":
            raise ValueError(
                "review_analysis is required. Please call the 'review' tool first "
                "to get review analysis, then provide it to this tool."
            )
        
        await ctx.info("Rewriting OpenFOAM files based on review analysis...")
        
        # Directly call rewrite_files - it now handles file reading internally
        from services.input_writer import rewrite_files
        
        result = rewrite_files(
            case_dir=request.case_dir,
            error_logs=request.error_logs,
            review_analysis=request.review_analysis,
            user_requirement=request.user_requirement
            # foamfiles and dir_structure will be read automatically if None
        )
        
        # Extract written file paths
        written_files = []
        if result.get("foamfiles") and hasattr(result["foamfiles"], "list_foamfile"):
            for foamfile in result["foamfiles"].list_foamfile:
                file_path = os.path.join(request.case_dir, foamfile.folder_name, foamfile.file_name)
                written_files.append(file_path)
        
        status = "ok" if written_files else "no_changes"
        
        await ctx.info(f"Successfully applied fixes. Updated {len(written_files)} file(s)")
        
        return ApplyFixesResponse(
            updated_files=written_files,
            status=status
        )
        
    except Exception as e:
        await ctx.error(f"Failed to apply fixes: {str(e)}")
        raise


# ============================================================================
# Tool: visualization
# ============================================================================

class VisualizationRequest(BaseModel):
    """Request to generate visualization."""
    case_dir: str = Field(description="Path to the case directory")
    quantity: str = Field(description="Quantity to visualize (e.g., 'velocity', 'pressure')")
    visualization_type: str = Field(default="pyvista", description="Visualization type")


class VisualizationResponse(BaseModel):
    """Response from visualization generation."""
    artifacts: List[str] = Field(description="List of generated visualization files")
    script: str = Field(description="Visualization script")


@mcp.tool(name="visualization")
async def visualization(
    request: VisualizationRequest,
    ctx: Context
) -> VisualizationResponse:
    """Generate visualization for the simulation results.
    
    This function creates visualization artifacts using PyVista.
    """
    try:
        await ctx.info(f"Generating visualization for case directory: {request.case_dir}")
        
        # Validate case directory exists
        if not os.path.exists(request.case_dir):
            raise ValueError(f"Case directory does not exist: {request.case_dir}")
        
        # Ensure foam file exists
        foam_file = ensure_foam_file(request.case_dir)
        
        # Generate visualization script
        script = generate_pyvista_script(
            case_dir=request.case_dir,
            foam_file=foam_file,
            user_requirement=request.quantity,
            previous_errors=[]
        )
        
        # Run visualization script
        ok, img, errs = run_pyvista_script(request.case_dir, script)
        
        if ok and img:
            artifacts = [img]
        else:
            # Try to fix the script
            fixed = fix_pyvista_script(foam_file, script, errs)
            ok2, img2, errs2 = run_pyvista_script(request.case_dir, fixed)
            artifacts = [img2] if ok2 and img2 else []
        
        await ctx.info(f"Generated {len(artifacts)} visualization artifact(s)")
        
        return VisualizationResponse(
            artifacts=artifacts,
            script=script
        )
        
    except Exception as e:
        await ctx.error(f"Failed to generate visualization: {str(e)}")
        raise



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastMCP OpenFOAM Agent Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="http",
        help="Transport method (default: http)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for HTTP transport (default: 7860)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for HTTP transport (default: localhost)"
    )
    
    args = parser.parse_args()
    
    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        # Configure uvicorn with correct websockets setting
        uvicorn_config = {"ws": "websockets"}
        mcp.run("http", host=args.host, port=args.port, uvicorn_config=uvicorn_config)


# run the server:
# python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860