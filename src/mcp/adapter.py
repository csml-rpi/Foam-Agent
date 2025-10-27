"""MCP adapter scaffolding.

This module exposes thin functions that validate inputs via models and call
service-layer wrappers. It does not implement an MCP server; you can import
these functions into your MCP runtime of choice.
"""

from typing import Dict
import os
from models import (
    CreateCaseIn, CreateCaseOut,
    PlanIn, PlanOut,
    GenerateFileIn, GenerateFileOut,
    HPCScriptIn, HPCScriptOut,
    RunIn, RunOut,
    JobStatusIn, JobStatusOut,
    LogsIn, LogsOut,
    ReviewIn, ReviewOut,
    ApplyFixIn, ApplyFixOut,
    VisualizationIn, VisualizationOut,
)
from services.cases import create_case
from services.plan import plan_simulation_structure
from services.run_local import run_simulation_local
from services.run_hpc import generate_hpc_script, run_simulation_hpc, check_job, extract_cluster_info_from_requirement, wait_for_job, check_logs_for_errors
from services.mesh import copy_custom_mesh, prepare_standard_mesh
from services.review import review_and_suggest_fix
from services.apply_fix import apply_fix as apply_fix_service
from services.visualization import ensure_foam_file, generate_pyvista_script, run_pyvista_script, fix_pyvista_script


def mcp_create_case(payload: Dict) -> Dict:
    out: CreateCaseOut = create_case(CreateCaseIn(**payload))
    return out.model_dump()


def mcp_plan_simulation_structure(payload: Dict, config) -> Dict:
    out: PlanOut = plan_simulation_structure(PlanIn(**payload), config)
    return out.model_dump()


def mcp_generate_file_content(payload: Dict, case_dir: str, tutorial_reference: str, case_solver: str) -> Dict:
    out: GenerateFileOut = generate_file_content(GenerateFileIn(**payload), case_dir, tutorial_reference, case_solver)
    return out.model_dump()


def mcp_generate_hpc_script(payload: Dict, case_dir: str) -> Dict:
    out: HPCScriptOut = generate_hpc_script(HPCScriptIn(**payload), case_dir)
    return out.model_dump()


def mcp_run_simulation(payload: Dict, config, case_dir: str, script_path: str = "") -> Dict:
    rin = RunIn(**payload)
    if rin.environment == "local":
        out: RunOut = run_simulation_local(rin, config, case_dir)
    else:
        out: RunOut = run_simulation_hpc(script_path)
    return out.model_dump()


def mcp_check_job_status(payload: Dict) -> Dict:
    out: JobStatusOut = check_job(JobStatusIn(**payload))
    return out.model_dump()


# Additional endpoints

def mcp_generate_mesh(payload: Dict, llm, case_dir: str) -> Dict:
    method = (payload.get('mesh_config') or {}).get('method', 'standard')
    if method == 'custom':
        custom_mesh_path = payload['mesh_config'].get('path')
        return copy_custom_mesh(custom_mesh_path, payload.get('user_requirement', ''), case_dir)
    elif method == 'standard':
        return prepare_standard_mesh(payload.get('user_requirement', ''), case_dir)
    else:
        return {"error": f"Unsupported mesh method: {method}"}


def mcp_get_simulation_logs(payload: Dict, case_dir: str) -> Dict:
    logs = {}
    out_path = os.path.join(case_dir, 'Allrun.out')
    err_path = os.path.join(case_dir, 'Allrun.err')
    if os.path.exists(out_path):
        logs['Allrun.out'] = open(out_path, 'r', errors='ignore').read()
    if os.path.exists(err_path):
        logs['Allrun.err'] = open(err_path, 'r', errors='ignore').read()
    logs['errors'] = check_logs_for_errors(case_dir)
    return logs


def mcp_review_and_suggest_fix(payload: Dict, tutorial_reference: str, foamfiles, user_requirement: str) -> Dict:
    return review_and_suggest_fix(ReviewIn(**payload), tutorial_reference, foamfiles, user_requirement).model_dump()


def mcp_apply_fix(payload: Dict, case_dir: str) -> Dict:
    return apply_fix_service(ApplyFixIn(**payload), case_dir).model_dump()


def mcp_generate_visualization(payload: Dict, case_dir: str) -> Dict:
    foam_file = ensure_foam_file(case_dir)
    script = generate_pyvista_script(case_dir, foam_file, payload.get('quantity', ''), [])
    ok, img, errs = run_pyvista_script(case_dir, script)
    if ok and img:
        return {"artifacts": [img], "job_id": None}
    fixed = fix_pyvista_script(foam_file, script, errs)
    ok2, img2, errs2 = run_pyvista_script(case_dir, fixed)
    return {"artifacts": [img2] if ok2 and img2 else [], "errors": errs + errs2}


