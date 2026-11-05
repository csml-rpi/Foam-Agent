"""Run solver-only Allrun in a separately verified Slurm attempt."""
from pathlib import Path
from services.stage_plan import check_stage_inputs, check_stage_script, stage_outcome
from services.run_hpc import HpcWorkflowStop, run_stage_hpc, check_logs_for_errors
from services.run_local import _cleanup_run_artifacts, _execution_output
from services.solver_execution import snapshot_result_fields
from openfoam_target import runtime_openfoam_target


def hpc_runner_node(state):
    current = dict(state)
    case = current["case_dir"]
    errors = check_stage_inputs(case, current["solver_operations"], "solver")
    if not errors:
        errors = check_stage_script(case, current['solver_operations'], 'solver')
    if errors:
        return {**stage_outcome(state, "solver", {"executed": False, "error_logs": errors}),
                "error_logs": errors, "solver_execution": {"executed": False, "error_logs": errors}}
    _cleanup_run_artifacts(case)
    fields_before = snapshot_result_fields(case)
    try:
        result = run_stage_hpc(current, str(Path(case) / "Allrun"), "solver", current["config"].max_time_limit)
    except HpcWorkflowStop as stop:
        return {**stop.state_update(), "solver_jobs": current.get("solver_jobs", []),
                "loop_count": current.get("loop_count", 0), "cluster_info": current.get("cluster_info")}
    except Exception as error:
        return {"error_logs": [{"file": "Allrun", "error_content": str(error)}],
                "solver_jobs": current.get("solver_jobs", []), "loop_count": current.get("loop_count", 0)}
    errors = check_logs_for_errors(case, fields_before=fields_before,
                                   openfoam_target=runtime_openfoam_target(current["config"]),
                                   timeout=current["config"].max_time_limit)
    if result.get("returncode"):
        # Attempt paths and job IDs remain in solver_jobs; the repair input is
        # the actual failure, so identical retries retain the same fingerprint.
        details = (
            f"HPC Allrun exited with return code {result['returncode']}; "
            f"Slurm state: {result.get('status')}.\n"
            f"{result.get('diagnostic') or ''}\n{_execution_output(case)}"
        )
        errors.append({"file": "Allrun", "error_content": details.strip()})
    return {**stage_outcome(current, "solver", {"execution": result, "error_logs": errors}),
            "solver_execution": result, "error_logs": errors, "solver_jobs": current.get("solver_jobs", []),
            "loop_count": current.get("loop_count", 0),
            "job_id": result.get("job_id"), "cluster_info": current.get("cluster_info"),
            "slurm_script_path": result.get("script"),
            "workflow_status": "success" if not errors else "running"}
