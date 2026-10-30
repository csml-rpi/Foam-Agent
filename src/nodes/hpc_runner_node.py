"""Run solver-only Allrun in a separately verified Slurm attempt."""
from pathlib import Path
from services.run_hpc import HpcWorkflowStop, run_stage_hpc, check_logs_for_errors
from services.run_local import _cleanup_run_artifacts, _execution_output


def hpc_runner_node(state):
    current = dict(state)
    case = current["case_dir"]
    _cleanup_run_artifacts(case)
    try:
        result = run_stage_hpc(current, str(Path(case) / "Allrun"), "solver", current["config"].max_time_limit)
    except HpcWorkflowStop as stop:
        return {**stop.state_update(), "solver_jobs": current.get("solver_jobs", []),
                "loop_count": current.get("loop_count", 0), "cluster_info": current.get("cluster_info")}
    except Exception as error:
        return {"error_logs": [{"file": "Allrun", "error_content": str(error)}],
                "solver_jobs": current.get("solver_jobs", []), "loop_count": current.get("loop_count", 0)}
    errors = check_logs_for_errors(case)
    if result.get("returncode"):
        # Attempt paths and job IDs remain in solver_jobs; the repair input is
        # the actual failure, so identical retries retain the same fingerprint.
        details = (
            f"HPC Allrun exited with return code {result['returncode']}; "
            f"Slurm state: {result.get('status')}.\n"
            f"{result.get('diagnostic') or ''}\n{_execution_output(case)}"
        )
        errors.append({"file": "Allrun", "error_content": details.strip()})
    return {"error_logs": errors, "solver_jobs": current.get("solver_jobs", []),
            "loop_count": current.get("loop_count", 0),
            "job_id": result.get("job_id"), "cluster_info": current.get("cluster_info"),
            "slurm_script_path": result.get("script"),
            "workflow_status": "success" if not errors else "running"}
