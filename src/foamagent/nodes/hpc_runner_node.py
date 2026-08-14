# hpc_runner_node.py
import os
from foamagent.utils import (
    remove_files, remove_file, remove_numeric_folders
)
from foamagent.services.run_hpc import (
    extract_cluster_info_from_requirement,
    run_simulation_hpc,
    wait_for_job,
    check_logs_for_errors,
    create_slurm_script_with_error_context,
    create_slurm_script,
)
from foamagent.logger import log_review

from foamagent.logger import get_logger

logger = get_logger(__name__)


def hpc_runner_node(state):
    """
    HPC Runner node: Extract cluster info from user requirement, create SLURM script,
    submit job to cluster, wait for completion, and check for errors.
    Retries submission on failure up to max_loop times, regenerating script based on errors.
    """
    config = state["config"]
    case_dir = state["case_dir"]
    allrun_file_path = os.path.join(case_dir, "Allrun")
    max_loop = config.max_loop
    current_attempt = 0
    
    logger.info("<hpc_runner>")

    # Clean up any previous log and error files.
    out_file = os.path.join(case_dir, "Allrun.out")
    err_file = os.path.join(case_dir, "Allrun.err")
    remove_files(case_dir, prefix="log")
    remove_file(err_file)
    remove_file(out_file)
    remove_numeric_folders(case_dir)
    
    # Extract cluster information using service
    logger.info("Extracting cluster information from user requirement...")
    cluster_info = extract_cluster_info_from_requirement(state["user_requirement"], case_dir)
    logger.info(f"<cluster_info>{cluster_info}</cluster_info>")
    
    # Submit the job with retry logic
    last_error_msg = ""
    while current_attempt < max_loop:
        current_attempt += 1
        logger.info(f"Attempt {current_attempt}/{max_loop}: Creating and submitting SLURM job...")
        
        # Create SLURM script
        if current_attempt == 1:
            logger.info("Creating initial SLURM script...")
            script_path = create_slurm_script(case_dir, cluster_info)
        else:
            logger.info(f"Regenerating SLURM script based on previous error...")
            try:
                with open(script_path, 'r') as f:
                    prev = f.read()
            except Exception:
                prev = ""
            # Use service helper for regeneration
            script_path = create_slurm_script_with_error_context(case_dir, cluster_info, last_error_msg, prev)
        
        logger.info(f"SLURM script created at: {script_path}")
        
        # Submit via service
        run_out = run_simulation_hpc(script_path)
        job_id = run_out.job_id
        success = run_out.status == "submitted"
        error_msg = "" if success else run_out.status
        
        if success:
            logger.info(f"Job submitted successfully with ID: {job_id}")
            break
        else:
            logger.info(f"Attempt {current_attempt} failed: {error_msg}")
            last_error_msg = error_msg  # Store error for next iteration
            if current_attempt < max_loop:
                logger.info(f"Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                logger.info(f"Maximum attempts ({max_loop}) reached. Job submission failed.")
                error_logs = [f"Job submission failed after {max_loop} attempts. Last error: {error_msg}"]
                log_review(str(error_logs), "error_logs")
                logger.info("</hpc_runner>")
                return {
                    **state,
                    "error_logs": error_logs,
                    "job_id": None,
                    "cluster_info": cluster_info,
                    "slurm_script_path": script_path
                }
    
    # Wait for job completion via service
    logger.info("Waiting for job completion...")
    status, status_success, status_error = wait_for_job(job_id)
    if not status_success:
        error_logs = [f"Status check failed: {status_error}"]
        log_review(str(error_logs), "error_logs")
        logger.info("</hpc_runner>")
        return {
            **state,
            "error_logs": error_logs,
            "job_id": job_id,
            "cluster_info": cluster_info,
            "slurm_script_path": script_path
        }
    logger.info(f"<job_status>{status}</job_status>")

    if status != "COMPLETED":
        error_logs = [f"HPC job finished with non-success status: {status}"]
        log_review(str(error_logs), "error_logs")
        logger.info("</hpc_runner>")
        return {
            **state,
            "error_logs": error_logs,
            "job_id": job_id,
            "cluster_info": cluster_info,
            "slurm_script_path": script_path
        }

    # Check for errors in log files (similar to local_runner)
    logger.info("Checking for errors in log files...")
    error_logs = check_logs_for_errors(case_dir)
    
    if len(error_logs) > 0:
        logger.error("Errors detected in the HPC Allrun execution.")
        log_review(str(error_logs), "error_logs")
    else:
        logger.info("HPC Allrun executed successfully without errors.")

    logger.info("</hpc_runner>")

    # Return updated state
    return {
        **state,
        "error_logs": error_logs,
        "job_id": job_id,
        "cluster_info": cluster_info,
        "slurm_script_path": script_path
    }
