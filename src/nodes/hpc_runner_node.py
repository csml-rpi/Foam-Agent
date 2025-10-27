# hpc_runner_node.py
import os
from utils import (
    remove_files, remove_file, remove_numeric_folders
)
from services.run_hpc import (
    extract_cluster_info_from_requirement,
    run_simulation_hpc,
    wait_for_job,
    check_logs_for_errors,
    create_slurm_script_with_error_context,
    create_slurm_script,
    submit_slurm_job,
    check_job_status,
)


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
    
    print(f"============================== HPC Runner ==============================")
    
    # Clean up any previous log and error files.
    out_file = os.path.join(case_dir, "Allrun.out")
    err_file = os.path.join(case_dir, "Allrun.err")
    remove_files(case_dir, prefix="log")
    remove_file(err_file)
    remove_file(out_file)
    remove_numeric_folders(case_dir)
    
    # Extract cluster information using service
    print("Extracting cluster information from user requirement...")
    cluster_info = extract_cluster_info_from_requirement(state["user_requirement"], case_dir)
    print(f"Cluster info extracted: {cluster_info}")
    
    # Submit the job with retry logic
    while current_attempt < max_loop:
        current_attempt += 1
        print(f"Attempt {current_attempt}/{max_loop}: Creating and submitting SLURM job...")
        
        # Create SLURM script
        if current_attempt == 1:
            print("Creating initial SLURM script...")
            script_path = create_slurm_script(case_dir, cluster_info)
        else:
            print(f"Regenerating SLURM script based on previous error...")
            try:
                with open(script_path, 'r') as f:
                    prev = f.read()
            except Exception:
                prev = ""
            # Use service helper for regeneration
            script_path = create_slurm_script_with_error_context(case_dir, cluster_info, last_error_msg, prev)
        
        print(f"SLURM script created at: {script_path}")
        
        # Submit via service
        run_out = run_simulation_hpc(script_path)
        job_id = run_out.job_id
        success = run_out.status == "submitted"
        error_msg = "" if success else run_out.status
        
        if success:
            print(f"Job submitted successfully with ID: {job_id}")
            break
        else:
            print(f"Attempt {current_attempt} failed: {error_msg}")
            last_error_msg = error_msg  # Store error for next iteration
            if current_attempt < max_loop:
                print(f"Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                print(f"Maximum attempts ({max_loop}) reached. Job submission failed.")
                error_logs = [f"Job submission failed after {max_loop} attempts. Last error: {error_msg}"]
                return {
                    **state,
                    "error_logs": error_logs,
                    "job_id": None,
                    "cluster_info": cluster_info,
                    "slurm_script_path": script_path
                }
    
    # Wait for job completion via service
    print("Waiting for job completion...")
    status, status_success, status_error = wait_for_job(job_id)
    if not status_success:
        error_logs = [f"Status check failed: {status_error}"]
        return {
            **state,
            "error_logs": error_logs,
            "job_id": job_id,
            "cluster_info": cluster_info,
            "slurm_script_path": script_path
        }
    print(f"Job finished with status: {status}")
    
    # Check for errors in log files (similar to local_runner)
    print("Checking for errors in log files...")
    error_logs = check_logs_for_errors(case_dir)
    
    if len(error_logs) > 0:
        print("Errors detected in the HPC Allrun execution.")
        print(error_logs)
    else:
        print("HPC Allrun executed successfully without errors.")
    
    state['loop_count'] += 1
    
    # Return updated state
    return {
        **state,
        "error_logs": error_logs,
        "job_id": job_id,
        "cluster_info": cluster_info,
        "slurm_script_path": script_path
    }
