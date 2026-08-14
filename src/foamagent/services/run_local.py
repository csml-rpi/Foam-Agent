import os
import re
from typing import List, Any
from foamagent.models import RunIn, RunOut
from foamagent.utils import remove_files, remove_file, remove_numeric_folders, run_command, check_foam_errors

from foamagent.logger import get_logger

logger = get_logger(__name__)

def run_allrun_and_collect_errors(
    case_dir: str,
    timeout: int = 3600,
    max_retries: int = 1
) -> List[str]:
    """
    Execute the Allrun script and collect any error logs from the simulation.
    
    This function runs the Allrun script in the specified case directory,
    captures the output and error streams, and parses the results to identify
    any OpenFOAM errors that occurred during execution.
    
    Args:
        case_dir (str): Directory path containing the OpenFOAM case and Allrun script
        timeout (int, optional): Maximum execution time in seconds. Defaults to 3600.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
    
    Returns:
        List[str]: List of error messages found in the simulation logs.
                 Empty list indicates successful execution with no errors.
    
    Raises:
        FileNotFoundError: If Allrun script does not exist in case_dir
        RuntimeError: If Allrun script execution fails repeatedly
        TimeoutError: If execution exceeds timeout limit
    
    Example:
        >>> errors = run_allrun_and_collect_errors(
        ...     case_dir="/path/to/case",
        ...     timeout=1800,
        ...     max_retries=2
        ... )
        >>> if not errors:
        ...     logger.info("Simulation completed successfully")
        >>> else:
        ...     logger.info(f"Found {len(errors)} errors")
    """
    allrun_file_path = os.path.join(case_dir, "Allrun")
    if not os.path.exists(allrun_file_path):
        return [f"Allrun script not found at {allrun_file_path}"]
    
    out_file = os.path.join(case_dir, "Allrun.out")
    err_file = os.path.join(case_dir, "Allrun.err")

    # Cleanup
    remove_files(case_dir, prefix="log")
    remove_file(err_file)
    remove_file(out_file)
    remove_numeric_folders(case_dir)

    last_error_logs = []

    # Run with retries
    for attempt in range(1, max_retries + 1):
        logger.info(f"Running Allrun (attempt {attempt}/{max_retries})")
        run_command(allrun_file_path, out_file, err_file, case_dir, timeout)

        # Inspect
        error_logs = check_foam_errors(case_dir)
        if len(error_logs) == 0:
            return []

        last_error_logs = error_logs
        if attempt < max_retries:
            logger.info("Allrun reported errors; retrying after cleanup...")
            remove_files(case_dir, prefix="log")
            remove_file(err_file)
            remove_file(out_file)
            remove_numeric_folders(case_dir)

    return last_error_logs


def run_simulation_local(
    case_id: str,
    case_dir: str,
    timeout: int = 3600,
    max_retries: int = 1
) -> RunOut:
    """
    Run OpenFOAM simulation locally and return execution status.
    
    This function executes the Allrun script in the specified case directory
    and returns the execution status along with any job information.
    For local execution, job_id is always None.
    
    Args:
        case_id (str): Unique identifier for the case
        case_dir (str): Directory path containing the OpenFOAM case
        timeout (int, optional): Maximum execution time in seconds. Defaults to 3600.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
    
    Returns:
        RunOut: Contains:
            - job_id (None): Always None for local execution
            - status (str): Execution status ("completed" or "failed")
    
    Raises:
        FileNotFoundError: If case directory or Allrun script does not exist
        RuntimeError: If simulation execution fails
    
    Example:
        >>> result = run_simulation_local(
        ...     case_id="test_case",
        ...     case_dir="/path/to/case",
        ...     timeout=1800
        ... )
        >>> logger.info(f"Simulation status: {result.status}")
    """
    errors = run_allrun_and_collect_errors(case_dir, timeout, max_retries)
    status = "completed" if len(errors) == 0 else "failed"
    return RunOut(job_id=None, status=status)


