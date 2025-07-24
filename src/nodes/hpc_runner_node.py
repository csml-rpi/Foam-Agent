# hpc_runner_node.py
from typing import List
import os
from pydantic import BaseModel, Field
import re
from utils import (
    save_file, remove_files, remove_file,
    run_command, check_foam_errors, retrieve_faiss, remove_numeric_folders
)


def hpc_runner_node(state):
    """
    HPC Runner node: Execute an Allrun script on HPC cluster, and check for errors.
    On error, update state.error_command and state.error_content.
    """
    config = state["config"]
    case_dir = state["case_dir"]
    allrun_file_path = os.path.join(case_dir, "Allrun")
    
    print(f"============================== HPC Runner ==============================")
    
    # Clean up any previous log and error files.
    out_file = os.path.join(case_dir, "Allrun.out")
    err_file = os.path.join(case_dir, "Allrun.err")
    remove_files(case_dir, prefix="log")
    remove_file(err_file)
    remove_file(out_file)
    remove_numeric_folders(case_dir)
    
    # Execute the Allrun script on HPC cluster.
    # TODO: Implement HPC-specific execution logic
    pass
    
    # Check for errors.
    # TODO: Implement HPC-specific error checking
    pass
    
    # Mock error logs for testing
    error_logs = []
    
    if len(error_logs) > 0:
        print("Errors detected in the HPC Allrun execution.")
        print(error_logs)
    else:
        print("HPC Allrun executed successfully without errors.")
    
    # Return updated state
    return {
        **state,
        "error_logs": error_logs
    }
