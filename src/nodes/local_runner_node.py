# runner_node.py
from typing import List
import os
from pydantic import BaseModel, Field
import re

from services.run_local import run_allrun_and_collect_errors


def local_runner_node(state):
    """
    Runner node: Execute an Allrun script, and check for errors.
    On error, update state.error_command and state.error_content.
    """
    config = state["config"]
    case_dir = state["case_dir"]
    max_time_limit = state["config"].max_time_limit
    
    print(f"============================== Runner ==============================")
    
    # Execute using service and collect errors
    error_logs = run_allrun_and_collect_errors(case_dir, max_time_limit)

    if len(error_logs) > 0:
        print("Errors detected in the Allrun execution.")
        print(error_logs)
    else:
        print("Allrun executed successfully without errors.")
    
    state['loop_count'] += 1
    # Return updated state
    return {
        **state,
        "error_logs": error_logs
    }
        