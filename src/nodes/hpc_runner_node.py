# hpc_runner_node.py
from typing import List
import os
import subprocess
import json
from pydantic import BaseModel, Field
import re
from utils import (
    save_file, remove_files, remove_file,
    run_command, check_foam_errors, retrieve_faiss, remove_numeric_folders
)
from services.run_hpc import (
    extract_cluster_info_from_requirement,
    generate_hpc_script,
    run_simulation_hpc,
    wait_for_job,
    check_logs_for_errors,
    create_slurm_script_with_error_context,
)


def extract_cluster_info(state) -> dict:
    """
    Extract cluster information from user requirement using LLM.
    
    Args:
        state: Current graph state containing user requirement and LLM service
        
    Returns:
        dict: Dictionary containing cluster_name, account_number, and other cluster details
    """
    user_requirement = state["user_requirement"]
    case_dir = state["case_dir"]
    
    # Check if decomposeParDict exists and read its content
    decompose_par_dict_content = ""
    decompose_par_dict_path = os.path.join(case_dir, "system", "decomposeParDict")
    if os.path.exists(decompose_par_dict_path):
        try:
            with open(decompose_par_dict_path, 'r') as f:
                decompose_par_dict_content = f.read()
        except Exception as e:
            print(f"Warning: Could not read decomposeParDict: {e}")
    
    system_prompt = (
        "You are an expert in HPC cluster analysis. "
        "Analyze the user requirement to extract cluster information. "
        "Look for keywords like: cluster name, account number, partition, queue, "
        "specific cluster names (e.g., Stampede2, Frontera, Summit, etc.), "
        "account numbers, project codes, or any mention of specific HPC systems. "
        ""
        "IMPORTANT: If a decomposeParDict file is provided, analyze it to determine "
        "the appropriate number of tasks per node (ntasks_per_node) based on the "
        "decomposition settings. The number of tasks should match the total number "
        "of subdomains or processes specified in the decomposeParDict."
        ""
        "Return a JSON object with the following structure: "
        "{"
        "  'cluster_name': 'name of the cluster or HPC system', "
        "  'account_number': 'account number or project code', "
        "  'partition': 'partition name (e.g., normal, debug, gpu)', "
        "  'nodes': 'number of nodes (default: 1)', "
        "  'ntasks_per_node': 'number of tasks per node (determine from decomposeParDict if available)', "
        "  'time_limit': 'time limit in hours (default: 24)', "
        "  'memory': 'memory per node in GB (default: 64)'"
        "}"
        "If any information is not specified, use reasonable defaults based on your expertise. "
        "Only return valid JSON. Don't include any other text."
    )
    
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
    )
    
    if decompose_par_dict_content:
        user_prompt += (
            f"decomposeParDict content:\n{decompose_par_dict_content}\n\n"
            "Analyze the decomposeParDict to determine the appropriate number of tasks per node "
            "based on the decomposition settings. "
        )
    
    user_prompt += "Extract cluster information and return as JSON object."
    
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    
    # Try to parse the JSON response
    try:
        # Clean up the response to extract JSON
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        cluster_info = json.loads(response)
        
        # Set defaults for missing values
        defaults = {
            'cluster_name': 'default_cluster',
            'account_number': 'default_account',
            'partition': 'normal',
            'nodes': 1,
            'ntasks_per_node': 1,
            'time_limit': 24,
            'memory': 64
        }
        
        for key, default_value in defaults.items():
            if key not in cluster_info or cluster_info[key] is None:
                cluster_info[key] = default_value
                
        return cluster_info
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing cluster info from LLM response: {e}")
        print(f"LLM response: {response}")
        # Return default values if parsing fails
        return {
            'cluster_name': 'default_cluster',
            'account_number': 'default_account',
            'partition': 'normal',
            'nodes': 1,
            'ntasks_per_node': 1,
            'time_limit': 24,
            'memory': 64
        }


def create_slurm_script(case_dir: str, cluster_info: dict, state) -> str:
    """
    Create a SLURM script for OpenFOAM simulation using LLM.
    
    Args:
        case_dir: Directory containing the OpenFOAM case
        cluster_info: Dictionary containing cluster configuration
        state: Current graph state containing LLM service
        
    Returns:
        str: Path to the created SLURM script
    """
    system_prompt = (
        "You are an expert in HPC cluster job submission and SLURM scripting. "
        "Create a complete SLURM script for running OpenFOAM simulations. "
        "The script should include:"
        "1. Proper SLURM directives (#SBATCH) based on the cluster information provided"
        "2. Do not load openfoam"
        "3. Load libaraies for openfoam for run in parallel"
        "4. Directory navigation and execution of the Allrun script"
        "5. Error handling and status reporting"
        "6. Any cluster-specific optimizations or requirements"
        "7. Use your understanding of the documentation of the cluster and figure out the syntax of their jobscript."
        ""
        "Return ONLY the complete SLURM script content. Do not include any explanations or markdown formatting."
        "Make sure the script is executable and follows best practices for the specified cluster."
    )
    
    user_prompt = (
        f"Create a SLURM script for OpenFOAM simulation with the following parameters:\n"
        f"Cluster: {cluster_info['cluster_name']}\n"
        f"Account: {cluster_info['account_number']}\n"
        f"Partition: {cluster_info['partition']}\n"
        f"Nodes: {cluster_info['nodes']}\n"
        f"Tasks per node: {cluster_info['ntasks_per_node']}\n"
        f"Time limit: {cluster_info['time_limit']} hours\n"
        f"Memory: {cluster_info['memory']} GB per node\n"
        f"Case directory: {case_dir}\n"
        f""
        f"Generate a complete SLURM script that will run the OpenFOAM simulation using the Allrun script."
    )
    
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    
    # Clean up the response to extract just the script content
    script_content = response.strip()
    if script_content.startswith('```bash'):
        script_content = script_content[7:]
    elif script_content.startswith('```'):
        script_content = script_content[3:]
    if script_content.endswith('```'):
        script_content = script_content[:-3]
    script_content = script_content.strip()
    
    # Ensure the script starts with shebang
    if not script_content.startswith('#!/bin/bash'):
        script_content = '#!/bin/bash\n' + script_content
    
    script_path = os.path.join(case_dir, "submit_job.slurm")
    save_file(script_path, script_content)
    return script_path


def create_slurm_script_with_error_context(case_dir: str, cluster_info: dict, state, error_message: str = "", previous_script_content: str = "") -> str:
    """
    Create a SLURM script for OpenFOAM simulation using LLM, with error context for retries.
    
    Args:
        case_dir: Directory containing the OpenFOAM case
        cluster_info: Dictionary containing cluster configuration
        state: Current graph state containing LLM service
        error_message: Error message from previous submission attempt
        previous_script_content: Content of the previous failed SLURM script
        
    Returns:
        str: Path to the created SLURM script
    """
    system_prompt = (
        "You are an expert in HPC cluster job submission and SLURM scripting. "
        "Create a complete SLURM script for running OpenFOAM simulations. "
        "The script should include:"
        "1. Proper SLURM directives (#SBATCH) based on the cluster information provided"
        "2. Do not load OpenFOAM"
        "3. Load libaraies for openfoam for run in parallel"
        "4. Directory navigation and execution of the Allrun script"
        "5. Error handling and status reporting"
        "6. Any cluster-specific optimizations or requirements"
        "7. Use your understanding of the documentation of the cluster and figure out the syntax of their jobscript."
        ""
        "If a previous script and error message are provided, analyze the error and the script "
        "to identify what went wrong and fix it. Common issues to consider:"
        "- Invalid account numbers or partitions"
        "- Insufficient resources (memory, time, nodes)"
        "- Missing modules or environment variables"
        "- Incorrect file paths or permissions"
        "- Cluster-specific requirements or restrictions"
        "- Syntax errors in SLURM directives"
        "- Incorrect module names or versions"
        ""
        "Compare the previous script with the error message to identify the specific issue "
        "and create a corrected version."
        ""
        "Return ONLY the complete SLURM script content. Do not include any explanations or markdown formatting."
        "Make sure the script is executable and follows best practices for the specified cluster."
    )
    
    user_prompt = (
        f"Create a SLURM script for OpenFOAM simulation with the following parameters:\n"
        f"Cluster: {cluster_info['cluster_name']}\n"
        f"Account: {cluster_info['account_number']}\n"
        f"Partition: {cluster_info['partition']}\n"
        f"Nodes: {cluster_info['nodes']}\n"
        f"Tasks per node: {cluster_info['ntasks_per_node']}\n"
        f"Time limit: {cluster_info['time_limit']} hours\n"
        f"Memory: {cluster_info['memory']} GB per node\n"
        f"Case directory: {case_dir}\n"
    )
    
    if error_message and previous_script_content:
        user_prompt += f"\nPrevious submission failed with error: {error_message}\n"
        user_prompt += f"Previous SLURM script that failed:\n```bash\n{previous_script_content}\n```\n"
        user_prompt += "Please analyze this error and the previous script to identify the issue and create a corrected version."
    
    user_prompt += f"\nGenerate a complete SLURM script that will run the OpenFOAM simulation using the Allrun script. Return ONLY the complete SLURM script content. Do not include any explanations or markdown formatting."
    
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    
    # Clean up the response to extract just the script content
    script_content = response.strip()
    if script_content.startswith('```bash'):
        script_content = script_content[7:]
    elif script_content.startswith('```'):
        script_content = script_content[3:]
    if script_content.endswith('```'):
        script_content = script_content[:-3]
    script_content = script_content.strip()
    
    # Ensure the script starts with shebang
    if not script_content.startswith('#!/bin/bash'):
        script_content = '#!/bin/bash\n' + script_content
    
    script_path = os.path.join(case_dir, "submit_job.slurm")
    save_file(script_path, script_content)
    return script_path


def submit_slurm_job(script_path: str) -> tuple:
    """
    Submit a SLURM job and return job ID.
    
    Args:
        script_path: Path to the SLURM script
        
    Returns:
        tuple: (job_id, success, error_message)
    """
    try:
        # Submit the job
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract job ID from output
        output = result.stdout.strip()
        job_id_match = re.search(r'Submitted batch job (\d+)', output)
        
        if job_id_match:
            job_id = job_id_match.group(1)
            return job_id, True, ""
        else:
            return None, False, f"Could not extract job ID from output: {output}"
            
    except subprocess.CalledProcessError as e:
        return None, False, f"Failed to submit job: {e.stderr}"
    except Exception as e:
        return None, False, f"Unexpected error: {str(e)}"


def check_job_status(job_id: str) -> tuple:
    """
    Check the status of a SLURM job.
    
    Args:
        job_id: SLURM job ID
        
    Returns:
        tuple: (status, success, error_message)
    """
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "-o", "%T"],
            capture_output=True,
            text=True,
            check=True
        )
        
        status = result.stdout.strip()
        if status:
            return status, True, ""
        else:
            return "COMPLETED", True, ""  # Job not in queue, likely completed
            
    except subprocess.CalledProcessError as e:
        return None, False, f"Failed to check job status: {e.stderr}"
    except Exception as e:
        return None, False, f"Unexpected error: {str(e)}"


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
    cluster_info = extract_cluster_info_from_requirement(state["user_requirement"], case_dir, state["llm_service"])
    print(f"Cluster info extracted: {cluster_info}")
    
    # Submit the job with retry logic
    while current_attempt < max_loop:
        current_attempt += 1
        print(f"Attempt {current_attempt}/{max_loop}: Creating and submitting SLURM job...")
        
        # Create SLURM script
        if current_attempt == 1:
            print("Creating initial SLURM script...")
            script_path = generate_hpc_script({"case_id": "n/a", "hpc_config": cluster_info}, state["llm_service"], case_dir).script_path
        else:
            print(f"Regenerating SLURM script based on previous error...")
            try:
                with open(script_path, 'r') as f:
                    prev = f.read()
            except Exception:
                prev = ""
            # Use service helper for regeneration
            script_path = create_slurm_script_with_error_context(case_dir, cluster_info, state["llm_service"], last_error_msg, prev)
        
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
