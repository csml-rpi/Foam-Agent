import os
import sys
import subprocess
import glob
from typing import Dict, List, Tuple
from utils import save_file
from . import global_llm_service


def ensure_foam_file(case_dir: str) -> str:
    """
    Ensure a .foam file exists in the case directory for OpenFOAM visualization.
    
    This function creates or updates a .foam file in the specified case directory.
    The .foam file is required for OpenFOAM visualization tools to recognize
    the directory as a valid OpenFOAM case.
    
    Args:
        case_dir (str): Directory path containing the OpenFOAM case
    
    Returns:
        str: Name of the .foam file (typically "{case_name}.foam")
    
    Raises:
        OSError: If directory cannot be accessed or file cannot be created
    
    Example:
        >>> foam_name = ensure_foam_file("/path/to/case")
        >>> print(f"Foam file: {foam_name}")  # "case.foam"
    """
    case_dir = os.path.abspath(case_dir)
    foam = f"{os.path.basename(case_dir)}.foam"
    foam_path = os.path.join(case_dir, foam)
    
    # Create or update the .foam file
    if not os.path.exists(foam_path):
        with open(foam_path, 'w') as f:
            pass
    else:
        # Update timestamp if file exists
        os.utime(foam_path, None)
    
    return foam


def generate_pyvista_script(
    case_dir: str,
    foam_file: str,
    user_requirement: str,
    previous_errors: List[str]
) -> str:
    """
    Generate PyVista visualization script for OpenFOAM case using LLM.
    
    This function uses LLM to generate a Python script that uses PyVista
    to visualize OpenFOAM simulation results. The script loads the .foam file,
    renders geometry with appropriate coloring, and saves visualization images.
    
    Args:
        case_dir (str): Directory path containing the OpenFOAM case
        foam_file (str): Name of the .foam file for the case
        user_requirement (str): User requirements for visualization context
        previous_errors (List[str]): List of previous visualization errors for context
    
    Returns:
        str: Generated Python script code for PyVista visualization
    
    Raises:
        RuntimeError: If LLM service fails to generate script
    
    Example:
        >>> script = generate_pyvista_script(
        ...     case_dir="/path/to/case",
        ...     foam_file="case.foam",
        ...     user_requirement="Visualize velocity field",
        ...     previous_errors=[]
        ... )
        >>> print("Generated PyVista script")
    """
    system_prompt = (
        "You are an expert in OpenFOAM post-processing and PyVista Python scripting. "
        "Generate a PyVista script that loads the .foam file, renders geometry colored by requested field, uses coolwarm colormap, and saves a PNG. "
        "Return ONLY Python code, no markdown."
    )
    prompt = (
        f"<case_directory>{case_dir}</case_directory>\n"
        f"<foam_file>{foam_file}</foam_file>\n"
        f"<visualization_requirements>{user_requirement}</visualization_requirements>\n"
        f"<previous_errors>{previous_errors}</previous_errors>\n"
    )
    return global_llm_service.invoke(prompt, system_prompt)


def run_pyvista_script(case_dir: str, script: str, filename: str = "visualization.py") -> Tuple[bool, str, List[str]]:
    script_path = os.path.join(case_dir, filename)
    save_file(script_path, script)
    try:
        result = subprocess.run([sys.executable, script_path], cwd=case_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        png_files = glob.glob(os.path.join(case_dir, "*.png"))
        if png_files:
            return True, png_files[0], []
        return False, "", ["Visualization script executed but no PNG output image was created"]
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout)
        error_msg = f"PyVista script execution failed (exit code {e.returncode})\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        return False, "", [error_msg]
    except FileNotFoundError:
        return False, "", [f"Python interpreter not found: {sys.executable}"]
    except Exception as e:
        return False, "", [f"Unexpected error running visualization script: {str(e)}"]


def fix_pyvista_script(foam_file: str, original_script: str, error_logs: List[str]) -> str:
    system_prompt = (
        "You are an expert in PyVista visualization. Fix the provided script to load the .foam file, render geometry, and save a PNG with colorbar. Return ONLY Python code."
    )
    prompt = (
        f"<error_logs>{error_logs}</error_logs>\n"
        f"<foam_file>{foam_file}</foam_file>\n"
        f"<original_script>{original_script}</original_script>\n"
    )
    return global_llm_service.invoke(prompt, system_prompt)


