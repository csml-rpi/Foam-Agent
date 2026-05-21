import os
import sys
import subprocess
from typing import List, Tuple, Optional
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


def run_pyvista_script(
    case_dir: str,
    script: str,
    *,
    filename: str = "visualization.py",
    expected_png: Optional[str] = None,
    timeout_s: int = 180,
) -> Tuple[bool, str, List[str]]:
    """Run a generated visualization script deterministically.

    Key behaviors (to avoid flaky bugs):
      - If expected_png is provided, we only consider success if that file exists after execution.
      - Apply a timeout so headless/VTK hangs don't block forever.
    """
    case_dir = os.path.abspath(case_dir)
    script_path = os.path.join(case_dir, filename)
    save_file(script_path, script)

    expected_png_abs = os.path.abspath(os.path.join(case_dir, expected_png)) if expected_png else None

    try:
        completed = subprocess.run(
            [sys.executable, script_path],
            cwd=case_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )

        if expected_png_abs:
            if os.path.exists(expected_png_abs) and os.path.getsize(expected_png_abs) > 0:
                return True, expected_png_abs, []
            return False, "", [
                "Visualization script executed but expected PNG was not created",
                f"expected_png={expected_png_abs}",
            ]

        # Backward-compatible behavior (non-deterministic): no expected output specified.
        return False, "", [
            "Visualization script executed but no expected_png was specified; please pass expected_png for deterministic artifact detection"
        ]

    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout)
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
        return False, "", [
            f"PyVista script timed out after {timeout_s}s",
            f"STDOUT:\n{out}",
            f"STDERR:\n{err}",
        ]

    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout)
        error_msg = (
            f"PyVista script execution failed (exit code {e.returncode})\n"
            f"STDOUT:\n{out}\n"
            f"STDERR:\n{err}"
        )
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


def generate_deterministic_pyvista_script(
    *,
    foam_file: str,
    output_png: str,
    field_preference: str = "U",
) -> str:
    """Generate a minimal, deterministic PyVista script.

    Goals:
      - Works in headless environments (off-screen)
      - Always writes to output_png (relative to case_dir)
      - Tries to color by field_preference, but falls back to any available scalar
    """
    # Note: keep this as a plain string (no f-strings with user-provided code).
    return f"""import os
import sys

# Force headless rendering early
os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')

import pyvista as pv

try:
    pv.OFF_SCREEN = True
except Exception:
    pass

try:
    pv.start_xvfb()
except Exception:
    # start_xvfb is optional and may be unavailable
    pass

foam_path = os.path.abspath({foam_file!r})
out_png = os.path.abspath({output_png!r})

reader = pv.OpenFOAMReader(foam_path)
# Many OpenFOAM readers expose available times; use the last one when present
try:
    reader.set_active_time_value(reader.time_values[-1])
except Exception:
    pass

data = reader.read()

# data can be a MultiBlock; merge to a single mesh for robust plotting
mesh = data
try:
    if hasattr(data, 'combine'):
        mesh = data.combine()
except Exception:
    # fallback: try first block
    try:
        mesh = data[0]
    except Exception:
        mesh = data

# Determine a scalar to plot
scalar_name = None
preferred = {field_preference!r}

# Try point data first then cell data
try:
    if preferred in getattr(mesh, 'point_data', {{}}):
        scalar_name = preferred
    elif preferred in getattr(mesh, 'cell_data', {{}}):
        scalar_name = preferred
except Exception:
    pass

if scalar_name is None:
    # pick any available scalar
    try:
        keys = list(getattr(mesh, 'point_data', {{}}).keys())
        scalar_name = keys[0] if keys else None
    except Exception:
        scalar_name = None

if scalar_name is None:
    try:
        keys = list(getattr(mesh, 'cell_data', {{}}).keys())
        scalar_name = keys[0] if keys else None
    except Exception:
        scalar_name = None

plotter = pv.Plotter(off_screen=True)
plotter.set_background('white')

if scalar_name is not None:
    plotter.add_mesh(mesh, scalars=scalar_name, cmap='coolwarm', show_scalar_bar=True)
else:
    plotter.add_mesh(mesh, color='lightgray')

plotter.view_isometric()
plotter.show(auto_close=False)
plotter.screenshot(out_png)
plotter.close()

print('Wrote', out_png)
"""


