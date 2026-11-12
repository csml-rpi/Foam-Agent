from logger import print
import os
import json
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from utils import save_file, reject_symbolic_links, run_owned_process, resolve_case_file
from . import get_llm_service
from .case_paths import safe_case_path
from .execution_logs import check_execution_outputs


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
    reject_symbolic_links(foam_path)
    
    # Create or update the .foam file
    if not os.path.exists(foam_path):
        with open(foam_path, "w", encoding="utf-8"):
            pass
    else:
        # Update timestamp if file exists
        os.utime(foam_path, None)
    
    return foam


def run_pyvista_script(
    case_dir: str,
    script: str,
    *,
    filename: str = ".foamagent/visualization.py",
    expected_pngs: List[str],
    timeout_s: int = 180,
    protected_inputs=(),
    source_input_dir: str | None = None,
) -> Tuple[bool, str, List[str]]:
    """Run a generated visualization script deterministically.

    Key behaviors (to avoid flaky bugs):
      - Check every declared PNG in expected_pngs (a single image uses a one-item list).
      - Return the first image path in the existing tuple interface.
      - Apply a timeout so headless/VTK hangs don't block forever.
    """
    case_dir = os.path.abspath(case_dir)
    case_root = Path(case_dir)

    def output_path(name):
        requested_path = case_root / name
        reject_symbolic_links(requested_path)
        path = requested_path.resolve()
        try:
            path.relative_to(case_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Visualization output must remain inside the case directory: {name}") from exc
        if path.exists() and not path.is_file():
            raise ValueError(f"Visualization output is not a regular file: {name}")
        relative = path.relative_to(case_root.resolve())
        conflicts = check_execution_outputs([relative.as_posix()], protected_inputs)
        if conflicts:
            raise ValueError(conflicts[0]['error_content'])
        if source_input_dir:
            source = safe_case_path(source_input_dir, str(relative.parent), relative.name)
            if resolve_case_file(source).exists():
                raise ValueError(f"Visualization output conflicts with canonical input: {relative}")
        return str(path)

    try:
        if not expected_pngs:
            raise ValueError("Specify a nonempty list of unique PNG output paths.")
        script_path = output_path(filename)
        expected_paths = []
        for name in expected_pngs:
            if not name or Path(name).suffix.lower() != ".png":
                raise ValueError(f"Expected a PNG output path: {name}")
            expected_paths.append(output_path(name))
        all_outputs = [script_path, *expected_paths]
        if len(set(all_outputs)) != len(all_outputs):
            raise ValueError("Visualization output paths must identify unique targets.")
    except (OSError, ValueError) as exc:
        return False, "", [str(exc)]

    save_file(script_path, script)
    for path in expected_paths:
        if Path(path).exists():
            Path(path).unlink()

    try:
        run_owned_process(
            [sys.executable, '-c',
             'import os, runpy, sys; sys.path.insert(0, os.path.dirname(sys.argv[1])); '
             'runpy.run_path(sys.argv[1], run_name="__main__")', script_path],
            cwd=case_dir,
            check=True,
            timeout=timeout_s,
        )

        for expected_png_abs in expected_paths:
            try:
                output_path(expected_png_abs)
            except (OSError, ValueError) as exc:
                return False, "", [str(exc)]
            if not os.path.isfile(expected_png_abs):
                return False, "", [f"Visualization did not create the expected PNG: {expected_png_abs}"]
            from PIL import Image
            try:
                with Image.open(expected_png_abs) as image:
                    if image.format != "PNG":
                        return False, "", [f"Output is not a PNG image: {expected_png_abs}"]
                    image.verify()
            except (OSError, ValueError) as exc:
                return False, "", [f"Invalid PNG {expected_png_abs}: {exc}"]
        return True, expected_paths[0], []

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

    except OSError as e:
        return False, "", [f"Unable to run visualization script: {e}"]


class VisualizationPlot(BaseModel):
    output_png: str = Field(description="Unique PNG path relative to the case directory")
    field_name: str = Field(description="Actual field or derived quantity shown; geometry for geometry-only plots")
    time_step: str = Field(description="Requested time selection, e.g. latest or 100")
    description: str = Field(description="What this image shows and which user requirement it implements")


class VisualizationScript(BaseModel):
    script: str = Field(description="Complete executable PyVista Python code, without markdown")
    plots: List[VisualizationPlot] = Field(min_length=1, description="Every requested output image")


class VisualizationReview(BaseModel):
    meets_requirements: bool
    feedback: str = Field(description="Explain requirement coverage or concrete corrections needed")


def visualize_case(
    case_dir: str | None,
    user_requirement: str,
    *,
    max_loop: int = 2,
    llm_service: Optional[Any] = None,
    timeout_s: int = 180,
    config: Optional[Any] = None,
    protected_inputs=(),
    source_input_dir: str | None = None,
) -> Dict[str, Any]:
    """Generate requested plots and retry execution or requirement-review failures."""
    if not case_dir:
        return _visualization_failure(case_dir=None, error_message="Missing case_dir")
    case_dir = os.path.abspath(case_dir)
    if not os.path.isdir(case_dir):
        return _visualization_failure(case_dir=case_dir, error_message=f"Case directory does not exist: {case_dir}")

    foam_file = ensure_foam_file(case_dir)
    llm_client = get_llm_service(llm_service, config)
    error_logs: List[str] = []
    previous_script = ""
    previous_plots = []
    system_prompt = (
        "Generate a complete off-screen PyVista script and an output plot manifest for the user's "
        "OpenFOAM visualization requirements. Inspect actual fields and time values at runtime. "
        "Implement every requested quantity, view, slice, time and image count; use multiple PNGs "
        "when needed. If visualization details are unspecified, choose a useful view of an available "
        "field. Do not silently substitute another field when an explicitly requested field is missing: "
        "raise a descriptive error instead. Manifest fields and times must match the code. "
        "Write all declared PNGs inside the case directory; do not modify simulation inputs or results. "
        "Use off-screen rendering and screenshot(), not an interactive show(); if DISPLAY is absent, "
        "start Xvfb with pv.start_xvfb() before rendering. "
        "Return executable code without markdown. When feedback is present, repair the previous "
        "script and manifest while preserving all original requirements."
    )
    for attempt in range(1, max_loop + 1):
        print(f"LLM visualization attempt {attempt} of {max_loop}")
        prompt = json.dumps({
            "case_directory": case_dir,
            "foam_file": foam_file,
            "user_requirement": user_requirement,
            "previous_script": previous_script,
            "previous_plots": previous_plots,
            "feedback": error_logs,
        }, ensure_ascii=False)
        response = llm_client.invoke(prompt, system_prompt, pydantic_obj=VisualizationScript)
        proposal = VisualizationScript.model_validate(response)
        previous_script = proposal.script
        previous_plots = [plot.model_dump() for plot in proposal.plots]
        output_names = [plot.output_png for plot in proposal.plots]
        success, _, errors = run_pyvista_script(
            case_dir, proposal.script,
            expected_pngs=output_names, timeout_s=timeout_s,
            protected_inputs=protected_inputs, source_input_dir=source_input_dir,
        )
        if not success:
            error_logs.extend(errors)
            continue

        from PIL import Image
        artifacts = []
        for plot in proposal.plots:
            output_path = str((Path(case_dir) / plot.output_png).resolve())
            with Image.open(output_path) as image:
                artifacts.append({**plot.model_dump(), "output_path": output_path, "width": image.width, "height": image.height})
        review_response = llm_client.invoke(
            json.dumps({"user_requirement": user_requirement, "script": proposal.script, "artifacts": artifacts}, ensure_ascii=False),
            "Review whether the executed visualization code and verified PNG artifact metadata implement "
            "ALL original visualization requirements. Check quantities, time selection, slices, views, "
            "labels and requested image count. A created PNG alone is not sufficient. Check the actual "
            "code, not just its declared manifest; reject omitted requirements or silent substitutions. "
            "You have text evidence only, not image pixels: do not claim visual inspection. "
            "Return meets_requirements=false with actionable feedback when requirements are not implemented.",
            pydantic_obj=VisualizationReview,
        )
        review = VisualizationReview.model_validate(review_response)
        if not review.meets_requirements:
            error_logs.append(f"Visualization does not meet requirements: {review.feedback}")
            continue
        result = _visualization_success(
            case_dir=case_dir, plots=artifacts, script=proposal.script,
            strategy="llm_script" if attempt == 1 else "llm_fixed_script",
        )
        result["pyvista_visualization"]["requirement_review"] = review.model_dump()
        result["pyvista_visualization"]["review_basis"] = "script_and_artifact_metadata"
        return result

    return _visualization_failure(
        case_dir=case_dir,
        error_message=f"Visualization failed execution or requirement review after {max_loop} attempts",
        error_logs=error_logs,
    )


def _visualization_success(
    *,
    case_dir: str,
    plots: List[Dict[str, Any]],
    script: str,
    strategy: str,
) -> Dict[str, Any]:
    """Return all accepted outputs while retaining the single-image compatibility field."""
    outputs = [plot["output_path"] for plot in plots]
    return {
        "plot_configs": [
            {"plot_type": "pyvista", "field_name": plot["field_name"],
             "time_step": plot["time_step"], "description": plot["description"],
             "output_format": "png", "output_path": plot["output_path"]}
            for plot in plots
        ],
        "plot_outputs": outputs,
        "visualization_summary": {
            "total_plots_generated": len(outputs),
            "plot_types": ["pyvista"],
            "fields_visualized": list(dict.fromkeys(plot["field_name"] for plot in plots)),
            "output_directory": case_dir,
            "pyvista_success": True,
            "used": strategy,
        },
        "pyvista_visualization": {
            "success": True, "output_image": outputs[0], "output_images": outputs,
            "script": script, "used": strategy,
        },
    }


def _visualization_failure(
    *,
    case_dir: str | None,
    error_message: str,
    error_logs: List[str] | None = None,
) -> Dict[str, Any]:
    """Build the stable workflow payload for a failed visualization."""
    logs = error_logs or []
    summary: Dict[str, Any] = {
        "total_plots_generated": 0,
        "plot_types": [],
        "fields_visualized": [],
        "output_directory": case_dir,
        "pyvista_success": False,
        "error": error_message,
    }
    visualization: Dict[str, Any] = {"success": False, "error": error_message}
    if logs:
        summary["error_logs"] = logs
        visualization["error_logs"] = logs
    return {
        "plot_configs": [],
        "plot_outputs": [],
        "visualization_summary": summary,
        "pyvista_visualization": visualization,
    }
