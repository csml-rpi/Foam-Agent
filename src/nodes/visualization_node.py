# visualization_node.py
import os
import subprocess
import sys
from typing import List, Optional
from pydantic import BaseModel, Field
from utils import save_file

# Helper to get the .foam file name
def get_foam_file(case_dir):
    case_dir_name = os.path.basename(os.path.normpath(case_dir))
    return f"{case_dir_name}.foam"

VISUALIZATION_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM post-processing and PyVista Python scripting. "
    "Your task is to generate a PyVista Python script that visualizes the specified data from the OpenFOAM case. "
    "The script should load the OpenFOAM case data by reading the .foam file (e.g., 'runs.foam') in the case directory using PyVista, display the geometry, and color the surface by the specified field (e.g., 'U' for velocity). "
    "Ensure the script shows the geometry, sets up the colorbar, and saves the visualization as a PNG file. "
    "Use coolwarm colormap."
    "The script must save the visualization as a PNG file, and the output image must contain the geometry and the colorbar, not just the colorbar. "
    "IMPORTANT: Return ONLY the Python code without any markdown formatting, code block markers, or explanatory text. "
    "The script should start with the necessary imports, read the .foam file using PyVista, and end with the screenshot saving."
)

ERROR_FIX_SYSTEM_PROMPT = (
    "You are an expert in PyVista Python scripting and OpenFOAM visualization. "
    "Your task is to fix the provided PyVista Python script that encountered an error. "
    "Ensure the script loads the OpenFOAM case data by reading the .foam file (e.g., 'runs.foam') in the case directory using PyVista, displays the geometry, and colors the surface by the specified field. "
    "Make sure the script shows the geometry, sets up the colorbar, and saves the visualization as a PNG file. "
    "Use coolwarm colormap."
    "The script must save the visualization as a PNG file, and the output image must contain the geometry and the colorbar, not just the colorbar. "
    "IMPORTANT: Return ONLY the Python code without any markdown formatting, code block markers, or explanatory text. "
    "The script should start with the necessary imports, read the .foam file using PyVista, and end with the screenshot saving."
)

class PlotConfigPydantic(BaseModel):
    """Configuration for plotting parameters"""
    plot_type: str = Field(description="Type of plot (e.g., 'contour', 'vector', 'streamline', 'time_series')")
    field_name: str = Field(description="Field to plot (e.g., 'U', 'p', 'T', 'rho')")
    time_step: Optional[str] = Field(default=None, description="Time step to plot (if None, use latest)")
    output_format: str = Field(default="png", description="Output format for plots")
    output_path: str = Field(description="Path to save the plot")

class VisualizationPlanPydantic(BaseModel):
    """Plan for visualization tasks"""
    plots: List[PlotConfigPydantic] = Field(description="List of plots to generate")

class VisualizationAnalysisPydantic(BaseModel):
    """Analysis of user requirements for visualization needs"""
    primary_field: str = Field(description="Primary field to visualize (e.g., 'U', 'p', 'T', 'rho')")
    plot_type: str = Field(description="Type of plot requested (e.g., 'contour', 'vector', 'streamline', 'time_series')")
    time_step: Optional[str] = Field(default=None, description="Specific time step to plot (if mentioned)")
    plane_info: Optional[str] = Field(default=None, description="Plane information if 2D slice is requested (e.g., 'Z plane', 'X=0.5')")
    additional_fields: List[str] = Field(default=[], description="Additional fields that might be useful to visualize")
    visualization_priority: str = Field(description="Priority of visualization (e.g., 'high', 'medium', 'low')")

def visualization_node(state):
    """
    Visualization node: Creates PyVista visualizations from the successfully generated OpenFOAM case.
    This node uses the successfully generated code and user_requirement to create PyVista visualizations.
    
    Updates state with:
      - plot_configs: List of plot configurations
      - plot_outputs: List of generated plot file paths
      - visualization_summary: Summary of generated visualizations
      - pyvista_visualization: PyVista visualization results
    """
    config = state["config"]
    user_requirement = state["user_requirement"]
    case_dir = state["case_dir"]
    
    print(f"============================== Visualization (PyVista) ==============================")
    
    # Ensure case_dir is absolute
    case_dir = os.path.abspath(case_dir)
    if not os.path.exists(case_dir):
        print(f"Case directory does not exist: {case_dir}")
        return {
            **state,
            "plot_configs": [],
            "plot_outputs": [],
            "visualization_summary": {"error": f"Case directory does not exist: {case_dir}"},
            "pyvista_visualization": {"success": False, "error": f"Case directory does not exist: {case_dir}"}
        }
    
    # Touch the .foam file before generating the visualization script
    foam_file = get_foam_file(case_dir)
    foam_file_path = os.path.join(case_dir, foam_file)
    with open(foam_file_path, 'a'):
        os.utime(foam_file_path, None)
    
    # Initialize loop counter
    current_loop = 0
    error_logs = []
    max_loop = state['config'].max_loop
    
    while current_loop < max_loop:
        current_loop += 1
        print(f"Attempt {current_loop} of {max_loop}")
        
        # Create visualization script
        viz_prompt = (
            f"<case_directory>{case_dir}</case_directory>\n"
            f"<foam_file>{foam_file}</foam_file>\n"
            f"<visualization_requirements>{state['user_requirement']}</visualization_requirements>\n"
            f"<previous_errors>{error_logs}</previous_errors>\n"
            f"Please create a PyVista Python script that visualizes the specified data by reading the .foam file ('{foam_file}') and saves it as a PNG file named visualization.png."
        )
        
        viz_script = state["llm_service"].invoke(viz_prompt, VISUALIZATION_SYSTEM_PROMPT)
        
        # Save the visualization script
        script_path = os.path.join(case_dir, "visualization.py")
        save_file(script_path, viz_script)
        
        # Execute the script using Python
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=case_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f"Finished command: Return Code {result.returncode}")
            error_logs = []
            
            # Check if the output image was created
            output_image = os.path.join(case_dir, "visualization.png")
            if os.path.exists(output_image):
                print(f"PyVista visualization created successfully: {output_image}")
                
                # Create plot configs and outputs in the expected format
                plot_configs = [
                    {
                        "plot_type": "pyvista_2d",
                        "field_name": "U",  # Default field, could be enhanced to detect from script
                        "time_step": "latest",
                        "output_format": "png",
                        "output_path": output_image
                    }
                ]
                
                plot_outputs = [output_image]
                
                visualization_summary = {
                    "total_plots_generated": len(plot_outputs),
                    "plot_types": ["pyvista_2d"],
                    "fields_visualized": ["U"],
                    "output_directory": case_dir,
                    "pyvista_success": True
                }
                
                pyvista_result = {
                    "success": True,
                    "output_image": output_image,
                    "script": viz_script
                }
                
                print(f"Generated {len(plot_outputs)} plots")
                print(f"PyVista visualization saved to: {output_image}")
                print("============================== Visualization Complete ==============================")
                
                return {
                    **state,
                    "plot_configs": plot_configs,
                    "plot_outputs": plot_outputs,
                    "visualization_summary": visualization_summary,
                    "pyvista_visualization": pyvista_result
                }
            else:
                error_logs.append("Visualization script executed but no output image was created")
                
        except subprocess.CalledProcessError as e:
            error_message = f"Error executing visualization script: {str(e)}"
            if e.stdout:
                error_message += f"\nSTDOUT:\n{e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout}"
            if e.stderr:
                error_message += f"\nSTDERR:\n{e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}"
            error_logs.append(error_message)
        
        # If we have errors and haven't reached max loops, try to fix them
        if error_logs and current_loop < max_loop:
            error_fix_prompt = (
                f"<error_logs>{error_logs}</error_logs>\n"
                f"<foam_file>{foam_file}</foam_file>\n"
                f"<original_script>{viz_script}</original_script>\n"
                f"<attempt_number>{current_loop}</attempt_number>\n"
                f"Please fix the PyVista Python script based on the error messages. The script should read the .foam file ('{foam_file}') in the case directory."
            )
            
            fixed_script = state["llm_service"].invoke(error_fix_prompt, ERROR_FIX_SYSTEM_PROMPT)
            
            # Save the fixed script
            save_file(script_path, fixed_script)
            
            # Try executing the fixed script
            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    cwd=case_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                print(f"Finished command: Return Code {result.returncode}")
                error_logs = []
                
                # Check if the output image was created
                output_image = os.path.join(case_dir, "visualization.png")
                if os.path.exists(output_image):
                    print(f"PyVista visualization created successfully: {output_image}")
                    
                    # Create plot configs and outputs in the expected format
                    plot_configs = [
                        {
                            "plot_type": "pyvista_3d",
                            "field_name": "U",  # Default field, could be enhanced to detect from script
                            "time_step": "latest",
                            "output_format": "png",
                            "output_path": output_image
                        }
                    ]
                    
                    plot_outputs = [output_image]
                    
                    visualization_summary = {
                        "total_plots_generated": len(plot_outputs),
                        "plot_types": ["pyvista_3d"],
                        "fields_visualized": ["U"],
                        "output_directory": case_dir,
                        "pyvista_success": True
                    }
                    
                    pyvista_result = {
                        "success": True,
                        "output_image": output_image,
                        "script": fixed_script
                    }
                    
                    print(f"Generated {len(plot_outputs)} plots")
                    print(f"PyVista visualization saved to: {output_image}")
                    print("============================== Visualization Complete ==============================")
                    
                    return {
                        **state,
                        "plot_configs": plot_configs,
                        "plot_outputs": plot_outputs,
                        "visualization_summary": visualization_summary,
                        "pyvista_visualization": pyvista_result
                    }
                else:
                    error_logs.append("Fixed visualization script executed but no output image was created")
                    
            except subprocess.CalledProcessError as e:
                error_message = f"Error executing fixed visualization script: {str(e)}"
                if e.stdout:
                    error_message += f"\nSTDOUT:\n{e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout}"
                if e.stderr:
                    error_message += f"\nSTDERR:\n{e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}"
                error_logs.append(error_message)
    
    # If we've exhausted all attempts
    if current_loop >= max_loop:
        print(f"Failed to create visualization after {max_loop} attempts")
        error_message = f"Maximum number of attempts ({max_loop}) reached without success"
        error_logs.append(error_message)
    
    # Return failure state in the expected format
    plot_configs = []
    plot_outputs = []
    
    visualization_summary = {
        "total_plots_generated": 0,
        "plot_types": [],
        "fields_visualized": [],
        "output_directory": case_dir,
        "pyvista_success": False,
        "error": error_message if 'error_message' in locals() else "Unknown error"
    }
    
    pyvista_result = {
        "success": False,
        "error": error_message if 'error_message' in locals() else "Unknown error",
        "error_logs": error_logs
    }
    
    print("============================== Visualization Failed ==============================")
    
    return {
        **state,
        "plot_configs": plot_configs,
        "plot_outputs": plot_outputs,
        "visualization_summary": visualization_summary,
        "pyvista_visualization": pyvista_result
    }
