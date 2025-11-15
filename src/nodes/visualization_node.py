# visualization_node.py
import os
import subprocess
import sys
from typing import List, Optional
from pydantic import BaseModel, Field
from utils import save_file
from services.visualization import ensure_foam_file, generate_pyvista_script, run_pyvista_script, fix_pyvista_script
import glob

# Helper to get the .foam file name
def get_foam_file(case_dir):
    case_dir_name = os.path.basename(os.path.normpath(case_dir))
    return f"{case_dir_name}.foam"

VISUALIZATION_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM post-processing and PyVista Python scripting. "
    "Your task is to generate a PyVista Python script that visualizes the specified data from the OpenFOAM case. "
    "The script should load the OpenFOAM case data by reading the .foam file (e.g., 'runs.foam') in the case directory using PyVista, display the geometry, and color the surface by the specified field (e.g., 'U' for velocity). "
    "Ensure the script shows the geometry, sets up the colorbar, and saves the visualization as a PNG file. "
    "Use coolwarm colormap by default."
    "The script must save the visualization as a PNG file, and the output image must contain the geometry and the colorbar, not just the colorbar. "
    "IMPORTANT: Return ONLY the Python code without any markdown formatting, code block markers, or explanatory text. "
    "The script should start with the necessary imports, read the .foam file using PyVista, and end with the screenshot saving."
)

ERROR_FIX_SYSTEM_PROMPT = (
    "You are an expert in PyVista Python scripting and OpenFOAM visualization. "
    "Your task is to fix the provided PyVista Python script that encountered an error. "
    "Ensure the script loads the OpenFOAM case data by reading the .foam file (e.g., 'runs.foam') in the case directory using PyVista, displays the geometry, and colors the surface by the specified field. "
    "Make sure the script shows the geometry, sets up the colorbar, and saves the visualization as a PNG file. "
    "Use coolwarm colormap by default."
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
    
    # Ensure .foam file exists
    foam_file = ensure_foam_file(case_dir)
    
    # Initialize loop counter
    current_loop = 0
    error_logs = []
    max_loop = state['config'].max_loop
    
    while current_loop < max_loop:
        current_loop += 1
        print(f"Attempt {current_loop} of {max_loop}")
        
        # Create visualization script
        viz_script = generate_pyvista_script(case_dir, foam_file, state['user_requirement'], error_logs)
        success, output_image, errs = run_pyvista_script(case_dir, viz_script)
        print(f"Error: {errs}")
        
        if success and output_image:
            error_logs = []
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
            error_logs.extend(errs)
        
        # If we have errors and haven't reached max loops, try to fix them
        if error_logs and current_loop < max_loop:
            fixed_script = fix_pyvista_script(foam_file, viz_script, error_logs)
            success, output_image, errs = run_pyvista_script(case_dir, fixed_script, filename="visualization.py")

            print(f"Error: {errs}")

            if success and output_image:
                print(f"Finished command: Return Code 0")
                error_logs = []
                print(f"PyVista visualization created successfully: {output_image}")
                
                # Create plot configs and outputs in the expected format
                plot_configs = [
                    {
                        "plot_type": "pyvista_3d",
                        "field_name": "U",
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
                error_logs.extend(errs)
    
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
