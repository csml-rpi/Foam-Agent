# planner_node.py
import os
import re
from typing import Dict, Any, List, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
import shutil
from utils import save_file, retrieve_faiss, parse_directory_structure, LLMService
from services.plan import generate_simulation_plan
from services import global_llm_service
from router_func import llm_requires_custom_mesh

class CaseSummaryPydantic(BaseModel):
    case_name: str = Field(description="name of the case")
    case_domain: str = Field(description="domain of the case, case domain must be one of [basic,combustion,compressible,discreteMethods,DNS,electromagnetics,financial,heatTransfer,incompressible,lagrangian,mesh,multiphase,resources,stressAnalysis].")
    case_category: str = Field(description="category of the case")
    case_solver: str = Field(description="solver of the case")


class SubtaskPydantic(BaseModel):
    file_name: str = Field(description="Name of the OpenFOAM input file")
    folder_name: str = Field(description="Name of the folder where the foamfile should be stored")

class OpenFOAMPlanPydantic(BaseModel):
    subtasks: List[SubtaskPydantic] = Field(description="List of subtasks, each with its corresponding file and folder names")


def planner_node(state):
    """
    Planner node: Parse the user requirement to a standard case description,
    finds a similar reference case from the FAISS databases, and splits the work into subtasks.
    Updates state with:
      - case_dir, tutorial, case_name, subtasks.
    """
    config = state["config"]
    user_requirement = state["user_requirement"]

    # Generate simulation plan using the core planning logic
    plan_data = generate_simulation_plan(
        user_requirement=user_requirement,
        case_stats=state["case_stats"],
        case_dir=getattr(config, "case_dir", ""),
        searchdocs=getattr(config, "searchdocs", 2),
        file_dependency_threshold=getattr(config, "file_dependency_threshold", 3000)
    )
    
    # Extract plan data
    case_name = plan_data["case_name"]
    case_domain = plan_data["case_domain"]
    case_category = plan_data["case_category"]
    case_solver = plan_data["case_solver"]
    case_dir = plan_data["case_dir"]
    faiss_detailed = plan_data["tutorial_reference"]
    case_path_reference = plan_data["case_path_reference"]
    dir_structure_reference = plan_data["dir_structure_reference"]
    allrun_reference = plan_data["allrun_reference"]
    subtasks = plan_data["subtasks"]
    file_dependency_flag = plan_data["file_dependency_flag"]
    
    print(f"Parsed case name: {case_name}")
    print(f"Parsed case domain: {case_domain}")
    print(f"Parsed case category: {case_category}")
    print(f"Parsed case solver: {case_solver}")
    print(f"Created case directory: {case_dir}")
    print(f"Retrieved similar case structure: {dir_structure_reference}")
    print(f"Generated {len(subtasks)} subtasks.")

    # Handle case directory creation/cleanup
    if os.path.exists(case_dir):
        print(f"Warning: Case directory {case_dir} already exists. Overwriting.")
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)
    
    # Save reference file
    save_file(case_path_reference, f"{faiss_detailed}\n\n\n{allrun_reference}")

    # Determine mesh type
    mesh_type = llm_requires_custom_mesh(state)
    if mesh_type == 1:
        mesh_type_value = "custom_mesh"
        print("Planner determined: Custom mesh requested.")
    elif mesh_type == 2:
        mesh_type_value = "gmsh_mesh"
        print("Planner determined: GMSH mesh requested.")
    else:
        mesh_type_value = "standard_mesh"
        print("Planner determined: Standard mesh generation.")
    
    print(f"Planner set mesh_type to: {mesh_type_value}")

    # Return updated state
    case_info = f"case name: {case_name}\ncase domain: {case_domain}\ncase category: {case_category}\ncase solver: {case_solver}"
    return {
        "case_name": case_name,
        "case_domain": case_domain,
        "case_category": case_category,
        "case_solver": case_solver,
        "case_dir": case_dir,
        "tutorial_reference": faiss_detailed,
        "case_path_reference": case_path_reference,
        "dir_structure_reference": dir_structure_reference,
        "case_info": case_info,
        "allrun_reference": allrun_reference,
        "subtasks": subtasks,
        "mesh_type": mesh_type_value,
        "file_dependency_flag": file_dependency_flag
    }
