# architect_node.py
import os
import re
from utils import save_file, retrieve_faiss, parse_directory_structure
from services.architect import (
    parse_requirement_to_case_info,
    resolve_case_dir,
    retrieve_references,
    decompose_to_subtasks,
)
from pydantic import BaseModel, Field
from typing import List
import shutil
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


def architect_node(state):
    """
    Architect node: Parse the user requirement to a standard case description,
    finds a similar reference case from the FAISS databases, and splits the work into subtasks.
    Updates state with:
      - case_dir, tutorial, case_name, subtasks.
    """
    config = state["config"]
    user_requirement = state["user_requirement"]

    # Step 1: Translate user requirement (service)
    info = parse_requirement_to_case_info(user_requirement, state["case_stats"])
    case_name = info["case_name"]
    case_domain = info["case_domain"]
    case_category = info["case_category"]
    case_solver = info["case_solver"]
    
    print(f"Parsed case name: {case_name}")
    print(f"Parsed case domain: {case_domain}")
    print(f"Parsed case category: {case_category}")
    print(f"Parsed case solver: {case_solver}")
    
    # Step 2: Determine case directory (service)
    case_dir = resolve_case_dir(
        case_name=case_name,
        case_dir=getattr(config, "case_dir", ""),
        run_times=getattr(config, "run_times", 1),
        run_directory=str(getattr(config, "run_directory", "")) or str(config.run_directory)
    )
    
    if os.path.exists(case_dir):
        print(f"Warning: Case directory {case_dir} already exists. Overwriting.")
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)
    
    
    print(f"Created case directory: {case_dir}")

    # Step 3: Retrieve references (service)
    faiss_detailed, dir_structure, dir_counts_str, allrun_reference, file_dependency_flag = retrieve_references(
        case_name=case_name,
        case_solver=case_solver,
        case_domain=case_domain,
        case_category=case_category,
        searchdocs=getattr(config, "searchdocs", 2),
        file_dependency_threshold=getattr(config, "file_dependency_threshold", 3000)
    )
    print(f"Retrieved similar case structure: {dir_structure}")
    print(dir_counts_str)
    
    case_path = os.path.join(case_dir, "similar_case.txt")
    
    tutorial_reference = faiss_detailed
    case_path_reference = case_path
    dir_structure_reference = dir_structure
    allrun_reference = allrun_reference
    
    save_file(case_path, f"{faiss_detailed}\n\n\n{allrun_reference}")
        

    # Step 4: Break down into subtasks (service)
    subtasks = decompose_to_subtasks(user_requirement, dir_structure, dir_counts_str)

    if len(subtasks) == 0:
        print("Failed to generate subtasks.")
        raise ValueError("Failed to generate subtasks.")

    print(f"Generated {len(subtasks)} subtasks.")

    mesh_type = llm_requires_custom_mesh(state)
    if mesh_type == 1:
        mesh_type_value = "custom_mesh"
        print("Architect determined: Custom mesh requested.")
    elif mesh_type == 2:
        mesh_type_value = "gmsh_mesh"
        print("Architect determined: GMSH mesh requested.")
    else:
        mesh_type_value = "standard_mesh"
        print("Architect determined: Standard mesh generation.")
    
    print(f"Architect set mesh_type to: {mesh_type_value}")

    # Return updated state
    case_info = f"case name: {case_name}\ncase domain: {case_domain}\ncase category: {case_category}\ncase solver: {case_solver}"
    return {
        "case_name": case_name,
        "case_domain": case_domain,
        "case_category": case_category,
        "case_solver": case_solver,
        "case_dir": case_dir,
        "tutorial_reference": tutorial_reference,
        "case_path_reference": case_path_reference,
        "dir_structure_reference": dir_structure_reference,
        "case_info": case_info,
        "allrun_reference": allrun_reference,
        "subtasks": subtasks,
        "mesh_type": mesh_type_value,
        "file_dependency_flag": file_dependency_flag
    }
