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
from router_func import llm_requires_custom_mesh, llm_requires_hpc, llm_requires_visualization
from logger import setup_logging

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
    similar_case_advice = plan_data.get("similar_case_advice")
    
    print(f"Parsed case name: {case_name}")
    print(f"Parsed case domain: {case_domain}")
    print(f"Parsed case category: {case_category}")
    print(f"Parsed case solver: {case_solver}")
    print(f"Created case directory: {case_dir}")
    print(f"Retrieved similar case structure: {dir_structure_reference}")
    print(f"Generated {len(subtasks)} subtasks.")
    if similar_case_advice:
        print(f"Similar case advice: {similar_case_advice}")

    # Handle case directory creation/cleanup.
    # Preserve dataset_log_path across rmtree when it lives inside case_dir,
    # so plan-phase LLM records (already written this run) aren't destroyed.
    dataset_log_path = getattr(config, "dataset_log_path", "") or ""
    preserved_log_bytes = None
    if dataset_log_path and os.path.exists(case_dir) and os.path.exists(dataset_log_path):
        try:
            rel = os.path.relpath(os.path.abspath(dataset_log_path), os.path.abspath(case_dir))
            if not rel.startswith("..") and not os.path.isabs(rel):
                with open(dataset_log_path, "rb") as _f:
                    preserved_log_bytes = _f.read()
        except Exception as _e:
            print(f"Warning: failed to snapshot dataset_log_path before rmtree: {_e}")
    if os.path.exists(case_dir):
        print(f"Warning: Case directory {case_dir} already exists. Overwriting.")
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)

    if preserved_log_bytes is not None:
        os.makedirs(os.path.dirname(dataset_log_path), exist_ok=True)
        with open(dataset_log_path, "wb") as _f:
            _f.write(preserved_log_bytes)

    # Initialize logging now that case_dir exists
    setup_logging(case_dir)

    # Save reference file
    save_file(case_path_reference, f"{faiss_detailed}\n\n\n{allrun_reference}")

    # Determine mesh type
    mesh_type = llm_requires_custom_mesh(state)
    if mesh_type == 1:
        mesh_type_value = "custom_mesh"
        print("<mesh_type>custom_mesh - Custom mesh requested.</mesh_type>")
    elif mesh_type == 2:
        mesh_type_value = "gmsh_mesh"
        print("<mesh_type>gmsh_mesh - GMSH mesh requested.</mesh_type>")
    else:
        mesh_type_value = "standard_mesh"
        print("<mesh_type>standard_mesh - Standard mesh generation.</mesh_type>")

    # Cache routing decisions to avoid repeated LLM calls in routing.
    requires_hpc = llm_requires_hpc(state)
    requires_visualization = llm_requires_visualization(state)
    print(f"<routing_decisions>requires_hpc={requires_hpc}, requires_visualization={requires_visualization}</routing_decisions>")
    print("</planner>")

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
        "requires_hpc": requires_hpc,
        "requires_visualization": requires_visualization,
        "similar_case_advice": similar_case_advice,
    }
