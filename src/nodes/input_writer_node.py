# input_writer_node.py
import os
from utils import save_file, parse_context, retrieve_faiss, FoamPydantic, FoamfilePydantic
from services.input_writer import initial_write, build_allrun, rewrite_files
import re
from typing import List
from pydantic import BaseModel, Field

# System prompts for different modes
INITIAL_WRITE_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM simulation and numerical modeling."
    f"Your task is to generate a complete and functional file named: <file_name>{{file_name}}</file_name> within the <folder_name>{{folder_name}}</folder_name> directory. "
    "Ensure all required values are present and match with the files content already generated."
    "Before finalizing the output, ensure:\n"
    "- All necessary fields exist (e.g., if `nu` is defined in `constant/transportProperties`, it must be used correctly in `0/U`).\n"
    "- Cross-check field names between different files to avoid mismatches.\n"
    "- Ensure units and dimensions are correct** for all physical variables.\n"
    f"- Ensure case solver settings are consistent with the user's requirements. Available solvers are: {{case_solver}}.\n"
    "Provide only the code—no explanations, comments, or additional text."
)
        

def parse_allrun(text: str) -> str:
    match = re.search(r'```(.*?)```', text, re.DOTALL)
    
    return match.group(1).strip() 

def retrieve_commands(command_path) -> str:
    with open(command_path, 'r') as file:
        commands = file.readlines()
    
    return f"[{', '.join([command.strip() for command in commands])}]"
    
class CommandsPydantic(BaseModel):
    commands: List[str] = Field(description="List of commands")

def input_writer_node(state):
    """
    InputWriter node: Generate the complete OpenFOAM foamfile.
    
    Args:
        state: The current state containing all necessary information
    """

    mode = state["input_writer_mode"]
    
    if mode == "rewrite":
        return _rewrite_mode(state)
    else:
        return _initial_write_mode(state)

def _rewrite_mode(state):
    """Rewrite mode: delegate to service to modify files based on review analysis."""
    print(f"============================== Rewrite Mode ==============================")
    if not state.get("review_analysis"):
        print("No review analysis available for rewrite mode.")
        return state
    out = rewrite_files(
        case_dir=state["case_dir"],
        foamfiles=state.get("foamfiles"),
        error_logs=state.get("error_logs", []),
        review_analysis=state.get("review_analysis", ""),
        user_requirement=state.get("user_requirement", ""),
        dir_structure=state.get("dir_structure", {}),
    )
    return out

def _initial_write_mode(state):
    """
    Initial write mode: Generate files from scratch
    """
    print(f"============================== Initial Write Agent ==============================")
    
    config = state["config"]
    write_out = initial_write(
        case_dir=state["case_dir"],
        subtasks=state["subtasks"],
        user_requirement=state["user_requirement"],
        tutorial_reference=state["tutorial_reference"],
        case_solver=state['case_stats']['case_solver'],
        file_dependency_flag=state["file_dependency_flag"],
    )

    dir_structure = write_out["dir_structure"]
    foamfiles = write_out["foamfiles"]

    # Build Allrun via service
    mesh_type = state.get("mesh_type")
    mesh_commands = state.get("mesh_commands") or []
    allrun_out = build_allrun(
        case_dir=state["case_dir"],
        database_path=config.database_path,
        searchdocs=config.searchdocs,
        dir_structure=dir_structure,
        case_info=state["case_info"],
        allrun_reference=state["allrun_reference"],
        mesh_type=mesh_type,
        mesh_commands=mesh_commands,
    )

    return {
        "dir_structure": dir_structure,
        "commands": [],
        "foamfiles": foamfiles,
    }
