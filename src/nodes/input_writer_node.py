# input_writer_node.py
import os
from utils import save_file, parse_context, retrieve_faiss, FoamPydantic, FoamfilePydantic
from services.input_writer import initial_write, build_allrun, rewrite_files
import re
from typing import List
from pydantic import BaseModel, Field




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
    print("<input_writer mode=\"rewrite\">")
    if not state.get("review_analysis"):
        print("No review analysis available for rewrite mode.")
        print("</input_writer>")
        return state

    return rewrite_files(
        case_dir=state["case_dir"],
        error_logs=state.get("error_logs", []),
        review_analysis=state.get("review_analysis", ""),
        rewrite_plan=None,
        user_requirement=state.get("user_requirement", ""),
        foamfiles=state.get("foamfiles"),
        dir_structure=state.get("dir_structure", {}),
        loop_count=state.get("loop_count", 0),
    )

def _initial_write_mode(state):
    """
    Initial write mode: Generate files from scratch
    """
    print("<input_writer mode=\"initial\">")
    
    config = state["config"]
    write_out = initial_write(
        case_dir=state["case_dir"],
        subtasks=state["subtasks"],
        user_requirement=state["user_requirement"],
        tutorial_reference=state["tutorial_reference"],
        case_solver=state['case_stats']['case_solver'],
        generation_mode=getattr(config, "input_writer_generation_mode", "sequential_dependency"),
        similar_case_advice=state.get("similar_case_advice"),
        reuse_generated_dir=getattr(config, "reuse_generated_dir", ""),
    )

    dir_structure = write_out["dir_structure"]
    foamfiles = write_out["foamfiles"]

    # Build Allrun via service
    mesh_type = state.get("mesh_type")
    mesh_commands = state.get("mesh_commands") or []
    advice = state.get("similar_case_advice")
    advice_d = advice.model_dump() if hasattr(advice, "model_dump") else (advice if isinstance(advice, dict) else {})
    pre_solver_steps = advice_d.get("pre_solver_steps") if advice_d else None
    allrun_out = build_allrun(
        case_dir=state["case_dir"],
        database_path=config.database_path,
        searchdocs=config.searchdocs,
        dir_structure=dir_structure,
        case_info=state["case_info"],
        allrun_reference=state["allrun_reference"],
        mesh_type=mesh_type,
        mesh_commands=mesh_commands,
        pre_solver_steps=pre_solver_steps,
    )

    print("</input_writer>")

    return {
        "dir_structure": dir_structure,
        "commands": [],
        "foamfiles": foamfiles,
    }
