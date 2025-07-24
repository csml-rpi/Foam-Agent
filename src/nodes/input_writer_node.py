# input_writer_node.py
import os
from utils import save_file, parse_context, retrieve_faiss, FoamPydantic, FoamfilePydantic
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

REWRITE_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM simulation and numerical modeling. "
    "Your task is to modify and rewrite the necessary OpenFOAM files to fix the reported error. "
    "Please do not propose solutions that require modifying any parameters declared in the user requirement, try other approaches instead."
    "The user will provide the error content, error command, reviewer's suggestions, and all relevant foam files. "
    "Only return files that require rewriting, modification, or addition; do not include files that remain unchanged. "
    "Return the complete, corrected file contents in the following JSON format: "
    "list of foamfile: [{file_name: 'file_name', folder_name: 'folder_name', content: 'content'}]. "
    "Ensure your response includes only the modified file content with no extra text, as it will be parsed using Pydantic."
)

def compute_priority(subtask):
    if subtask["folder_name"] == "system":
        return 0
    elif subtask["folder_name"] == "constant":
        return 1
    elif subtask["folder_name"] == "0":
        return 2
    else:
        return 3
        

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
    """
    Rewrite mode: Fix errors based on reviewer analysis
    """
    print(f"============================== Rewrite Mode ==============================")

    config = state["config"]
    
    if not state.get("review_analysis"):
        print("No review analysis available for rewrite mode.")
        return state
    
    rewrite_user_prompt = (
        f"<foamfiles>{str(state['foamfiles'])}</foamfiles>\n"
        f"<error_logs>{state['error_logs']}</error_logs>\n"
        f"<reviewer_analysis>{state['review_analysis']}</reviewer_analysis>\n\n"
        f"<user_requirement>{state['user_requirement']}</user_requirement>\n\n"
        "Please update the relevant OpenFOAM files to resolve the reported errors, ensuring that all modifications strictly adhere to the specified formats. Ensure all modifications adhere to user requirement."
    )
    rewrite_response = state["llm_service"].invoke(rewrite_user_prompt, REWRITE_SYSTEM_PROMPT, pydantic_obj=FoamPydantic)

    print(f"============================== Rewrite ==============================")
    # Prepare updated dir_structure and foamfiles without mutating state
    dir_structure = dict(state["dir_structure"]) if state.get("dir_structure") else {}
    foamfiles_list = list(state["foamfiles"].list_foamfile) if state.get("foamfiles") and hasattr(state["foamfiles"], "list_foamfile") else []

    for foamfile in rewrite_response.list_foamfile:
        print(f"Modified the file: {foamfile.file_name} in folder: {foamfile.folder_name}")
        file_path = os.path.join(state["case_dir"], foamfile.folder_name, foamfile.file_name)
        save_file(file_path, foamfile.content)
        
        if foamfile.folder_name not in dir_structure:
            dir_structure[foamfile.folder_name] = []
        if foamfile.file_name not in dir_structure[foamfile.folder_name]:
            dir_structure[foamfile.folder_name].append(foamfile.file_name)
        
        foamfiles_list = [f for f in foamfiles_list if not (f.folder_name == foamfile.folder_name and f.file_name == foamfile.file_name)]
        foamfiles_list.append(foamfile)

    foamfiles = FoamPydantic(list_foamfile=foamfiles_list)
    return {
        "dir_structure": dir_structure,
        "foamfiles": foamfiles,
        "error_logs": []
    }

def _initial_write_mode(state):
    """
    Initial write mode: Generate files from scratch
    """
    print(f"============================== Initial Write Mode ==============================")
    
    config = state["config"]
    subtasks = state["subtasks"]
    subtasks = sorted(subtasks, key=compute_priority)
    
    writed_files = []
    dir_structure = {}
    
    for subtask in subtasks:
        file_name = subtask["file_name"]
        folder_name = subtask["folder_name"]
        
        if folder_name not in dir_structure:
            dir_structure[folder_name] = []
        dir_structure[folder_name].append(file_name)
        
        print(f"Generating file: {file_name} in folder: {folder_name}")
        
        if not file_name or not folder_name:
            raise ValueError(f"Invalid subtask format: {subtask}")

        file_path = os.path.join(state["case_dir"], folder_name, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Retrieve a similar reference foamfile from the tutorial.
        similar_file_text = state["tutorial_reference"]
        
        # Generate the complete foamfile.
        code_system_prompt = INITIAL_WRITE_SYSTEM_PROMPT.format(
            file_name=file_name,
            folder_name=folder_name,
            case_solver=state['case_stats']['case_solver']
        )

        code_user_prompt = (
            f"User requirement: {state['user_requirement']}\n"
            f"Refer to the following similar case file content to ensure the generated file aligns with the user requirement:\n<similar_case_reference>{similar_file_text}</similar_case_reference>\n"
            f"Similar case reference is always correct. If you find the user requirement is very consistent with the similar case reference, you should use the similar case reference as the template to generate the file."
            f"Just modify the necessary parts to make the file complete and functional."
            "Please ensure that the generated file is complete, functional, and logically sound."
            "Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence."
            "When generating controlDict, do not include anything to preform post processing. Just include the necessary settings to run the simulation."
        )
        if len(writed_files) > 0:
            code_user_prompt += f"The following are files content already generated: {str(writed_files)}\n\n\nYou should ensure that the new file is consistent with the previous files. Such as boundary conditions, mesh settings, etc."

        generation_response = state["llm_service"].invoke(code_user_prompt, code_system_prompt)
        
        code_context = parse_context(generation_response)
        save_file(file_path, code_context)
        
        writed_files.append(FoamfilePydantic(file_name=file_name, folder_name=folder_name, content=code_context))
    
    # Write the Allrun script.
    case_dir = state["case_dir"]
    allrun_file_path = os.path.join(case_dir, "Allrun")
    if os.path.exists(allrun_file_path):
        print("Warning: Allrun file exists. Overwriting.")
    
    # Retrieve available commands from the FAISS "Commands" database.
    commands = retrieve_commands(f"{config.database_path}/raw/openfoam_commands.txt")
    
    # Include mesh commands if custom mesh is used
    mesh_commands_info = ""
    if state.get("custom_mesh_used") and state.get("mesh_commands"):
        mesh_commands_info = f"\nCustom mesh commands to include: {state['mesh_commands']}"
        print(f"Including custom mesh commands: {state['mesh_commands']}")
    
    command_system_prompt = (
        "You are an expert in OpenFOAM. The user will provide a list of available commands. "
        "Your task is to generate only the necessary OpenFOAM commands required to create an Allrun script for the given user case, based on the provided directory structure. "
        "Return only the list of commands—no explanations, comments, or additional text."
    )

    if state.get("mesh_type") == "custom_mesh":
        command_system_prompt += "If custom mesh commands are provided, include them in the appropriate order (typically after blockMesh or instead of blockMesh if custom mesh is used). "
    
    command_user_prompt = (
        f"Available OpenFOAM commands for the Allrun script: {commands}\n"
        f"Case directory structure: {dir_structure}\n"
        f"User case information: {state['case_info']}\n"
        f"Reference Allrun scripts from similar cases: {state['allrun_reference']}\n"
        "Generate only the required OpenFOAM command list—no extra text."
    )

    if state.get("mesh_type") == "custom_mesh":
        command_user_prompt += f"{mesh_commands_info}\n"
    
    command_response = state["llm_service"].invoke(command_user_prompt, command_system_prompt, pydantic_obj=CommandsPydantic)

    if len(command_response.commands) == 0:
        print("Failed to generate subtasks.")
        raise ValueError("Failed to generate subtasks.")

    print(f"Need {len(command_response.commands)} commands.")
    
    commands_help = []
    for command in command_response.commands:
        command_help = retrieve_faiss("openfoam_command_help", command, topk=config.searchdocs)
        commands_help.append(command_help[0]['full_content'])
    commands_help = "\n".join(commands_help)


    allrun_system_prompt = (
        "You are an expert in OpenFOAM. Generate an Allrun script based on the provided details."
        f"Available commands with descriptions: {commands_help}\n\n"
        f"Reference Allrun scripts from similar cases: {state['allrun_reference']}\n\n"
        "If custom mesh commands are provided, make sure to include them in the appropriate order in the Allrun script. "
        "CRITICAL: Do not include any post processing commands in the Allrun script."
    )

    if state.get("mesh_mode") == "custom":
        allrun_system_prompt += "CRITICAL: Do not include any other mesh commands other than the custom mesh commands.\n"
        allrun_system_prompt += "CRITICAL: Do not include any gmshToFoam commands in the Allrun script."
    
    allrun_user_prompt = (
        f"User requirement: {state['user_requirement']}\n"
        f"Case directory structure: {dir_structure}\n"
        f"User case infomation: {state['case_info']}\n"
        f"{mesh_commands_info}\n"
        "All run scripts for these similar cases are for reference only and may not be correct, as you might be a different case solver or have a different directory structure. " 
        "You need to rely on your OpenFOAM and physics knowledge to discern this, and pay more attention to user requirements, " 
        "as your ultimate goal is to fulfill the user's requirements and generate an allrun script that meets those requirements."
        "CRITICAL: Do not include any post processing commands in the Allrun script."
        "Generate the Allrun script strictly based on the above information. Do not include explanations, comments, or additional text. Put the code in ``` tags."
    )

    if state.get("mesh_mode") == "custom":
        allrun_user_prompt += "CRITICAL: Do not include any other mesh commands other than the custom mesh commands.\n"
        allrun_user_prompt += "CRITICAL: Do not include any gmshToFoam commands in the Allrun script."


    
    allrun_response = state["llm_service"].invoke(allrun_user_prompt, allrun_system_prompt)
    
    allrun_script = parse_allrun(allrun_response)
    save_file(allrun_file_path, allrun_script)
    
    writed_files.append(FoamfilePydantic(file_name="Allrun", folder_name="./", content=allrun_script))
    foamfiles = FoamPydantic(list_foamfile=writed_files)
    
    # Return updated state
    return {
        "dir_structure": dir_structure,
        "commands": command_response.commands,
        "foamfiles": foamfiles
    }
