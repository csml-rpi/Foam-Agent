# input_writer_node.py
import os
from utils import save_file, parse_context, retrieve_faiss, FoamPydantic, FoamfilePydantic
import re
from typing import List
from pydantic import BaseModel, Field


def compute_priority(subtask):
    if subtask.folder_name == "system":
        return 0
    elif subtask.folder_name == "constant":
        return 1
    elif subtask.folder_name == "0":
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
    """
    config = state.config
    subtasks = state.subtasks
    
    subtasks = sorted(subtasks, key=compute_priority)
    
    writed_files = []
    dir_structure = {}
    
    for subtask in subtasks:
        file_name = subtask.file_name
        folder_name = subtask.folder_name
        
        if folder_name not in dir_structure:
            dir_structure[folder_name] = []
        dir_structure[folder_name].append(file_name)
        
        print(f"Generating file: {file_name} in folder: {folder_name}")
        
        if not file_name or not folder_name:
            raise ValueError(f"Invalid subtask format: {subtask}")

        file_path = os.path.join(state.case_dir, folder_name, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Retrieve a similar reference foamfile from the tutorial.
        similar_file_text = state.tutorial_reference
        
        # Generate the complete foamfile.
        code_system_prompt = (
            "You are an expert in OpenFOAM simulation and numerical modeling."
            f"Your task is to generate a complete and functional file named: <file_name>{file_name}</file_name> within the <folder_name>{folder_name}</folder_name> directory. "
            "Ensure all required values are present and match with the files content already generated."
            "Before finalizing the output, ensure:\n"
            "- All necessary fields exist (e.g., if `nu` is defined in `constant/transportProperties`, it must be used correctly in `0/U`).\n"
            "- Cross-check field names between different files to avoid mismatches.\n"
            "- Ensure units and dimensions are correct** for all physical variables.\n"
            f"- Ensure case solver settings are consistent with the user's requirements. Available solvers are: {state.case_stats['case_solver']}.\n"
            "Provide only the code—no explanations, comments, or additional text."
        )

        code_user_prompt = (
            f"User requirement: {state.user_requirement}\n"
            f"Refer to the following similar case file content to ensure the generated file aligns with the user requirement:\n<similar_case_reference>{similar_file_text}</similar_case_reference>\n"
            f"Similar case reference is always correct. If you find the user requirement is very consistent with the similar case reference, you should use the similar case reference as the template to generate the file."
            f"Just modify the necessary parts to make the file complete and functional."
            "Please ensure that the generated file is complete, functional, and logically sound."
            "Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence."
            "When generating controlDict, do not include anything to preform post processing. Just include the necessary settings to run the simulation."
        )
        if len(writed_files) > 0:
            code_user_prompt += f"The following are files content already generated: {str(writed_files)}\n\n\nYou should ensure that the new file is consistent with the previous files. Such as boundary conditions, mesh settings, etc."

        generation_response = state.llm_service.invoke(code_user_prompt, code_system_prompt)
        
        code_context = parse_context(generation_response)
        save_file(file_path, code_context)
        
        writed_files.append(FoamfilePydantic(file_name=file_name, folder_name=folder_name, content=code_context))
    
    state.dir_structure = dir_structure
    
    # Write the Allrun script.
    case_dir = state.case_dir
    allrun_file_path = os.path.join(case_dir, "Allrun")
    if os.path.exists(allrun_file_path):
        print("Warning: Allrun file exists. Overwriting.")
    
    # Retrieve available commands from the FAISS "Commands" database.
    commands = retrieve_commands(f"{state.config.database_path}/raw/openfoam_commands.txt")
    
    command_system_prompt = (
        "You are an expert in OpenFOAM. The user will provide a list of available commands. "
        "Your task is to generate only the necessary OpenFOAM commands required to create an Allrun script for the given user case, based on the provided directory structure. "
        "Return only the list of commands—no explanations, comments, or additional text."
    )
    
    command_user_prompt = (
        f"Available OpenFOAM commands for the Allrun script: {commands}\n"
        f"Case directory structure: {state.dir_structure}\n"
        f"User case information: {state.case_info}\n"
        f"Reference Allrun scripts from similar cases: {state.allrun_reference}\n"
        "Generate only the required OpenFOAM command list—no extra text."
    )
    
    command_response = state.llm_service.invoke(command_user_prompt, command_system_prompt, pydantic_obj=CommandsPydantic)

    if len(command_response.commands) == 0:
        print("Failed to generate subtasks.")
        raise ValueError("Failed to generate subtasks.")

    print(f"Need {len(command_response.commands)} commands.")
    
    state.commands = command_response.commands
    
    commands_help = []
    for command in command_response.commands:
        command_help = retrieve_faiss("openfoam_command_help", command, topk=state.config.searchdocs)
        commands_help.append(command_help[0]['full_content'])
    commands_help = "\n".join(commands_help)


    allrun_system_prompt = (
        "You are an expert in OpenFOAM. Generate an Allrun script based on the provided details."
        f"Available commands with descriptions: {commands_help}\n\n"
        f"Reference Allrun scripts from similar cases: {state.allrun_reference}\n\n"
        "Identify if the mesh is provided as an external file from the user requirements. If the mesh is provided as an external file, dont include any command in Allrun to generate the mesh."
        "Do not include any command in Allrunto convert .msh file to foam format."
    )
    
    allrun_user_prompt = (
        f"User requirement: {state.user_requirement}\n"
        f"Case directory structure: {state.dir_structure}\n"
        f"User case infomation: {state.case_info}\n"
        "All run scripts for these similar cases are for reference only and may not be correct, as you might be a different case solver or have a different directory structure. " 
        "You need to rely on your OpenFOAM and physics knowledge to discern this, and pay more attention to user requirements, " 
        "as your ultimate goal is to fulfill the user's requirements and generate an allrun script that meets those requirements."
        "Generate the Allrun script strictly based on the above information. Do not include explanations, comments, or additional text. Put the code in ``` tags."
    )
    
    allrun_response = state.llm_service.invoke(allrun_user_prompt, allrun_system_prompt)
    
    allrun_script = parse_allrun(allrun_response)
    save_file(allrun_file_path, allrun_script)
    
    writed_files.append(FoamfilePydantic(file_name="Allrun", folder_name="./", content=allrun_script))
    state.foamfiles = FoamPydantic(list_foamfile=writed_files)
    
    return {"goto": "runner"}
