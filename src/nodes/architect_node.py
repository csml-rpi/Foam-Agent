# architect_node.py
import os
import re
from utils import save_file, retrieve_faiss, parse_directory_structure
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

    # Step 1: Translate user requirement.
    parse_system_prompt = ("Please transform the following user requirement into a standard case description using a structured format."
                           "The key elements should include case name, case domain, case category, and case solver."
                           f"Note: case domain must be one of {state['case_stats']['case_domain']}."
                           f"Note: case category must be one of {state['case_stats']['case_category']}."
                           f"Note: case solver must be one of {state['case_stats']['case_solver']}."
                           )
    parse_user_prompt = f"User requirement: {user_requirement}."
    
    parse_response = state["llm_service"].invoke(parse_user_prompt, parse_system_prompt, pydantic_obj=CaseSummaryPydantic)
    
    case_name = parse_response.case_name.replace(" ", "_")
    case_domain = parse_response.case_domain
    case_category = parse_response.case_category
    case_solver = parse_response.case_solver
    
    print(f"Parsed case name: {case_name}")
    print(f"Parsed case domain: {case_domain}")
    print(f"Parsed case category: {case_category}")
    print(f"Parsed case solver: {case_solver}")
    
    # Step 2: Determine case directory.
    if config.case_dir != "":
        case_dir = config.case_dir
    else:
        if config.run_times > 1:
            case_dir = os.path.join(config.run_directory, f"{case_name}_{config.run_times}")
        else:
            case_dir = os.path.join(config.run_directory, case_name)
    
    if os.path.exists(case_dir):
        print(f"Warning: Case directory {case_dir} already exists. Overwriting.")
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)
    
    
    print(f"Created case directory: {case_dir}")

    # Step 3: Retrieve a similar reference case from the FAISS databases.
    # Retrieve by case info
    case_info = f"case name: {case_name}\ncase domain: {case_domain}\ncase category: {case_category}\ncase solver: {case_solver}"
    
    faiss_structure = retrieve_faiss("openfoam_tutorials_structure", case_info, topk=config.searchdocs)
    faiss_structure = faiss_structure[0]['full_content']
    faiss_structure = re.sub(r"\n{3}", '\n', faiss_structure)
    
    # Retrieve by case info + directory structure
    faiss_detailed = retrieve_faiss("openfoam_tutorials_details", faiss_structure, topk=config.searchdocs)
    faiss_detailed = faiss_detailed[0]['full_content']

    # cutoff similar case length for file dependency
    config.file_dependency = (faiss_detailed.count('\n') < config.file_dependency_threshold)
    if (config.file_dependency):
        print("File-dependency will be used by input writer.")
    else:
        print("No file-dependency in input writer.")
    
    dir_structure = re.search(r"<directory_structure>(.*?)</directory_structure>", faiss_detailed, re.DOTALL).group(1).strip()
    print(f"Retrieved similar case structure: {dir_structure}")
    
    dir_counts = parse_directory_structure(dir_structure)
    dir_counts_str = ',\n'.join([f"There are {count} files in Directory: {directory}" for directory, count in dir_counts.items()])
    print(dir_counts_str)
    
    # Retrieve a reference Allrun script from the FAISS "Allrun" database.
    index_content = f"<index>\ncase name: {case_name}\ncase solver: {case_solver}\n</index>\n<directory_structure>\n{dir_structure}\n</directory_structure>"
    faiss_allrun = retrieve_faiss("openfoam_allrun_scripts", index_content, topk=config.searchdocs)
    allrun_reference = "Similar cases are ordered, with smaller numbers indicating greater similarity. For example, similar_case_1 is more similar than similar_case_2, and similar_case_2 is more similar than similar_case_3.\n"
    for idx, item in enumerate(faiss_allrun):
        allrun_reference += f"<similar_case_{idx + 1}>{item['full_content']}</similar_case_{idx + 1}>\n\n\n"
    
    case_path = os.path.join(case_dir, "similar_case.txt")
    
    tutorial_reference = faiss_detailed
    case_path_reference = case_path
    dir_structure_reference = dir_structure
    allrun_reference = allrun_reference
    
    save_file(case_path, f"{faiss_detailed}\n\n\n{allrun_reference}")
        

    # Step 4: Break down the work into smaller, manageable subtasks.
    decompose_system_prompt = (
        "You are an experienced Planner specializing in OpenFOAM projects. "
        "Your task is to break down the following user requirement into a series of smaller, manageable subtasks. "
        "For each subtask, identify the file name of the OpenFOAM input file (foamfile) and the corresponding folder name where it should be stored. "
        "Your final output must strictly follow the JSON schema below and include no additional keys or information:\n\n"
        "```\n"
        "{\n"
        "  \"subtasks\": [\n"
        "    {\n"
        "      \"file_name\": \"<string>\",\n"
        "      \"folder_name\": \"<string>\"\n"
        "    }\n"
        "    // ... more subtasks\n"
        "  ]\n"
        "}\n"
        "```\n\n"
        "Make sure that your output is valid JSON and strictly adheres to the provided schema."
        "Make sure you generate all the necessary files for the user's requirements."
    )

    decompose_user_prompt = (
        f"User Requirement: {user_requirement}\n\n"
        f"Reference Directory Structure (similar case): {dir_structure}\n\n{dir_counts_str}\n\n"        
        "Make sure you generate all the necessary files for the user's requirements."
        "Do not include any gmsh files like .geo etc. in the subtasks."
        "Only include blockMesh or snappyHexMesh if the user hasnt requested for gmsh mesh or user isnt using an external uploaded custom mesh"
        "Please generate the output as structured JSON."
    )
    
    decompose_resposne = state["llm_service"].invoke(decompose_user_prompt, decompose_system_prompt, pydantic_obj=OpenFOAMPlanPydantic)

    if len(decompose_resposne.subtasks) == 0:
        print("Failed to generate subtasks.")
        raise ValueError("Failed to generate subtasks.")

    print(f"Generated {len(decompose_resposne.subtasks)} subtasks.")

    subtasks = decompose_resposne.subtasks

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
        "subtasks": [{"file_name": subtask.file_name, "folder_name": subtask.folder_name} for subtask in subtasks],
        "mesh_type": mesh_type_value,
    }
