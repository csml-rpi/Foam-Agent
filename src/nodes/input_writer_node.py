# input_writer_node.py
import os
from utils import save_file, parse_context, retrieve_faiss, FoamPydantic, FoamfilePydantic
import re
from typing import List
from pydantic import BaseModel, Field

# 计算子任务的优先级，优先级用于排序subtasks，保证system、constant、0等文件夹优先生成
# system > constant > 0 > 其他
# 这样做可以保证依赖关系的正确性，例如0/U依赖constant/transportProperties
# 这里建议print每个subtask的优先级，方便调试

def compute_priority(subtask):
    priority = None
    if subtask["folder_name"] == "system":
        priority = 0
    elif subtask["folder_name"] == "constant":
        priority = 1
    elif subtask["folder_name"] == "0":
        priority = 2
    else:
        priority = 3
    print(f"subtask: {subtask}, priority: {priority}")
    return priority
        
# 解析Allrun脚本内容，只提取```包裹的代码部分
# 注意：如果没有匹配到，match为None，直接group会报错
# 建议加异常处理

def parse_allrun(text: str) -> str:
    match = re.search(r'```(.*?)```', text, re.DOTALL)
    if match is None:
        print(f"[parse_allrun] 未找到代码块, text={text[:100]}")
        return text.strip()  # fallback: 返回原始内容
    return match.group(1).strip() 

# 读取命令文件，返回命令字符串列表
# 这里假设每一行是一个命令

def retrieve_commands(command_path) -> str:
    with open(command_path, 'r') as file:
        commands = file.readlines()
    print(f"[retrieve_commands] commands: {commands}")
    return f"[{', '.join([command.strip() for command in commands])}]"
    
# 用于pydantic校验命令列表
class CommandsPydantic(BaseModel):
    commands: List[str] = Field(description="List of commands")

    
    
def input_writer_node(state):
    """
    InputWriter node: Generate the complete OpenFOAM foamfile.
    """
    # 1. 读取配置和子任务
    config = state["config"]
    subtasks = state["subtasks"]
    
    # 2. 按优先级排序subtasks，保证依赖顺序
    subtasks = sorted(subtasks, key=compute_priority)
    print(f"[input_writer_node] sorted subtasks: {subtasks}")
    
    writed_files = []  # 已生成的文件列表
    dir_structure = {}  # 目录结构，便于后续Allrun生成
    
    # 3. 依次生成每个foam文件
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
        
        # 取相似案例的参考文件内容
        similar_file_text = state["tutorial_reference"]
        
        # 生成完整foam文件的system prompt
        code_system_prompt = (
            "You are an expert in OpenFOAM simulation and numerical modeling."
            f"Your task is to generate a complete and functional file named: <file_name>{file_name}</file_name> within the <folder_name>{folder_name}</folder_name> directory. "
            "Ensure all required values are present and match with the files content already generated."
            "Before finalizing the output, ensure:\n"
            "- All necessary fields exist (e.g., if `nu` is defined in `constant/transportProperties`, it must be used correctly in `0/U`).\n"
            "- Cross-check field names between different files to avoid mismatches.\n"
            "- Ensure units and dimensions are correct** for all physical variables.\n"
            f"- Ensure case solver settings are consistent with the user's requirements. Available solvers are: {state['case_stats']['case_solver']}.\n"
            "Provide only the code—no explanations, comments, or additional text."
        )

        # 生成user prompt，包含用户需求、相似案例内容、已生成文件内容
        code_user_prompt = (
            f"User requirement: {state['user_requirement']}\n"
            f"Refer to the following similar case file content to ensure the generated file aligns with the user requirement:\n<similar_case_reference>{similar_file_text}</similar_case_reference>\n"
            f"Similar case reference is always correct. If you find the user requirement is very consistent with the similar case reference, you should use the similar case reference as the template to generate the file."
            f"Just modify the necessary parts to make the file complete and functional."
            "Please ensure that the generated file is complete, functional, and logically sound."
            "Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence."
        )
        if len(writed_files) > 0:
            code_user_prompt += f"The following are files content already generated: {str(writed_files)}\n\n\nYou should ensure that the new file is consistent with the previous files. Such as boundary conditions, mesh settings, etc."

        # 4. 调用llm_service生成foam文件内容
        print(f"[input_writer_node] code_user_prompt: {code_user_prompt[:200]}...")
        generation_response = state["llm_service"].invoke(code_user_prompt, code_system_prompt)
        print(f"[input_writer_node] generation_response: {generation_response[:200]}...")
        
        # 首先对解析生成内容，去除多余内容
        code_context = parse_context(generation_response)
        save_file(file_path, code_context)
        print(f"[input_writer_node] save file: {file_path}")
        
        #然后还需要用pydantic校验一下，确保生成的内容符合要求
        writed_files.append(FoamfilePydantic(file_name=file_name, folder_name=folder_name, content=code_context))
    
    # 5. 生成Allrun脚本
    case_dir = state["case_dir"]
    allrun_file_path = os.path.join(case_dir, "Allrun")
    print(f"[input_writer_node] allrun_file_path: {allrun_file_path}")
    if os.path.exists(allrun_file_path):
        print("Warning: Allrun file exists. Overwriting.")
    
    # 从数据库读取可用命令
    commands = retrieve_commands(f"{config.database_path}/raw/openfoam_commands.txt")
    print(f"[input_writer_node] available commands: {commands}")
    
    # 如果有自定义网格命令，加入提示
    mesh_commands_info = ""
    if state.get("custom_mesh_used") and state.get("mesh_commands"):
        mesh_commands_info = f"\nCustom mesh commands to include: {state['mesh_commands']}"
        print(f"Including custom mesh commands: {state['mesh_commands']}")
    
    # 生成Allrun命令列表的system prompt
    command_system_prompt = (
        "You are an expert in OpenFOAM. The user will provide a list of available commands. "
        "Your task is to generate only the necessary OpenFOAM commands required to create an Allrun script for the given user case, based on the provided directory structure. "
        "If custom mesh commands are provided, include them in the appropriate order (typically after blockMesh or instead of blockMesh if custom mesh is used). "
        "Return only the list of commands—no explanations, comments, or additional text."
    )
    
    # 生成Allrun命令列表的user prompt
    command_user_prompt = (
        f"Available OpenFOAM commands for the Allrun script: {commands}\n"
        f"Case directory structure: {dir_structure}\n"
        f"User case information: {state['case_info']}\n"
        f"Reference Allrun scripts from similar cases: {state['allrun_reference']}\n"
        f"{mesh_commands_info}\n"
        "Generate only the required OpenFOAM command list—no extra text."
    )
    print("--------------------------------")
    print(f"[input_writer_node] command_user_prompt: {command_user_prompt}")
    print("--------------------------------")

    # 6. 调用llm_service生成命令列表
    command_response = state["llm_service"].invoke(command_user_prompt, command_system_prompt, pydantic_obj=CommandsPydantic)
    print(f"[input_writer_node] command_response: {command_response}")

    if len(command_response.commands) == 0:
        print("Failed to generate subtasks.")
        raise ValueError("Failed to generate subtasks.")

    print(f"Need {len(command_response.commands)} commands.")
    
    # 7. 查询每个命令的帮助信息，便于Allrun脚本生成
    commands_help = []
    for command in command_response.commands:
        command_help = retrieve_faiss("openfoam_command_help", command, topk=config.searchdocs)
        print(f"[input_writer_node] command: {command}, help: {command_help}") # 比如对于简单的case，可能只有blockMesh和icoFoam，所以这里可能只有两个命令
        commands_help.append(command_help[0]['full_content'])
    commands_help = "\n".join(commands_help)

    # 8. 生成Allrun脚本的system prompt
    allrun_system_prompt = (
        "You are an expert in OpenFOAM. Generate an Allrun script based on the provided details."
        f"Available commands with descriptions: {commands_help}\n\n"
        f"Reference Allrun scripts from similar cases: {state['allrun_reference']}\n\n"
        "If custom mesh commands are provided, make sure to include them in the appropriate order in the Allrun script."
    )
    
    # 生成Allrun脚本的user prompt
    allrun_user_prompt = (
        f"User requirement: {state['user_requirement']}\n"
        f"Case directory structure: {dir_structure}\n"
        f"User case infomation: {state['case_info']}\n"
        f"{mesh_commands_info}\n"
        "All run scripts for these similar cases are for reference only and may not be correct, as there might be a different case solver or have a different directory structure. " 
        "You need to rely on your OpenFOAM and physics knowledge to discern this, and pay more attention to user requirements, " 
        "as your ultimate goal is to fulfill the user's requirements and generate an allrun script that meets those requirements."
        "Generate the Allrun script strictly based on the above information. Do not include explanations, comments, or additional text. Put the code in ``` tags."
    )
    print(f"[input_writer_node] allrun_user_prompt: {allrun_user_prompt}")
    
    # 9. 调用llm_service生成Allrun脚本
    allrun_response = state["llm_service"].invoke(allrun_user_prompt, allrun_system_prompt)
    print(f"[input_writer_node] allrun_response: {allrun_response[:200]}...")
    
    # 解析Allrun脚本内容
    allrun_script = parse_allrun(allrun_response)
    save_file(allrun_file_path, allrun_script)
    print(f"[input_writer_node] save Allrun: {allrun_file_path}")
    
    writed_files.append(FoamfilePydantic(file_name="Allrun", folder_name="./", content=allrun_script))
    foamfiles = FoamPydantic(list_foamfile=writed_files)
    
    # Return updated state
    return {
        **state,
        "dir_structure": dir_structure,
        "commands": command_response.commands,
        "foamfiles": foamfiles
    }
