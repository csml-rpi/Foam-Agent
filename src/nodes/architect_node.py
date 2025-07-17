# architect_node.py
"""
架构节点模块：负责解析用户需求、查找相似案例并分解任务
主要功能：
1. 将用户需求转换为标准化的案例描述
2. 从FAISS数据库中检索相似的参考案例
3. 将工作分解为可管理的子任务
4. 创建案例目录结构
"""

import os
import re
from utils import save_file, retrieve_faiss, parse_directory_structure
from pydantic import BaseModel, Field
from typing import List
import shutil

# 数据模型定义
class CaseSummaryPydantic(BaseModel):
    """
    案例摘要数据模型：用于标准化用户需求解析结果
    包含案例的基本信息：名称、领域、类别和求解器
    BaseModel：Pydantic的基础类，提供数据验证和序列化功能
    Field(description=...)：为每个字段提供描述，这些描述会被传递给LLM作为输出格式指导
    字段类型str：限制输出必须是字符串类型
    """
    case_name: str = Field(description="name of the case")
    case_domain: str = Field(description="domain of the case, case domain must be one of [basic,combustion,compressible,discreteMethods,DNS,electromagnetics,financial,heatTransfer,incompressible,lagrangian,mesh,multiphase,resources,stressAnalysis].")
    case_category: str = Field(description="category of the case")
    case_solver: str = Field(description="solver of the case")


class SubtaskPydantic(BaseModel):
    """
    子任务数据模型：定义每个子任务的文件和文件夹信息
    """
    file_name: str = Field(description="Name of the OpenFOAM input file")
    folder_name: str = Field(description="Name of the folder where the foamfile should be stored")

class OpenFOAMPlanPydantic(BaseModel):
    """
    OpenFOAM计划数据模型：包含所有子任务的列表
    """
    subtasks: List[SubtaskPydantic] = Field(description="List of subtasks, each with its corresponding file and folder names")


def architect_node(state):
    """
    Architect node: Parse the user requirement to a standard case description,
    finds a similar reference case from the FAISS databases, and splits the work into subtasks.
    Updates state with:
      - case_dir, tutorial, case_name, subtasks.
    
    架构节点主函数：解析用户需求、查找相似案例并分解任务
    执行流程：
    1. 解析用户需求为标准化案例描述
    2. 确定案例目录
    3. 从FAISS数据库检索相似参考案例
    4. 将工作分解为子任务
    """
    # 从状态中获取配置和用户需求
    config = state["config"]
    user_requirement = state["user_requirement"]
    
    print(f"=== 开始架构节点处理 ===")
    print(f"用户需求: {user_requirement}")
    print(f"配置信息: {config}")

    # 步骤1: 将用户需求转换为标准化案例描述
    print(f"\n--- 步骤1: 解析用户需求 ---")
    parse_system_prompt = ("Please transform the following user requirement into a standard case description using a structured format."
                           "The key elements should include case name, case domain, case category, and case solver."
                           f"Note: case domain must be one of {state['case_stats']['case_domain']}."
                           f"Note: case category must be one of {state['case_stats']['case_category']}."
                           f"Note: case solver must be one of {state['case_stats']['case_solver']}."
                           )
    parse_user_prompt = f"User requirement: {user_requirement}."
    
    # 调用LLM服务解析用户需求
    parse_response = state["llm_service"].invoke(parse_user_prompt, parse_system_prompt, pydantic_obj=CaseSummaryPydantic)
    
    # 提取解析结果并处理案例名称（替换空格为下划线）
    case_name = parse_response.case_name.replace(" ", "_")
    case_domain = parse_response.case_domain
    case_category = parse_response.case_category
    case_solver = parse_response.case_solver
    
    # 打印解析结果
    print(f"解析结果:")
    print(f"  - 案例名称: {case_name}")
    print(f"  - 案例领域: {case_domain}")
    print(f"  - 案例类别: {case_category}")
    print(f"  - 案例求解器: {case_solver}")
    
    # 步骤2: 确定案例目录
    print(f"\n--- 步骤2: 确定案例目录 ---")
    if config.case_dir != "":
        # 如果配置中指定了案例目录，直接使用
        case_dir = config.case_dir
        print(f"使用配置指定的案例目录: {case_dir}")
    else:
        # 否则根据运行次数生成目录名
        if config.run_times > 1:
            case_dir = os.path.join(config.run_directory, f"{case_name}_{config.run_times}")
        else:
            case_dir = os.path.join(config.run_directory, case_name)
        print(f"生成案例目录: {case_dir}")
    
    # 如果目录已存在，删除并重新创建
    if os.path.exists(case_dir):
        print(f"警告: 案例目录 {case_dir} 已存在，正在覆盖...")
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)
    
    print(f"成功创建案例目录: {case_dir}")

    # 步骤3: 从FAISS数据库检索相似参考案例
    print(f"\n--- 步骤3: 检索相似参考案例 ---")
    
    # 构建案例信息查询字符串
    case_info = f"case name: {case_name}\ncase domain: {case_domain}\ncase category: {case_category}\ncase solver: {case_solver}"
    print(f"案例信息查询: {case_info}")
    
    # 第一次检索：基于案例信息检索结构
    print(f"正在检索案例结构...")
    faiss_structure_results = retrieve_faiss("openfoam_tutorials_structure", case_info, topk=config.searchdocs)
    print(f"检索到 {len(faiss_structure_results)} 个结构文档:")
    for i, result in enumerate(faiss_structure_results):
        print(f"  文档 {i+1}: case_name={result['case_name']}, case_category={result['case_category']}, case_solver={result['case_solver']}")
    faiss_structure = faiss_structure_results[0]['full_content']
    print(f"检索到结构信息长度: {len(faiss_structure)} 字符")
    
    # 第二次检索：基于结构信息检索详细内容
    print(f"正在检索详细内容...")
    faiss_detailed_results = retrieve_faiss("openfoam_tutorials_details", faiss_structure, topk=config.searchdocs)
    print(f"检索到 {len(faiss_detailed_results)} 个详细文档:")
    for i, result in enumerate(faiss_detailed_results):
        print(f"  文档 {i+1}: case_name={result['case_name']}, case_category={result['case_category']}, case_solver={result['case_solver']}")
    faiss_detailed = faiss_detailed_results[0]['full_content'] # 选择第一个文档，后续可以优化选择最相似的文档
    print(f"检索到详细内容长度: {len(faiss_detailed)} 字符")
    
    # 提取目录结构信息
    dir_structure = re.search(r"<directory_structure>(.*?)</directory_structure>", faiss_detailed, re.DOTALL).group(1).strip()
    print(f"提取的目录结构: {dir_structure}")
    
    # 解析目录结构，统计各目录文件数量
    dir_counts = parse_directory_structure(dir_structure)
    dir_counts_str = ',\n'.join([f"There are {count} files in Directory: {directory}" for directory, count in dir_counts.items()])
    print(f"目录文件统计: {dir_counts_str}")
    
    # 检索Allrun脚本参考
    print(f"正在检索Allrun脚本...")
    index_content = f"<index>\ncase name: {case_name}\ncase solver: {case_solver}</index>\n<directory_structure>{dir_structure}</directory_structure>"
    faiss_allrun = retrieve_faiss("openfoam_allrun_scripts", index_content, topk=config.searchdocs)
    print(f"检索到 {len(faiss_allrun)} 个Allrun脚本:")
    for i, result in enumerate(faiss_allrun):
        print(f"  脚本 {i+1}: case_name={result['case_name']}, case_category={result['case_category']}, case_solver={result['case_solver']}")
    
    # 构建Allrun参考信息
    allrun_reference = "Similar cases are ordered, with smaller numbers indicating greater similarity. For example, similar_case_1 is more similar than similar_case_2, and similar_case_2 is more similar than similar_case_3.\n"
    for idx, item in enumerate(faiss_allrun):
        allrun_reference += f"<similar_case_{idx + 1}>{item['full_content']}</similar_case_{idx + 1}>\n\n\n"
    
    # 保存参考案例信息到文件
    case_path = os.path.join(case_dir, "similar_case.txt")
    print(f"保存参考案例信息到: {case_path}")

    # exit()  # 注释掉这行，让程序继续执行

    # TODO update all information to faiss_detailed
    tutorial_reference = faiss_detailed
    case_path_reference = case_path
    dir_structure_reference = dir_structure
    allrun_reference = allrun_reference
    
    save_file(case_path, f"{faiss_detailed}\n\n\n{allrun_reference}") # 所以它这里其实保存的是第一个faiss_detailed的文档，以及allrun_reference的文档
        

    # 步骤4: 将工作分解为可管理的子任务
    print(f"\n--- 步骤4: 分解子任务 ---")
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
        "Please generate the output as structured JSON."
    )
    
    print(f"正在生成子任务...")
    decompose_resposne = state["llm_service"].invoke(decompose_user_prompt, decompose_system_prompt, pydantic_obj=OpenFOAMPlanPydantic)

    # 验证子任务生成结果
    if len(decompose_resposne.subtasks) == 0:
        print("错误: 未能生成子任务")
        raise ValueError("Failed to generate subtasks.")

    print(f"成功生成 {len(decompose_resposne.subtasks)} 个子任务:")
    for i, subtask in enumerate(decompose_resposne.subtasks):
        print(f"  子任务 {i+1}: 文件={subtask.file_name}, 文件夹={subtask.folder_name}")

    subtasks = decompose_resposne.subtasks

    # 返回更新后的状态
    print(f"\n--- 架构节点处理完成 ---")
    print(f"返回状态包含以下关键信息:")
    print(f"  - case_name: {case_name}")
    print(f"  - case_dir: {case_dir}")
    print(f"  - subtasks数量: {len(subtasks)}")
    
    return {
        **state,
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
        "subtasks": [{"file_name": subtask.file_name, "folder_name": subtask.folder_name} for subtask in subtasks]
    }
