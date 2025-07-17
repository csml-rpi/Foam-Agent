# main.py
# Foam-Agent 主程序文件
# 该文件实现了基于LangGraph的OpenFOAM工作流自动化系统
# 通过多智能体协作完成从用户需求到CFD仿真的全流程自动化

from dataclasses import dataclass, field
from typing import List, Optional, TypedDict, Literal
from langgraph.graph import StateGraph, START, END  # LangGraph工作流图相关组件
from langgraph.types import Command
import argparse  # 命令行参数解析
from pathlib import Path  # 路径处理
from utils import LLMService  # LLM服务封装

# 导入配置和各个工作流节点
from config import Config
from nodes.architect_node import architect_node      # 架构师节点：解析用户需求并规划工作
from nodes.meshing_node import meshing_node          # 网格节点：处理自定义网格文件
from nodes.input_writer_node import input_writer_node # 输入文件编写节点：生成OpenFOAM配置文件
from nodes.local_runner_node import local_runner_node # 本地运行节点：执行本地仿真
from nodes.reviewer_node import reviewer_node        # 审查节点：分析错误并提供修复建议
from nodes.visualization_node import visualization_node # 可视化节点：生成结果图表
from nodes.hpc_runner_node import hpc_runner_node    # HPC运行节点：在高性能计算集群上执行仿真

# 导入路由函数和工作流状态定义
from router_func import (
    GraphState,                    # 工作流状态数据结构
    route_after_architect,         # 架构师节点后的路由逻辑
    route_after_input_writer,      # 输入文件编写节点后的路由逻辑
    route_after_runner,            # 运行节点后的路由逻辑
    route_after_reviewer           # 审查节点后的路由逻辑
)
import json

def create_foam_agent_graph() -> StateGraph:
    """
    创建OpenFOAM智能体工作流图
    
    该函数定义了整个CFD仿真工作流的节点和边，构建了一个有向图结构：
    - START -> architect: 从架构师节点开始
    - architect -> meshing/input_writer: 根据是否需要自定义网格决定路径
    - meshing -> input_writer: 网格处理后进入输入文件编写
    - input_writer -> local_runner/hpc_runner: 根据配置选择运行环境
    - runner -> reviewer/visualization/END: 根据运行结果决定下一步
    - reviewer -> input_writer/visualization/END: 错误修复后重新运行或结束
    
    Returns:
        StateGraph: 编译后的工作流图对象
    """
    
    # 创建工作流图实例，使用GraphState作为状态类型
    workflow = StateGraph(GraphState)
    
    # 添加所有工作流节点
    workflow.add_node("architect", architect_node)        # 架构师：需求分析和规划
    workflow.add_node("meshing", meshing_node)            # 网格处理：自定义网格文件处理
    workflow.add_node("input_writer", input_writer_node)  # 输入文件编写：生成OpenFOAM配置
    workflow.add_node("local_runner", local_runner_node)  # 本地运行：本地环境执行仿真
    workflow.add_node("hpc_runner", hpc_runner_node)      # HPC运行：集群环境执行仿真
    workflow.add_node("reviewer", reviewer_node)          # 审查：错误分析和修复
    workflow.add_node("visualization", visualization_node) # 可视化：结果图表生成
    
    # 定义工作流的边（节点间的连接关系）
    workflow.add_edge(START, "architect")  # 工作流从架构师节点开始
    
    # 条件边：根据架构师节点的输出决定下一步路径
    workflow.add_conditional_edges("architect", route_after_architect)
    
    # 固定边：网格处理后直接进入输入文件编写
    workflow.add_edge("meshing", "input_writer")
    
    # 条件边：根据输入文件编写结果决定运行环境
    workflow.add_conditional_edges("input_writer", route_after_input_writer)
    
    # 条件边：根据运行结果决定下一步（错误检查、可视化或结束）
    workflow.add_conditional_edges("hpc_runner", route_after_runner)
    workflow.add_conditional_edges("local_runner", route_after_runner)
    
    # 条件边：根据审查结果决定是否重新运行或结束
    workflow.add_conditional_edges("reviewer", route_after_reviewer)
    
    # 固定边：可视化完成后结束工作流
    workflow.add_edge("visualization", END)

    # # 打印工作流图结构
    # print(workflow.draw_mermaid_png())
    # # ascii graph
    # print(workflow.get_graph().draw_ascii())
    
    return workflow

def initialize_state(user_requirement: str, config: Config) -> GraphState:
    """
    初始化工作流状态
    
    该函数创建并初始化工作流的初始状态，包括：
    - 用户需求文本
    - 系统配置参数
    - 案例统计信息（从数据库加载）
    - 各种中间状态字段的初始值
    
    Args:
        user_requirement (str): 用户的CFD仿真需求描述
        config (Config): 系统配置对象
        
    Returns:
        GraphState: 初始化完成的工作流状态对象
    """
    # 从数据库加载OpenFOAM案例统计信息
    # 这些统计信息用于帮助LLM理解可用的案例类型和配置选项
    case_stats_file_path = f"{config.database_path}/raw/openfoam_case_stats.json"
    print(f"📊 正在加载案例统计信息...")
    print(f"    📁 文件路径: {case_stats_file_path}")
    
    try:
        with open(case_stats_file_path, "r", encoding="utf-8") as f:
            case_stats = json.load(f)
        
        print(f"✅ 成功加载案例统计信息:")
        print(f"    🏷️  案例领域 (case_domain): {len(case_stats['case_domain'])} 个")
        print(f"        📋 可选值: {case_stats['case_domain']}")
        print(f"    🏷️  案例类别 (case_category): {len(case_stats['case_category'])} 个")
        print(f"        📋 可选值: {case_stats['case_category']}")
        print(f"    🏷️  案例名称 (case_name): {len(case_stats['case_name'])} 个")
        print(f"        📋 可选值: {case_stats['case_name']}")
        print(f"    🏷️  案例求解器 (case_solver): {len(case_stats['case_solver'])} 个")
        print(f"        📋 可选值: {case_stats['case_solver']}")
        
        print(f"    💡 这些统计信息来自OpenFOAM教程案例的自动分析")
        print(f"    💡 用于限制LLM输出格式，确保生成的案例信息符合现有案例库")
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到案例统计文件 {case_stats_file_path}")
        print(f"    💡 请先运行数据库初始化脚本生成统计信息")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ 错误: 案例统计文件格式错误: {e}")
        raise
    except Exception as e:
        print(f"❌ 错误: 加载案例统计信息失败: {e}")
        raise
    
    # 创建初始状态对象，包含所有必要的字段
    state = GraphState(
        # 核心输入参数
        user_requirement=user_requirement,  # 用户需求文本
        config=config,                      # 系统配置
        
        # 案例基本信息（将在工作流中逐步填充）
        case_dir="",                        # 案例目录路径
        tutorial="",                        # 参考教程名称
        case_name="",                       # 案例名称
        subtasks=[],                        # 子任务列表
        current_subtask_index=0,            # 当前子任务索引
        
        # 错误处理相关
        error_command=None,                 # 出错的命令
        error_content=None,                 # 错误内容
        loop_count=0,                       # 循环次数（用于防止无限循环）
        
        # LLM服务实例
        llm_service=LLMService(config),     # 大语言模型服务
        
        # 参考信息（从FAISS数据库检索）
        case_stats=case_stats,              # 案例统计信息
        tutorial_reference=None,            # 教程参考信息
        case_path_reference=None,           # 案例路径参考
        dir_structure_reference=None,       # 目录结构参考
        case_info=None,                     # 案例详细信息
        allrun_reference=None,              # allrun脚本参考
        
        # 工作流中间状态
        dir_structure=None,                 # 目录结构
        commands=None,                      # 执行的命令列表
        foamfiles=None,                     # OpenFOAM文件内容
        error_logs=None,                    # 错误日志
        history_text=None,                  # 历史文本
        
        # 案例分类信息
        case_domain=None,                   # 案例领域（如incompressible, compressible等）
        case_category=None,                 # 案例类别
        case_solver=None,                   # 求解器类型
        
        # 网格相关字段
        mesh_info=None,                     # 网格信息
        mesh_commands=None,                 # 网格处理命令
        mesh_file_destination=None,         # 网格文件目标位置
        custom_mesh_used=None               # 是否使用自定义网格
    )
    
    return state

def main(user_requirement: str, config: Config):
    """
    主函数：运行OpenFOAM工作流
    
    这是整个系统的核心入口函数，负责：
    1. 创建工作流图并编译
    2. 初始化工作流状态
    3. 执行工作流
    4. 处理结果和统计信息
    
    Args:
        user_requirement (str): 用户的CFD仿真需求描述
        config (Config): 系统配置对象
    """
    
    # 步骤1：创建并编译工作流图
    workflow = create_foam_agent_graph()
    app = workflow.compile()  # 编译工作流图，生成可执行的应用
    
    # 步骤2：初始化工作流状态
    initial_state = initialize_state(user_requirement, config)
    
    print("Starting Foam-Agent...")  # 开始执行提示
    
    # 步骤3：执行工作流
    try:
        # 调用工作流，传入初始状态
        result = app.invoke(initial_state)
        print("Workflow completed successfully!")  # 成功完成提示
        
        # 步骤4：输出最终统计信息
        # 如果LLM服务存在，打印使用统计（如API调用次数、token使用量等）
        if result.get("llm_service"):
            result["llm_service"].print_statistics()
        
        # 注释掉的调试信息：打印最终状态
        # print(f"Final state: {result}")
        
    except Exception as e:
        # 异常处理：捕获并报告工作流执行错误
        print(f"Workflow failed with error: {e}")
        raise  # 重新抛出异常，便于调试

if __name__ == "__main__":
    """
    程序入口点：命令行接口
    
    当直接运行此脚本时，会：
    1. 解析命令行参数
    2. 加载用户需求文件
    3. 初始化配置
    4. 启动主工作流
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Run the OpenFOAM workflow"  # 参数解析器描述
    )
    
    # 定义命令行参数
    parser.add_argument(
        "--prompt_path",  # 用户需求文件路径参数
        type=str,
        default=f"{Path(__file__).parent.parent}/user_requirement.txt",  # 默认路径
        help="User requirement file path for the workflow.",  # 帮助信息，
    )
    parser.add_argument(
        "--output_dir",  # 输出目录参数
        type=str,
        default="",  # 默认为空，使用配置中的默认目录
        help="Output directory for the workflow.",  # 帮助信息
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    print(args)  # 打印解析后的参数
    
    # 初始化系统配置
    config = Config()
    
    # 如果指定了输出目录，更新配置
    if args.output_dir != "":
        config.case_dir = args.output_dir
    
    # 读取用户需求文件
    with open(args.prompt_path, 'r') as f:
        user_requirement = f.read()
    
    # 启动主工作流
    main(user_requirement, config)
