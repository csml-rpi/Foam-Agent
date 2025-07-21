# main.py
# Foam-Agent 主程序文件
# 该文件实现了基于LangGraph的OpenFOAM工作流自动化系统
# 通过多智能体协作完成从用户需求到CFD仿真的全流程自动化

from dataclasses import dataclass, field
from typing import List, Optional, TypedDict, Literal
from langgraph.graph import StateGraph, START, END  # LangGraph工作流图相关组件
from langgraph.types import Command
import argparse
from pathlib import Path
from utils import LLMService, GraphState

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
    route_after_architect, 
    route_after_input_writer, 
    route_after_runner, 
    route_after_reviewer
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

def initialize_state(user_requirement: str, config: Config, custom_mesh_path: Optional[str] = None) -> GraphState:
    case_stats = json.load(open(f"{config.database_path}/raw/openfoam_case_stats.json", "r"))
    mesh_type = "custom_mesh" if custom_mesh_path else "standard_mesh"
    state = GraphState(
        user_requirement=user_requirement,
        config=config,
        case_dir="",
        tutorial="",
        case_name="",
        subtasks=[],
        current_subtask_index=0,
        error_command=None,
        error_content=None,
        loop_count=0,
        llm_service=LLMService(config),
        case_stats=case_stats,
        tutorial_reference=None,
        case_path_reference=None,
        dir_structure_reference=None,
        case_info=None,
        allrun_reference=None,
        dir_structure=None,
        commands=None,
        foamfiles=None,
        error_logs=None,
        history_text=None,
        case_domain=None,
        case_category=None,
        case_solver=None,
        mesh_info=None,
        mesh_commands=None,
        custom_mesh_used=None,
        mesh_type=mesh_type,
        custom_mesh_path=custom_mesh_path,
        review_analysis=None,
        input_writer_mode="initial"
    )
    if custom_mesh_path:
        print(f"Custom mesh path: {custom_mesh_path}")
    else:
        print("No custom mesh path provided.")
    return state

def main(user_requirement: str, config: Config, custom_mesh_path: Optional[str] = None):
    """Main function to run the OpenFOAM workflow."""
    
    # 步骤1：创建并编译工作流图
    workflow = create_foam_agent_graph()
    app = workflow.compile()  # 编译工作流图，生成可执行的应用
    
    # Initialize the state
    initial_state = initialize_state(user_requirement, config, custom_mesh_path)
    
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
        description="Run the OpenFOAM workflow"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=f"{Path(__file__).parent.parent}/user_requirement.txt",
        help="User requirement file path for the workflow.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for the workflow.",
    )
    parser.add_argument(
        "--custom_mesh_path",
        type=str,
        default=None,
        help="Path to custom mesh file (e.g., .msh, .stl, .obj). If not provided, no custom mesh will be used.",
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
    
    main(user_requirement, config, args.custom_mesh_path)
