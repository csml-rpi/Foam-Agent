from typing import TypedDict, List, Optional
from config import Config
from utils import LLMService, GraphState
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


def llm_requires_custom_mesh(state: GraphState) -> int:
    """
    使用LLM判断用户是否需要自定义网格
    
    功能：分析用户需求，确定是否需要使用自定义网格文件
    执行逻辑：
    1. 提取用户需求文本
    2. 构建系统提示词，定义判断标准
    3. 调用LLM服务进行分析
    4. 根据LLM响应判断是否需要自定义网格
    
    Args:
        state: 包含用户需求和LLM服务的当前图状态
        
    Returns:
        int: 1 if custom mesh is required, 2 if gmsh mesh is required, 0 otherwise
    """
    user_requirement = state["user_requirement"]
    print(f"[DEBUG] 分析用户需求是否需要自定义网格: {user_requirement[:100]}...")
    
    # 系统提示词：定义LLM的角色和判断标准
    system_prompt = (
        "You are an expert in OpenFOAM workflow analysis. "
        "Analyze the user requirement to determine if they want to use a custom mesh file. "
        "Look for keywords like: custom mesh, mesh file, .msh, .stl, .obj, gmsh, snappyHexMesh, "
        "or any mention of importing/using external mesh files. "
        "If the user explicitly mentions or implies they want to use a custom mesh file, return 'custom_mesh'. "
        "If they want to use standard OpenFOAM mesh generation (blockMesh, snappyHexMesh with STL, etc.), return 'standard_mesh'. "
        "Look for keywords like gmsh and determine if they want to create mesh using gmsh. If they want to create mesh using gmsh, return 'gmsh_mesh'. "
        "Be conservative - if unsure, assume 'standard_mesh' unless clearly specified otherwise."
        "Only return 'custom_mesh' or 'standard_mesh' or 'gmsh_mesh'. Don't return anything else."
    )
    
    # 用户提示词：具体的问题描述
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "Determine if the user wants to use 'custom_mesh' or 'standard_mesh' or 'gmsh_mesh'. "
    )
    
    # 检查LLM服务是否可用
    if state["llm_service"] is None:
        print("[WARNING] LLM服务不可用，默认使用标准网格生成")
        return False
    
    # 调用LLM服务进行分析
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    if "custom_mesh" in response.lower():
        return 1
    elif "gmsh_mesh" in response.lower():
        return 2
    else:
        return 0


def llm_requires_hpc(state: GraphState) -> bool:
    """
    使用LLM判断用户是否需要HPC/集群执行
    
    功能：分析用户需求，确定是否需要在高性能计算集群上运行
    执行逻辑：
    1. 提取用户需求文本
    2. 构建系统提示词，定义HPC相关的关键词
    3. 调用LLM服务进行分析
    4. 根据LLM响应判断是否需要HPC执行
    
    Args:
        state: 包含用户需求和LLM服务的当前图状态
        
    Returns:
        bool: 如果需要HPC执行返回True，否则返回False
    """
    user_requirement = state["user_requirement"]
    print(f"[DEBUG] 分析用户需求是否需要HPC执行: {user_requirement[:100]}...")
    
    # 系统提示词：定义HPC相关的判断标准
    system_prompt = (
        "You are an expert in OpenFOAM workflow analysis. "
        "Analyze the user requirement to determine if they want to run the simulation on HPC (High Performance Computing) or locally. "
        "Look for keywords like: HPC, cluster, supercomputer, SLURM, PBS, job queue, "
        "parallel computing, distributed computing, or any mention of running on remote systems. "
        "If the user explicitly mentions or implies they want to run on HPC/cluster, return 'hpc_run'. "
        "If they want to run locally or don't specify, return 'local_run'. "
        "Be conservative - if unsure, assume local run unless clearly specified otherwise."
        "Only return 'hpc_run' or 'local_run'. Don't return anything else."
    )
    
    # 用户提示词：具体的问题描述
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "return 'hpc_run' or 'local_run'"
    )
    
    # 检查LLM服务是否可用
    if state["llm_service"] is None:
        print("[WARNING] LLM服务不可用，默认本地执行")
        return False
    
    # 调用LLM服务进行分析
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    print(f"[DEBUG] LLM响应: {response}")
    
    result = "hpc_run" in response.lower()
    print(f"[DEBUG] 是否需要HPC执行: {result}")
    return result


def llm_requires_visualization(state: GraphState) -> bool:
    """
    使用LLM判断用户是否需要可视化
    
    功能：分析用户需求，确定是否需要结果可视化
    执行逻辑：
    1. 提取用户需求文本
    2. 构建系统提示词，定义可视化相关的关键词
    3. 调用LLM服务进行分析
    4. 根据LLM响应判断是否需要可视化
    
    Args:
        state: 包含用户需求和LLM服务的当前图状态
        
    Returns:
        bool: 如果需要可视化返回True，否则返回False
    """
    user_requirement = state["user_requirement"]
    print(f"[DEBUG] 分析用户需求是否需要可视化: {user_requirement[:100]}...")
    
    # 系统提示词：定义可视化相关的判断标准
    system_prompt = (
        "You are an expert in OpenFOAM workflow analysis. "
        "Analyze the user requirement to determine if they want visualization of results. "
        "Look for keywords like: plot, visualize, graph, chart, contour, streamlines, paraview, post-processing."
        "Only if the user explicitly mentions they want visualization, return 'yes_visualization'. "
        "If they don't mention visualization or only want to run the simulation, return 'no_visualization'. "
        "Be conservative - if unsure, assume visualization is wanted unless clearly specified otherwise."
        "Only return 'yes_visualization' or 'no_visualization'. Don't return anything else."
    )
    
    # 用户提示词：具体的问题描述
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "return 'yes_visualization' or 'no_visualization'"
    )
    
    # 检查LLM服务是否可用
    if state["llm_service"] is None:
        print("[WARNING] LLM服务不可用，默认需要可视化")
        return True
    
    # 调用LLM服务进行分析
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    return "yes_visualization" in response.lower()


def route_after_architect(state: GraphState):
    """
    Route after architect node based on whether user wants custom mesh.
    For current version, if user wants custom mesh, user should be able to provide a path to the mesh file.
    """
    mesh_type = state.get("mesh_type", "standard_mesh")
    if mesh_type == "custom_mesh":
        print("Router: Custom mesh requested. Routing to meshing node.")
        return "meshing"
    elif mesh_type == "gmsh_mesh":
        print("Router: GMSH mesh requested. Routing to meshing node.")
        return "meshing"
    else:
        print("Router: Standard mesh generation. Routing to input_writer node.")
        return "input_writer"


def route_after_input_writer(state: GraphState):
    """
    输入文件编写节点后的路由决策
    
    功能：根据用户是否需要HPC执行来决定下一步的执行路径
    执行逻辑：
    1. 调用LLM判断是否需要HPC执行
    2. 如果需要HPC执行，路由到HPC运行节点
    3. 否则路由到本地运行节点
    
    Args:
        state: 当前图状态
        
    Returns:
        str: 下一个节点的名称
    """
    print("[DEBUG] 执行输入文件编写节点后的路由决策")
    
    if llm_requires_hpc(state):
        print("LLM determined: HPC run requested. Routing to hpc_runner node.")
        return "hpc_runner"
    else:
        print("LLM determined: Local run requested. Routing to local_runner node.")
        return "local_runner"


def route_after_runner(state: GraphState):
    """
    运行节点后的路由决策
    
    功能：根据运行结果和用户需求决定下一步的执行路径
    执行逻辑：
    1. 检查是否有错误日志，如果有则路由到审查节点
    2. 如果没有错误且用户需要可视化，路由到可视化节点
    3. 否则结束工作流
    
    Args:
        state: 当前图状态
        
    Returns:
        str: 下一个节点的名称或END
    """
    print("[DEBUG] 执行运行节点后的路由决策")
    
    # 检查错误日志
    error_logs = state.get("error_logs")
    print(f"[DEBUG] 错误日志数量: {len(error_logs) if error_logs else 0}")
    
    if error_logs and len(error_logs) > 0:
        print(f"[DEBUG] 发现错误，路由到审查节点。错误数量: {len(error_logs)}")
        return "reviewer"
    elif llm_requires_visualization(state):
        print("[DEBUG] 无错误且需要可视化，路由到可视化节点")
        return "visualization"
    else:
        print("[DEBUG] 无错误且不需要可视化，结束工作流")
        return END


def route_after_reviewer(state: GraphState):
    """
    审查节点后的路由决策
    
    功能：根据循环次数和用户需求决定是否继续修复错误或结束工作流
    执行逻辑：
    1. 检查当前循环次数是否达到最大限制
    2. 如果达到最大限制，根据用户需求决定是否进行可视化后结束
    3. 如果未达到最大限制，增加循环计数并继续修复错误
    
    Args:
        state: 当前图状态
        
    Returns:
        str: 下一个节点的名称或END
    """
    print("[DEBUG] 执行审查节点后的路由决策")
    
    # 获取当前循环次数和最大循环次数
    loop_count = state.get("loop_count", 0)
    max_loop = state["config"].max_loop
    print(f"[DEBUG] 当前循环次数: {loop_count}, 最大循环次数: {max_loop}")
    
    if loop_count >= max_loop:
        print(f"Maximum loop count ({max_loop}) reached. Ending workflow.")
        if llm_requires_visualization(state):
            print("[DEBUG] 达到最大循环次数但需要可视化，路由到可视化节点")
            return "visualization"
        else:
            print("[DEBUG] 达到最大循环次数且不需要可视化，结束工作流")
            return END
    print(f"Loop {loop_count}: Continuing to fix errors.")

    return "input_writer"
