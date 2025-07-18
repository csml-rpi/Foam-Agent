from typing import TypedDict, List, Optional
from config import Config
from utils import LLMService
from langgraph.graph import StateGraph, START, END

# 定义图状态的数据结构，用于在LangGraph工作流中传递数据
class GraphState(TypedDict):
    # 用户输入的需求描述
    user_requirement: str
    # 配置信息
    config: Config
    # 案例目录路径
    case_dir: str
    # 教程内容
    tutorial: str
    # 案例名称
    case_name: str
    # 子任务列表
    subtasks: List[dict]
    # 当前子任务索引
    current_subtask_index: int
    # 错误命令（可选）
    error_command: Optional[str]
    # 错误内容（可选）
    error_content: Optional[str]
    # 循环计数，用于防止无限循环
    loop_count: int
    # 以下字段在执行过程中会被添加
    # LLM服务实例
    llm_service: Optional[LLMService]
    # 案例统计信息
    case_stats: Optional[dict]
    # 教程参考信息
    tutorial_reference: Optional[str]
    # 案例路径参考
    case_path_reference: Optional[str]
    # 目录结构参考
    dir_structure_reference: Optional[str]
    # 案例信息
    case_info: Optional[str]
    # allrun脚本参考
    allrun_reference: Optional[str]
    # 目录结构
    dir_structure: Optional[dict]
    # 命令列表
    commands: Optional[List[str]]
    # OpenFOAM文件信息
    foamfiles: Optional[dict]
    # 错误日志列表
    error_logs: Optional[List[str]]
    # 历史文本
    history_text: Optional[List[str]]
    # 案例领域（如流体力学、热传导等）
    case_domain: Optional[str]
    # 案例类别
    case_category: Optional[str]
    # 求解器类型
    case_solver: Optional[str]
    # 网格相关状态字段
    mesh_info: Optional[dict]
    # 网格生成命令
    mesh_commands: Optional[List[str]]
    # 网格文件目标路径
    mesh_file_destination: Optional[str]
    # 是否使用自定义网格
    custom_mesh_used: Optional[bool]


def llm_requires_custom_mesh(state: GraphState) -> bool:
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
        bool: 如果需要自定义网格返回True，否则返回False
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
        "Be conservative - if unsure, assume standard mesh unless clearly specified otherwise."
    )
    
    # 用户提示词：具体的问题描述
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "Determine if the user wants to use a custom mesh file. "
        "Return exactly 'custom_mesh' if they want to use a custom mesh file, "
        "or 'standard_mesh' if they want standard OpenFOAM mesh generation."
    )
    
    # 检查LLM服务是否可用
    if state["llm_service"] is None:
        print("[WARNING] LLM服务不可用，默认使用标准网格生成")
        return False
    
    # 调用LLM服务进行分析
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    print(f"[DEBUG] LLM响应: {response}")
    
    result = "custom_mesh" in response.lower()
    print(f"[DEBUG] 是否需要自定义网格: {result}")
    return result


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
    )
    
    # 用户提示词：具体的问题描述
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "Determine if the user wants to run the simulation on HPC/cluster. "
        "Return exactly 'hpc_run' if they want to use HPC/cluster, "
        "or 'local_run' if they want to run locally."
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
        "Look for keywords like: plot, visualize, graph, chart, contour, streamlines, "
        "paraview, post-processing, results analysis, or any mention of viewing/displaying results. "
        "If the user explicitly mentions or implies they want visualization, return 'visualization'. "
        "If they don't mention visualization or only want to run the simulation, return 'no_visualization'. "
        "Be conservative - if unsure, assume visualization is wanted unless clearly specified otherwise." 
        # "Be strict - only return 'visualization' if the user explicitly requests it or uses clear visualization keywords."
    )
    
    # 用户提示词：具体的问题描述
    user_prompt = (
        f"User requirement: {user_requirement}\n\n"
        "Determine if the user wants visualization of simulation results. "
        "Return exactly 'visualization' if they want to visualize results, "
        "or 'no_visualization' if they don't need visualization."
    )
    
    # 检查LLM服务是否可用
    if state["llm_service"] is None:
        print("[WARNING] LLM服务不可用，默认需要可视化")
        return True
    
    # 调用LLM服务进行分析
    response = state["llm_service"].invoke(user_prompt, system_prompt)
    print(f"[DEBUG] LLM响应: {response}")
    
    result = "visualization" in response.lower()
    print(f"[DEBUG] 是否需要可视化: {result}")
    return result


def route_after_architect(state: GraphState):
    """
    架构节点后的路由决策
    
    功能：根据用户是否需要自定义网格来决定下一步的执行路径
    执行逻辑：
    1. 调用LLM判断是否需要自定义网格
    2. 如果需要自定义网格，路由到网格生成节点
    3. 否则路由到输入文件编写节点
    
    Args:
        state: 当前图状态
        
    Returns:
        str: 下一个节点的名称
    """
    print("[DEBUG] 执行架构节点后的路由决策")
    
    if llm_requires_custom_mesh(state):
        print("LLM determined: Custom mesh requested. Routing to meshing node.")
        return "meshing"
    else:
        print("LLM determined: Standard mesh generation. Routing to input_writer node.")
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
        
        # 即使达到最大循环次数，如果用户需要可视化，仍然进行可视化
        if llm_requires_visualization(state):
            print("[DEBUG] 达到最大循环次数但需要可视化，路由到可视化节点")
            return "visualization"
        else:
            print("[DEBUG] 达到最大循环次数且不需要可视化，结束工作流")
            return END
    
    # 注意：在路由函数中直接修改state不会生效
    # 循环计数的增加应该在reviewer_node中完成
    print(f"Loop {loop_count + 1}: Continuing to fix errors.")
    
    return "input_writer"
