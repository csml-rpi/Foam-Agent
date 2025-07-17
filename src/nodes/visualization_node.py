# visualization_node.py
"""
OpenFOAM可视化节点

本模块实现了OpenFOAM案例的可视化功能，包括：
1. 分析用户需求中的可视化要求
2. 检查可用的OpenFOAM数据文件和时间步
3. 生成可视化计划
4. 执行OpenFOAM后处理工具
5. 生成图表和可视化结果
6. 保存可视化结果并生成摘要

作者: OpenFOAM Agent Team
版本: 1.0
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field

class PlotConfigPydantic(BaseModel):
    """Configuration for plotting parameters"""
    plot_type: str = Field(description="Type of plot (e.g., 'contour', 'vector', 'streamline', 'time_series')")
    field_name: str = Field(description="Field to plot (e.g., 'U', 'p', 'T', 'rho')")
    time_step: Optional[str] = Field(default=None, description="Time step to plot (if None, use latest)")
    output_format: str = Field(default="png", description="Output format for plots")
    output_path: str = Field(description="Path to save the plot")

class VisualizationPlanPydantic(BaseModel):
    """Plan for visualization tasks"""
    plots: List[PlotConfigPydantic] = Field(description="List of plots to generate")

class VisualizationAnalysisPydantic(BaseModel):
    """Analysis of user requirements for visualization needs"""
    primary_field: str = Field(description="Primary field to visualize (e.g., 'U', 'p', 'T', 'rho')")
    plot_type: str = Field(description="Type of plot requested (e.g., 'contour', 'vector', 'streamline', 'time_series')")
    time_step: Optional[str] = Field(default=None, description="Specific time step to plot (if mentioned)")
    plane_info: Optional[str] = Field(default=None, description="Plane information if 2D slice is requested (e.g., 'Z plane', 'X=0.5')")
    additional_fields: List[str] = Field(default=[], description="Additional fields that might be useful to visualize")
    visualization_priority: str = Field(description="Priority of visualization (e.g., 'high', 'medium', 'low')")

def visualization_node(state):
    """
    Visualization node: Creates plots and visualizations from the successfully generated OpenFOAM case.
    This nodeuses the successfully generated code and user_requirement to create plots.
    
    Updates state with:
      - plot_configs: List of plot configurations
      - plot_outputs: List of generated plot file paths
      - visualization_summary: Summary of generated visualizations
    """
    config = state["config"]
    user_requirement = state["user_requirement"]
    case_dir = state["case_dir"]
    
    print(f"============================== Visualization Node ==============================")
    print(f"[visualization_node] 开始可视化处理...")
    print(f"[visualization_node] 案例名称: {state.get('case_name', 'Unknown')}")
    print(f"[visualization_node] 案例目录: {case_dir}")
    print(f"[visualization_node] 用户需求长度: {len(user_requirement)} 字符")
    
    # 步骤1：分析用户需求以确定需要生成哪些图表
    print("Step 1: Analyzing user requirements for visualization needs...")
    print(f"[visualization_node] 步骤1：分析用户需求中的可视化要求...")
    
    # 可视化分析的系统提示词 - 用于从用户需求中提取可视化相关信息
    visualization_system_prompt = (
        "You are an expert in OpenFOAM visualization and computational fluid dynamics. "
        "Your task is to analyze user requirements and extract key information needed for visualization. "
        "Focus specifically on visualization-related information and ignore other simulation setup details. "
        "Extract the following key elements:\n"
        "- Primary field to visualize (e.g., velocity magnitude, pressure, temperature)\n"
        "- Type of plot requested (e.g., contour, vector, streamline, time series)\n"
        "- Time step information (if specified)\n"
        "- Plane or slice information (if 2D visualization is requested)\n"
        "- Additional fields that might be useful to visualize\n"
        "- Priority of the visualization request\n\n"
        "Your output must strictly follow the JSON schema provided and include no additional information. "
        "If specific visualization details are not mentioned, make reasonable assumptions based on the simulation type."
    )
    
    # 构建用户提示词，要求LLM分析用户需求中的可视化需求
    visualization_user_prompt = (
        f"User Requirement: {user_requirement}\n\n"
        "Please analyze this user requirement and extract the visualization needs. "
        "Focus on what the user wants to visualize, what fields are important, and what type of plots would be most useful. "
        "Ignore simulation setup details like boundary conditions, mesh information, or solver settings unless they directly relate to visualization."
    )
    
    print(f"[visualization_node] 调用LLM服务分析可视化需求...")
    print(f"[visualization_node] 用户提示词长度: {len(visualization_user_prompt)} 字符")
    
    # 调用LLM服务分析用户需求中的可视化要求
    visualization_analysis = state["llm_service"].invoke(
        visualization_user_prompt, 
        visualization_system_prompt, 
        pydantic_obj=VisualizationAnalysisPydantic
    )
    
    # 输出分析结果的关键信息
    print(f"[visualization_node] LLM分析完成，提取的可视化信息:")
    print(f"[visualization_node] 主要可视化字段: {visualization_analysis.primary_field}")
    print(f"[visualization_node] 请求的图表类型: {visualization_analysis.plot_type}")
    if visualization_analysis.time_step:
        print(f"[visualization_node] 指定时间步: {visualization_analysis.time_step}")
    if visualization_analysis.plane_info:
        print(f"[visualization_node] 平面信息: {visualization_analysis.plane_info}")
    if visualization_analysis.additional_fields:
        print(f"[visualization_node] 附加字段: {visualization_analysis.additional_fields}")
    print(f"[visualization_node] 可视化优先级: {visualization_analysis.visualization_priority}")
    
    # 步骤2：检查可用的OpenFOAM数据文件和时间步
    print(f"[visualization_node] 步骤2：检查可用的OpenFOAM数据文件和时间步...")
    print(f"[visualization_node] TODO: 扫描案例目录以获取可用的时间步和字段数据")
    # TODO: Scan case directory for available time steps and field data
    pass
    
    # 步骤3：基于可用数据和用户需求生成可视化计划
    print(f"[visualization_node] 步骤3：基于可用数据和用户需求生成可视化计划...")
    print(f"[visualization_node] TODO: 创建结构化计划，确定需要生成哪些图表")
    # TODO: Create structured plan for what plots to generate
    pass
    
    # 步骤4：使用OpenFOAM工具执行绘图命令
    print(f"[visualization_node] 步骤4：使用OpenFOAM工具执行绘图命令...")
    print(f"[visualization_node] TODO: 运行OpenFOAM后处理工具（postProcess, sample等）")
    # TODO: Run OpenFOAM post-processing tools (postProcess, sample, etc.)
    pass
    
    # 步骤5：使用matplotlib、pyvista或其他可视化库生成图表
    print(f"[visualization_node] 步骤5：使用可视化库生成图表...")
    print(f"[visualization_node] TODO: 从处理后的数据创建实际图表")
    # TODO: Create actual plots from the processed data
    pass
    
    # 步骤6：保存图表并生成摘要
    print(f"[visualization_node] 步骤6：保存图表并生成摘要...")
    print(f"[visualization_node] TODO: 将图表保存到输出目录并创建摘要报告")
    # TODO: Save plots to output directory and create summary report
    pass
    
    # 为骨架代码提供模拟返回值
    print(f"[visualization_node] 生成模拟的可视化配置...")
    
    # 模拟的图表配置列表，我草，这里其实应该用pydantic来生成，而不是用字典
    # 感觉它先hard code了
    plot_configs = [
        {
            "plot_type": "contour",
            "field_name": "p",
            "time_step": "latest",
            "output_format": "png",
            "output_path": os.path.join(case_dir, "plots", "pressure_contour.png")
        },
        {
            "plot_type": "vector",
            "field_name": "U",
            "time_step": "latest", 
            "output_format": "png",
            "output_path": os.path.join(case_dir, "plots", "velocity_vectors.png")
        }
    ]
    
    print(f"[visualization_node] 图表配置数量: {len(plot_configs)}")
    for i, config in enumerate(plot_configs):
        print(f"[visualization_node] 配置 {i+1}: 类型={config['plot_type']}, 字段={config['field_name']}, 输出={config['output_path']}")
    
    # 模拟的图表输出文件路径列表
    plot_outputs = [
        os.path.join(case_dir, "plots", "pressure_contour.png"),
        os.path.join(case_dir, "plots", "velocity_vectors.png")
    ]
    
    print(f"[visualization_node] 图表输出文件数量: {len(plot_outputs)}")
    for i, output in enumerate(plot_outputs):
        print(f"[visualization_node] 输出文件 {i+1}: {output}")
    
    # 可视化摘要信息
    visualization_summary = {
        "total_plots_generated": len(plot_outputs),
        "plot_types": ["contour", "vector"],
        "fields_visualized": ["p", "U"],
        "output_directory": os.path.join(case_dir, "plots")
    }
    
    print(f"[visualization_node] 可视化摘要:")
    print(f"[visualization_node] - 生成的图表总数: {visualization_summary['total_plots_generated']}")
    print(f"[visualization_node] - 图表类型: {visualization_summary['plot_types']}")
    print(f"[visualization_node] - 可视化字段: {visualization_summary['fields_visualized']}")
    print(f"[visualization_node] - 输出目录: {visualization_summary['output_directory']}")
    
    print(f"Generated {len(plot_outputs)} plots")
    print(f"Plots saved to: {os.path.join(case_dir, 'plots')}")
    print("============================== Visualization Complete ==============================")
    
    # 返回更新后的状态
    print(f"[visualization_node] 返回更新后的状态...")
    return {
        **state,
        "visualization_analysis": visualization_analysis,
        "plot_configs": plot_configs,
        "plot_outputs": plot_outputs,
        "visualization_summary": visualization_summary
    }
