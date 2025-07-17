# reviewer_node.py
"""
OpenFOAM错误审查和修复节点

本模块实现了OpenFOAM案例执行错误的自动诊断和修复功能。
主要功能：
1. 分析OpenFOAM执行过程中的错误日志
2. 基于相似案例参考和用户需求提供修复建议
3. 自动重写和修改相关的OpenFOAM文件
4. 维护错误修复的历史记录

"""

import os
from utils import save_file, FoamPydantic
from pydantic import BaseModel, Field
from typing import List

# 审查者系统提示词 - 用于分析错误日志并提供修复建议
REVIEWER_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM simulation and numerical modeling. "
    "Your task is to review the provided error logs and diagnose the underlying issues. "
    "You will be provided with a similar case reference, which is a list of similar cases that are ordered by similarity. You can use this reference to help you understand the user requirement and the error."
    "When an error indicates that a specific keyword is undefined (for example, 'div(phi,(p|rho)) is undefined'), your response must propose a solution that simply defines that exact keyword as shown in the error log. "
    "Do not reinterpret or modify the keyword (e.g., do not treat '|' as 'or'); instead, assume it is meant to be taken literally. "
    "Propose ideas on how to resolve the errors, but do not modify any files directly. "
    "Please do not propose solutions that require modifying any parameters declared in the user requirement, try other approaches instead. Do not ask the user any questions."
    "The user will supply all relevant foam files along with the error logs, and within the logs, you will find both the error content and the corresponding error command indicated by the log file name."
)

# 重写者系统提示词 - 用于实际修改OpenFOAM文件
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



def reviewer_node(state):
    """
    审查者节点：审查错误日志并确定错误是否与输入文件相关
    
    主要执行流程：
    1. 检查是否有错误需要审查
    2. 根据是否有历史记录构建不同的用户提示词
    3. 调用LLM服务分析错误并提供修复建议
    4. 记录修复历史
    5. 调用LLM服务重写相关文件
    6. 保存修改的文件并更新状态
    
    Args:
        state (dict): 包含案例信息、错误日志、文件结构等的状态字典
        
    Returns:
        dict: 更新后的状态字典，包含修复历史和清空的错误日志
    """
    config = state["config"]
    
    print(f"============================== Reviewer Analysis ==============================")
    print(f"[reviewer_node] 开始审查错误日志...")
    print(f"[reviewer_node] 当前错误日志数量: {len(state['error_logs'])}")
    
    # 检查是否有错误需要审查
    if len(state["error_logs"]) == 0:
        print("No error to review.")
        return state
    
    # 分析错误原因并提供修复方法
    # 根据是否有历史记录构建不同的用户提示词
    if state.get("history_text") and state["history_text"]:
        print(f"[reviewer_node] 检测到历史记录，构建包含历史信息的提示词")
        print(f"[reviewer_node] 历史记录长度: {len(state['history_text'])} 行")
        
        # 如果有历史记录，说明这是多次尝试修复，需要包含历史信息
        reviewer_user_prompt = (
            f"<similar_case_reference>{state['tutorial_reference']}</similar_case_reference>\n"
            f"<foamfiles>{str(state['foamfiles'])}</foamfiles>\n"
            f"<current_error_logs>{state['error_logs']}</current_error_logs>\n"
            f"<history>\n"
            f"{chr(10).join(state['history_text'])}\n"
            f"</history>\n\n"
            f"<user_requirement>{state['user_requirement']}</user_requirement>\n\n"
            f"I have modified the files according to your previous suggestions. If the error persists, please provide further guidance. Make sure your suggestions adhere to user requirements and do not contradict it. Also, please consider the previous attempts and try a different approach."
        )
    else:
        print(f"[reviewer_node] 首次修复，构建基础提示词")
        
        # 首次修复，使用基础提示词
        reviewer_user_prompt = (
            f"<similar_case_reference>{state['tutorial_reference']}</similar_case_reference>\n"
            f"<foamfiles>{str(state['foamfiles'])}</foamfiles>\n"
            f"<error_logs>{state['error_logs']}</error_logs>\n"
            f"<user_requirement>{state['user_requirement']}</user_requirement>\n"
            "Please review the error logs and provide guidance on how to resolve the reported errors. Make sure your suggestions adhere to user requirements and do not contradict it."
        ) 
    
    print(f"[reviewer_node] 调用LLM服务分析错误...")
    print(f"[reviewer_node] 用户提示词长度: {len(reviewer_user_prompt)} 字符")
    
    # 调用LLM服务分析错误并提供修复建议
    review_response = state["llm_service"].invoke(reviewer_user_prompt, REVIEWER_SYSTEM_PROMPT)
    review_content = review_response
    
    print(f"[reviewer_node] LLM分析完成，响应长度: {len(review_content)} 字符")
    
    # 初始化历史记录文本列表
    if not state.get("history_text"):
        print(f"[reviewer_node] 初始化历史记录")
        history_text = []
    else:
        print(f"[reviewer_node] 使用现有历史记录")
        history_text = state["history_text"]
        
    # 将当前尝试添加到历史记录中
    current_attempt = [
        f"<Attempt {len(history_text)//4 + 1}>\n"
        f"<Error_Logs>\n{state['error_logs']}\n</Error_Logs>",
        f"<Review_Analysis>\n{review_content}\n</Review_Analysis>",
        f"</Attempt>\n"  # 结束标签，包含空行
    ]
    history_text.extend(current_attempt)
    
    print(f"[reviewer_node] 当前尝试次数: {len(history_text)//4}")
    print(f"[reviewer_node] 历史记录总长度: {len(history_text)} 行")
    
    print(review_content)

    # 返回重写的foamfile内容
    print(f"[reviewer_node] 构建重写提示词...")
    
    # 构建用于重写文件的用户提示词
    rewrite_user_prompt = (
        f"<foamfiles>{str(state['foamfiles'])}</foamfiles>\n"
        f"<error_logs>{state['error_logs']}</error_logs>\n"
        f"<reviewer_analysis>{review_content}</reviewer_analysis>\n\n"
        f"<user_requirement>{state['user_requirement']}</user_requirement>\n\n"
        "Please update the relevant OpenFOAM files to resolve the reported errors, ensuring that all modifications strictly adhere to the specified formats. Ensure all modifications adhere to user requirement."
    )
    
    print(f"[reviewer_node] 调用LLM服务重写文件...")
    print(f"[reviewer_node] 重写提示词长度: {len(rewrite_user_prompt)} 字符")
    
    # 调用LLM服务重写相关文件，使用FoamPydantic进行结构化输出
    rewrite_response = state["llm_service"].invoke(rewrite_user_prompt, REWRITE_SYSTEM_PROMPT, pydantic_obj=FoamPydantic)
    
    print(f"[reviewer_node] 文件重写完成，需要修改的文件数量: {len(rewrite_response.list_foamfile)}")
    
    # 保存修改的文件
    print(f"============================== Rewrite ==============================")
    for foamfile in rewrite_response.list_foamfile:
        print(f"Modified the file: {foamfile.file_name} in folder: {foamfile.folder_name}")
        print(f"[reviewer_node] 文件路径: {foamfile.folder_name}/{foamfile.file_name}")
        print(f"[reviewer_node] 文件内容长度: {len(foamfile.content)} 字符")
        
        # 构建完整的文件路径
        file_path = os.path.join(state["case_dir"], foamfile.folder_name, foamfile.file_name)
        print(f"[reviewer_node] 完整文件路径: {file_path}")
        
        # 保存文件到磁盘
        save_file(file_path, foamfile.content)
        
        # 更新状态中的目录结构
        if foamfile.folder_name not in state["dir_structure"]:
            print(f"[reviewer_node] 新增目录: {foamfile.folder_name}")
            state["dir_structure"][foamfile.folder_name] = []
        if foamfile.file_name not in state["dir_structure"][foamfile.folder_name]:
            print(f"[reviewer_node] 新增文件到目录: {foamfile.folder_name}/{foamfile.file_name}")
            state["dir_structure"][foamfile.folder_name].append(foamfile.file_name)
        
        # 更新状态中的foamfiles列表
        # 移除旧的文件记录
        for f in state["foamfiles"].list_foamfile:
            if f.folder_name == foamfile.folder_name and f.file_name == foamfile.file_name:
                print(f"[reviewer_node] 移除旧文件记录: {f.folder_name}/{f.file_name}")
                state["foamfiles"].list_foamfile.remove(f)
                break
            
        # 添加新的文件记录
        state["foamfiles"].list_foamfile.append(foamfile)
        print(f"[reviewer_node] 添加新文件记录: {foamfile.folder_name}/{foamfile.file_name}")
    
    print(f"[reviewer_node] 文件修改完成，总共修改了 {len(rewrite_response.list_foamfile)} 个文件")
    print(f"[reviewer_node] 当前foamfiles列表长度: {len(state['foamfiles'].list_foamfile)}")
    
    # 返回更新后的状态
    # 注意：清空错误日志，因为已经尝试修复
    return {
        **state,
        "history_text": history_text,
        "error_logs": []  # 清空错误日志，因为已经尝试修复
    }
