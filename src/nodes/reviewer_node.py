# reviewer_node.py
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

def reviewer_node(state):
    """
    Reviewer node: Reviews the error logs and provides analysis and suggestions
    for fixing the errors. This node only focuses on analysis, not file modification.
    """
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



    # Return updated state with review analysis
    return {
        "history_text": history_text,
        "review_analysis": review_content,
        "loop_count": state.get("loop_count", 0) + 1,
        "input_writer_mode": "rewrite",
    }
