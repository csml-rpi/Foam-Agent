# hpc_runner_node.py
"""
HPC运行节点模块：负责在HPC集群上执行Allrun脚本并检查错误
主要功能：
1. 清理旧的日志和错误文件
2. 执行Allrun脚本（HPC集群）
3. 检查执行过程中的错误
4. 返回执行结果和错误信息
"""
from typing import List
import os
from pydantic import BaseModel, Field
import re
from utils import (
    save_file, remove_files, remove_file,
    run_command, check_foam_errors, retrieve_faiss, remove_numeric_folders
)


def hpc_runner_node(state):
    """
    HPC Runner node: Execute an Allrun script on HPC cluster, and check for errors.
    On error, update state.error_command and state.error_content.
    """
    # 从state中获取配置和案例目录
    config = state["config"]
    case_dir = state["case_dir"]
    allrun_file_path = os.path.join(case_dir, "Allrun")
    
    print(f"============================== HPC Runner ==============================")
    print(f"案例目录: {case_dir}")
    print(f"Allrun脚本路径: {allrun_file_path}")
    
    # 步骤1：清理旧的日志和错误文件
    print("--- 步骤1：清理旧日志和错误文件 ---")
    out_file = os.path.join(case_dir, "Allrun.out")
    err_file = os.path.join(case_dir, "Allrun.err")
    print(f"准备清理: {out_file}, {err_file} 以及所有log*文件和数字文件夹")
    remove_files(case_dir, prefix="log")  # 删除所有以log开头的文件
    remove_file(err_file)                  # 删除旧的错误文件
    remove_file(out_file)                  # 删除旧的输出文件
    remove_numeric_folders(case_dir)       # 删除所有纯数字命名的文件夹（OpenFOAM计算结果）
    print("清理完成")
    
    # 步骤2：在HPC集群上执行Allrun脚本
    print("--- 步骤2：执行Allrun脚本（HPC集群） ---")
    # TODO: 实现HPC集群的具体执行逻辑
    # 这里可以调用run_command等工具，结合HPC调度系统（如SLURM/PBS）
    # 当前未实现，留空
    pass
    
    # 步骤3：检查执行过程中的错误
    print("--- 步骤3：检查执行错误 ---")
    # TODO: 实现HPC集群的错误检查逻辑
    # 可以分析Allrun.out/Allrun.err或log文件内容
    # 当前未实现，留空
    pass
    
    # 步骤4：模拟错误日志（测试用）
    error_logs = []
    print(f"错误日志内容: {error_logs}")
    
    if len(error_logs) > 0:
        print("检测到HPC Allrun执行错误！")
        print(error_logs)
    else:
        print("HPC Allrun执行成功，无错误。")
    
    # 返回更新后的state，包含错误日志
    print("--- 返回状态 ---")
    print(f"error_logs: {error_logs}")
    return {
        **state,
        "error_logs": error_logs
    }
