# runner_node.py
from typing import List
import os
from pydantic import BaseModel, Field
import re
from utils import (
    save_file, remove_files, remove_file,
    run_command, check_foam_errors, retrieve_faiss, remove_numeric_folders
)

# 本节点用于本地执行OpenFOAM的Allrun脚本，并检查执行过程中的错误。
# 主要流程：
# 1. 清理旧的log和错误文件，保证环境干净。
# 2. 执行Allrun脚本，将输出和错误分别重定向到Allrun.out和Allrun.err。
# 3. 检查执行过程中是否有OpenFOAM相关的错误日志。
# 4. 返回包含错误日志的state。

def local_runner_node(state):
    """
    Runner node: Execute an Allrun script, and check for errors.
    On error, update state.error_command and state.error_content.
    """
    config = state["config"]
    case_dir = state["case_dir"]
    allrun_file_path = os.path.join(case_dir, "Allrun")
    
    print(f"============================== Runner ==============================")
    print(f"[local_runner_node] case_dir: {case_dir}")
    print(f"[local_runner_node] allrun_file_path: {allrun_file_path}")
    
    # 1. 清理旧的log和错误文件，保证环境干净
    out_file = os.path.join(case_dir, "Allrun.out")
    err_file = os.path.join(case_dir, "Allrun.err")
    print(f"[local_runner_node] 清理log和错误文件: {out_file}, {err_file}")
    remove_files(case_dir, prefix="log")
    remove_file(err_file)
    remove_file(out_file)
    remove_numeric_folders(case_dir)
    
    # 2. 执行Allrun脚本，将输出和错误分别重定向到Allrun.out和Allrun.err
    print(f"[local_runner_node] 开始执行Allrun脚本...")
    run_command(allrun_file_path, out_file, err_file, case_dir, config)
    print(f"[local_runner_node] Allrun执行完毕，输出文件: {out_file}, 错误文件: {err_file}")
    
    # 3. 检查执行过程中是否有OpenFOAM相关的错误日志
    error_logs = check_foam_errors(case_dir)
    print(f"[local_runner_node] error_logs: {error_logs}")

    if len(error_logs) > 0:
        print("Errors detected in the Allrun execution.")
        print(error_logs)
    else:
        print("Allrun executed successfully without errors.")
    
    # 4. 返回包含错误日志的state
    return {
        **state,
        "error_logs": error_logs
    }
        