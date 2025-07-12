import os
import subprocess
import sys
import argparse
import shlex
import shutil
    
# 加载.env文件中的环境变量
from dotenv import load_dotenv
load_dotenv()

def parse_args():
    """
    解析命令行参数
    
    返回:
        argparse.Namespace: 包含解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description="Benchmark Workflow Interface")
    parser.add_argument(
        '--openfoam_path',
        type=str,
        required=True,
        help="Path to OpenFOAM installation (WM_PROJECT_DIR)"
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Base output directory for benchmark results"
    )
    parser.add_argument(
        '--prompt_path',
        type=str,
        required=True,
        help="User requirement file path for the benchmark"
    )
    return parser.parse_args()

def run_command(command_str):
    """
    执行命令字符串，使用当前终端的输入/输出
    
    该函数会：
    1. 将命令字符串分割成参数列表
    2. 设置工作目录为当前脚本所在目录
    3. 执行命令并实时显示输出
    4. 处理执行错误
    
    参数:
        command_str (str): 要执行的命令，例如 "python main.py --output_dir xxxx" 
                          或 "bash xxxxx.sh"
    
    异常:
        SystemExit: 当命令执行失败时退出程序
    """
    # 将命令字符串分割成参数列表
    args = shlex.split(command_str)
    # 设置工作目录为当前脚本所在目录
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # 执行命令，实时显示输出
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,  # 如果命令返回非零状态码则抛出异常
            stdout=sys.stdout,  # 实时显示标准输出
            stderr=sys.stderr,  # 实时显示错误输出
            stdin=sys.stdin     # 允许用户输入
        )
        print(f"Finished command: Return Code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(e.returncode)  # 使用命令的返回码退出程序

def main():
    """
    主函数：执行完整的基准测试工作流程
    
    工作流程包括：
    1. 解析命令行参数
    2. 设置环境变量
    3. 检查必要的环境变量（如OPENAI_API_KEY）
    4. 创建输出目录
    5. 根据文件存在情况决定执行哪些预处理脚本
    6. 执行主要的基准测试脚本
    """
    # 解析命令行参数
    args = parse_args()
    print(args)

    # 设置OpenFOAM环境变量
    WM_PROJECT_DIR = args.openfoam_path
    
    # 检查OPENAI_API_KEY环境变量是否可用
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY is not set in the environment.")
        sys.exit(1)

    # 创建输出文件夹（如果不存在）
    os.makedirs(args.output, exist_ok=True)

    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"script_dir: {script_dir}")

    # 定义要执行的脚本列表
    # 每个脚本都是Python或shell脚本
    SCRIPTS = []
    
    # # 我想删除faiss和raw文件夹下的所有文件，并重新创建
    # if os.path.exists(f"{script_dir}/database/faiss"):
    #     shutil.rmtree(f"{script_dir}/database/faiss")
    # os.makedirs(f"{script_dir}/database/faiss", exist_ok=True)
    # if os.path.exists(f"{script_dir}/database/raw"):
    #     shutil.rmtree(f"{script_dir}/database/raw")
    # os.makedirs(f"{script_dir}/database/raw", exist_ok=True)

    # 预处理OpenFOAM教程 - 只在文件不存在时执行
    # 1. 解析OpenFOAM教程详细信息
    if not os.path.exists(f"{script_dir}/database/raw/openfoam_tutorials_details.txt"):
        SCRIPTS.append(f"python database/script/tutorial_parser.py --output_dir=./database/raw --wm_project_dir={WM_PROJECT_DIR}")
    
    # 2. 创建OpenFOAM命令帮助的FAISS索引
    if not os.path.exists(f"{script_dir}/database/faiss/openfoam_command_help"):
        SCRIPTS.append(f"python database/script/faiss_command_help.py --database_path=./database")
    
    # 3. 创建OpenFOAM Allrun脚本的FAISS索引
    if not os.path.exists(f"{script_dir}/database/faiss/openfoam_allrun_scripts"):
        SCRIPTS.append(f"python database/script/faiss_allrun_scripts.py --database_path=./database")
    
    # 4. 创建OpenFOAM教程结构的FAISS索引
    if not os.path.exists(f"{script_dir}/database/faiss/openfoam_tutorials_structure"):
        SCRIPTS.append(f"python database/script/faiss_tutorials_structure.py --database_path=./database")
    
    # 5. 创建OpenFOAM教程详细信息的FAISS索引
    if not os.path.exists(f"{script_dir}/database/faiss/openfoam_tutorials_details"):
        SCRIPTS.append(f"python database/script/faiss_tutorials_details.py --database_path=./database")
    
    print(f"python src/main.py --prompt_path='{args.prompt_path}' --output_dir='{args.output}'")
    
    # 添加主要的基准测试工作流程脚本
    SCRIPTS.extend([
        f"python src/main.py --prompt_path='{args.prompt_path}' --output_dir='{args.output}'"
    ])

    # 按顺序执行所有脚本
    print("Starting workflow...")
    for script in SCRIPTS:
        run_command(script)
    print("Workflow completed successfully.")

if __name__ == "__main__":
    # 使用示例：
    # python foambench_main.py --openfoam_path $WM_PROJECT_DIR --output ./output --prompt_path "./user_requirement.txt"
    main()
