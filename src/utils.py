# utils.py
"""
OpenFOAM 智能代理工具模块

本模块提供了OpenFOAM案例处理、LLM服务、文件操作、FAISS向量数据库检索等核心功能。
主要包含以下组件：
1. LLMService: 支持多种LLM提供商的服务类
2. FAISS数据库缓存: 预加载的OpenFOAM知识库
3. 文件操作工具: 保存、读取、删除文件等
4. OpenFOAM案例处理: 运行脚本、检查错误、解析结构等
5. 向量检索: 基于FAISS的相似案例检索

作者: OpenFOAM Agent Team
版本: 1.0
"""

import re
import subprocess
import os
from typing import Optional, Any, Type
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_anthropic import ChatAnthropic
from pathlib import Path
import tracking_aws
import requests
from pydantic import BaseModel, Field
from typing import List
import time
import random
from botocore.exceptions import ClientError
import shutil
from config import Config
from langchain_ollama import ChatOllama

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 全局FAISS数据库缓存字典，避免重复加载
FAISS_DB_CACHE = {}
DATABASE_DIR = f"{Path(__file__).resolve().parent.parent}/database/faiss"

print(f"[DEBUG] 正在加载FAISS数据库，路径: {DATABASE_DIR}")

# 预加载所有FAISS数据库到内存中，提高检索效率
# 注意：这里使用了allow_dangerous_deserialization=True，需要确保数据库文件的安全性
FAISS_DB_CACHE = {
    "openfoam_allrun_scripts": FAISS.load_local(f"{DATABASE_DIR}/openfoam_allrun_scripts", OpenAIEmbeddings(model="text-embedding-3-small"), allow_dangerous_deserialization=True),
    "openfoam_tutorials_structure": FAISS.load_local(f"{DATABASE_DIR}/openfoam_tutorials_structure", OpenAIEmbeddings(model="text-embedding-3-small"), allow_dangerous_deserialization=True),
    "openfoam_tutorials_details": FAISS.load_local(f"{DATABASE_DIR}/openfoam_tutorials_details", OpenAIEmbeddings(model="text-embedding-3-small"), allow_dangerous_deserialization=True),
    "openfoam_command_help": FAISS.load_local(f"{DATABASE_DIR}/openfoam_command_help", OpenAIEmbeddings(model="text-embedding-3-small"), allow_dangerous_deserialization=True)
}

print(f"[DEBUG] FAISS数据库加载完成，共加载 {len(FAISS_DB_CACHE)} 个数据库")

class FoamfilePydantic(BaseModel):
    """OpenFOAM文件的数据模型，用于结构化输出"""
    file_name: str = Field(description="Name of the OpenFOAM input file")
    folder_name: str = Field(description="Folder where the foamfile should be stored")
    content: str = Field(description="Content of the OpenFOAM file, written in OpenFOAM dictionary format")

class FoamPydantic(BaseModel):
    """OpenFOAM文件列表的数据模型"""
    list_foamfile: List[FoamfilePydantic] = Field(description="List of OpenFOAM configuration files")

class ResponseWithThinkPydantic(BaseModel):
    """包含思考过程的响应数据模型，主要用于DeepSeek模型"""
    think: str = Field(description="Thought process of the LLM")
    response: str = Field(description="Response of the LLM")
    
class LLMService:
    """
    LLM服务类，支持多种LLM提供商
    
    支持的提供商：
    - OpenAI (GPT系列)
    - Anthropic (Claude系列)
    - AWS Bedrock
    - Ollama (本地部署)
    
    功能特性：
    - 自动重试机制（针对限流错误）
    - 详细的token使用统计
    - 结构化输出支持
    - 多种模型配置
    """
    
    def __init__(self, config: object):
        """
        初始化LLM服务
        
        Args:
            config: 配置对象，包含模型版本、温度、提供商等信息
        """
        # 从配置对象中获取参数，提供默认值
        self.model_version = getattr(config, "model_version", "gpt-4o")
        self.temperature = getattr(config, "temperature", 0)
        self.model_provider = getattr(config, "model_provider", "openai")
        
        print(f"[DEBUG] 初始化LLM服务 - 提供商: {self.model_provider}, 模型: {self.model_version}, 温度: {self.temperature}")
        
        # 初始化统计信息
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.failed_calls = 0
        self.retry_count = 0
        
        # 根据提供商初始化相应的LLM
        if self.model_provider.lower() == "bedrock":
            print("[DEBUG] 使用AWS Bedrock服务")
            bedrock_runtime = tracking_aws.new_default_client()
            self.llm = ChatBedrockConverse(
                client=bedrock_runtime, 
                model_id=self.model_version, 
                temperature=self.temperature, 
                max_tokens=8192
            )
        elif self.model_provider.lower() == "anthropic":
            print("[DEBUG] 使用Anthropic Claude服务")
            self.llm = ChatAnthropic(
                model=self.model_version, 
                temperature=self.temperature
            )
        elif self.model_provider.lower() == "openai":
            print("[DEBUG] 使用OpenAI服务")
            self.llm = init_chat_model(
                self.model_version, 
                model_provider=self.model_provider, 
                temperature=self.temperature
            )
        elif self.model_provider.lower() == "ollama":
            print("[DEBUG] 使用Ollama本地服务")
            try:
                # 检查Ollama服务是否运行
                response = requests.get("http://localhost:11434/api/version", timeout=2)
                print("[DEBUG] Ollama服务已运行")
            except requests.exceptions.RequestException:
                print("[DEBUG] Ollama服务未运行，正在启动...")
                # 启动Ollama服务
                subprocess.Popen(["ollama", "serve"], 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
                # 等待服务启动
                time.sleep(5)  # 给服务5秒时间初始化

            self.llm = ChatOllama(
                model=self.model_version, 
                temperature=self.temperature,
                num_predict=-1,  # 无限制预测
                num_ctx=131072,  # 上下文窗口大小
                base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"{self.model_provider} 是不支持的模型提供商")
    
    def invoke(self, 
              user_prompt: str, 
              system_prompt: Optional[str] = None, 
              pydantic_obj: Optional[Type[BaseModel]] = None,
              max_retries: int = 10) -> Any:
        """
        调用LLM并返回响应
        
        Args:
            user_prompt: 用户提示词
            system_prompt: 可选的系统提示词
            pydantic_obj: 可选的Pydantic模型，用于结构化输出
            max_retries: 最大重试次数（针对限流错误）
            
        Returns:
            LLM响应，包含token使用统计
            
        Raises:
            Exception: 当达到最大重试次数或发生其他错误时
        """
        self.total_calls += 1
        print(f"[DEBUG] 第 {self.total_calls} 次调用LLM")
        print(f"[DEBUG] 用户提示词长度: {len(user_prompt)} 字符")
        
        # 构建消息列表
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            print(f"[DEBUG] 系统提示词长度: {len(system_prompt)} 字符")
        messages.append({"role": "user", "content": user_prompt})
        
        # 计算提示词token数量
        prompt_tokens = 0
        for message in messages:
            prompt_tokens += self.llm.get_num_tokens(message["content"])
        
        print(f"[DEBUG] 提示词token数量: {prompt_tokens}")
        
        retry_count = 0
        while True:
            try:
                if pydantic_obj:
                    print("[DEBUG] 使用结构化输出模式")
                    structured_llm = self.llm.with_structured_output(pydantic_obj)
                    response = structured_llm.invoke(messages)
                else:
                    if self.model_version.startswith("deepseek"):
                        print("[DEBUG] 使用DeepSeek模型，提取思考过程")
                        structured_llm = self.llm.with_structured_output(ResponseWithThinkPydantic)
                        response = structured_llm.invoke(messages)

                        # 提取响应内容，去除思考过程
                        response = response.response
                    else:
                        print("[DEBUG] 使用标准输出模式")
                        response = self.llm.invoke(messages)
                        response = response.content

                # 计算完成token数量
                response_content = str(response)
                completion_tokens = self.llm.get_num_tokens(response_content)
                total_tokens = prompt_tokens + completion_tokens
                
                print(f"[DEBUG] 响应token数量: {completion_tokens}, 总token数量: {total_tokens}")
                
                # 更新统计信息
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                
                return response
                
            except ClientError as e:
                # 处理AWS Bedrock的限流错误
                if e.response['Error']['Code'] == 'Throttling' or e.response['Error']['Code'] == 'TooManyRequestsException':
                    retry_count += 1
                    self.retry_count += 1
                    
                    if retry_count > max_retries:
                        self.failed_calls += 1
                        print(f"[ERROR] 达到最大重试次数 {max_retries}")
                        raise Exception(f"Maximum retries ({max_retries}) exceeded: {str(e)}")
                    
                    # 指数退避策略，带随机抖动
                    base_delay = 1.0
                    max_delay = 60.0
                    delay = min(max_delay, base_delay * (2 ** (retry_count - 1)))
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = delay + jitter
                    
                    print(f"[WARNING] 发生限流错误: {str(e)}. {sleep_time:.2f}秒后进行第{retry_count}/{max_retries}次重试")
                    time.sleep(sleep_time)
                else:
                    self.failed_calls += 1
                    print(f"[ERROR] AWS Bedrock错误: {str(e)}")
                    raise e
            except Exception as e:
                self.failed_calls += 1
                print(f"[ERROR] LLM调用失败: {str(e)}")
                raise e
    
    def get_statistics(self) -> dict:
        """
        获取LLM服务的当前统计信息
        
        Returns:
            包含各种统计信息的字典
        """
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "retry_count": self.retry_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "average_prompt_tokens": self.total_prompt_tokens / self.total_calls if self.total_calls > 0 else 0,
            "average_completion_tokens": self.total_completion_tokens / self.total_calls if self.total_calls > 0 else 0,
            "average_tokens": self.total_tokens / self.total_calls if self.total_calls > 0 else 0
        }
    
    def print_statistics(self) -> None:
        """
        打印LLM服务的当前统计信息
        """
        stats = self.get_statistics()
        print("\n<LLM Service Statistics>")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Failed calls: {stats['failed_calls']}")
        print(f"Total retries: {stats['retry_count']}")
        print(f"Total prompt tokens: {stats['total_prompt_tokens']}")
        print(f"Total completion tokens: {stats['total_completion_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Average prompt tokens per call: {stats['average_prompt_tokens']:.2f}")
        print(f"Average completion tokens per call: {stats['average_completion_tokens']:.2f}")
        print(f"Average tokens per call: {stats['average_tokens']:.2f}\n")
        print("</LLM Service Statistics>")

def tokenize(text: str) -> str:
    """
    对文本进行分词处理，用于向量检索前的预处理
    
    Args:
        text: 输入文本
        
    Returns:
        处理后的文本（小写，下划线替换为空格，驼峰命名分割）
    """
    print(f"[DEBUG] 原始文本: {text[:100]}...")
    # 将下划线替换为空格
    text = text.replace('_', ' ')
    # 在小写字母和大写字母之间插入空格（全局匹配）
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    result = text.lower()
    print(f"[DEBUG] 分词后文本: {result[:100]}...")
    return result

def save_file(path: str, content: str) -> None:
    """
    保存文件到指定路径
    
    Args:
        path: 文件路径
        content: 文件内容
    """
    print(f"[DEBUG] 保存文件到: {path}")
    print(f"[DEBUG] 文件内容长度: {len(content)} 字符")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Saved file at {path}")

def read_file(path: str) -> str:
    """
    读取文件内容
    
    Args:
        path: 文件路径
        
    Returns:
        文件内容，如果文件不存在则返回空字符串
    """
    print(f"[DEBUG] 读取文件: {path}")
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = f.read()
        print(f"[DEBUG] 文件内容长度: {len(content)} 字符")
        return content
    print(f"[WARNING] 文件不存在: {path}")
    return ""

def list_case_files(case_dir: str) -> str:
    """
    列出案例目录中的所有文件
    
    Args:
        case_dir: 案例目录路径
        
    Returns:
        文件名的逗号分隔字符串
    """
    print(f"[DEBUG] 列出案例文件: {case_dir}")
    files = [f for f in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, f))]
    result = ", ".join(files)
    print(f"[DEBUG] 找到 {len(files)} 个文件: {result}")
    return result

def remove_files(directory: str, prefix: str) -> None:
    """
    删除指定目录中具有特定前缀的文件
    
    Args:
        directory: 目录路径
        prefix: 文件前缀
    """
    print(f"[DEBUG] 删除文件，目录: {directory}, 前缀: {prefix}")
    removed_count = 0
    for file in os.listdir(directory):
        if file.startswith(prefix):
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            removed_count += 1
            print(f"[DEBUG] 删除文件: {file_path}")
    print(f"Removed {removed_count} files with prefix '{prefix}' in {directory}")

def remove_file(path: str) -> None:
    """
    删除指定文件
    
    Args:
        path: 文件路径
    """
    print(f"[DEBUG] 删除文件: {path}")
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed file {path}")
    else:
        print(f"[WARNING] 文件不存在: {path}")

def remove_numeric_folders(case_dir: str) -> None:
    """
    删除案例目录中表示数值的文件夹，包括带小数点的文件夹，但保留"0"文件夹
    
    Args:
        case_dir: 案例目录路径
    """
    print(f"[DEBUG] 删除数值文件夹，目录: {case_dir}")
    removed_count = 0
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if os.path.isdir(item_path) and item != "0":
            try:
                # 尝试转换为浮点数来检查是否为数值
                float(item)
                # 如果转换成功，说明是数值文件夹
                try:
                    shutil.rmtree(item_path)
                    removed_count += 1
                    print(f"[DEBUG] 删除数值文件夹: {item_path}")
                except Exception as e:
                    print(f"[ERROR] 删除文件夹 {item_path} 时出错: {str(e)}")
            except ValueError:
                # 不是数值，保留这个文件夹
                print(f"[DEBUG] 保留非数值文件夹: {item_path}")
                pass
    print(f"[DEBUG] 共删除 {removed_count} 个数值文件夹")

def run_command(script_path: str, out_file: str, err_file: str, working_dir: str, config : Config) -> None:
    """
    在指定工作目录中执行OpenFOAM脚本
    
    Args:
        script_path: 脚本路径
        out_file: 标准输出文件路径
        err_file: 标准错误文件路径
        working_dir: 工作目录
        config: 配置对象，包含超时时间等参数
    """
    print(f"[DEBUG] 执行脚本: {script_path}")
    print(f"[DEBUG] 工作目录: {working_dir}")
    print(f"[DEBUG] 输出文件: {out_file}")
    print(f"[DEBUG] 错误文件: {err_file}")
    
    # 设置脚本执行权限
    os.chmod(script_path, 0o777)
    
    # 获取OpenFOAM环境变量
    openfoam_dir = os.getenv("WM_PROJECT_DIR")
    print(f"[DEBUG] OpenFOAM目录: {openfoam_dir}")
    
    # 构建命令：先加载OpenFOAM环境，然后执行脚本
    command = f"source {openfoam_dir}/etc/bashrc && bash {os.path.abspath(script_path)}"
    timeout_seconds = config.max_time_limit
    
    print(f"[DEBUG] 执行命令: {command}")
    print(f"[DEBUG] 超时时间: {timeout_seconds} 秒")

    with open(out_file, 'w') as out, open(err_file, 'w') as err:
        process = subprocess.Popen(
            ['bash', "-c", command],
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            out.write(stdout)
            err.write(stderr)
            print(f"[DEBUG] 脚本执行完成，返回码: {process.returncode}")
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            timeout_message = (
                "OpenFOAM execution took too long. "
                "This case, if set up right, does not require such large execution times.\n"
            )
            out.write(timeout_message + stdout)
            err.write(timeout_message + stderr)
            print(f"[WARNING] 脚本执行超时: {script_path}")

    print(f"Executed script {script_path}")

def check_foam_errors(directory: str) -> list:
    """
    检查OpenFOAM日志文件中的错误

    参数:
        directory: 包含日志文件的目录路径

    返回:
        错误日志列表，每个元素是一个字典，包含文件名和错误内容
    """
    print(f"[DEBUG] 检查OpenFOAM错误，目录: {directory}")
    error_logs = []
    # 定义正则表达式，匹配以"ERROR:"开头的内容，DOTALL模式下'.'也能匹配换行符
    pattern = re.compile(r"ERROR:(.*)", re.DOTALL)
    
    # 遍历目录下所有文件
    for file in os.listdir(directory):
        # 只处理以"log"开头的文件（通常是OpenFOAM的日志文件，如log.blockMesh、log.icoFoam等）
        if file.startswith("log"):
            filepath = os.path.join(directory, file)
            print(f"[DEBUG] 检查日志文件: {filepath}")
            with open(filepath, 'r') as f:
                content = f.read()
            
            # 用正则表达式查找第一个"ERROR:"及其后面的所有内容
            match = pattern.search(content)
            if match:
                error_content = match.group(0).strip()
                # 记录错误日志，包含文件名和错误内容
                error_logs.append({"file": file, "error_content": error_content})
                print(f"[DEBUG] 发现错误: {file}")
            # 如果没有匹配到"ERROR:"，但内容中包含"error"（不区分大小写），也给出警告
            elif "error" in content.lower():
                print(f"[WARNING] 文件 {file} 包含'error'但不匹配预期格式")
    
    print(f"[DEBUG] 发现 {len(error_logs)} 个错误")
    return error_logs

def extract_commands_from_allrun_out(out_file: str) -> list:
    """
    从allrun脚本的输出文件中提取执行的命令
    
    Args:
        out_file: 输出文件路径
        
    Returns:
        命令列表
    """
    print(f"[DEBUG] 从输出文件提取命令: {out_file}")
    commands = []
    if not os.path.exists(out_file):
        print(f"[WARNING] 输出文件不存在: {out_file}")
        return commands
    
    with open(out_file, 'r') as f:
        for line in f:
            if line.startswith("Running "):
                parts = line.split(" ")
                if len(parts) > 1:
                    command = parts[1].strip()
                    commands.append(command)
                    print(f"[DEBUG] 提取到命令: {command}")
    
    print(f"[DEBUG] 共提取到 {len(commands)} 个命令")
    return commands

def parse_case_name(text: str) -> str:
    """
    从文本中解析案例名称
    
    Args:
        text: 包含案例名称的文本
        
    Returns:
        案例名称，如果未找到则返回"default_case"
    """
    print(f"[DEBUG] 解析案例名称，文本: {text[:100]}...")
    match = re.search(r'case name:\s*(.+)', text, re.IGNORECASE)
    if match:
        case_name = match.group(1).strip()
        print(f"[DEBUG] 解析到案例名称: {case_name}")
        return case_name
    else:
        print(f"[WARNING] 未找到案例名称，使用默认值")
        return "default_case"

def split_subtasks(text: str) -> list:
    """
    从文本中分割子任务
    
    Args:
        text: 包含子任务信息的文本
        
    Returns:
        子任务列表
    """
    print(f"[DEBUG] 分割子任务，文本长度: {len(text)}")
    header_match = re.search(r'splits into (\d+) subtasks:', text, re.IGNORECASE)
    if not header_match:
        print("[WARNING] 在响应中未找到子任务头部信息")
        return []
    
    num_subtasks = int(header_match.group(1))
    print(f"[DEBUG] 预期子任务数量: {num_subtasks}")
    
    subtasks = re.findall(r'subtask\d+:\s*(.*)', text, re.IGNORECASE)
    if len(subtasks) != num_subtasks:
        print(f"[WARNING] 预期 {num_subtasks} 个子任务但找到 {len(subtasks)} 个")
    
    print(f"[DEBUG] 实际找到 {len(subtasks)} 个子任务")
    for i, subtask in enumerate(subtasks):
        print(f"[DEBUG] 子任务 {i+1}: {subtask[:50]}...")
    
    return subtasks

def parse_context(text: str) -> str:
    """
    从文本中解析OpenFOAM文件内容，提取FoamFile字典格式的部分
    
    该函数用于从LLM生成的文本中提取OpenFOAM文件的实际内容。
    LLM可能会在回答中包含解释、注释或其他非OpenFOAM内容，此函数
    通过正则表达式提取从"FoamFile"开始到代码块结束或文本结束的部分。
    
    Args:
        text (str): 包含FoamFile信息的文本，可能包含LLM的解释内容
        
    Returns:
        str: 解析后的OpenFOAM文件内容，如果解析失败则返回原文本
        
    Example:
        >>> text = "这是一个OpenFOAM文件：\nFoamFile\n{\n    format ascii;\n    class dictionary;\n}\n```"
        >>> parse_context(text)
        "FoamFile\n{\n    format ascii;\n    class dictionary;\n}"
    """
    print(f"[DEBUG] 解析FoamFile上下文，文本长度: {len(text)}")
    
    # 使用正则表达式匹配FoamFile内容
    # r'FoamFile\s*\{.*?(?=```|$)' 的含义：
    # - FoamFile: 匹配字面量"FoamFile"
    # - \s*: 匹配零个或多个空白字符
    # - \{: 匹配左大括号
    # - .*?: 非贪婪匹配任意字符（包括换行符，因为使用了DOTALL标志）
    # - (?=```|$): 正向预查，匹配到```代码块结束符或文本结尾时停止
    # - re.DOTALL: 让.也能匹配换行符
    # - re.IGNORECASE: 忽略大小写，匹配FoamFile、foamfile等
    match = re.search(r'FoamFile\s*\{.*?(?=```|$)', text, re.DOTALL | re.IGNORECASE)
    
    if match:
        # 提取匹配到的内容并去除首尾空白字符
        context = match.group(0).strip()
        print(f"[DEBUG] 成功解析FoamFile上下文，长度: {len(context)}")
        return context
    
    # 如果正则匹配失败，返回原文本作为fallback
    print("[WARNING] 无法解析上下文，返回原文本")
    return text

def parse_file_name(subtask: str) -> str:
    """
    从子任务中解析文件名
    
    Args:
        subtask: 子任务文本
        
    Returns:
        文件名，如果未找到则返回空字符串
    """
    print(f"[DEBUG] 解析文件名，子任务: {subtask[:100]}...")
    match = re.search(r'openfoam\s+(.*?)\s+foamfile', subtask, re.IGNORECASE)
    if match:
        file_name = match.group(1).strip()
        print(f"[DEBUG] 解析到文件名: {file_name}")
        return file_name
    else:
        print(f"[WARNING] 未找到文件名")
        return ""

def parse_folder_name(subtask: str) -> str:
    """
    从子任务中解析文件夹名
    
    Args:
        subtask: 子任务文本
        
    Returns:
        文件夹名，如果未找到则返回空字符串
    """
    print(f"[DEBUG] 解析文件夹名，子任务: {subtask[:100]}...")
    match = re.search(r'foamfile in\s+(.*?)\s+folder', subtask, re.IGNORECASE)
    if match:
        folder_name = match.group(1).strip()
        print(f"[DEBUG] 解析到文件夹名: {folder_name}")
        return folder_name
    else:
        print(f"[WARNING] 未找到文件夹名")
        return ""

def find_similar_file(description: str, tutorial: str) -> str:
    """
    在教程文本中查找相似的文件描述
    
    Args:
        description: 文件描述
        tutorial: 教程文本
        
    Returns:
        找到的相似文件内容，如果未找到则返回"None"
    """
    print(f"[DEBUG] 查找相似文件，描述: {description[:50]}...")
    start_pos = tutorial.find(description)
    if start_pos == -1:
        print(f"[WARNING] 未找到描述: {description}")
        return "None"
    
    end_marker = "input_file_end."
    end_pos = tutorial.find(end_marker, start_pos)
    if end_pos == -1:
        print(f"[WARNING] 未找到结束标记")
        return "None"
    
    result = tutorial[start_pos:end_pos + len(end_marker)]
    print(f"[DEBUG] 找到相似文件，长度: {len(result)}")
    return result

def read_commands(file_path: str) -> str:
    """
    读取命令文件内容
    
    Args:
        file_path: 命令文件路径
        
    Returns:
        非空行的逗号分隔字符串
        
    Raises:
        FileNotFoundError: 当文件不存在时
    """
    print(f"[DEBUG] 读取命令文件: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Commands file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        # 连接非空行，用逗号分隔
        lines = [line.strip() for line in f if line.strip()]
        result = ", ".join(lines)
        print(f"[DEBUG] 读取到 {len(lines)} 个命令: {result}")
        return result

def find_input_file(case_dir: str, command: str) -> str:
    """
    在案例目录中查找包含指定命令的输入文件
    
    Args:
        case_dir: 案例目录
        command: 命令名称
        
    Returns:
        找到的文件路径，如果未找到则返回空字符串
    """
    print(f"[DEBUG] 查找输入文件，案例目录: {case_dir}, 命令: {command}")
    for root, _, files in os.walk(case_dir):
        for file in files:
            if command in file:
                file_path = os.path.join(root, file)
                print(f"[DEBUG] 找到输入文件: {file_path}")
                return file_path
    
    print(f"[WARNING] 未找到包含命令 {command} 的输入文件")
    return ""

def retrieve_faiss(database_name: str, query: str, topk: int = 1) -> dict:
    """
    从FAISS数据库中检索相似案例
    
    Args:
        database_name: 数据库名称
        query: 查询文本
        topk: 返回的相似文档数量
        
    Returns:
        格式化的检索结果字典
        
    Raises:
        ValueError: 当数据库不存在或未找到文档时
    """
    print(f"[DEBUG] FAISS检索 - 数据库: {database_name}, 查询: {query[:50]}..., topk: {topk}")
    
    if database_name not in FAISS_DB_CACHE:
        raise ValueError(f"Database '{database_name}' is not loaded.")
    
    # 对查询进行分词处理
    query = tokenize(query)
    
    vectordb = FAISS_DB_CACHE[database_name]
    docs = vectordb.similarity_search(query, k=topk)
    if not docs:
        raise ValueError(f"No documents found for query: {query}")
    
    print(f"[DEBUG] 找到 {len(docs)} 个相似文档")
    
    formatted_results = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata or {}
        print(f"[DEBUG] 文档 {i+1} 元数据字段: {list(metadata.keys())}")
        # print(f"[DEBUG] 文档 {i+1} 详细元数据:")
        # for key, value in metadata.items():
        #     if key == 'full_content':
        #         print(f"    {key}: {str(value)[:100]}... (长度: {len(str(value))})")
        #     else:
        #         print(f"    {key}: {value}")
        # print()
        
        if database_name == "openfoam_allrun_scripts":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "allrun_script": metadata.get("allrun_script", "N/A")
            })
        elif database_name == "openfoam_command_help":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "command": metadata.get("command", "unknown"),
                "help_text": metadata.get("help_text", "unknown")
            })
        elif database_name == "openfoam_tutorials_structure":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown")
            })
        elif database_name == "openfoam_tutorials_details":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "tutorials": metadata.get("tutorials", "N/A")
            })
        else:
            raise ValueError(f"Unknown database name: {database_name}")
    
    print(f"[DEBUG] 格式化完成，返回 {len(formatted_results)} 个结果")
    return formatted_results

def parse_directory_structure(data: str) -> dict:
    """
    解析目录结构字符串，返回字典，其中：
      - 键：目录名
      - 值：该目录中的文件数量
    
    Args:
        data: 包含目录结构信息的字符串
        
    Returns:
        目录文件数量字典
    """
    print(f"[DEBUG] 解析目录结构，数据长度: {len(data)}")
    directory_file_counts = {}

    # 在输入字符串中查找所有 <dir>...</dir> 块
    dir_blocks = re.findall(r'<dir>(.*?)</dir>', data, re.DOTALL)
    print(f"[DEBUG] 找到 {len(dir_blocks)} 个目录块")

    for i, block in enumerate(dir_blocks):
        print(f"[DEBUG] 处理目录块 {i+1}")
        # 提取目录名（"directory name:" 之后到第一个句号之前的所有内容）
        dir_name_match = re.search(r'directory name:\s*(.*?)\.', block)
        # 提取方括号中的文件名列表
        files_match = re.search(r'File names in this directory:\s*\[(.*?)\]', block)
        
        if dir_name_match and files_match:
            dir_name = dir_name_match.group(1).strip()
            files_str = files_match.group(1)
            # 按逗号分割文件名，去除周围的空白字符
            file_list = [filename.strip() for filename in files_str.split(',')]
            directory_file_counts[dir_name] = len(file_list)
            print(f"[DEBUG] 目录: {dir_name}, 文件数量: {len(file_list)}")
        else:
            print(f"[WARNING] 目录块 {i+1} 格式不正确")

    print(f"[DEBUG] 解析完成，共 {len(directory_file_counts)} 个目录")
    return directory_file_counts
