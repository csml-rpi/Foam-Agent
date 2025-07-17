# config.py
# 配置文件模块 - 定义Foam-Agent项目的全局配置参数
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """
    Foam-Agent项目配置类
    
    使用dataclass装饰器自动生成__init__、__repr__等方法
    所有参数都有默认值，便于配置管理
    """
    
    # 最大循环次数 - 控制AI代理的最大执行轮次，防止无限循环
    max_loop: int = 2 # 50
    
    # 批处理大小 - 每次处理文档或任务的数量
    batchsize: int = 10
    
    # 搜索文档数量 - 每次搜索时返回的相关文档数量
    searchdocs: int = 2
    
    # 运行次数 - 当前运行的编号，用于目录命名和区分不同运行
    run_times: int = 1  # current run number (for directory naming)
    
    # 数据库路径 - 存储项目数据的目录路径
    # 使用Path(__file__).resolve().parent.parent获取项目根目录
    database_path: Path = Path(__file__).resolve().parent.parent / "database"
    
    # 运行目录 - 存储运行结果和日志的目录路径
    run_directory: Path = Path(__file__).resolve().parent.parent / "runs"
    
    # 案例目录 - 特定案例的目录路径，默认为空字符串
    case_dir: str = ""
    
    # 最大时间限制 - OpenFOAM运行的最大时间限制（秒），超时后自动终止，这个很有意思
    max_time_limit = 36000 # Max time limit after which the openfoam run will be terminated
    
    # 模型提供商 - 指定使用的AI模型服务提供商
    model_provider: str = "openai" # [openai, ollama]
    
    # 模型版本 - 指定使用的具体AI模型版本
    # model_version should be in ["gpt-4o", "deepseek-r1:32b-qwen-distill-fp16", "qwen2.5:32b-instruct"]
    model_version: str = "gpt-4o"
    
    # 温度参数 - 控制AI模型输出的随机性，值越低输出越确定性
    temperature: float = 0.1
    