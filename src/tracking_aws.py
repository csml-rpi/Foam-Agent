"""
AWS Bedrock使用跟踪模块

功能：跟踪和记录AWS Bedrock API的使用情况，包括token消耗和成本计算
用途：监控LLM服务的使用量，帮助成本控制和资源管理

# 使用方法

初始化客户端如下：

```python
import tracking_aws

bedrock_runtime = tracking_aws.new_default_client()
```

您的使用情况将被跟踪并记录在名为 `usage_nrel_aws.json` 的本地文件中。
"""
from __future__ import annotations
from math import nan
import itertools
import os
import io
import json
import pathlib
from typing import Union, Dict
import pathlib
from typing import Any
from contextlib import contextmanager

import boto3

# 定义使用情况的数据类型：包含整数或浮点数的字典
Usage = Dict[str, Union[int, float]]  # TODO: 可以使用Counter类来简化
# 默认使用情况文件路径
default_usage_file = pathlib.Path("usage_nrel_aws.json")

# AWS Bedrock模型ARN定义
# 这些是NREL的特定模型ARN，用于访问Claude模型
CLAUDE_3_5_HAIKU = 'arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/g47vfd2xvs5w'
CLAUDE_3_5_SONNET = 'arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/56i8iq1vib3e'

# 价格表：每1000个token的价格（美元）
# 价格基于2025年1月的AWS Bedrock定价，但不考虑缓存或批处理的折扣
# 注意：并非所有模型都列在此文件中，也不包括微调API
pricing = {  # 每1000个token的价格（美元）
    CLAUDE_3_5_HAIKU: {'input': 0.0008, 'output': 0.004},    # 输入0.0008美元/1000 tokens，输出0.004美元/1000 tokens
    CLAUDE_3_5_SONNET: {'input': 0.003, 'output': 0.015},    # 输入0.003美元/1000 tokens，输出0.015美元/1000 tokens
}

# 默认模型设置
# 即使被评估的系统使用便宜的default_model，也可能需要使用更昂贵的default_eval_model进行仔细评估
default_model = CLAUDE_3_5_HAIKU        # 默认使用的模型
default_eval_model = CLAUDE_3_5_HAIKU   # 默认评估模型


# 上下文管理器：允许在代码块中临时更改默认模型
# 可以这样使用：
#     with use_model('arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/g47vfd2xvs5w'):
#        ...
#
#     with use_model(eval_model='arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/g47vfd2xvs5w'):
#        ...
@contextmanager
def use_model(model: str = default_model, eval_model: str = default_eval_model):
    """
    临时更改默认模型的上下文管理器
    
    功能：在代码块执行期间临时更改默认模型，执行完毕后恢复原设置
    执行逻辑：
    1. 保存当前的默认模型设置
    2. 设置新的默认模型
    3. 执行代码块
    4. 恢复原来的模型设置
    
    Args:
        model: 临时使用的模型ARN
        eval_model: 临时使用的评估模型ARN
    """
    global default_model, default_eval_model
    print(f"[DEBUG] 临时更改模型设置 - 原模型: {default_model}, 新模型: {model}")
    
    # 保存当前设置
    save_model, save_eval_model = default_model, default_eval_model
    # 设置新模型
    default_model, default_eval_model = model, eval_model
    
    try:
        yield  # 执行代码块
    finally:
        # 恢复原设置
        default_model, default_eval_model = save_model, save_eval_model
        print(f"[DEBUG] 恢复模型设置 - 模型: {default_model}")


def track_usage(client: boto3.client, path: pathlib.Path = default_usage_file) -> boto3.client:
    """
    为boto3客户端添加使用情况跟踪功能
    
    功能：修改客户端，使其API调用能够记录token消耗到指定文件
    执行逻辑：
    1. 保存原始的invoke_model方法
    2. 创建新的跟踪方法，在每次调用后记录使用情况
    3. 替换客户端的invoke_model方法
    4. 返回修改后的客户端
    
    Args:
        client: 要添加跟踪功能的boto3客户端
        path: 使用情况记录文件路径
        
    Returns:
        boto3.client: 添加了跟踪功能的客户端
        
    Example:
        >>> client = boto3.client('bedrock')
        >>> track_usage(client, "example_usage_file.json")
        >>> type(client)
        <class 'botocore.client.BaseClient'>
    """
    print(f"[DEBUG] 为客户端添加使用情况跟踪，记录文件: {path}")
    
    # 保存原始的invoke_model方法
    old_invoke_model = client.invoke_model
    print(f"[DEBUG] 原始invoke_model方法: {old_invoke_model}")

    def tracked_invoke_model(*args, **kwargs) -> Any:
        """
        跟踪版本的invoke_model方法
        
        功能：在调用原始方法后，记录token使用情况和成本
        执行逻辑：
        1. 调用原始的invoke_model方法
        2. 从响应中提取使用情况信息
        3. 读取现有的使用情况记录
        4. 合并新的使用情况
        5. 写入更新后的记录
        """
        print(f"[DEBUG] 调用跟踪的invoke_model，模型ID: {kwargs.get('modelId', 'Unknown')}")
        
        # 调用原始方法
        response = old_invoke_model(*args, **kwargs)
        
        # 读取现有使用情况
        old: Usage = read_usage(path)
        print(f"[DEBUG] 现有使用情况: {old}")
        
        # 获取新的使用情况
        new, response_body = get_usage(response, model=kwargs.get('modelId', None))
        print(f"[DEBUG] 本次使用情况: {new}")
        
        # 合并并写入使用情况
        merged_usage = _merge_usage(old, new)
        _write_usage(merged_usage, path)
        print(f"[DEBUG] 更新后总使用情况: {merged_usage}")
        
        return response_body

    # 替换客户端的invoke_model方法
    client.invoke_model = tracked_invoke_model  # type:ignore
    print(f"[DEBUG] 已替换客户端的invoke_model方法")
    
    return client


def get_usage(response, model=None) -> tuple[Usage, dict]:
    """
    从AWS Bedrock响应中提取使用情况信息
    
    功能：解析API响应，计算token消耗和成本
    执行逻辑：
    1. 解析响应体，提取token数量
    2. 根据模型查找价格信息
    3. 计算总成本
    4. 返回使用情况字典和响应体
    
    Args:
        response: AWS Bedrock API响应
        model: 模型ARN，用于查找价格信息
        
    Returns:
        tuple: (使用情况字典, 响应体)
        
    Raises:
        ValueError: 当模型价格信息未知时
    """
    print(f"[DEBUG] 解析使用情况，模型: {model}")
    
    # 解析响应体
    response_body = json.loads(response['body'].read().decode())
    print(f"[DEBUG] 响应体: {response_body}")
    
    # 提取token使用情况
    usage: Usage = {
        'input_tokens': response_body['usage']['input_tokens'],
        'output_tokens': response_body['usage']['output_tokens']
    }
    print(f"[DEBUG] Token使用情况: 输入={usage['input_tokens']}, 输出={usage['output_tokens']}")

    # 计算成本
    try:
        costs = pricing[model]  # 根据模型查找价格
        print(f"[DEBUG] 模型价格: {costs}")
    except KeyError:
        error_msg = f"未知模型价格: {model} 或 {response.model}"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    # 计算总成本（每1000个token的价格）
    cost = (usage.get('input_tokens', 0) * costs['input'] 
            + usage.get('output_tokens', 0) * costs['output']) / 1_000
    usage['cost'] = cost
    
    print(f"[DEBUG] 计算成本: ${cost:.6f}")
    return usage, response_body


def read_usage(path: pathlib.Path = default_usage_file) -> Usage:
    """
    从文件中读取总使用情况
    
    功能：读取JSON文件中的使用情况记录
    执行逻辑：
    1. 检查文件是否存在
    2. 如果存在，读取并解析JSON
    3. 如果不存在，返回空字典
    
    Args:
        path: 使用情况文件路径
        
    Returns:
        Usage: 使用情况字典
    """
    print(f"[DEBUG] 读取使用情况文件: {path}")
    
    if os.path.exists(path):
        with open(path, "rt") as f:
            usage_data = json.load(f)
            print(f"[DEBUG] 读取到的使用情况: {usage_data}")
            return usage_data
    else:
        print(f"[DEBUG] 文件不存在，返回空使用情况")
        return {}


def _write_usage(u: Usage, path: pathlib.Path):
    """
    将使用情况写入文件
    
    功能：将使用情况字典以JSON格式写入文件
    执行逻辑：
    1. 以写入模式打开文件
    2. 将字典转换为JSON格式
    3. 写入文件
    
    Args:
        u: 使用情况字典
        path: 文件路径
    """
    print(f"[DEBUG] 写入使用情况到文件: {path}")
    print(f"[DEBUG] 写入内容: {u}")
    
    with open(path, "wt") as f:
        json.dump(u, f, indent=4)
    
    print(f"[DEBUG] 使用情况已写入文件")


def _merge_usage(u1: Usage, u2: Usage) -> Usage:
    """
    合并两个使用情况字典
    
    功能：将两个使用情况字典的对应值相加
    执行逻辑：
    1. 获取所有键的并集
    2. 对每个键，将两个字典中的值相加（如果不存在则为0）
    3. 返回合并后的字典
    
    Args:
        u1: 第一个使用情况字典
        u2: 第二个使用情况字典
        
    Returns:
        Usage: 合并后的使用情况字典
    """
    print(f"[DEBUG] 合并使用情况: {u1} + {u2}")
    
    merged = {k: u1.get(k, 0) + u2.get(k, 0) for k in itertools.chain(u1, u2)}
    print(f"[DEBUG] 合并结果: {merged}")
    
    return merged


def new_default_client(default='boto3') -> boto3.client:
    """
    创建新的默认跟踪客户端
    
    功能：基于当前AWS凭据创建新的跟踪客户端
    执行逻辑：
    1. 创建bedrock-runtime客户端
    2. 为客户端添加使用情况跟踪功能
    3. 设置全局默认客户端
    4. 返回跟踪客户端
    
    注意：如果您的凭据发生变化，应该再次调用此方法
    
    Args:
        default: 客户端类型（当前未使用）
        
    Returns:
        boto3.client: 添加了跟踪功能的客户端
    """
    print(f"[DEBUG] 创建新的默认跟踪客户端")
    
    global default_client
    
    # 创建bedrock-runtime客户端并添加跟踪功能
    default_client = track_usage(
        boto3.client('bedrock-runtime', region_name='us-west-2')
    )
    
    print(f"[DEBUG] 默认客户端已设置: {default_client}")
    return default_client


# 在导入模块时立即设置默认客户端
# new_default_client()

if __name__ == "__main__":
    import doctest
    print("[DEBUG] 运行doctest测试")
    doctest.testmod()

# 这个代码实现了一个实用的AWS Bedrock使用监控系统，特别适合需要控制LLM服务成本的场景。
# 通过自动跟踪和成本计算，可以帮助用户更好地管理API使用量。
