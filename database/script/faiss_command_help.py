import os
import re
import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

def tokenize(text: str) -> str:
    """
    对文本进行预处理和标准化
    
    该函数会：
    1. 将下划线替换为空格
    2. 在驼峰命名法的小写字母和大写字母之间插入空格
    3. 将文本转换为小写
    
    参数:
        text (str): 输入的文本，通常是OpenFOAM命令名
    
    返回:
        str: 处理后的标准化文本
    
    示例:
        "blockMesh" -> "block mesh"
        "interFoam" -> "inter foam"
        "simpleFoam" -> "simple foam"
    """
    print(f"🔤 原始文本: '{text}'")
    
    # 将下划线替换为空格
    text = text.replace('_', ' ')
    print(f"  🧹 替换下划线后: '{text}'")
    
    # 在驼峰命名法的小写字母和大写字母之间插入空格
    # (?<=[a-z]) - 正向向后查找，匹配小写字母
    # (?=[A-Z]) - 正向向前查找，匹配大写字母
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    print(f"  🔤 处理驼峰命名后: '{text}'")
    
    # 转换为小写
    text = text.lower()
    print(f"  📝 最终标准化文本: '{text}'")
    
    return text

def main():
    """
    主函数：处理OpenFOAM命令帮助数据并创建FAISS向量索引
    
    工作流程：
    1. 解析命令行参数
    2. 读取OpenFOAM命令帮助文件
    3. 提取命令和帮助信息
    4. 创建文档对象
    5. 生成向量嵌入
    6. 保存FAISS索引
    """
    print("🚀 开始处理OpenFOAM命令帮助数据...")
    
    # 步骤1：解析命令行参数
    parser = argparse.ArgumentParser(
        description="Process OpenFOAM case data and store embeddings in FAISS."
    )
    parser.add_argument(
        "--database_path",
        type=str,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the database directory (default: '../../')",
    )
        
    args = parser.parse_args()
    database_path = args.database_path
    print(f"📂 数据库路径: {database_path}")
        
    # 步骤2：读取输入文件
    database_allrun_path = os.path.join(database_path, "raw/openfoam_command_help.txt")
    print(f"📄 读取文件: {database_allrun_path}")
    
    if not os.path.exists(database_allrun_path):
        raise FileNotFoundError(f"文件未找到: {database_allrun_path}")

    with open(database_allrun_path, "r", encoding="utf-8") as file:
        file_content = file.read()
    
    print(f"📊 文件大小: {len(file_content)} 字符")
    print(f"📋 文件内容预览: {file_content[:200]}...")

    # 步骤3：使用正则表达式提取 `<command_begin> ... </command_end>` 片段
    # 这个模式会匹配所有被 <command_begin> 和 </command_end> 包围的内容
    pattern = re.compile(r"<command_begin>(.*?)</command_end>", re.DOTALL)
    matches = pattern.findall(file_content)
    
    print(f"🔍 找到 {len(matches)} 个命令片段")
    
    if not matches:
        raise ValueError("输入文件中未找到任何命令。请检查文件内容。")

    documents = []
    print(f"📝 开始处理 {len(matches)} 个命令...")

    for i, match in enumerate(matches):
        print(f"\n📋 处理命令 {i+1}/{len(matches)}:")
        
        # 提取命令名称
        command_match = re.search(r"<command>(.*?)</command>", match, re.DOTALL)
        if command_match:
            command = command_match.group(1).strip()
            print(f"  🔧 命令名称: {command}")
        else:
            print(f"  ❌ 无法提取命令名称")
            continue
        
        # 提取帮助文本
        help_match = re.search(r"<help_text>(.*?)</help_text>", match, re.DOTALL)
        if help_match:
            help_text = help_match.group(1).strip()
            print(f"  📖 帮助文本长度: {len(help_text)} 字符")
            print(f"  📖 帮助文本预览: {help_text[:100]}...")
        else:
            print(f"  ❌ 无法提取帮助文本")
            continue
        
        full_content = match.strip()  # 存储完整的命令片段
        print(f"  📄 完整内容长度: {len(full_content)} 字符")
        
        # 对命令名称进行标准化处理
        tokenized_command = tokenize(command)
        print(f"  🔤 标准化命令名: '{tokenized_command}'")
        
        # 创建Document实例
        # page_content: 用于向量化的文本（标准化的命令名）
        # metadata: 存储额外的元数据信息
        doc = Document(
            page_content=tokenized_command, 
            metadata={
                "full_content": full_content,  # 完整的原始内容
                "command": command,            # 原始命令名
                "help_text": help_text         # 帮助文本
            }
        )
        
        documents.append(doc)
        print(f"  ✅ 文档创建成功")

    print(f"\n📊 文档处理完成，共创建 {len(documents)} 个文档")
    
    # 显示前几个文档的示例
    if documents:
        print(f"📋 文档示例:")
        for i, doc in enumerate(documents[:3]):
            print(f"  {i+1}. 命令: '{doc.metadata['command']}' -> 标准化: '{doc.page_content}'")

    # 步骤4：计算嵌入向量并存储在FAISS中
    print(f"\n🧠 开始生成向量嵌入...")
    print(f"🔧 使用模型: text-embedding-3-small")
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 从文档创建FAISS向量数据库
    # 这会为每个文档的page_content生成向量嵌入
    vectordb = FAISS.from_documents(documents, embedding_model)
    
    print(f"✅ 向量嵌入生成完成")
    print(f"📊 向量数据库大小: {vectordb.index.ntotal} 个向量")

    # 步骤5：本地保存FAISS索引
    persist_directory = os.path.join(database_path, "faiss/openfoam_command_help")
    print(f"💾 保存FAISS索引到: {persist_directory}")
    
    # 确保目录存在
    os.makedirs(persist_directory, exist_ok=True)
    
    # 保存向量数据库
    vectordb.save_local(persist_directory)

    print(f"🎉 成功索引 {len(documents)} 个命令！")
    print(f"💾 保存位置: {persist_directory}")
    print(f"📈 索引包含元数据，可用于后续的语义搜索")

if __name__ == "__main__":
    main()


'''
==================== FAISS 索引文件说明 ====================

本脚本运行后，会在 faiss/openfoam_command_help/ 目录下生成两个主要文件：

1. index.faiss
   - 类型：二进制文件
   - 作用：存储所有命令帮助文本经过嵌入（embedding）后生成的高维向量，以及用于高效相似性搜索的FAISS内部索引结构。
   - 类比：就像“查找相似命令”的加速器，里面存的是“命令的数学特征”和“如何快速查找最像的命令”的算法数据。

2. index.pkl
   - 类型：Python pickle序列化文件
   - 作用：存储每个向量对应的原始信息，比如命令名、帮助文本等（即metadata部分）。
   - 类比：就像“查到相似命令后，怎么把它还原成人话”的说明书，里面存的是“命令的原文、帮助内容”等。

【配合使用】
- 检索时，先用 .faiss 文件查找最相似的向量（速度极快，适合大规模数据）。
- 查到“第N个向量”后，再用 .pkl 文件找到第N个命令的原始信息，实现“数学结果”到“人类可读内容”的还原。

【形象比喻】
- .faiss：图书馆的索引卡片柜，帮你快速定位到最相关的书。
- .pkl：书的目录和内容，拿到卡片号后去书架上把书拿出来看。

===========================================================
'''

