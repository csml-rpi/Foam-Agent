#!/usr/bin/env python
import os
import re
import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

def extract_field(field_name: str, text: str) -> str:
    """
    从文本中提取指定字段的内容。
    例如：field_name="case name"，text包含"case name: cavity"
    """
    match = re.search(fr"{field_name}:\s*(.*)", text)
    return match.group(1).strip() if match else "Unknown"

def tokenize(text: str) -> str:
    """
    对文本进行标准化处理：
    1. 下划线转空格
    2. 驼峰分词
    3. 转小写
    """
    text = text.replace('_', ' ')
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    return text.lower()

def main():
    """
    主流程：
    1. 解析参数
    2. 读取Allrun脚本原始数据
    3. 解析每个案例的结构和元数据
    4. 生成向量并存入FAISS
    5. 保存索引
    """
    # 1. 解析命令行参数
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
    print(f"📂 Allrun数据库路径: {database_path}")

    # 2. 读取输入文件
    database_allrun_path = os.path.join(database_path, "raw/openfoam_allrun_scripts.txt")
    print(f"📄 读取文件: {database_allrun_path}")
    if not os.path.exists(database_allrun_path):
        raise FileNotFoundError(f"File not found: {database_allrun_path}")

    with open(database_allrun_path, "r", encoding="utf-8") as file:
        file_content = file.read()
    print(f"📊 文件大小: {len(file_content)} 字符")
    print(f"📋 文件内容预览: {file_content[:200]}...")

    # 3. 用正则提取每个<case_begin>...</case_end>片段
    pattern = re.compile(r"<case_begin>(.*?)</case_end>", re.DOTALL)
    matches = pattern.findall(file_content)
    print(f"🔍 找到 {len(matches)} 个案例片段")
    if not matches:
        raise ValueError("No cases found in the input file. Please check the file content.")

    documents = []
    for i, match in enumerate(matches):
        print(f"\n📋 处理案例 {i+1}/{len(matches)}")
        # 提取<index>内容
        index_match = re.search(r"<index>(.*?)</index>", match, re.DOTALL)
        if not index_match:
            print("  ❌ 未找到<index>标签，跳过")
            continue
        index_content = index_match.group(0).strip()
        full_content = match.strip()

        # 提取目录结构
        dir_match = re.search(r"<directory_structure>(.*?)</directory_structure>", match, re.DOTALL)
        dir_structure = dir_match.group(0).strip() if dir_match else "Unknown"
        print(f"  📂 目录结构长度: {len(dir_structure)} 字符")
        print(f"  📂 目录结构预览: {dir_structure[:100]}...")

        # 提取元数据字段
        case_name = extract_field("case name", index_content)
        case_domain = extract_field("case domain", index_content)
        case_category = extract_field("case category", index_content)
        case_solver = extract_field("case solver", index_content)
        print(f"  🏷️ 名称: {case_name}, 域: {case_domain}, 类别: {case_category}, 求解器: {case_solver}")

        # 只保留部分index内容用于向量化
        index_content = f"<index>\ncase name: {case_name}\ncase solver: {case_solver}</index>"
        print(f"  📝 用于向量化的index内容: {index_content}")

        # 提取Allrun脚本内容
        script_match = re.search(r"<allrun_script>([\\s\\S]*?)</allrun_script>", full_content)
        case_allrun_script = script_match.group(1).strip() if script_match else "Unknown"
        print(f"  📜 Allrun脚本长度: {len(case_allrun_script)} 字符")

        # 生成Document对象
        doc = Document(
            page_content=tokenize(index_content + dir_structure),
            metadata={
                "full_content": full_content,
                "case_name": case_name,
                "case_domain": case_domain,
                "case_category": case_category,
                "case_solver": case_solver,
                "dir_structure": dir_structure,
                "allrun_script": case_allrun_script,
            },
        )
        documents.append(doc)
        print(f"  ✅ 文档创建成功")

    print(f"\n📊 共创建 {len(documents)} 个文档，开始生成向量嵌入...")

    # 4. 计算嵌入并存入FAISS
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = FAISS.from_documents(documents, embeddings)
    print(f"✅ 向量嵌入生成完成，数据库大小: {vectordb.index.ntotal}")

    # 5. 保存FAISS索引
    persist_directory = os.path.join(database_path, "faiss/openfoam_allrun_scripts")
    print(f"💾 保存FAISS索引到: {persist_directory}")
    vectordb.save_local(persist_directory)
    print(f"🎉 {len(documents)} cases indexed successfully with metadata! Saved at: {persist_directory}")

if __name__ == "__main__":
    main()


'''
==================== 为什么只保留部分index内容用于向量化？ ====================

【核心目的】
提升检索的相关性和效率，让向量空间更聚焦于案例的核心身份和用途。

1. 减少无关信息干扰：
   - 原始<index>里包含domain、category等字段，这些字段对“脚本内容的语义检索”帮助不大，反而可能引入噪声。
   - 只保留case name和case solver，让向量更聚焦于案例的本质。

2. 提升向量检索的区分度：
   - 如果把所有元数据都拼进去，很多案例的domain/category会重复，导致向量空间中不同案例的距离变小，检索时容易混淆。
   - 只用最能区分案例的字段（如名称和求解器），能让相似案例的向量距离更近，不相关的距离更远。

3. 避免高维稀疏噪声：
   - 文本越长，嵌入模型越容易“稀释”关键信息，反而不如短文本聚焦。
   - 只用关键信息，能让嵌入模型更好地捕捉语义。

4. 实际检索需求驱动：
   - 用户检索Allrun脚本时，最关心的是“案例名”和“用的求解器”，而不是它属于哪个领域或类别。
   - 这样检索出来的结果更贴合用户需求。

【例子】
原始index内容：
<index>
case name: cavity
case domain: incompressible
case category: basic
case solver: icoFoam
</index>

只保留部分后：
<index>
case name: cavity
case solver: icoFoam
</index>

这样，向量化时只关注“cavity+icoFoam”这个组合，能更精准地定位到你想要的案例。

【结论】
- 只保留部分index内容用于向量化，是为了让语义检索更精准、更高效、更贴合实际需求。
- 其它元数据依然保存在metadata里，后续可以用来展示或过滤，但不参与向量化。

==========================================================================
'''


