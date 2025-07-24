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
    2. 读取教程结构原始数据
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
    print(f"📂 教程结构数据库路径: {database_path}")

    # 2. 读取输入文件
    database_allrun_path = os.path.join(database_path, "raw/openfoam_tutorials_structure.txt")
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
        full_content = match.strip()  # 存储完整的案例内容

        # 提取<index>内容
        index_match = re.search(r"<index>(.*?)</index>", match, re.DOTALL)
        if not index_match:
            print("  ❌ 未找到<index>标签，跳过")
            continue
        index_content = index_match.group(1).strip()  # 只提取<index>内部内容
        print(f"  📝 index内容预览: {index_content[:100]}...")

        # 提取元数据字段
        case_name = extract_field("case name", index_content)
        case_domain = extract_field("case domain", index_content)
        case_category = extract_field("case category", index_content)
        case_solver = extract_field("case solver", index_content)
        print(f"  🏷️ 名称: {case_name}, 域: {case_domain}, 类别: {case_category}, 求解器: {case_solver}")

        # 提取目录结构
        dir_match = re.search(r"<directory_structure>([\s\S]*?)</directory_structure>", full_content)
        case_directory_structure = dir_match.group(1) if dir_match else "Unknown"
        print(f"  📂 目录结构长度: {len(case_directory_structure)} 字符")
        print(f"  📂 目录结构预览: {case_directory_structure[:100]}...")

        # 创建Document对象
        doc = Document(
            page_content=tokenize(index_content),  # 只用<index>内容做向量化
            metadata={
                "full_content": full_content,  # 存完整内容
                "case_name": case_name,
                "case_domain": case_domain,
                "case_category": case_category,
                "case_solver": case_solver,
                'dir_structure': case_directory_structure
            }
        )
        documents.append(doc)
        print(f"  ✅ 文档创建成功")

    print(f"\n📊 共创建 {len(documents)} 个文档，开始生成向量嵌入...")

    # 4. 计算嵌入并存入FAISS
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = FAISS.from_documents(documents, embedding_model)
    print(f"✅ 向量嵌入生成完成，数据库大小: {vectordb.index.ntotal}")

    # 5. 保存FAISS索引
    persist_directory = os.path.join(database_path, "faiss/openfoam_tutorials_structure")
    print(f"💾 保存FAISS索引到: {persist_directory}")
    vectordb.save_local(persist_directory)
    print(f"🎉 {len(documents)} cases indexed successfully with metadata! Saved at: {persist_directory}")

if __name__ == "__main__":
    main()
    
