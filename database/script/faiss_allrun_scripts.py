#!/usr/bin/env python
import os
import re
import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document


def extract_field(field_name: str, text: str) -> str:
    """Extract the specified field from the given text."""
    match = re.search(fr"{field_name}:\s*(.*)", text)
    return match.group(1).strip() if match else "Unknown"

def tokenize(text: str) -> str:
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    # Insert a space between a lowercase letter and an uppercase letter (global match)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    return text.lower()

def main():
    # Step 1: Parse command-line arguments
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
    print(f"Database path: {database_path}")

    # Step 2: Read the input file
    database_allrun_path = os.path.join(database_path, "raw/openfoam_allrun_scripts.txt")
    if not os.path.exists(database_allrun_path):
        raise FileNotFoundError(f"File not found: {database_allrun_path}")

    with open(database_allrun_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    # Step 3: Extract segments using regex
    pattern = re.compile(r"<case_begin>(.*?)</case_end>", re.DOTALL)
    matches = pattern.findall(file_content)
    if not matches:
        raise ValueError("No cases found in the input file. Please check the file content.")

    documents = []
    for match in matches:
        # Extract <index> content
        index_match = re.search(r"<index>(.*?)</index>", match, re.DOTALL)
        if not index_match:
            continue
        index_content = index_match.group(0).strip()
        full_content = match.strip()

        # Extract directory structure
        dir_match = re.search(r"<directory_structure>(.*?)</directory_structure>", match, re.DOTALL)
        dir_structure = dir_match.group(0).strip() if dir_match else "Unknown"

        # Extract metadata fields from index_content
        case_name = extract_field("case name", index_content)
        case_domain = extract_field("case domain", index_content)
        case_category = extract_field("case category", index_content)
        case_solver = extract_field("case solver", index_content)
        
        # allrun script content is not sensitive to case domain and category
        index_content = f"<index>\ncase name: {case_name}\ncase solver: {case_solver}\n</index>\n"

        # Extract allrun script content from full_content
        script_match = re.search(r"<allrun_script>([\s\S]*?)</allrun_script>", full_content)
        case_allrun_script = script_match.group(1).strip() if script_match else "Unknown"

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

    # Step 4: Compute embeddings and store in FAISS
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = FAISS.from_documents(documents, embeddings)

    # Step 5: Save the FAISS index locally
    persist_directory = os.path.join(database_path, "faiss/openfoam_allrun_scripts")
    vectordb.save_local(persist_directory)

    print(f"{len(documents)} cases indexed successfully with metadata! Saved at: {persist_directory}")


if __name__ == "__main__":
    main()
