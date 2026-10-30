import os
import re
import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

# Function to extract specific fields from text
def extract_field(field_name: str, text: str) -> str:
    """Extracts the specified field from the given text."""
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
        default=Path(__file__).resolve().parent.parent / "foundation-v10",
        help="Path to one target corpus (default: '../foundation-v10')",
    )
        
    parser.add_argument(
        "--embedding_provider",
        type=str,
        default="openai",
        choices=["openai", "huggingface", "ollama"],
        help="Embedding provider",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model name",
    )
        
    args = parser.parse_args()
    database_path = args.database_path
    embedding_provider = args.embedding_provider
    embedding_model = args.embedding_model

    print(f"Database path: {database_path}")
    print(f"Provider: {embedding_provider}, Model: {embedding_model}")
        
    # Step 2: Read the input file
    database_allrun_path = os.path.join(database_path, "raw/openfoam_tutorials_details.txt")
    if not os.path.exists(database_allrun_path):
        raise FileNotFoundError(f"File not found: {database_allrun_path}")

    with open(database_allrun_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    # Step 3: Extract `<case_begin> ... </case_end>` segments using regex
    pattern = re.compile(r"<case_begin>(.*?)</case_end>", re.DOTALL)
    matches = pattern.findall(file_content)

    if not matches:
        raise ValueError("No cases found in the input file. Please check the file content.")

    documents = []

    for match in matches:
        full_content = match.strip()  # Store the complete case
        
        index_match = re.search(r"<index>(.*?)</index>", match, re.DOTALL)
        index_content = index_match.group(1).strip()  # Extract `<index>` content
        index = index_match.group(0) # include the tags for indexing
        
        # Extract metadata fields
        case_name = extract_field("case name", index_content)
        case_domain = extract_field("case domain", index_content)
        case_category = extract_field("case category", index_content)
        case_solver = extract_field("case solver", index_content)
        directory_structure = re.search(r"<directory_structure>([\s\S]*?)</directory_structure>", full_content)
        case_directory_structure = directory_structure.group(1) # keep only the content for metadata
        dir_structure = directory_structure.group(0) # include the tags for indexing
        detailed_tutorial = re.search(r"<tutorials>([\s\S]*?)</tutorials>", full_content).group(1)

        # Create a Document instance
        documents.append(Document(
            page_content=tokenize(index + '\n' + dir_structure),
            metadata={
                "full_content": full_content,  # Store full `<case_begin> ... </case_end>`
                "case_name": case_name,
                "case_domain": case_domain,
                "case_category": case_category,
                "case_solver": case_solver,
                'dir_structure': case_directory_structure,
                'tutorials': detailed_tutorial
            }
        ))

    # Step 4: Compute embeddings and store them in FAISS
    if embedding_provider == "openai":
        embeddings = OpenAIEmbeddings(model=embedding_model)
    elif embedding_provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        except ImportError:
             raise ImportError("Please install langchain-huggingface")
    elif embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model=embedding_model)
    else:
        raise ValueError(f"Unknown provider: {embedding_provider}")

    vectordb = FAISS.from_documents(documents, embeddings)

    # Step 5: Save FAISS index locally
    model_dir_name = embedding_model.replace("/", "_").replace(":", "_")
    persist_directory = os.path.join(database_path, f"faiss/{model_dir_name}/openfoam_tutorials_details")
    vectordb.save_local(persist_directory)

    print(f"{len(documents)} cases indexed successfully with metadata! Saved at: {persist_directory}")

if __name__ == "__main__":
    main()
    