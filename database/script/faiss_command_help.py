import os
import re
import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

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
    database_allrun_path = os.path.join(database_path, "raw/openfoam_command_help.txt")
    if not os.path.exists(database_allrun_path):
        raise FileNotFoundError(f"File not found: {database_allrun_path}")

    with open(database_allrun_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    # Step 3: Extract `<command_begin> ... </command_end>` segments using regex
    pattern = re.compile(r"<command_begin>(.*?)</command_end>", re.DOTALL)
    matches = pattern.findall(file_content)

    if not matches:
        raise ValueError("No cases found in the input file. Please check the file content.")

    documents = []

    for match in matches:
        command = re.search(r"<command>(.*?)</command>", match, re.DOTALL).group(1).strip()
        help_text = re.search(r"<help_text>(.*?)</help_text>", match, re.DOTALL).group(1).strip()
        full_content = match.strip()  # Store the complete case
        
        # Create a Document instance
        documents.append(Document(
            page_content=tokenize(command), 
            metadata={
                "full_content": full_content,
                "command": command,
                "help_text": help_text
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
    persist_directory = os.path.join(database_path, f"faiss/{model_dir_name}/openfoam_command_help")
    vectordb.save_local(persist_directory)

    print(f"{len(documents)} cases indexed successfully with metadata! Saved at: {persist_directory}")

if __name__ == "__main__":
    main()
    