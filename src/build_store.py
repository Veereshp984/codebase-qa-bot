
from pathlib import Path
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.load_files import load_files, convert_to_documents
from src.split_code import split_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "db"


def generate_chunk_ids(chunks):

    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk_index = chunk.metadata["chunk_index"]

        chunk_id = f"{source}_{chunk_index}"

        yield chunk_id


def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store(
    chunks,
    repo_id,
    rebuild=False
):

    embedding_model = get_embedding_model()

    db_path = DB_DIR / repo_id

    if rebuild and db_path.exists():
        shutil.rmtree(db_path)
        print("Existing database deleted!")

    if db_path.exists():

        print("Existing database found. Loading it...")

        vector_store = Chroma(
            collection_name=repo_id,
            embedding_function=embedding_model,
            persist_directory=str(db_path)
        )

    else:

        print("Creating new vector database...")

        vector_store = Chroma(
            collection_name=repo_id,
            embedding_function=embedding_model,
            persist_directory=str(db_path),
            collection_metadata={"hnsw:space": "cosine"}
        )

        chunk_ids = list(generate_chunk_ids(chunks))

        vector_store.add_documents(
            documents=chunks,
            ids=chunk_ids
        )

    return vector_store


def process_repository(repo_path, repo_id, rebuild=False):

    repo_path = Path(repo_path)

    if not repo_path.exists():
        raise FileNotFoundError(
            f"Repository not found: {repo_path}"
        )

    print("Processing repository:", repo_id)

    # 1. Load repository files
    files = load_files(repo_path)

    if not files:
        raise ValueError("No supported source files found.")

    # 2. Convert files into Documents
    documents = convert_to_documents(files)

    # 3. Split documents into chunks
    chunks = split_documents(documents)

    print("Total files:", len(files))
    print("Total chunks:", len(chunks))

    # 4. Create vector store
    vector_store = create_vector_store(
        chunks=chunks,
        repo_id=repo_id,
        rebuild=rebuild
    )

    print("Vector store created successfully!")
    print("Total stored chunks:", vector_store._collection.count())

    return vector_store


if __name__ == "__main__":

    repo_name = "ChargeFlow-EV-Charging-Management-Platform"
    owner = "Veereshp984"

    repo_id = f"{owner}_{repo_name}"

    repo_path = (
        PROJECT_ROOT
        / "repos"
        / owner
        / repo_name
    )

    process_repository(
        repo_path=repo_path,
        repo_id=repo_id,
        rebuild=True
    )
