from pathlib import Path
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from load_files import load_files, convert_to_documents
from split_code import split_documents


def generate_chunk_ids(chunks):

    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk_index = chunk.metadata["chunk_index"]

        chunk_id = f"{source}_{chunk_index}"

        yield chunk_id
        
        
        

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model



def create_vector_store(chunks, repo_name, rebuild=False):

    embedding_model = get_embedding_model()

    db_path = Path("../db") / repo_name

    # Delete existing database if rebuild is True
    if rebuild and db_path.exists():
        shutil.rmtree(db_path)
        print("Existing database deleted!")

    # Load existing database
    if db_path.exists():
        print("Existing database found. Loading it...")

        vector_store = Chroma(
            collection_name=repo_name,
            embedding_function=embedding_model,
            persist_directory=str(db_path)
        )

    # Create new database
    else:
        print("Creating new vector database...")

        vector_store = Chroma(
            collection_name=repo_name,
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



def main():

    repo_name = "ChargeFlow-EV-Charging-Management-Platform"

    repo_path = Path("../repos") / repo_name

    # 1. Load repository files
    files = load_files(repo_path)

    # 2. Convert files into Documents
    documents = convert_to_documents(files)

    # 3. Split Documents into chunks
    chunks = split_documents(documents)

    print("Total files:", len(files))
    print("Total chunks:", len(chunks))

    # 4. Create ChromaDB vector store
    vector_store = create_vector_store(
    chunks,
    repo_name,
    rebuild=False
)

    
    print("Vector store created successfully!")
    
    print("Total stored chunks:", vector_store._collection.count())
    
    results = vector_store.similarity_search_with_score(
    "How does the booking system work?",
    k=3
)

    for doc, score in results:
        print("\nScore:", score)
        print("Source:", doc.metadata["source"])
        print("Chunk:", doc.metadata["chunk_index"])
        print("Content:", doc.page_content[:300])

    

if __name__ == "__main__":
    main()