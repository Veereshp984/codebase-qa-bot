
from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


REPO_NAME = "ChargeFlow-EV-Charging-Management-Platform"


# 1. Load embedding model
def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# 2. Load existing vector database
def load_vector_store():

    embedding_model = get_embedding_model()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    db_path = PROJECT_ROOT / "db" / REPO_NAME

    vector_store = Chroma(
        collection_name=REPO_NAME,
        embedding_function=embedding_model,
        persist_directory=str(db_path)
    )

    return vector_store


# 3. Retrieve relevant chunks using semantic search
def retrieve_chunks(
    vector_store,
    query,
    k=4,
    language=None,
    use_mmr=False
):

    search_filter = None

    if language:
        search_filter = {"language": language}

    if use_mmr:

        results = vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=20,
            filter=search_filter
        )

        return [(doc, None) for doc in results]

    else:

        results = vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=search_filter
        )

        return results


# 4. Retrieve neighbouring chunks
def get_neighbouring_chunks(vector_store, retrieved_chunks, window=1):

    expanded_by_source = {}

    for doc, score in retrieved_chunks:

        source = doc.metadata["source"]
        chunk_index = doc.metadata["chunk_index"]

        # Create a group for this file if it doesn't exist
        if source not in expanded_by_source:
            expanded_by_source[source] = {}

        # Get neighbouring chunks
        for offset in range(-window, window + 1):

            neighbour_index = chunk_index + offset

            if neighbour_index < 0:
                continue

            results = vector_store.get(
                where={
                    "$and": [
                        {"source": {"$eq": source}},
                        {"chunk_index": {"$eq": neighbour_index}}
                    ]
                },
                include=["documents", "metadatas"]
            )

            # Store chunks using their index as the key
            for index, metadata in enumerate(results["metadatas"]):

                chunk_index = metadata["chunk_index"]

                expanded_by_source[source][chunk_index] = Document(
                    page_content=results["documents"][index],
                    metadata=metadata
                )

    # Combine groups while maintaining chunk order
    final_chunks = []

    for source, chunks in expanded_by_source.items():

        sorted_chunks = sorted(chunks.items())

        for chunk_index, doc in sorted_chunks:
            final_chunks.append(doc)

    return final_chunks

def find_function_source(vector_store, function_name):

    results = vector_store.get(
        where_document={
            "$contains": f"def {function_name}"
        },
        include=["documents", "metadatas"]
    )

    for index, content in enumerate(results["documents"]):

        if f"def {function_name}" in content:

            source = results["metadatas"][index]["source"]

            return source

    return None


# 5. Retrieve a complete function by its name
def retrieve_function_by_name(
    vector_store,
    function_name,
    source
):

    results = vector_store.get(
        where={"source": {"$eq": source}},
        include=["documents", "metadatas"]
    )

    # Sort chunks in their original order
    chunks = sorted(
        zip(results["documents"], results["metadatas"]),
        key=lambda item: item[1]["chunk_index"]
    )

    function_chunks = []
    found_function = False

    for content, metadata in chunks:

        # Find function definitions in the current chunk
        function_matches = re.findall(
            r"(?m)^\s*def\s+(\w+)\s*\(",
            content
        )

        if function_matches:

            # Target function found
            if function_name in function_matches:
                found_function = True

            # Another function starts
            elif found_function:
                break

        if found_function:

            function_chunks.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )

    return function_chunks


# 6. Test retrieval independently
def main():

    vector_store = load_vector_store()

    query = "get_my_bookings"

    retrieved_chunks = retrieve_chunks(
        vector_store=vector_store,
        query=query,
        k=6,
        language="python",
        use_mmr=False
    )

    expanded_chunks = get_neighbouring_chunks(
        vector_store=vector_store,
        retrieved_chunks=retrieved_chunks,
        window=1
    )

    print("\nOriginal chunks:", len(retrieved_chunks))
    print("Expanded chunks:", len(expanded_chunks))

    for index, doc in enumerate(expanded_chunks, start=1):

        print("\n----------------------------")
        print("Result:", index)
        print("Source:", doc.metadata["source"])
        print("Language:", doc.metadata["language"])
        print("Chunk:", doc.metadata["chunk_index"])
        print("Content:")
        print(doc.page_content)


if __name__ == "__main__":
    main()
