
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


REPO_NAME = "ChargeFlow-EV-Charging-Management-Platform"


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_vector_store():

    embedding_model = get_embedding_model()

    db_path = Path("../db") / REPO_NAME

    vector_store = Chroma(
        collection_name=REPO_NAME,
        embedding_function=embedding_model,
        persist_directory=str(db_path)
    )

    return vector_store


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

    else:

        results = vector_store.similarity_search(
            query,
            k=k,
            filter=search_filter
        )

    return results


def main():

    vector_store = load_vector_store()

    query = "How does the booking system work?"

    results = retrieve_chunks(
    vector_store=vector_store,
    query=query,
    k=4,
    language="python",
    use_mmr=True
)

    print("\nRetrieved chunks:", len(results))

    for index, doc in enumerate(results, start=1):

        print("\n----------------------------")
        print("Result:", index)
        print("Source:", doc.metadata["source"])
        print("Language:", doc.metadata["language"])
        print("Chunk:", doc.metadata["chunk_index"])
        print("Content:")
        print(doc.page_content[:500])


if __name__ == "__main__":
    main()
