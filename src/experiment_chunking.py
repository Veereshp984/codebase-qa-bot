
from load_files import load_files, convert_to_documents

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter
)


LANGUAGE_MAP = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "java": Language.JAVA,
    "markdown": Language.MARKDOWN
}


def run_experiment(documents, chunk_size, method):

    all_chunks = []

    for doc in documents:

        # Character-based splitting
        if method == "character":

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=100
            )

        # Language-aware splitting
        elif method == "language":

            language = doc.metadata.get("language")

            if language in LANGUAGE_MAP:

                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=LANGUAGE_MAP[language],
                    chunk_size=chunk_size,
                    chunk_overlap=100
                )

            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=100
                )

        # Split document
        chunks = splitter.split_text(doc.page_content)

        all_chunks.extend(chunks)

    # Calculate statistics
    total_chunks = len(all_chunks)

    if total_chunks == 0:
        return 0, 0, 0

    lengths = [len(chunk) for chunk in all_chunks]

    average_length = sum(lengths) / total_chunks
    maximum_length = max(lengths)

    return total_chunks, average_length, maximum_length


if __name__ == "__main__":

    repo_path = "../repos/ChargeFlow-EV-Charging-Management-Platform"

    # Load and convert files
    files = load_files(repo_path)
    documents = convert_to_documents(files)

    print("Total documents:", len(documents))

    chunk_sizes = [500, 1000, 1500]
    methods = ["character", "language"]

    print("\nCHUNKING EXPERIMENT RESULTS")
    print("-" * 85)

    print(
        f"{'Method':<15}"
        f"{'Size':<12}"
        f"{'Chunks':<12}"
        f"{'Average Length':<20}"
        f"{'Maximum Length':<15}"
    )

    print("-" * 85)

    for size in chunk_sizes:

        for method in methods:

            total, average, maximum = run_experiment(
                documents,
                size,
                method
            )

            print(
                f"{method:<15}"
                f"{size:<12}"
                f"{total:<12}"
                f"{average:<20.2f}"
                f"{maximum:<15}"
            )
