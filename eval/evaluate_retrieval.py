import json
from pathlib import Path

from src.retrieve import load_vector_store


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "eval" / "evaluation_dataset.json"


def load_evaluation_dataset():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset


def normalize_source(source):

    return source.replace("\\", "/").lstrip("./")


def evaluate_retrieval():

    dataset = load_evaluation_dataset()

    vector_store = load_vector_store()

    total_questions = len(dataset)
    successful_hits = 0

    for item in dataset:

        question = item["question"]
        expected_sources = item["expected_sources"]

        results = vector_store.similarity_search(
            question,
            k=3
        )
        
        if item["id"] in [7, 19]:

            print("\n========== DEBUGGING MISS ==========")

            for doc in results:
                print("\nSource:", doc.metadata["source"])
                print("Chunk:", doc.metadata["chunk_index"])
                print(doc.page_content)

        retrieved_sources = [
            normalize_source(doc.metadata["source"])
            for doc in results
        ]

        expected_found = any(
            any(
                source.endswith(expected)
                for source in retrieved_sources
            )
            for expected in expected_sources
        )

        if expected_found:
            successful_hits += 1

        print("\n--------------------------------")
        print("Question:", item["id"])
        print("Query:", question)
        print("Expected sources:", expected_sources)
        print("Retrieved sources:", retrieved_sources)
        print("Result:", "HIT" if expected_found else "MISS")

    hit_rate = successful_hits / total_questions

    print("\n================================")
    print("RETRIEVAL EVALUATION SUMMARY")
    print("================================")
    print("Total questions:", total_questions)
    print("Successful hits:", successful_hits)
    print("Misses:", total_questions - successful_hits)
    print(f"Hit Rate@3: {hit_rate:.2%}")


if __name__ == "__main__":
    evaluate_retrieval()
