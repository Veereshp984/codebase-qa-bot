
import json
from pathlib import Path

from src.postgres_store import retrieve_chunks


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "eval" / "evaluation_dataset.json"

REPO_ID = "Veereshp984_ChargeFlow-EV-Charging-Management-Platform"


def load_evaluation_dataset():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset


def normalize_source(source):

    return source.replace("\\", "/").lstrip("./")


def evaluate_retrieval():

    dataset = load_evaluation_dataset()

    total_questions = len(dataset)
    successful_hits = 0

    for item in dataset:

        question = item["question"]
        expected_sources = item["expected_sources"]

        # Retrieve top 3 chunks from PostgreSQL
        results = retrieve_chunks(
            query=question,
            repo_id=REPO_ID,
            k=3
        )

        if item["id"] in [7, 19]:

            print("\n========== DEBUGGING MISS ==========")

            for content, metadata, distance in results:
                print("\nSource:", metadata["source"])
                print("Chunk:", metadata["chunk_index"])
                print("Distance:", distance)
                print(content)

        retrieved_sources = [
            normalize_source(metadata["source"])
            for content, metadata, distance in results
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

    hit_rate = successful_hits / total_questions if total_questions else 0

    print("\n================================")
    print("POSTGRESQL RETRIEVAL EVALUATION")
    print("================================")
    print("Total questions:", total_questions)
    print("Successful hits:", successful_hits)
    print("Misses:", total_questions - successful_hits)
    print(f"Hit Rate@3: {hit_rate:.2%}")


if __name__ == "__main__":
    evaluate_retrieval()
