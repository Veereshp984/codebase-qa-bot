
from src.postgres_store import retrieve_chunks

repo_id = "Veereshp984_ChargeFlow-EV-Charging-Management-Platform"

query = "How does the application handle user login?"

results = retrieve_chunks(
    query=query,
    repo_id=repo_id,
    k=4
)

print("\nUser Question:", query)
print("\nRetrieved Results:", len(results))

for index, (content, metadata, distance) in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print("Source:", metadata.get("source"))
    print("Language:", metadata.get("language"))
    print("Chunk Index:", metadata.get("chunk_index"))
    print("Cosine Distance:", round(distance, 4))
    print("Content Preview:", content[:300])
