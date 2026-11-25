
from src.postgres_store import (
    retrieve_chunks,
    get_neighbouring_chunks
)

repo_id = "Veereshp984_ChargeFlow-EV-Charging-Management-Platform"

query = "How does the application handle user login?"

# Retrieve the top 4 matching chunks
results = retrieve_chunks(
    query=query,
    repo_id=repo_id,
    k=4
)

print("Retrieved chunks:", len(results))

# Expand each retrieved chunk with its neighbours
expanded_chunks = get_neighbouring_chunks(
    retrieved_chunks=results,
    repo_id=repo_id,
    window=1
)

print("Chunks after neighbour expansion:", len(expanded_chunks))

for index, chunk in enumerate(expanded_chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print("Source:", chunk.metadata.get("source"))
    print("Chunk Index:", chunk.metadata.get("chunk_index"))
    print("Content Preview:", chunk.page_content[:200])
