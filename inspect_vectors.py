from src.retrieve import load_vector_store

repo_id = "Veereshp984_ChargeFlow-EV-Charging-Management-Platform"

vector_store = load_vector_store(repo_id)

data = vector_store.get(
    limit=1,
    include=["documents", "metadatas", "embeddings"]
)

print("Document:")
print(data["documents"][0])

print("\nMetadata:")
print(data["metadatas"][0])

print("\nEmbedding:")
print(data["embeddings"][0])

print("\nEmbedding dimensions:")
print(len(data["embeddings"][0]))