
from pathlib import Path

from src.load_files import load_files, convert_to_documents
from src.split_code import split_documents
from src.postgres_store import store_chunks, get_chunk_count


repo_path = Path(
    "repos/Veereshp984/ChargeFlow-EV-Charging-Management-Platform"
)

repo_id = "Veereshp984_ChargeFlow-EV-Charging-Management-Platform"


print("Loading repository files...")

files = load_files(repo_path)
documents = convert_to_documents(files)

print("Splitting files into chunks...")

chunks = split_documents(documents)

print("Total chunks generated:", len(chunks))

print("Storing chunks in PostgreSQL...")

store_chunks(
    chunks=chunks,
    repo_id=repo_id,
    replace_repo=True
)

count = get_chunk_count(repo_id)

print("\nMigration completed!")
print("Total chunks in PostgreSQL:", count)
