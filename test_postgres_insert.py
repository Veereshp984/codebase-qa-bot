
from pathlib import Path

from src.load_files import load_files, convert_to_documents
from src.split_code import split_documents
from src.postgres_store import store_chunks, get_chunk_count


# Existing ChargeFlow repository
repo_path = Path(
    "repos/Veereshp984/ChargeFlow-EV-Charging-Management-Platform"
)

repo_id = "Veereshp984_ChargeFlow-EV-Charging-Management-Platform"

# Load and split repository files
files = load_files(repo_path)
documents = convert_to_documents(files)
chunks = split_documents(documents)

print("Total chunks generated:", len(chunks))

# Store only one chunk for testing
store_chunks(chunks[:1], repo_id)

# Verify database count
count = get_chunk_count(repo_id)

print("Chunks stored in PostgreSQL:", count)
