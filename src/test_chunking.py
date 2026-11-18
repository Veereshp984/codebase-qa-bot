
from load_files import load_files, convert_to_documents
from split_code import split_documents

repo_path = "../repos/ChargeFlow-EV-Charging-Management-Platform"

# Load files
files = load_files(repo_path)

# Convert to LangChain Documents
documents = convert_to_documents(files)

# Split documents into chunks
chunks = split_documents(documents)

# Inspect a booking service chunk
count = 0

for chunk in chunks:

    if "booking_service.py" in chunk.metadata["source"]:

        print("\n-------------------------")
        print("Chunk index:", chunk.metadata["chunk_index"])
        print("Characters:", len(chunk.page_content))
        print(chunk.page_content)

        count += 1

        if count == 3:
            break

print("Total files:", len(documents))
print("Total chunks:", len(chunks))

# Inspect first three chunks
for chunk in chunks[:3]:

    print("\n---------------------------")
    print("Source:", chunk.metadata["source"])
    print("Language:", chunk.metadata["language"])
    print("Chunk index:", chunk.metadata["chunk_index"])
    print("Characters:", len(chunk.page_content))
    print("Content:\n", chunk.page_content)
