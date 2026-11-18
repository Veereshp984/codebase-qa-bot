from langchain_core.documents import Document
from pathlib import Path

# 1. File extensions we want to load
ALLOWED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".java",
    ".md"
}

# 2. Directories we want to skip
SKIP_DIRS = {
    ".git",
    "node_modules",
    "venv",
    "__pycache__",
    ".venv",
    "dist",
    "build"
}

# 3. Files we want to skip
SKIP_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml"
}

# 4. Maximum file size: 200 KB
MAX_FILE_SIZE = 200 * 1024

# 5. Map extensions to programming languages
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".md": "markdown"
}


def load_files(repo_path):

    repo_path = Path(repo_path)

    loaded_files = []

    # Visit every file and directory recursively
    for file in repo_path.rglob("*"):

        # Skip unwanted directories
        if any(part in SKIP_DIRS for part in file.parts):
            continue

        # Skip directories
        if not file.is_file():
            continue

        # Skip unsupported extensions
        if file.suffix not in ALLOWED_EXTENSIONS:
            continue

        # Skip unwanted files
        if file.name in SKIP_FILES:
            continue

        # Skip files larger than 200 KB
        if file.stat().st_size > MAX_FILE_SIZE:
            continue

        try:
            # Read file content
            with open(
                file,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                content = f.read()

            # Store content and metadata
            loaded_files.append({
                "path": str(file),
                "content": content,
                "language": LANGUAGE_MAP.get(file.suffix)
            })

        except OSError as error:
            print(f"Could not read {file}: {error}")

    return loaded_files



def convert_to_documents(loaded_files):

    documents = []

    for file in loaded_files:

        doc = Document(
            page_content=file["content"],
            metadata={
                "source": file["path"],
                "language": file["language"]
            }
        )

        documents.append(doc)

    return documents


# 6. Run the function
if __name__ == "__main__":

    files = load_files(
        "../repos/ChargeFlow-EV-Charging-Management-Platform"
    )

    documents = convert_to_documents(files)

    print("Total files loaded:", len(files))
    print("Total documents created:", len(documents))

    # Inspect the first document
    print("\nFirst document:")
    print("Content:", documents[0].page_content[:200])
    print("Metadata:", documents[0].metadata)