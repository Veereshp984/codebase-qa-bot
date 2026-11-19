
import subprocess
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = PROJECT_ROOT / "repos"


def clone_repo(repo_url):

    # 1. Validate GitHub URL
    parsed_url = urlparse(repo_url)

    if parsed_url.scheme != "https" or parsed_url.netloc != "github.com":
        raise ValueError("Please provide a valid public GitHub URL.")

    parts = parsed_url.path.strip("/").split("/")

    if len(parts) != 2 or not all(parts):
        raise ValueError("Invalid GitHub repository URL.")

    owner, repo_name = parts
    repo_name = repo_name.removesuffix(".git")

    # 2. Create repository destination
    destination = REPOS_DIR / owner / repo_name

    if destination.exists():
        print("Repository already exists!")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    # 3. Clone repository
    command = [
        "git",
        "clone",
        "--depth",
        "1",
        f"https://github.com/{owner}/{repo_name}.git",
        str(destination)
    ]

    try:
        subprocess.run(command, check=True)
        print("Repository cloned successfully!")

    except subprocess.CalledProcessError as error:
        print("Error cloning repository:", error)
        return None

    return destination


if __name__ == "__main__":

    url = input("Enter GitHub repository URL: ")

    try:
        repo_path = clone_repo(url)

        if repo_path:
            print("Repository path:", repo_path)

    except ValueError as error:
        print("Error:", error)
