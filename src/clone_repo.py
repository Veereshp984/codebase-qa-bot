import subprocess
from pathlib import Path
from urllib.parse import urlparse


def clone_repo(repo_url):
    parsed_url = urlparse(repo_url)
    
    repo_name = Path(parsed_url.path).stem
    
    destination = Path("repos") / repo_name
    
    if destination.exists():
        print("Repository already exists!")
        return destination
    
    
    command = [
        "git",
        "clone",
        "--depth",
        "1",
        repo_url,
        str(destination)
    ]
    
    try:
        subprocess.run(command, check=True)
        print("Repository cloned successfully")
        
    except subprocess.CalledProcessError as error:
        print("Error cloning repository:", error)
        return None
    
    return destination