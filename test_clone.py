from src.clone_repo import clone_repo

repo_url = "https://github.com/Veereshp984/ChargeFlow-EV-Charging-Management-Platform"

repo_path = clone_repo(repo_url)

print("Repository location:", repo_path)


