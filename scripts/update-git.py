#!/usr/bin/env python3

import subprocess
import re


# Function to get the latest commit hash of a git repository
def get_latest_commit_hash(repo_path):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting commit hash for {repo_path}: {e}")
        return None


# Get the latest commit hashes for toktire and derivre
toktrie_commit = get_latest_commit_hash("../toktrie")
derivre_commit = get_latest_commit_hash("../derivre")

# Check if the commit hashes were retrieved successfully
if not toktrie_commit or not derivre_commit:
    print("Error retrieving commit hashes. Exiting.")
    exit(1)

# Read the contents of parser/Cargo.toml
cargo_toml_path = "parser/Cargo.toml"
with open(cargo_toml_path, "r") as file:
    cargo_toml_contents = file.read()

# Update the Cargo.toml file with the new commit hashes
toktrie_pattern = r'(toktrie\s*=\s*\{[^}]*rev\s*=\s*")[^"]*(")'
derivre_pattern = r'(derivre\s*=\s*\{[^}]*rev\s*=\s*")[^"]*(")'

cargo_toml_contents = re.sub(
    toktrie_pattern,
    lambda m: m.group(1) + toktrie_commit + m.group(2),
    cargo_toml_contents,
)
cargo_toml_contents = re.sub(
    derivre_pattern,
    lambda m: m.group(1) + derivre_commit + m.group(2),
    cargo_toml_contents,
)

# Write the updated contents back to parser/Cargo.toml
with open(cargo_toml_path, "w") as file:
    file.write(cargo_toml_contents)

print("Cargo.toml updated successfully.")
