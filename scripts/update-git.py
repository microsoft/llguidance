#!/usr/bin/env python3

import subprocess
import re
import sys


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


# Function to update a single Cargo.toml file with commit hashes
def update_cargo_toml(cargo_toml_path, derivre_commit):
    try:
        with open(cargo_toml_path, "r") as file:
            cargo_toml_contents = file.read()

        # Patterns for replacing the rev in Cargo.toml
        derivre_pattern = r'(derivre\s*=\s*\{[^}]*rev\s*=\s*")[^"]*(")'

        cargo_toml_contents = re.sub(
            derivre_pattern,
            lambda m: m.group(1) + derivre_commit + m.group(2),
            cargo_toml_contents,
        )

        # Write the updated contents back to the Cargo.toml file
        with open(cargo_toml_path, "w") as file:
            file.write(cargo_toml_contents)

        print(f"{cargo_toml_path} updated successfully.")
    except FileNotFoundError:
        print(f"Error: {cargo_toml_path} not found.")
    except Exception as e:
        print(f"Error updating {cargo_toml_path}: {e}")


# Get the latest commit hashes for toktire and derivre
derivre_commit = get_latest_commit_hash("../derivre")

# Check if the commit hashes were retrieved successfully
if not derivre_commit:
    print("Error retrieving commit hashes. Exiting.")
    sys.exit(1)

# List of Cargo.toml paths to update
cargo_toml_paths = [
    "parser/Cargo.toml",
]

# Update each Cargo.toml file
for cargo_toml_path in cargo_toml_paths:
    update_cargo_toml(cargo_toml_path, derivre_commit)

# Run cargo fetch for each path to update Cargo.lock
for path in cargo_toml_paths:
    subprocess.run(["cargo", "fetch", "--manifest-path", path], check=True)

print("All Cargo.toml files updated and cargo fetch run successfully.")
