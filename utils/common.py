import os

def ensure_directory_exists(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)