import json
import os
import shutil


def register_dir(dir_path):
    """
    Create a new directory for the given path. If the directory already exists, does nothing
    Args:
        dir_path: The path of the new directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
