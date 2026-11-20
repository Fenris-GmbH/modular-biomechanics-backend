from pathlib import Path

def get_project_root():
    """Returns the root directory of the project."""
    return Path(__file__).parents[3].resolve()