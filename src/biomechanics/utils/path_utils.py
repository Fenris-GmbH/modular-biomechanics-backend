from pathlib import Path

def get_project_root() -> Path:
    """Return the repository root inferred from this module's location."""
    return Path(__file__).parents[3].resolve()
