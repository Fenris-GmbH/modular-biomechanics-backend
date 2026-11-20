"""Check JSON files for object keys that contain uppercase letters.

Usage:
    python lower_case_json_keys.py PATH

The script only validates files. It never changes or writes JSON data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assert_lowercase_keys(value: Any, file_path: Path, location: str = "$") -> None:
    """Raise with the source file and key path if a key contains capitals."""
    if isinstance(value, dict):
        for key, child in value.items():
            key_location = f"{location}[{key!r}]"
            if key != key.lower():
                raise ValueError(
                    f"Uppercase letters found in {file_path} at key "
                    f"{key_location} ({key!r})"
                )
            assert_lowercase_keys(child, file_path, key_location)
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_lowercase_keys(item, file_path, f"{location}[{index}]")


def main() -> int:
    default_path = r"C:\Users\Marco Chow\Documents\GitHub\biomechanics_modular_backend\outputs\kb-dc1-36482-1792133275\basic_screen"
    default_path = Path(default_path)
    parser = argparse.ArgumentParser(
        description="Check JSON keys in one file or recursively in a directory."
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=default_path,
        help="JSON file or directory containing JSON files",
    )
    args = parser.parse_args()

    source = args.path.expanduser().resolve()
    if not source.exists():
        parser.error(f"Path does not exist: {source}")

    if source.is_file():
        files = [source] if source.suffix.lower() == ".json" else []
    else:
        files = sorted(source.rglob("*.json"))

    if not files:
        print(f"No JSON files found under: {source}")
        return 0

    for input_file in files:
        with input_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
        assert_lowercase_keys(data, input_file)
        print(f"OK: {input_file}")

    print(f"Validated {len(files)} JSON file(s); no files were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
