import json


def load_force_data(force_path):
    with open(force_path, "r") as f:
        force_data = json.load(f)
    return force_data


if __name__ == "__main__":
    import os
    from pathlib import Path

    current_dir = Path.cwd()
    parent_dir = current_dir.parent

    inputs_folder = os.path.join(parent_dir, r"inputs/")
    outputs_folder = os.path.join(parent_dir, r"outputs/")

    swing_id = r"m2c-dc1-1393-1750069325"
    # # Opening TRACKING - JSON file
    force_path = os.path.join(
        inputs_folder, swing_id + "_measured_force.json"
    )  # 3d data
    force = load_force_data(force_path)
