import json


def load_config(config_path):
    with open(config_path, "r") as f:
        config_data = json.load(f)
        config_data = config_data["configuration"]

    return config_data


if __name__ == "__main__":
    import os
    from pathlib import Path

    current_dir = Path.cwd()
    parent_dir = current_dir.parent

    inputs_folder = os.path.join(parent_dir, r"inputs/")
    outputs_folder = os.path.join(parent_dir, r"outputs/")

    swing_id = r"m2c-dc1-1393-1750069325"
    # # Opening TRACKING - JSON file
    cv_path = os.path.join(inputs_folder, swing_id + "_cv.json")  # 3d data
    config = load_config(cv_path)
