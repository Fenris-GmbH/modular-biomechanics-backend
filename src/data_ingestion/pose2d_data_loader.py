import json


def load_pose2d_data(pose2d_path):
    with open(pose2d_path, "r") as f:
        pose2d_data = json.load(f)

    if "data" not in pose2d_data:
        raise ValueError(f"Missing pose_2d data")

    return pose2d_data["data"]


if __name__ == "__main__":
    import os
    from pathlib import Path

    current_dir = Path.cwd()
    parent_dir = current_dir.parent

    inputs_folder = os.path.join(parent_dir, r"inputs/")
    outputs_folder = os.path.join(parent_dir, r"outputs/")

    swing_id = r"m2c-dc1-1393-1750069325"
    # # Opening TRACKING - JSON file
    json2d_path = os.path.join(inputs_folder, swing_id + "_2d.json")  # 2d data
    pose2d = load_pose2d_data(json2d_path)
