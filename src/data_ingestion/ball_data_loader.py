import json
from utils.logging_utils import setup_logging

logger = setup_logging()


def load_ball_position_data(pose3d_path):
    with open(pose3d_path, "r") as f:
        pose3d_data = json.load(f)

    if "data" not in pose3d_data:
        logger.error("Missing pose 3d data")
        raise ValueError(f"Missing pose 3d data")

    return pose3d_data["data"]["ball"]


# if __name__ == "__main__":
#     import os
#     from pathlib import Path

#     current_dir = Path.cwd()
#     parent_dir = current_dir.parent

#     inputs_folder = os.path.join(parent_dir, r"inputs/")
#     outputs_folder = os.path.join(parent_dir, r"outputs/")

#     swing_id = r"m2c-dc1-1393-1750069325"
#     # # Opening TRACKING - JSON file
#     json3d_path = os.path.join(inputs_folder, swing_id + "_3d.json")  # 3d data
#     pose3d = load_pose3d_data(json3d_path)
