import json


def load_club_data(club_path):
    with open(club_path, "r") as f:
        club_data = json.load(f)
    return club_data["data"]["club"]


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
    club = load_club_data(cv_path)
