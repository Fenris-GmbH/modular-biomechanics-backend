import json


def load_ball_club_speed_data(speed_path):
    with open(speed_path, "r") as f:
        ball_club_speed_data = json.load(f)
    return ball_club_speed_data["data"]


if __name__ == "__main__":
    import os
    from pathlib import Path

    current_dir = Path.cwd()
    parent_dir = current_dir.parent

    inputs_folder = os.path.join(parent_dir, r"inputs/")
    outputs_folder = os.path.join(parent_dir, r"outputs/")

    swing_id = r"m2c-dc1-1393-1750069325"
    # # Opening TRACKING - JSON file
    speed_path = os.path.join(inputs_folder, swing_id + "_ball.json")  # 3d data
    ball_club_speed = load_ball_club_speed_data(speed_path)
