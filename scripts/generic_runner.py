import json

from biomechanics.main import movement
from biomechanics.utils.path_utils import get_project_root
from biomechanics.utils.test_helpers import data_3d_bio_parser, numpy_int_parser

PROJECT_ROOT = get_project_root()
INPUT_DIR = PROJECT_ROOT / "inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def save_jsons(swing_id, metric, values, save=True):
    if save:
        output_path = OUTPUT_DIR / swing_id
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / f"{metric}.json").open("w", encoding="utf-8") as file:
            json.dump(values, file, indent=4, default=numpy_int_parser)


if __name__ == "__main__":
    swing_id = "m2c-dc1-690-1786440089_taps"
    recording_dir = INPUT_DIR / swing_id

    with (recording_dir / f"{swing_id}.json").open(encoding="utf-8") as file:
        recording = json.load(file)["data_3d_bio"]

    (
        pose3d,
        frame_rate,
        weight,
        orientation,
        handedness,
        movement_type,
        critical_events,
        _club_3d,
    ) = data_3d_bio_parser(recording)

    session = movement(
        pose3d=pose3d,
        movement_type=movement_type,
        frame_rate=frame_rate,
        weight=weight,
        orientation=orientation,
        handedness=handedness,
    )

    session_basic_time = session.get_basic_time()
    save_jsons(swing_id, "kinematic_sequence", session.calculate_kinematic_sequence())
    save_jsons(swing_id, "com", session.calculate_COM())
    save_jsons(swing_id, "segment_angles", session.calculate_segment_angles())
    save_jsons(
        swing_id, "segment_displacements", session.calculate_segment_displacements()
    )

    session.initialize_forces(
        include_club=False,
        critical_events=critical_events,
    )
    # session_all_forces = session.calculate_all_forces()

    pose2d = None
    pose2d_path = recording_dir / f"{swing_id}_2d.json"
    if pose2d_path.is_file():
        with pose2d_path.open(encoding="utf-8") as file:
            pose2d = json.load(file)

    force_results = (
        session.calculate_force_vectors(pose2d=pose2d),
        session.calculate_weight_shift(pose2d=pose2d),
        session.calculate_forces(),
        session.calculate_force_time(),
        session.calculate_force_vectors_3d(),
        session.calculate_values_for_COP_trajectory(),
        session.calculate_COM_MODELS(pose2d=pose2d),
        session.calculate_power(),
        session.calculate_energy(),
        session.calculate_impulse(),
        session.calculate_small_force_vectors(),
        session.calculate_COM_COP_traj_back_view(),
        session.calculate_critical_events(),
        session.calculate_frame_offset(),
    )
    for result in force_results:
        for key, value in (result or {}).items():
            if value is not None:
                save_jsons(swing_id, key, {key: value})

    session.initialize_club_fitting()
    # session_all_club = session.calculate_all_club_fitting()

    club_results = [
        session.calculate_movement_type(),
        session.calculate_M2C_power(),
        session.calculate_M2C_club_mechanics(),
        session.calculate_M2C_club_mechanics_highlights(),
        session.calculate_M2C_wrist(),
    ]
    available_keys = set(session._get_club_fitting().available_keys)
    for key, calculate in (
        ("M2C_power_left", session.calculate_M2C_power_left),
        ("M2C_power_right", session.calculate_M2C_power_right),
        ("M2C_club_mechanics_left", session.calculate_M2C_club_mechanics_left),
        ("M2C_club_mechanics_right", session.calculate_M2C_club_mechanics_right),
        (
            "M2C_club_mechanics_highlights_left",
            session.calculate_M2C_club_mechanics_highlights_left,
        ),
        (
            "M2C_club_mechanics_highlights_right",
            session.calculate_M2C_club_mechanics_highlights_right,
        ),
        ("M2C_wrist_left", session.calculate_M2C_wrist_left),
        ("M2C_wrist_right", session.calculate_M2C_wrist_right),
    ):
        if key in available_keys:
            club_results.append(calculate())

    for result in club_results:
        for key, value in result.items():
            save_jsons(swing_id, key, {key: value})
