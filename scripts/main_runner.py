from pathlib import Path
import json
from biomechanics.main import movement
from biomechanics.utils.test_helpers import (
    all_passed,
    numpy_int_parser,
)
from biomechanics.utils.path_utils import get_project_root

PROJECT_ROOT = get_project_root()
inputs_folder = PROJECT_ROOT / "inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REF_INPUT_DIR = PROJECT_ROOT / "ref_outputs"

def save_jsons(swing_id, metric, values, save=True):
    if save:
        output_path = OUTPUT_DIR / swing_id
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / f"{metric}.json", "w") as f:
            json.dump(values, f, indent=4, default=numpy_int_parser)

if __name__ == "__main__":

    swing_id = "kb-dc1-36482-1792132984-hitting"

    # ref output
    ref_bs_output = json.load(
        open(REF_INPUT_DIR / rf"{swing_id}\{swing_id}_basicscreen.json")
    )
    ref_fs_output = json.load(
        open(REF_INPUT_DIR / rf"{swing_id}\{swing_id}_forcescreen.json")
    )

    ref_cf_output = json.load(
        open(REF_INPUT_DIR / rf"{swing_id}\{swing_id}_clubfitting.json")
    )

    # Objects
    input_json = Path(inputs_folder / swing_id / f"{swing_id}.json")
    with open(input_json, "r") as f:
        json3d = json.load(f)


    #instantiate session
    session = movement(json3d["data_3d_bio"])

    # basic screen
    session_basic_time = session.get_basic_time()

    session_kin_seq = session.calculate_kinematic_sequence()
    save_jsons(swing_id, "kinematic_sequence", session_kin_seq)

    session_com = session.calculate_COM()
    save_jsons(swing_id, "com", session_com)

    session_segment_angles = session.calculate_segment_angles()
    save_jsons(swing_id, "segment_angles", session_segment_angles)

    session_segment_displacements = session.calculate_segment_displacements()
    save_jsons(swing_id, "segment_displacements", session_segment_displacements)

    # forces
    session_all_forces = session.calculate_all_forces()
    for force_key in session_all_forces.keys():
        save_jsons(swing_id, force_key, {force_key:session_all_forces[force_key]})

    # clubfitting
    session_all_club = session.calculate_all_club_fitting()
    for club_key in session_all_club.keys():
        save_jsons(swing_id, club_key, {club_key: session_all_club[club_key]})

    # # comparisons
    # all_passed(session_basic_time, ref_bs_output)
    # all_passed(session_kin_seq, ref_bs_output["kinematic_sequence"])
    # all_passed(session_com, ref_bs_output["com"])
    # all_passed(session_segment_angles, ref_bs_output["segment_angles"])
    # all_passed(session_segment_displacements, ref_bs_output["segment_displacements"])
    # all_passed(session_all_forces, ref_fs_output)
    # all_passed(session_all_club, ref_cf_output)
