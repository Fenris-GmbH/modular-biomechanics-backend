# """Run the modular biomechanics methods one at a time for selected swings."""

# import json
# from pathlib import Path
# from typing import Iterable, Optional

# from biomechanics.main import movement
# from biomechanics.utils.test_helpers import KEYS_BY_MOVEMENT, numpy_int_parser

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# INPUTS_DIR = PROJECT_ROOT / "inputs"
# OUTPUT_DIR = PROJECT_ROOT / "outputs"

# # Each force output has a separate movement method and, when enabled, a separate
# # JSON file. Keep this explicit so the runner shows precisely what was invoked.
# FORCE_METHODS = (
#     ("force_vectors", "calculate_force_vectors"),
#     ("weight_shift", "calculate_weight_shift"),
#     ("forces", "calculate_forces"),
#     ("time", "calculate_force_time"),
#     ("force_vectors_3d", "calculate_force_vectors_3d"),
#     ("values_for_COP_trajectory", "calculate_values_for_COP_trajectory"),
#     ("COM_MODELS", "calculate_COM_MODELS"),
#     ("power", "calculate_power"),
#     ("energy", "calculate_energy"),
#     ("impulse", "calculate_impulse"),
#     ("small_force_vectors", "calculate_small_force_vectors"),
#     ("COM_COP_traj_back_view", "calculate_COM_COP_traj_back_view"),
#     ("critical_events", "calculate_critical_events"),
#     ("frame_offset", "calculate_frame_offset"),
# )

# # Club fitting is called one output method at a time. Availability is checked
# # against the detected movement type before a method is invoked.
# CLUB_METHODS = (
#     ("M2C_insights", "calculate_M2C_insights"),
#     ("Performance_metrics", "calculate_Performance_metrics"),
#     ("M2C_power", "calculate_M2C_power"),
#     ("M2C_club_mechanics", "calculate_M2C_club_mechanics"),
#     ("M2C_club_mechanics_highlights", "calculate_M2C_club_mechanics_highlights"),
#     ("M2C_wrist", "calculate_M2C_wrist"),
#     ("M2C_force_vectors_hand_r_3d", "calculate_M2C_force_vectors_hand_r_3d"),
#     ("M2C_force_vectors_bat_3d", "calculate_M2C_force_vectors_bat_3d"),
#     ("M2C_skeleton_3d", "calculate_M2C_skeleton_3d"),
#     ("pitcher_metric_equivalents", "calculate_pitcher_metric_equivalents"),
#     ("inPlaneMetrics", "calculate_inPlaneMetrics"),
#     ("outOfPlaneMetrics", "calculate_outOfPlaneMetrics"),
#     ("M2C_inplane_vectors", "calculate_M2C_inplane_vectors"),
#     ("M2C_inplane_vectors_3d", "calculate_M2C_inplane_vectors_3d"),
#     ("M2C_kinetics_3d", "calculate_M2C_kinetics_3d"),
#     ("generic", "calculate_generic"),
#     ("generic_metric_equivalents", "calculate_generic_metric_equivalents"),
#     ("M2C_other_side_error", "calculate_M2C_other_side_error"),
#     ("generic_widgets_error", "calculate_generic_widgets_error"),
#     ("M2C_power_left", "calculate_M2C_power_left"),
#     ("M2C_power_right", "calculate_M2C_power_right"),
#     ("M2C_club_mechanics_left", "calculate_M2C_club_mechanics_left"),
#     ("M2C_club_mechanics_right", "calculate_M2C_club_mechanics_right"),
#     (
#         "M2C_club_mechanics_highlights_left",
#         "calculate_M2C_club_mechanics_highlights_left",
#     ),
#     (
#         "M2C_club_mechanics_highlights_right",
#         "calculate_M2C_club_mechanics_highlights_right",
#     ),
#     ("M2C_wrist_left", "calculate_M2C_wrist_left"),
#     ("M2C_wrist_right", "calculate_M2C_wrist_right"),
# )

# BASIC_METHODS = (
#     ("kinematic_sequence", "calculate_kinematic_sequence"),
#     ("COM", "calculate_COM"),
#     ("segment_angles", "calculate_segment_angles"),
#     ("segment_displacements", "calculate_segment_displacements"),
# )


# def _save_method_result(swing_id: str, method_name: str, result: dict) -> Path:
#     """Write one method's returned object to its own output JSON."""
#     output_path = OUTPUT_DIR / swing_id
#     output_path.mkdir(parents=True, exist_ok=True)
#     method_suffix = method_name.removeprefix("calculate_")
#     file_path = output_path / f"{swing_id}_{method_suffix}.json"
#     with file_path.open("w", encoding="utf-8") as stream:
#         json.dump(result, stream, indent=4, default=numpy_int_parser)
#     return file_path


# def _call_method(session, swing_id, label, method_name, save_json):
#     print(f"  CALL  session.{method_name}()  [{label}]")
#     try:
#         result = getattr(session, method_name)()
#     except Exception as exc:
#         print(f"  ERROR session.{method_name}(): {type(exc).__name__}: {exc}")
#         raise

#     if not result:
#         print(f"  EMPTY {method_name}: no output returned (often an optional 2D input)")
#         return {}

#     if save_json:
#         file_path = _save_method_result(swing_id, method_name, result)
#         print(f"  SAVED {file_path.relative_to(PROJECT_ROOT)}")
#     else:
#         returned_keys = (
#             list(result) if isinstance(result, dict) else [type(result).__name__]
#         )
#         print(f"  RETURN {method_name}: {returned_keys}")
#     return result


# def run_swing(swing_id: str, save_json: bool = False) -> dict:
#     """Run all basic, force, and applicable club methods for one input.

#     Args:
#         swing_id: Input directory name, which is also the JSON file stem.
#         save_json: If true, save each nonempty method result in its own JSON.

#     Returns:
#         Results grouped by ``basic``, ``forces``, and ``club``. Methods that
#         are unavailable for this movement type are reported and omitted.
#     """
#     input_path = INPUTS_DIR / swing_id / f"{swing_id}.json"
#     if not input_path.is_file():
#         raise FileNotFoundError(f"Input JSON not found: {input_path}")

#     with input_path.open(encoding="utf-8") as stream:
#         input_json = json.load(stream)

#     print(f"\n=== {swing_id} ===")
#     session = movement(input_json["data_3d_bio"])
#     results = {"basic": {}, "forces": {}, "club": {}}

#     print(" BASIC METHODS")
#     for output_key, method_name in BASIC_METHODS:
#         results["basic"].update(
#             _call_method(session, swing_id, output_key, method_name, save_json)
#         )

#     print(" FORCE METHODS (each called separately; shared computation is cached)")
#     for output_key, method_name in FORCE_METHODS:
#         results["forces"].update(
#             _call_method(session, swing_id, output_key, method_name, save_json)
#         )

#     print(" CLUB METHODS")
#     movement_result = _call_method(
#         session, swing_id, "movement_type", "calculate_movement_type", save_json
#     )
#     results["club"].update(movement_result)
#     movement_type = movement_result.get("movement_type", "unknown")
#     club_computation = session._get_club_fitting()
#     available_keys = set(club_computation.available_keys)
#     print(f"  MOVEMENT TYPE: {movement_type}")

#     for output_key, method_name in CLUB_METHODS:
#         if output_key not in available_keys:
#             applicable_types = [
#                 type_name
#                 for type_name, keys in KEYS_BY_MOVEMENT.items()
#                 if output_key in keys
#             ]
#             scope = (
#                 "available for " + ", ".join(applicable_types)
#                 if applicable_types
#                 else "not included in the registered output sets"
#             )
#             print(
#                 f"  SKIP  session.{method_name}()  [{scope}; "
#                 f"current movement_type={movement_type!r}]"
#             )
#             continue
#         results["club"].update(
#             _call_method(session, swing_id, output_key, method_name, save_json)
#         )

#     print(
#         f"=== Finished {swing_id}: basic={len(results['basic'])} keys, "
#         f"forces={len(results['forces'])} keys, club={len(results['club'])} keys ==="
#     )
#     return results


# def run_all(swing_ids: Optional[Iterable[str]] = None, save_json: bool = False) -> dict:
#     """Run ``run_swing`` for the supplied IDs, or every input directory."""
#     if swing_ids is None:
#         swing_ids = sorted({path.parent.name for path in INPUTS_DIR.glob("*/*.json")})
#     return {
#         swing_id: run_swing(swing_id, save_json=save_json) for swing_id in swing_ids
#     }


# def main(swing_ids: Optional[Iterable[str]] = None, save_json: bool = False) -> dict:
#     """Callable runner entry point. Saving is opt-in for imported callers."""
#     return run_all(swing_ids=swing_ids, save_json=save_json)


# if __name__ == "__main__":
#     # Preserve the old script behavior when launched directly. Call main() or
#     # run_swing(..., save_json=False) from Python to run without writing JSON.
#     main(
#         swing_ids=[
#             "m2c-dc1-690-1786440469_pitching",
#         ],
#         save_json=True,
#     )
from pathlib import Path
import json
from biomechanics.main import movement
from biomechanics.utils.test_helpers import (
    all_passed,
    numpy_int_parser,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
inputs_folder = PROJECT_ROOT / "inputs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REF_INPUT_DIR = Path(
    rf"C:\Users\Marco Chow\Documents\GitHub\biomechanics_modular_backend\ref_outputs"
)


def save_jsons(swing_id, metric, values):
    output_path = OUTPUT_DIR / swing_id
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"{swing_id}_{metric}.json", "w") as f:
        json.dump(values, f, indent=4, default=numpy_int_parser)


swing_id = "m2c-dc1-690-1786440469_pitching"
# swing_id = "m2c-dc1-690-1786440415-hitting"
swing_id = "kb-dc1-36482-1792133275"
# swing_id = "m2c-dc1-11064-1780909756-golf-lh"
# swing_id = "m2c-dc1-690-1786439971_jump"
# swing_id = "kb-dc1-36482-1792132984-hitting"

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

# basic screen

session = movement(json3d["data_3d_bio"])

session_basic_time = session.get_basic_time()

session_kin_seq = session.calculate_kinematic_sequence()
save_jsons(swing_id, "kinematic_sequence", session_kin_seq)

session_com = session.calculate_COM()
save_jsons(swing_id, "COM", session_com)

session_segment_angles = session.calculate_segment_angles()
save_jsons(swing_id, "segment_angles", session_segment_angles)

session_segment_displacements = session.calculate_segment_displacements()
save_jsons(swing_id, "segment_displacements", session_segment_displacements)

# forces
session_all_forces = session.calculate_all_forces()
for force_key in session_all_forces.keys():
    save_jsons(swing_id, force_key, session_all_forces[force_key])

# clubfitting
session_all_club = session.calculate_all_club_fitting()
for club_key in session_all_club.keys():
    save_jsons(swing_id, club_key, session_all_club[club_key])

# comparisons
all_passed(session_basic_time, ref_bs_output)
all_passed(session_kin_seq, ref_bs_output["kinematic_sequence"])
all_passed(session_com, ref_bs_output["com"])
all_passed(session_segment_angles, ref_bs_output["segment_angles"])
all_passed(session_segment_displacements, ref_bs_output["segment_displacements"])
all_passed(session_all_forces, ref_fs_output)
all_passed(session_all_club, ref_cf_output)

# Checking keys
assert set(session_all_club.keys()) == (set(ref_cf_output.keys()))
assert set(session_all_forces.keys()) == (set(ref_fs_output.keys()))
