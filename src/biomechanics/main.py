import copy
from typing import Any
import numpy as np
from dataclasses import dataclass
from biomechanics.utils.output_keys import lowercase_output_keys
from biomechanics.open_movements.open_kinematic_sequence import kinematic_sequence
from biomechanics.open_movements.open_segment_angles import segment_angles
from biomechanics.open_movements.open_com import com
from biomechanics.open_movements.open_segment_displacements import segment_displacements
from biomechanics.open_movements.forces.force_screen_modules import (
    ForcesScreen,
    force_vectors,
    weight_shift,
    forces,
    time,
    force_vectors_3d,
    values_for_COP_trajectory,
    COM_MODELS,
    power,
    energy,
    impulse,
    small_force_vectors,
    COM_COP_traj_back_view,
    critical_events,
    frame_offset,
)
from biomechanics.open_movements.club_fitting import (
    ClubFittingComputation,
    compute_all,
    movement_type,
    M2C_insights,
    Performance_metrics,
    M2C_power,
    M2C_club_mechanics,
    M2C_club_mechanics_highlights,
    M2C_wrist,
    M2C_force_vectors_hand_r_3d,
    M2C_force_vectors_bat_3d,
    # M2C_skeleton_3d,
    pitcher_metric_equivalents,
    inPlaneMetrics,
    outOfPlaneMetrics,
    M2C_inplane_vectors,
    M2C_inplane_vectors_3d,
    M2C_kinetics_3d,
    # generic,
    # generic_metric_equivalents,
    M2C_other_side_error,
    generic_widgets_error,
    M2C_power_left,
    M2C_power_right,
    M2C_club_mechanics_left,
    M2C_club_mechanics_right,
    M2C_club_mechanics_highlights_left,
    M2C_club_mechanics_highlights_right,
    M2C_wrist_left,
    M2C_wrist_right,
)

MARKERS_RIGHT = (
    "pelvis",
    "hip_r",
    "knee_r",
    "ankle_r",
    "foot_r",
    "midfoot_r",
    "hip_l",
    "knee_l",
    "ankle_l",
    "foot_l",
    "midfoot_l",
    "nose",
    "spine_low",
    "spine_high",
    "shoulder_r",
    "elbow_r",
    "wrist_r",
    "hand_r",
    "shoulder_l",
    "elbow_l",
    "wrist_l",
    "hand_l",
    "neck",
    "skullbase",
    "skull",
    "club_main",
    "club_head",
    "ear_r",
    "ear_l",
)
MARKERS_LEFT = (
    "pelvis",
    "hip_l",
    "knee_l",
    "ankle_l",
    "foot_l",
    "midfoot_l",
    "hip_r",
    "knee_r",
    "ankle_r",
    "foot_r",
    "midfoot_r",
    "nose",
    "spine_low",
    "spine_high",
    "shoulder_l",
    "elbow_l",
    "wrist_l",
    "hand_l",
    "shoulder_r",
    "elbow_r",
    "wrist_r",
    "hand_r",
    "neck",
    "skullbase",
    "skull",
    "club_main",
    "club_head",
    "ear_l",
    "ear_r",
)


@dataclass
class Context:
    """Prepared movement markers and metadata shared by calculation methods."""

    markers: dict[str, np.ndarray]
    frame_rate: float
    weight: float
    down_the_line: bool
    lefty: bool
    time: np.ndarray
    raw_json_3d: dict[str, Any]


def prepare(json3d: dict[str, Any]) -> Context:
    """Validate, orient, and prepare a 3D movement payload for calculations."""
    config = json3d.get("configuration")
    if config is None:
        raise Exception("No data")
    frame_rate = config["video"]["frame_rate"]
    weight = config["user"]["weight"]
    down_the_line = config["swing"]["orientation"] == "DownTheLine"
    lefty = config["swing"]["handedness"] == "leftHanded"
    # Copy to avoid mutating the input JSON while applying orientation transforms.
    pose = np.squeeze(np.asarray(json3d["data"]["pose"], dtype=float)).copy()
    if down_the_line:
        # Use simultaneous assignment; original code's x/y views alias pose.
        x, y = pose[:, :, 0].copy(), pose[:, :, 1].copy()
        pose[:, :, 0] = -y if lefty else y
        pose[:, :, 1] = -x
    elif lefty:
        pose[:, :, 1] *= -1
    names = MARKERS_LEFT if lefty else MARKERS_RIGHT
    markers = {name: np.squeeze(pose[i]) for i, name in enumerate(names)}
    nframes = len(np.squeeze(pose[0]))
    time = np.arange(1, nframes) / frame_rate  # original time convention
    return Context(markers, frame_rate, weight, down_the_line, lefty, time, json3d)


class movement:
    """Session facade for basic, force, and club-fitting movement outputs."""

    def __init__(self, json3d: dict[str, Any]) -> None:
        """Create a session from the inner ``data_3d_bio`` input object."""
        self.ctx = prepare(json3d)
        self.com = None
        self.force_computation = None
        self.club_fitting = None

    def _get_context(self) -> Context:
        """Return a deep copy of the prepared context for a calculation."""
        return copy.deepcopy(self.ctx)

    def get_basic_time(self) -> dict[str, Any]:
        """Return the basic-screen time vector."""
        return {"time": self.ctx.time.tolist()}

    def calculate_kinematic_sequence(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_kinematic_sequence calculation."""
        ctx = self._get_context()
        return kinematic_sequence(ctx)

    def calculate_COM(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_COM calculation."""
        ctx = self._get_context()
        self.com = com(ctx)
        return {"com":{"full_body": self.com.tolist()}}

    def calculate_segment_angles(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_segment_angles calculation."""
        ctx = self._get_context()
        if self.com is None:
            self.com = com(ctx)
        return segment_angles(ctx, self.com)

    def calculate_segment_displacements(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_segment_displacements calculation."""
        ctx = self._get_context()
        return segment_displacements(ctx)

    def _get_force_computation(self) -> ForcesScreen:
        """Create or return the cached force-screen computation."""
        if self.force_computation is None:
            ctx = self._get_context()
            self.force_computation = ForcesScreen(
                ctx.raw_json_3d,
                body_mass=ctx.weight,
                frame_rate=ctx.frame_rate,
            )
        return self.force_computation

    def calculate_force_vectors(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_force_vectors calculation."""
        return force_vectors(computation=self._get_force_computation())

    def calculate_weight_shift(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_weight_shift calculation."""
        return weight_shift(computation=self._get_force_computation())

    def calculate_forces(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_forces calculation."""
        return forces(computation=self._get_force_computation())

    def calculate_force_time(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_force_time calculation."""
        return time(computation=self._get_force_computation())

    def calculate_force_vectors_3d(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_force_vectors_3d calculation."""
        return force_vectors_3d(computation=self._get_force_computation())

    def calculate_values_for_COP_trajectory(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_values_for_COP_trajectory calculation."""
        return values_for_COP_trajectory(computation=self._get_force_computation())

    def calculate_COM_MODELS(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_COM_MODELS calculation."""
        return COM_MODELS(computation=self._get_force_computation())

    def calculate_power(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_power calculation."""
        return power(computation=self._get_force_computation())

    def calculate_energy(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_energy calculation."""
        return energy(computation=self._get_force_computation())

    def calculate_impulse(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_impulse calculation."""
        return impulse(computation=self._get_force_computation())

    def calculate_small_force_vectors(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_small_force_vectors calculation."""
        return small_force_vectors(computation=self._get_force_computation())

    def calculate_COM_COP_traj_back_view(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_COM_COP_traj_back_view calculation."""
        return COM_COP_traj_back_view(computation=self._get_force_computation())

    def calculate_critical_events(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_critical_events calculation."""
        return critical_events(computation=self._get_force_computation())

    def calculate_frame_offset(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_frame_offset calculation."""
        return frame_offset(computation=self._get_force_computation())

    def calculate_all_forces(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_all_forces calculation."""
        computation = self._get_force_computation()

        return lowercase_output_keys(
            {
                key: value
                for key in (
                    "force_vectors",
                    "weight_shift",
                    "forces",
                    "time",
                    "force_vectors_3d",
                    "values_for_COP_trajectory",
                    "COM_MODELS",
                    "power",
                    "energy",
                    "impulse",
                    "small_force_vectors",
                    "COM_COP_traj_back_view",
                    "critical_events",
                    "frame_offset",
                )
                if (value := computation.get(key)) is not None
            }
        )

    def _get_club_fitting(self) -> ClubFittingComputation:
        """Create or return the cached club-fitting computation."""
        if self.club_fitting is None:
            self.club_fitting = ClubFittingComputation(self._get_context())
        return self.club_fitting

    def calculate_movement_type(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_movement_type calculation."""
        return movement_type(computation=self._get_club_fitting())

    def calculate_M2C_insights(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_insights calculation."""
        return M2C_insights(computation=self._get_club_fitting())

    def calculate_Performance_metrics(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_Performance_metrics calculation."""
        return Performance_metrics(computation=self._get_club_fitting())

    def calculate_M2C_power(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_power calculation."""
        return M2C_power(computation=self._get_club_fitting())

    def calculate_M2C_club_mechanics(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_club_mechanics calculation."""
        return M2C_club_mechanics(computation=self._get_club_fitting())

    def calculate_M2C_club_mechanics_highlights(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_club_mechanics_highlights calculation."""
        return M2C_club_mechanics_highlights(computation=self._get_club_fitting())

    def calculate_M2C_wrist(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_wrist calculation."""
        return M2C_wrist(computation=self._get_club_fitting())

    def calculate_M2C_force_vectors_hand_r_3d(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_force_vectors_hand_r_3d calculation."""
        return M2C_force_vectors_hand_r_3d(computation=self._get_club_fitting())

    def calculate_M2C_force_vectors_bat_3d(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_force_vectors_bat_3d calculation."""
        return M2C_force_vectors_bat_3d(computation=self._get_club_fitting())

    # def calculate_M2C_skeleton_3d(self):
    #     return M2C_skeleton_3d(computation=self._get_club_fitting())

    def calculate_pitcher_metric_equivalents(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_pitcher_metric_equivalents calculation."""
        return pitcher_metric_equivalents(computation=self._get_club_fitting())

    def calculate_inPlaneMetrics(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_inPlaneMetrics calculation."""
        return inPlaneMetrics(computation=self._get_club_fitting())

    def calculate_outOfPlaneMetrics(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_outOfPlaneMetrics calculation."""
        return outOfPlaneMetrics(computation=self._get_club_fitting())

    def calculate_M2C_inplane_vectors(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_inplane_vectors calculation."""
        return M2C_inplane_vectors(computation=self._get_club_fitting())

    def calculate_M2C_inplane_vectors_3d(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_inplane_vectors_3d calculation."""
        return M2C_inplane_vectors_3d(computation=self._get_club_fitting())

    def calculate_M2C_kinetics_3d(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_kinetics_3d calculation."""
        return M2C_kinetics_3d(computation=self._get_club_fitting())

    # def calculate_generic(self):
    #     return generic(computation=self._get_club_fitting())

    # def calculate_generic_metric_equivalents(self):
    #     return generic_metric_equivalents(computation=self._get_club_fitting())

    def calculate_M2C_other_side_error(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_other_side_error calculation."""
        return M2C_other_side_error(computation=self._get_club_fitting())

    def calculate_generic_widgets_error(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_generic_widgets_error calculation."""
        return generic_widgets_error(computation=self._get_club_fitting())

    def calculate_M2C_power_left(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_power_left calculation."""
        return M2C_power_left(computation=self._get_club_fitting())

    def calculate_M2C_power_right(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_power_right calculation."""
        return M2C_power_right(computation=self._get_club_fitting())

    def calculate_M2C_club_mechanics_left(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_club_mechanics_left calculation."""
        return M2C_club_mechanics_left(computation=self._get_club_fitting())

    def calculate_M2C_club_mechanics_right(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_club_mechanics_right calculation."""
        return M2C_club_mechanics_right(computation=self._get_club_fitting())

    def calculate_M2C_club_mechanics_highlights_left(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_club_mechanics_highlights_left calculation."""
        return M2C_club_mechanics_highlights_left(computation=self._get_club_fitting())

    def calculate_M2C_club_mechanics_highlights_right(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_club_mechanics_highlights_right calculation."""
        return M2C_club_mechanics_highlights_right(computation=self._get_club_fitting())

    def calculate_M2C_wrist_left(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_wrist_left calculation."""
        return M2C_wrist_left(computation=self._get_club_fitting())

    def calculate_M2C_wrist_right(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_M2C_wrist_right calculation."""
        return M2C_wrist_right(computation=self._get_club_fitting())

    def calculate_all_club_fitting(self) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_all_club_fitting calculation."""
        return compute_all(computation=self._get_club_fitting())


if __name__ == "__main__":
    from pathlib import Path
    import json
    from biomechanics.utils.test_helpers import (
        all_passed,
        numpy_int_parser,
    )

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    inputs_folder = PROJECT_ROOT / "inputs"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

    def save_jsons(swing_id: str, metric: str, values: Any) -> None:
        """Write one metric result as formatted JSON beneath the output folder."""
        output_path = OUTPUT_DIR / swing_id
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / f"{swing_id}_{metric}.json", "w") as f:
            json.dump(values, f, indent=4, default=numpy_int_parser)

    #swing_id = "m2c-dc1-690-1786440469_pitching"
    # swing_id = "m2c-dc1-690-1786440415-hitting"
    #swing_id = "kb-dc1-36482-1792133275"
    swing_id = "m2c-dc1-11064-1780909756-golf-lh"
    # swing_id = "m2c-dc1-690-1786439971_jump"
    # swing_id = "kb-dc1-36482-1792132984-hitting"

    # ref output
    ref_bs_output = json.load(
        open(
            rf"C:\Users\Marco Chow\Documents\GitHub\biomechanics_modular_backend\ref_outputs\{swing_id}\{swing_id}_basicscreen.json"
        )
    )
    ref_fs_output = json.load(
        open(
            rf"C:\Users\Marco Chow\Documents\GitHub\biomechanics_modular_backend\ref_outputs\{swing_id}\{swing_id}_forcescreen.json"
        )
    )

    ref_cf_output = json.load(
        open(
            rf"C:\Users\Marco Chow\Documents\GitHub\biomechanics_modular_backend\ref_outputs\{swing_id}\{swing_id}_clubfitting.json"
        )
    )

    # Objects
    input_json = Path(inputs_folder / swing_id / f"{swing_id}.json")
    with open(input_json, "r") as f:
        json3d = json.load(f)

    # basic screen

    session = movement(json3d["data_3d_bio"])

    session_basic_time = session.get_basic_time()

    # session_kin_seq = session.calculate_kinematic_sequence()
    # save_jsons(swing_id, "kinematic_sequence", session_kin_seq)

    # session_com = session.calculate_COM()
    # save_jsons(swing_id, "com", session_com)

    # session_segment_angles = session.calculate_segment_angles()
    # save_jsons(swing_id, "segment_angles", session_segment_angles)

    # session_segment_displacements = session.calculate_segment_displacements()
    # save_jsons(swing_id, "segment_displacements", session_segment_displacements)

    # forces
    session_all_forces = session.calculate_all_forces()
    # for force_key in session_all_forces.keys():
    #     save_jsons(swing_id, force_key, {force_key:session_all_forces[force_key]})

    # clubfitting
    # session_all_club = session.calculate_all_club_fitting()
    # for club_key in session_all_club.keys():
    #     save_jsons(swing_id, club_key, {club_key: session_all_club[club_key]})

    # # comparisons
    # all_passed(session_basic_time, ref_bs_output)
    # all_passed(
    #     session_kin_seq,
    #     {"kinematic_sequence": ref_bs_output["kinematic_sequence"]},
    # )
    # all_passed(session_com, ref_bs_output["com"])
    # all_passed(session_segment_angles, ref_bs_output["segment_angles"])
    # all_passed(session_segment_displacements, ref_bs_output["segment_displacements"])
    # all_passed(session_all_forces, ref_fs_output)
    # all_passed(session_all_club, ref_cf_output)

    # # Checking keys
    # assert set(session_all_club.keys()) == (set(ref_cf_output.keys()))
    # assert set(session_all_forces.keys()) == (set(ref_fs_output.keys()))
