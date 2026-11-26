import copy
from typing import Any
import numpy as np
import logging
from dataclasses import dataclass
from biomechanics.utils.output_keys import lowercase_output_keys
from biomechanics.open_movements.basic import (
    kinematic_sequence,
    segment_displacements,
    segment_angles,
    com)
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
    pitcher_metric_equivalents,
    inPlaneMetrics,
    outOfPlaneMetrics,
    M2C_inplane_vectors,
    M2C_inplane_vectors_3d,
    M2C_kinetics_3d,
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
from biomechanics.constants import Camera, GolfClubs, GolfSwings, Player
logger = logging.getLogger(__file__)


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
    movement_type: str


def prepare(pose,frame_rate,weight,down_the_line,lefty,movement_type) -> Context:
    """Validate, orient, and prepare a 3D movement payload for calculations."""
    pose_raw = pose.copy()
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
    return Context(markers, frame_rate, weight, down_the_line, lefty, time, pose_raw, movement_type)


class movement:
    """Session facade for basic, force, and club-fitting movement outputs."""

    def __init__(self, pose3d=None,movement_type:str=None,frame_rate:int =None,weight:float =None,orientation:str =None,handedness:str =None) -> None:
        """Create a session from the inner ``data_3d_bio`` input object."""

        down_the_line = orientation.lower() == Camera.ORIENTATION_DOWN_THE_LINE.lower()
        lefty = "left" in handedness.lower()

        self.ctx = prepare(pose=pose3d,movement_type=movement_type,frame_rate=frame_rate,weight=weight,down_the_line=down_the_line,lefty=lefty)
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

    def initialize_forces(
        self,
        club_3d=None,
        include_club=None,
        club_type=None,
        critical_events=None,
    ) -> None:
        if self.force_computation is not None:
            raise RuntimeError("ForcesScreen is already initialized.")

        ctx = self._get_context()

        if include_club is None:
            include_club = (
                ctx.movement_type.lower() == GolfSwings.FULL_SWING
                and club_3d is not None
            )


        if include_club and club_type is None:
            logging.warning("Club type not specified for golf swing; falling back to iron.")
            club_type = GolfClubs.IRON

        if include_club and club_3d is None:
            raise ValueError("club_3d is required when include_club=True.")

        self.force_computation = ForcesScreen(
            pose3d=ctx.raw_json_3d,
            swing_type=ctx.movement_type,
            lefty=ctx.lefty,
            body_mass=ctx.weight,
            frame_rate=ctx.frame_rate,
            include_club=include_club,
            club_3d=club_3d,
            club_type=club_type,
            critical_events=critical_events,
        )

    def _get_force_computation(self) -> ForcesScreen:
        if self.force_computation is None:
            raise RuntimeError(
                "Force computation is not initialized. "
                "Call initialize_forces() first."
            )
        return self.force_computation

    def calculate_force_vectors(self,pose2d=None) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_force_vectors calculation."""
        return force_vectors(computation=self._get_force_computation(),pose2d=pose2d)

    def calculate_weight_shift(self,pose2d=None) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_weight_shift calculation."""
        return weight_shift(computation=self._get_force_computation(),pose2d=pose2d)

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

    def calculate_COM_MODELS(self,pose2d=None) -> dict[str, Any]:
        """Return the dictionary produced by the calculate_COM_MODELS calculation."""
        return COM_MODELS(computation=self._get_force_computation(),pose2d=pose2d)

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

    #CLUB
    def initialize_club_fitting(self):
        ctx = self._get_context()
        self.club_fitting = ClubFittingComputation(
            pose3d=ctx.raw_json_3d,
            movement_type=ctx.movement_type,
            lefty=ctx.lefty,
            body_mass=ctx.weight,
            frame_rate=ctx.frame_rate,
            down_the_line=ctx.down_the_line,
        )
    def _get_club_fitting(self) -> ClubFittingComputation:
        """Create or return the cached club-fitting computation."""
        if self.club_fitting is None:
            raise RuntimeError(f"Club Fitting is not initialized. "
                             "Call initialize_club_fitting() first.")

        return self.club_fitting


    def calculate_movement_type(self) -> dict[str, Any]:
        return movement_type(computation=self._get_club_fitting())

    def calculate_M2C_insights(
        self, *, club_3d=None, critical_events=None
    ) -> dict[str, Any]:
        return M2C_insights(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
            critical_events=critical_events,
        )

    def calculate_Performance_metrics(
        self, *, club_3d=None, bat_type=None, critical_events=None
    ) -> dict[str, Any]:
        return Performance_metrics(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
            bat_type=bat_type,
            critical_events=critical_events,
        )

    def calculate_M2C_power(
        self, *, club_3d=None, bat_type=None, critical_events=None
    ) -> dict[str, Any]:
        return M2C_power(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
            bat_type=bat_type,
            critical_events=critical_events,
        )

    def calculate_M2C_club_mechanics(
        self, *, club_3d=None, bat_type=None, critical_events=None
    ) -> dict[str, Any]:
        return M2C_club_mechanics(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
            bat_type=bat_type,
            critical_events=critical_events,
        )

    def calculate_M2C_club_mechanics_highlights(
        self, *, club_3d=None, bat_type=None, critical_events=None
    ) -> dict[str, Any]:
        return M2C_club_mechanics_highlights(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
            bat_type=bat_type,
            critical_events=critical_events,
        )

    def calculate_M2C_wrist(self, *, club_3d=None) -> dict[str, Any]:
        return M2C_wrist(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
        )

    def calculate_M2C_force_vectors_hand_r_3d(self) -> dict[str, Any]:
        return M2C_force_vectors_hand_r_3d(computation=self._get_club_fitting())

    def calculate_M2C_force_vectors_bat_3d(
        self, *, club_3d=None, bat_type=None, critical_events=None
    ) -> dict[str, Any]:
        return M2C_force_vectors_bat_3d(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
            bat_type=bat_type,
            critical_events=critical_events,
        )

    def calculate_pitcher_metric_equivalents(self) -> dict[str, Any]:
        return pitcher_metric_equivalents(computation=self._get_club_fitting())

    def calculate_inPlaneMetrics(
        self, *, club_3d=None, bat_type=None, critical_events=None
    ) -> dict[str, Any]:
        return inPlaneMetrics(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
            bat_type=bat_type,
            critical_events=critical_events,
        )

    def calculate_outOfPlaneMetrics(
        self, *, club_3d=None, bat_type=None, critical_events=None
    ) -> dict[str, Any]:
        return outOfPlaneMetrics(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
            bat_type=bat_type,
            critical_events=critical_events,
        )

    def calculate_M2C_inplane_vectors(
        self, *, club_3d=None, bat_type=None, critical_events=None
    ) -> dict[str, Any]:
        return M2C_inplane_vectors(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
            bat_type=bat_type,
            critical_events=critical_events,
        )

    def calculate_M2C_inplane_vectors_3d(
        self, *, club_3d=None, bat_type=None, critical_events=None
    ) -> dict[str, Any]:
        return M2C_inplane_vectors_3d(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
            bat_type=bat_type,
            critical_events=critical_events,
        )

    def calculate_M2C_kinetics_3d(
        self, *, club_3d=None, bat_type=None, critical_events=None
    ) -> dict[str, Any]:
        return M2C_kinetics_3d(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
            bat_type=bat_type,
            critical_events=critical_events,
        )

    def calculate_M2C_other_side_error(self) -> dict[str, Any]:
        return M2C_other_side_error(computation=self._get_club_fitting())

    def calculate_generic_widgets_error(self) -> dict[str, Any]:
        return generic_widgets_error(computation=self._get_club_fitting())

    def calculate_M2C_power_left(self) -> dict[str, Any]:
        return M2C_power_left(computation=self._get_club_fitting())

    def calculate_M2C_power_right(self) -> dict[str, Any]:
        return M2C_power_right(computation=self._get_club_fitting())

    def calculate_M2C_club_mechanics_left(self) -> dict[str, Any]:
        return M2C_club_mechanics_left(computation=self._get_club_fitting())

    def calculate_M2C_club_mechanics_right(self) -> dict[str, Any]:
        return M2C_club_mechanics_right(computation=self._get_club_fitting())

    def calculate_M2C_club_mechanics_highlights_left(self) -> dict[str, Any]:
        return M2C_club_mechanics_highlights_left(
            computation=self._get_club_fitting()
        )

    def calculate_M2C_club_mechanics_highlights_right(self) -> dict[str, Any]:
        return M2C_club_mechanics_highlights_right(
            computation=self._get_club_fitting()
        )

    def calculate_M2C_wrist_left(self) -> dict[str, Any]:
        return M2C_wrist_left(computation=self._get_club_fitting())

    def calculate_M2C_wrist_right(self) -> dict[str, Any]:
        return M2C_wrist_right(computation=self._get_club_fitting())


    def calculate_all_club_fitting(
        self, *, club_3d=None, bat_type=None, critical_events=None
    ) -> dict[str, Any]:
        return compute_all(
            computation=self._get_club_fitting(),
            club_3d=club_3d,
            bat_type=bat_type,
            critical_events=critical_events,
        )

class golf(movement):
    def __init__(self, pose3d=None,movement_type:str=None,frame_rate:int =None,weight:float =None,orientation:str =None,handedness:str =None) -> None:
        super().__init__(
            pose3d=pose3d,
            movement_type=movement_type,
            frame_rate=frame_rate,
            weight=weight,
            orientation=orientation,
            handedness=handedness,
        )
class baseball(movement):
    def __init__(self, pose3d=None,movement_type:str=None,frame_rate:int =None,weight:float =None,orientation:str =None,handedness:str =None) -> None:
        super().__init__(
            pose3d=pose3d,
            movement_type=movement_type,
            frame_rate=frame_rate,
            weight=weight,
            orientation=orientation,
            handedness=handedness,
        )



if __name__ == "__main__":
    from pathlib import Path
    import json
    from biomechanics.utils.path_utils import get_project_root
    from biomechanics.utils.test_helpers import data_3d_bio_parser

    PROJECT_ROOT = get_project_root()
    inputs_folder = PROJECT_ROOT / "inputs"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

    swing_id = "m2c-dc1-690-1786440089_taps"

    # Objects
    input_json = Path(inputs_folder / swing_id / f"{swing_id}.json")
    with open(input_json, "r") as f:
        json3d = json.load(f)

    #Parse parameters from data_3d_bio
    pose3d,frame_rate,weight,orientation,handedness,movement_name,input_critical_events,club_3d_markers = data_3d_bio_parser(json3d['data_3d_bio'])


    try:
        input_2d_json = Path(inputs_folder / swing_id / f"{swing_id}_2d.json")
        with open(input_2d_json, "r") as f:
            json2d = json.load(f)
    except Exception as e:
        print(f"Failed to load 2D input JSON: {e}")
        json2d = None

    # basic screen
    session = movement(pose3d=pose3d,movement_type=movement_name,frame_rate =frame_rate,weight=weight,orientation=orientation,handedness=handedness)

    session_basic_time = session.get_basic_time()
    session_kin_seq = session.calculate_kinematic_sequence()
    session_com = session.calculate_COM()
    session_segment_angles = session.calculate_segment_angles()
    session_segment_displacements = session.calculate_segment_displacements()

    # forces
    session.initialize_forces(club_3d=club_3d_markers,include_club=None,club_type=None,critical_events=input_critical_events) #NOTE Critical events are passed
    session_weight_shift = session.calculate_weight_shift(json2d)

    # clubfitting
    session.initialize_club_fitting()
    session.calculate_M2C_wrist_left()
