from typing import Any

import numpy as np
from scipy.signal import savgol_filter
from biomechanics.utils.normalization import safe_normalize


def kinematic_sequence(ctx: Any) -> dict[str, Any]:
    """Calculate rotational-speed series and heatmap values by body segment."""
    # ctx = self._get_context()
    dt = 1 / ctx.frame_rate
    m = ctx.markers
    hip_l = m["hip_l"]
    hip_r = m["hip_r"]
    spine_high = m["spine_high"]
    spine_low = m["spine_low"]
    shoulder_l = m["shoulder_l"]
    shoulder_r = m["shoulder_r"]
    elbow_l = m["elbow_l"]
    elbow_r = m["elbow_r"]
    wrist_l = m["wrist_l"]
    wrist_r = m["wrist_r"]
    #################################################################### KINEMATIC SEQUENCE ##################################################################

    def rotational_velocity_about_spine(
        proximal: np.ndarray,
        distal: np.ndarray,
        spine_high: np.ndarray,
        spine_low: np.ndarray,
        dt: float,
        filter_win: int,
        filter_ord: int = 2,
    ) -> np.ndarray:
        """
        Computes signed angular velocity (deg/s)
        around instantaneous spine axis.
        """

        segment_axis = safe_normalize(distal - proximal)

        spine_axis = safe_normalize(spine_high - spine_low)

        segment_axis_dot = np.gradient(segment_axis, dt, axis=0)

        omega_vector = np.cross(segment_axis, segment_axis_dot)

        omega_signed = np.einsum("ij,ij->i", omega_vector, spine_axis)

        omega_deg_s = np.degrees(omega_signed)

        if len(omega_deg_s) >= filter_win:
            omega_deg_s = savgol_filter(omega_deg_s, filter_win, filter_ord)

        return omega_deg_s

    pelvis_rot_speed = rotational_velocity_about_spine(
        hip_l,
        hip_r,
        spine_high,
        spine_low,
        dt,
        filter_win=41,
    )
    UT_rot_speed = rotational_velocity_about_spine(
        shoulder_l,
        shoulder_r,
        spine_high,
        spine_low,
        dt,
        filter_win=41,
    )
    arm_rot_speed = rotational_velocity_about_spine(
        shoulder_l,
        elbow_l,
        spine_high,
        spine_low,
        dt,
        filter_win=41,
    )
    # Lead forearm rotational velocity
    hand_rot_speed = rotational_velocity_about_spine(
        elbow_l,
        wrist_l,
        spine_high,
        spine_low,
        dt,
        filter_win=41,
    )
    trail_arm_rot_speed = rotational_velocity_about_spine(
        shoulder_r,
        elbow_r,
        spine_high,
        spine_low,
        dt,
        filter_win=41,
    )
    trail_hand_rot_speed = rotational_velocity_about_spine(
        elbow_r,
        wrist_r,
        spine_high,
        spine_low,
        dt,
        filter_win=41,
    )

    ########################################  Bat SPEED ###################################################################

    kinematic_sequence = {
        "pelvis_rotational_speed": pelvis_rot_speed.tolist(),
        "ut_rotational_speed": UT_rot_speed.tolist(),
        "lead_arm_rotational_speed": arm_rot_speed.tolist(),
        "lead_hand_rotational_speed": hand_rot_speed.tolist(),
        "trail_arm_rot_speed": trail_arm_rot_speed.tolist(),
        "trail_hand_rot_speed": trail_hand_rot_speed.tolist(),
    }

    # Heat Map Values for Kinematic Sequence
    def kin_seq_heatmap(
        kinematic_sequence: dict[str, Any],
    ) -> dict[str, list[float]]:
        """Normalize each available rotational-speed series for heatmap use."""
        heatmap_labels = [
            "pelvis_rotational_speed",
            "ut_rotational_speed",
            "lead_arm_rotational_speed",
            "lead_hand_rotational_speed",
            "club_rotational_speed",
            "trail_arm_rot_speed",
            "trail_hand_rot_speed",
        ]

        heatmap_data = {}

        for label in heatmap_labels:
            signal = np.asarray(kinematic_sequence.get(label, []), dtype=float)

            if signal.size == 0:
                heatmap_data[f"{label}_hm"] = []
                continue

            # Baseball: no backswing / downswing split around P4.
            # Normalize each rotational-speed signal globally over
            # the complete movement.
            signal_abs = np.abs(signal)

            signal_min = np.min(signal_abs)
            signal_max = np.max(signal_abs)

            norm_rot_speed = (signal_abs - signal_min) / (
                signal_max - signal_min + np.finfo(float).eps
            )

            heatmap_data[f"{label}_hm"] = norm_rot_speed.tolist()

        return heatmap_data

    kinematic_sequence_hm_dict = kin_seq_heatmap(
        kinematic_sequence,
    )
    kinematic_sequence.update(kinematic_sequence_hm_dict)
    return {"kinematic_sequence": kinematic_sequence}
