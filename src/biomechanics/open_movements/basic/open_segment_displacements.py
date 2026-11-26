from typing import Any

import numpy as np
from scipy import signal
from biomechanics.constants import Conversions


def segment_displacements(ctx: Any) -> dict[str, Any]:
    """Calculate filtered head, upper-trunk, and pelvis displacement series."""
    frame_rate = ctx.frame_rate
    m = ctx.markers
    pelvis = m["pelvis"]
    skullbase = m["skullbase"]
    spine_high = m["spine_high"]
    c, d = signal.butter(4, 5 / (frame_rate / 2))
    ######################################################### Segment displacements #############################################################
    # Pelvis displacements
    pelvis_sway = 100 * (pelvis[:, 1] - pelvis[0, 1])
    pelvis_thrust = 100 * (pelvis[:, 0] - pelvis[0, 0])
    pelvis_lift = 100 * (pelvis[:, 2] - pelvis[0, 2])
    pelvis_sway = signal.filtfilt(c, d, pelvis_sway)
    pelvis_thrust = signal.filtfilt(c, d, pelvis_thrust)
    pelvis_lift = signal.filtfilt(c, d, pelvis_lift)
    # 15.08.2024: Due to filtering the data once more after calculating our start point P1 = 0 is not happening anymore. That's why i am hard coding once more to start with 0 at p1
    pelvis_sway[0] = 0
    pelvis_thrust[0] = 0
    pelvis_lift[0] = 0
    pelvis_sway_inch = pelvis_sway * Conversions.CM_TO_INCH  # inch
    pelvis_thrust_inch = pelvis_thrust * Conversions.CM_TO_INCH  # inch
    pelvis_lift_inch = pelvis_lift * Conversions.CM_TO_INCH  # inch

    # Head displacements
    head_sway = 100 * (skullbase[:, 1] - skullbase[0, 1])
    head_thrust = 100 * (skullbase[:, 0] - skullbase[0, 0])
    head_lift = 100 * (skullbase[:, 2] - skullbase[0, 2])
    head_sway = signal.filtfilt(c, d, head_sway)
    head_thrust = signal.filtfilt(c, d, head_thrust)
    head_lift = signal.filtfilt(c, d, head_lift)
    # 15.08.2024: Due to filtering the data once more after calculating our start point P1 = 0 is not happening anymore. That's why i am hard coding once more to start with 0 at p1
    head_sway[0] = 0
    head_thrust[0] = 0
    head_lift[0] = 0
    head_sway_inch = head_sway * Conversions.CM_TO_INCH  # inch
    head_thrust_inch = head_thrust * Conversions.CM_TO_INCH  # inch
    head_lift_inch = head_lift * Conversions.CM_TO_INCH  # inch

    # Chest displacements
    chest_sway = 100 * (spine_high[:, 1] - spine_high[0, 1])
    chest_thrust = 100 * (spine_high[:, 0] - spine_high[0, 0])
    chest_lift = 100 * (spine_high[:, 2] - spine_high[0, 2])
    chest_sway = signal.filtfilt(c, d, chest_sway)
    chest_thrust = signal.filtfilt(c, d, chest_thrust)
    chest_lift = signal.filtfilt(c, d, chest_lift)
    # 15.08.2024: Due to filtering the data once more after calculating our start point P1 = 0 is not happening anymore. That's why i am hard coding once more to start with 0 at p1
    chest_sway[0] = 0
    chest_thrust[0] = 0
    chest_lift[0] = 0
    chest_sway_inch = chest_sway * Conversions.CM_TO_INCH  # inch
    chest_thrust_inch = chest_thrust * Conversions.CM_TO_INCH  # inch
    chest_lift_inch = chest_lift * Conversions.CM_TO_INCH  # inch
    return {
        "segment_displacements": {
            "head_sway": np.around(head_sway, 2).tolist(),
            "head_thrust": np.around(head_thrust, 2).tolist(),
            "head_lift": np.around(head_lift, 2).tolist(),
            "head_sway_inch": np.around(head_sway_inch, 2).tolist(),
            "head_thrust_inch": np.around(head_thrust_inch, 2).tolist(),
            "head_lift_inch": np.around(head_lift_inch, 2).tolist(),
            "ut_sway": np.around(chest_sway, 2).tolist(),
            "ut_thrust": np.around(chest_thrust, 2).tolist(),
            "ut_lift": np.around(chest_lift, 2).tolist(),
            "ut_sway_inch": np.around(chest_sway_inch, 2).tolist(),
            "ut_thrust_inch": np.around(chest_thrust_inch, 2).tolist(),
            "ut_lift_inch": np.around(chest_lift_inch, 2).tolist(),
            "pelvis_sway": np.around(pelvis_sway, 2).tolist(),
            "pelvis_thrust": np.around(pelvis_thrust, 2).tolist(),
            "pelvis_lift": np.around(pelvis_lift, 2).tolist(),
            "pelvis_sway_inch": np.around(pelvis_sway_inch, 2).tolist(),
            "pelvis_thrust_inch": np.around(pelvis_thrust_inch, 2).tolist(),
            "pelvis_lift_inch": np.around(pelvis_lift_inch, 2).tolist(),
        }
    }
