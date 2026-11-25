import numpy as np
from scipy.signal import savgol_filter

from .math_helpers import safe_normalize


def rotational_velocity_about_spine(
    proximal,
    distal,
    spine_high,
    spine_low,
    dt,
    filter_win=21,
    filter_ord=2,
):
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
