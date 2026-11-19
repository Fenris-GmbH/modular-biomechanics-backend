from .math_helpers import *

def calculate(state):
    """Compute rotational stage with cached dependencies."""
    distal = state.get('distal')
    dt = state.get('dt')
    elbow_l = state.get('elbow_l')
    elbow_r = state.get('elbow_r')
    filter_ord = state.get('filter_ord')
    filter_win = state.get('filter_win')
    frame_rate = state.get('frame_rate')
    hip_l = state.get('hip_l')
    hip_r = state.get('hip_r')
    omega_deg_s = state.get('omega_deg_s')
    omega_signed = state.get('omega_signed')
    omega_vector = state.get('omega_vector')
    proximal = state.get('proximal')
    rotational_velocity_about_spine = state.get('rotational_velocity_about_spine')
    segment_axis = state.get('segment_axis')
    segment_axis_dot = state.get('segment_axis_dot')
    shoulder_l = state.get('shoulder_l')
    shoulder_r = state.get('shoulder_r')
    spine_axis = state.get('spine_axis')
    spine_high = state.get('spine_high')
    spine_low = state.get('spine_low')
    wrist_l = state.get('wrist_l')
    wrist_r = state.get('wrist_r')
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

        segment_axis = safe_normalize(
            distal - proximal
        )

        spine_axis = safe_normalize(
            spine_high - spine_low
        )

        segment_axis_dot = np.gradient(
            segment_axis,
            dt,
            axis=0
        )

        omega_vector = np.cross(
            segment_axis,
            segment_axis_dot
        )

        omega_signed = np.einsum(
            "ij,ij->i",
            omega_vector,
            spine_axis
        )

        omega_deg_s = np.degrees(
            omega_signed
        )

        if len(omega_deg_s) >= filter_win:
            omega_deg_s = savgol_filter(
                omega_deg_s,
                filter_win,
                filter_ord
            )

        return omega_deg_s
    dt = 1 / frame_rate
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
    out = locals().copy()
    out.pop("state", None)
    return out
