"""Body-only widgets shared by pitching and generic recordings."""

import numpy as np

from .helpers import WIDGET_EQUIVALENTS, _ang3, _lp, _seg_com
from .math_helpers import PITCH_KIN_FC_HZ


def filter_body_com(body_com, frame_rate):
    return np.column_stack(
        [
            _lp(body_com[:, 0], 3, frame_rate),
            _lp(body_com[:, 1], 3, frame_rate),
            _lp(body_com[:, 2], 5, frame_rate),
        ]
    )


def vertical_power(filtered_com, dt, W):
    velocity = np.gradient(filtered_com - filtered_com[0], dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    fx, fy = W * acceleration[:, 0], W * acceleration[:, 1]
    fz = W * (acceleration[:, 2] + 9.81)
    fz[fz < 0] = 0
    return fx * velocity[:, 0] + fy * velocity[:, 1] + fz * velocity[:, 2]


def arm_power(shoulder, elbow, wrist, hand, frame_rate, dt, W):
    cutoff = min(PITCH_KIN_FC_HZ, 0.4 * frame_rate / 2.0)
    total = np.zeros(len(hand))
    distal = np.zeros(len(hand))
    segments = [
        ("arm", shoulder, elbow, 0.5772, 0.0271),
        ("forearm", elbow, wrist, 0.4574, 0.0162),
        ("hand", wrist, hand, 0.79, 0.0061),
    ]
    for segment in segments:
        center = _lp(_seg_com(segment), cutoff, frame_rate)
        velocity = np.gradient(center, dt, axis=0)
        acceleration = np.gradient(velocity, dt, axis=0)
        power = W * segment[4] * np.sum(acceleration * velocity, axis=1)
        total += power
        if segment[0] != "arm":
            distal += power
    return total, distal


def hand_power(hand, hand_force, frame_rate):
    velocity = np.vstack([np.zeros((1, 3)), np.diff(hand, axis=0) * frame_rate])
    return np.sum(hand_force * velocity, axis=1)


def power(vertical_power, hand_power, arm_power, distal_power, window_start, window_stop, W):
    return {
        "vertical_power": vertical_power.tolist(),
        "hand_power": hand_power.tolist(),
        "arm_power": arm_power.tolist(),
        "distal_segment_power": distal_power.tolist(),
        "power_index": float(np.max(arm_power[window_start:window_stop]) / max(W, 1e-6)),
    }


def elbow_kinetics(elbow, wrist, hand, frame_rate, dt, W):
    cutoff = min(PITCH_KIN_FC_HZ, 0.4 * frame_rate / 2.0)
    elb_f = _lp(elbow, cutoff, frame_rate)
    wri_f = _lp(wrist, cutoff, frame_rate)
    han_f = _lp(hand, cutoff, frame_rate)
    m_fore, m_hand = W * 0.0162, W * 0.0061
    m_comp = m_fore + m_hand
    c_fore = elb_f + 0.4574 * (wri_f - elb_f)
    c_hand = wri_f + 0.79 * (han_f - wri_f)
    c_comp = (m_fore * c_fore + m_hand * c_hand) / m_comp
    a_comp = np.gradient(np.gradient(c_comp, dt, axis=0), dt, axis=0)
    F_elbow_vec = m_comp * (a_comp + np.array([0.0, 0.0, 9.81]))
    seg_len = np.maximum(np.linalg.norm(han_f - elb_f, axis=1), 1e-6)
    e_ax = (han_f - elb_f) / seg_len[:, None]
    e_dd = np.gradient(np.gradient(e_ax, dt, axis=0), dt, axis=0)
    I_comp = m_comp * (0.303 * seg_len) ** 2
    r_el = elb_f - c_comp
    M_elbow_vec = I_comp[:, None] * np.cross(e_ax, e_dd) - np.cross(r_el, F_elbow_vec)
    return np.linalg.norm(F_elbow_vec, axis=1), np.linalg.norm(M_elbow_vec, axis=1)


def mechanics(tangential_force, axial_force, elbow_force, hand_magnitude, elbow_torque):
    return {
        "Tangential_Force": tangential_force.tolist(),
        "Centripetal_Force": np.abs(axial_force).tolist(),
        "Elbow_Force": elbow_force.tolist(),
        "F_hand": hand_magnitude.tolist(),
        "Elbow_torque": elbow_torque.tolist(),
    }


def highlights(
    tangential_force,
    axial_force,
    elbow_force,
    hand_magnitude,
    elbow_torque,
    window_start,
    window_end,
    reference,
    frame_rate,
):
    def peak(arr):
        segment = arr[window_start : window_end + 1]
        index = int(np.argmax(segment))
        return (
            float(segment[index]),
            index,
            float(1000.0 * (index + window_start - reference) / frame_rate),
        )

    pk_tan = peak(tangential_force)
    pk_axl = peak(np.abs(axial_force))
    pk_elf = peak(elbow_force)
    pk_hnd = peak(hand_magnitude)
    pk_elt = peak(elbow_torque)
    order_idx = np.argsort([pk_tan[1], pk_axl[1], pk_elt[1]])
    return {
        "peak_club_mechanics": {
            "peak_tangential_force": pk_tan[0],
            "peak_centripetal_force": pk_axl[0],
            "peak_Elbow_force": pk_elf[0],
            "peak_F_hand": pk_hnd[0],
            "peak_Elbow_torque": pk_elt[0],
        },
        "time_to_impact": {
            "time_to_impact_tangential_force": pk_tan[2],
            "time_to_impact_centripetal_force": pk_axl[2],
            "time_to_impact_Elbow_force": pk_elf[2],
            "time_to_impact_F_hand": pk_hnd[2],
            "time_to_impact_Elbow_torque": pk_elt[2],
        },
        "kinetic_order": {
            "peak_force_order_F_tangential_force": int(np.where(order_idx == 0)[0][0]) + 1,
            "peak_force_order_F_centripetal_force": int(np.where(order_idx == 1)[0][0]) + 1,
            "peak_force_order_Elbow_torque": int(np.where(order_idx == 2)[0][0]) + 1,
        },
    }


def elbow_flexion(shoulder, elbow, wrist):
    return 180.0 - _ang3(shoulder, elbow, wrist)


def wrist(elbow, wrist, hand, elbow_flexion, frame_rate, dt):
    wrist_angle = _ang3(hand, wrist, elbow)
    wrist_speed = (
        np.concatenate([[0], np.linalg.norm(np.diff(wrist, axis=0) * frame_rate, axis=1)])
        * 2.23693629
    )
    displacement = 100 * (wrist - wrist[0])

    return {
        "throwing_elbow_flexion": elbow_flexion.tolist(),
        "throwing_wrist_extension_flexion": wrist_angle.tolist(),
        "throwing_wrist_extension_flexion_vel": np.gradient(wrist_angle, dt).tolist(),
        "throwing_wrist_speed": wrist_speed.tolist(),
        "throwing_wrist_thrust": displacement[:, 0].tolist(),
        "throwing_wrist_sway": displacement[:, 1].tolist(),
        "throwing_wrist_lift": displacement[:, 2].tolist(),
    }


def pitcher_insights(arc_width, arc_width_inch, elbow_flexion, x_factor, x_factor_stretch):
    return {
        "arc_width": arc_width.tolist(),
        "arc_width_inch": arc_width_inch.tolist(),
        "throwing_elbow_flexion": elbow_flexion.tolist(),
        "x_factor": x_factor.tolist(),
        "x_factor_stretch": x_factor_stretch,
    }


def generic_equivalents(primary_side, left_reference, right_reference, handedness_specified, dt):
    side = {"l": "left", "r": "right"}
    other = "r" if primary_side == "l" else "l"
    return dict(
        WIDGET_EQUIVALENTS,
        primary_side={
            "side": side[primary_side],
            "selected_by": "config handedness"
            if handedness_specified
            else "arm with the higher peak wrist speed",
        },
        other_side={
            "side": side[other],
            "keys_suffix": "_" + side[other],
            "note": "same metrics as the primary side; each arm is referenced to its own peak wrist speed",
        },
        time_to_impact={
            "meaning": "ms relative to the reference event (peak wrist speed of that arm), negative = before it",
            "reference_frame": {"left": left_reference, "right": right_reference},
            "reference_time_s": {"left": left_reference * dt, "right": right_reference * dt},
        },
        analysis_window="whole clip",
    )
