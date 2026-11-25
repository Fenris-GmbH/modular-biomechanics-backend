import numpy as np

from .math_helpers import safe_normalize


def performance(
    UT_rot_speed,
    throwing_arm_rot_speed,
    throwing_hand_rot_speed,
    ankle_l,
    ankle_r,
    arm_rot_speed,
    dt,
    elbow_l,
    elbow_r,
    frame_rate,
    hand_rot_speed,
    hip_l,
    hip_r,
    knee_l,
    knee_r,
    lefty,
    p1,
    p4,
    p5,
    p7,
    p8,
    pelvis_rot_speed,
    shoulder_l,
    shoulder_r,
    wrist_l,
    wrist_r,
):
    Pitcher_performance_metrics = {}

    if lefty == 1:
        throwing_wrist = wrist_l
        throwing_elbow = elbow_l
        throwing_shoulder = shoulder_l

        lead_leg = ankle_r
        lead_knee = knee_r

    else:
        throwing_wrist = wrist_r
        throwing_elbow = elbow_r
        throwing_shoulder = shoulder_r

        lead_leg = ankle_l
        lead_knee = knee_l

    hand_velocity = np.gradient(throwing_wrist, dt, axis=0)

    hand_speed = np.linalg.norm(hand_velocity, axis=1)

    peak_hand_frame = np.argmax(hand_speed[p4:p7]) + p4

    Pitcher_performance_metrics["peak_hand_speed"] = {
        "value_mps": float(hand_speed[peak_hand_frame]),
        "frame": int(peak_hand_frame),
        "time_to_release_ms": float((peak_hand_frame - p7) / frame_rate * 1000),
    }

    release_position = throwing_wrist[p7]

    Pitcher_performance_metrics["release_height"] = {
        "value_m": float(release_position[2]),
        "frame": int(p7),
    }

    extension_vector = throwing_wrist[p7] - lead_leg[p7]

    release_extension = abs(extension_vector[0])

    Pitcher_performance_metrics["release_extension"] = {
        "value_m": float(release_extension),
        "frame": int(p7),
    }

    trunk_rot_velocity = abs(UT_rot_speed)

    #################################################################
    # Peak trunk rotation velocity
    #################################################################

    trunk_peak_idx = np.argmax(trunk_rot_velocity[p1 : p8 + 1]) + p1

    trunk_peak = {
        "peak_velocity_deg_s": float(trunk_rot_velocity[trunk_peak_idx]),
        "time_to_release_ms": float(1000 * (trunk_peak_idx - p7) / frame_rate),
    }

    #################################################################
    # Trunk rotation range
    #################################################################

    trunk_rotation_angle = np.cumsum(UT_rot_speed * dt)

    trunk_rotation_window = trunk_rotation_angle[p1 : p8 + 1]

    trunk_rotation_range = float(np.max(trunk_rotation_window) - np.min(trunk_rotation_window))

    Pitcher_performance_metrics["trunk_rotation"] = {
        "peak_velocity_deg_s": trunk_peak["peak_velocity_deg_s"],
        "peak_time_ms": trunk_peak["time_to_release_ms"],
        "rotation_range_deg": trunk_rotation_range,
    }

    #################################################################
    # Throwing shoulder rotational velocity
    #################################################################

    #################################################################
    # Peak shoulder rotation velocity
    #################################################################

    shoulder_peak_idx = np.argmax(abs(throwing_arm_rot_speed[p1 : p8 + 1])) + p1

    shoulder_peak_velocity = throwing_arm_rot_speed[shoulder_peak_idx]

    Pitcher_performance_metrics["shoulder_rotation"] = {
        "peak_velocity_deg_s": float(abs(shoulder_peak_velocity)),
        "peak_time_ms": float(1000 * (shoulder_peak_idx - p7) / frame_rate),
        "peak_frame": int(shoulder_peak_idx),
    }

    upper_arm = safe_normalize(throwing_shoulder - throwing_elbow)

    forearm = safe_normalize(throwing_wrist - throwing_elbow)

    elbow_angle = np.degrees(np.arccos(np.clip(np.sum(upper_arm * forearm, axis=1), -1, 1)))

    elbow_velocity = np.gradient(elbow_angle, dt)

    idx = np.argmax(np.abs(elbow_velocity[p4:p7])) + p4

    Pitcher_performance_metrics["elbow_extension_velocity"] = {
        "peak_velocity_deg_s": float(elbow_velocity[idx]),
        "peak_time_ms": float((idx - p7) / frame_rate * 1000),
        "extension_range_deg": float(np.max(elbow_angle) - np.min(elbow_angle)),
    }

    if lefty == 1:
        lead_knee = knee_r
        lead_ankle = ankle_r
        lead_hip = hip_r

    else:
        lead_knee = knee_l
        lead_ankle = ankle_l
        lead_hip = hip_l

    thigh = lead_hip - lead_knee
    shank = lead_ankle - lead_knee

    thigh = thigh / (np.linalg.norm(thigh, axis=1, keepdims=True) + 1e-8)

    shank = shank / (np.linalg.norm(shank, axis=1, keepdims=True) + 1e-8)

    knee_angle = np.degrees(np.arccos(np.clip(np.sum(thigh * shank, axis=1), -1, 1)))

    knee_extension_velocity = np.gradient(knee_angle, dt)

    knee_extension_acceleration = np.gradient(knee_extension_velocity, dt)

    release_frame = p7
    start_frame = max(p5, 0)

    window = slice(start_frame, release_frame + 1)

    knee_accel_window = knee_extension_acceleration[window]

    if knee_accel_window.size == 0:
        # Invalid/inverted event frames (e.g. p5 >= p7) - skip metric for this trial
        print(f"Warning: empty knee bracing window (p5={p5}, p7={p7}) - skipping lead_leg_bracing")
        peak_bracing = np.nan
        lead_leg_bracing = np.nan
    else:
        peak_bracing = np.max(knee_accel_window)

        lead_leg_bracing = np.clip(100 * peak_bracing / 15000, 0, 100)

    Pitcher_performance_metrics["lead_leg_bracing"] = {
        "score": float(lead_leg_bracing),
        "peak_extension_acceleration_deg_s2": float(peak_bracing),
    }

    #################################################################
    # Throwing forearm rotational velocity (handedness corrected)
    #################################################################

    #################################################################
    # Angular acceleration
    #################################################################

    arm_acceleration = np.gradient(throwing_hand_rot_speed, dt)

    #################################################################
    # Late acceleration window before release
    #################################################################

    late_window = slice(max(release_frame - 12, p1), release_frame + 1)

    #################################################################
    # Peak acceleration magnitude
    #################################################################

    peak_idx = np.argmax(abs(arm_acceleration[late_window])) + late_window.start

    peak_acceleration = arm_acceleration[peak_idx]

    Pitcher_performance_metrics["late_arm_acceleration"] = {
        "peak_deg_s2": float(abs(peak_acceleration)),
        "time_to_release_ms": float(1000 * (peak_idx - release_frame) / frame_rate),
        "peak_frame": int(peak_idx),
    }

    pelvis_peak = np.max(np.abs(pelvis_rot_speed[p1 : p8 + 1]))

    trunk_peak = np.max(np.abs(UT_rot_speed[p1 : p8 + 1]))

    arm_peak = np.max(np.abs(arm_rot_speed[p1 : p8 + 1]))

    hand_peak = np.max(np.abs(hand_rot_speed[p1 : p8 + 1]))

    transfer = np.mean([trunk_peak / pelvis_peak, arm_peak / trunk_peak, hand_peak / arm_peak])

    energy_transfer = np.clip(100 * transfer, 0, 100)

    Pitcher_performance_metrics["energy_transfer"] = {"score": float(energy_transfer)}

    #################################################################
    # Throwing hand trajectory
    # Already dominance corrected:
    # throwing_wrist = wrist_l or wrist_r
    #################################################################

    hand_path = throwing_wrist

    #################################################################
    # Pitching acceleration phase
    # P4 = maximum leg lift / start of drive
    # P7 = ball release
    #################################################################

    hand_segment = hand_path[p4 : p7 + 1]

    #################################################################
    # Actual hand path travelled
    #################################################################

    segment_lengths = np.linalg.norm(np.diff(hand_segment, axis=0), axis=1)

    actual_path = np.sum(segment_lengths)

    #################################################################
    # Direct displacement
    #################################################################

    direct_path = np.linalg.norm(hand_segment[-1] - hand_segment[0])

    #################################################################
    # Efficiency calculation
    #################################################################

    if actual_path > 1e-6:
        efficiency = direct_path / actual_path

    else:
        efficiency = 0.0

    #################################################################
    # Additional interpretation values
    #################################################################

    path_waste = 1.0 - efficiency

    Pitcher_performance_metrics["hand_path_efficiency"] = {
        "value": float(efficiency),
        "percent": float(efficiency * 100),
        "actual_path_length_m": float(actual_path),
        "direct_displacement_m": float(direct_path),
        "path_waste_percent": float(path_waste * 100),
        "window": {"start_frame": int(p4), "release_frame": int(p7)},
    }

    #################################################################
    # Throwing forearm rotational velocity (handedness corrected)
    #################################################################

    #################################################################
    # Release window stability
    #################################################################

    release_window = slice(max(release_frame - 3, p1), min(release_frame + 4, p8))

    release_velocity_variability = np.std(throwing_hand_rot_speed[release_window])

    #################################################################
    # Stability score
    #################################################################

    stability = np.clip(100 - release_velocity_variability / 10, 0, 100)

    if stability > 90:
        label = "Excellent"

    elif stability > 75:
        label = "Good"

    elif stability > 60:
        label = "Average"

    else:
        label = "Poor"

    Pitcher_performance_metrics["release_stability"] = {
        "score": float(stability),
        "classification": label,
        "release_velocity_sd_deg_s": float(release_velocity_variability),
        "release_window_frames": [int(max(release_frame - 3, p1)), int(min(release_frame + 4, p8))],
    }
    return Pitcher_performance_metrics


def throwing_hand_vectors(
    hand, hand_force, tangential, axial, tangent_axis, axial_axis, frame_rate
):
    # These legacy vectors use an unconditional order-four 3/5 Hz filter.
    # The widgets also support shorter recordings through a filter fallback.
    if len(hand) <= 15 or frame_rate <= 10:
        raise ValueError(
            "throwing-hand force vectors require more than 15 frames and frame_rate above 10 Hz"
        )
    return {
        "start_3d": hand.tolist(),
        "end_rad_force_3d": (hand + axial_axis * axial[:, None] * 0.01).tolist(),
        "end_tan_force_3d": (hand + tangent_axis * tangential[:, None] * 0.01).tolist(),
        "end_combined_force_3d": (hand + hand_force * 0.01).tolist(),
    }
