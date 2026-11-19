import numpy as np
import math
from scipy import signal
from numpy.linalg import norm
from biomechanics.constants import SEGMENT_MASS_FRACTIONS


def segment_angles(ctx, COM):
    frame_rate, W = ctx.frame_rate, ctx.weight
    downtheLine = int(ctx.down_the_line)
    m = ctx.markers
    pelvis = m["pelvis"]
    hip_l = m["hip_l"]
    knee_l = m["knee_l"]
    ankle_l = m["ankle_l"]
    foot_l = m["foot_l"]
    midfoot_l = m["midfoot_l"]
    hip_r = m["hip_r"]
    knee_r = m["knee_r"]
    ankle_r = m["ankle_r"]
    foot_r = m["foot_r"]
    midfoot_r = m["midfoot_r"]
    nose = m["nose"]
    spine_low = m["spine_low"]
    spine_high = m["spine_high"]
    shoulder_l = m["shoulder_l"]
    elbow_l = m["elbow_l"]
    wrist_l = m["wrist_l"]
    hand_l = m["hand_l"]
    shoulder_r = m["shoulder_r"]
    elbow_r = m["elbow_r"]
    wrist_r = m["wrist_r"]
    hand_r = m["hand_r"]
    neck = m["neck"]
    skullbase = m["skullbase"]
    skull = m["skull"]
    club_main = m["club_main"]
    club_head = m["club_head"]
    ear_l = m["ear_l"]
    ear_r = m["ear_r"]
    #######################################################  HAND PATH (cm) ######################################################################
    zero_array = np.array([0, 0, 0])
    # Three components of hand path
    hand_path_pre = hand_l[1:] - hand_l[0:-1]  # in m
    hand_path_pre = np.insert(hand_path_pre, 0, zero_array, axis=0)

    # Resultant hand path (and conversion to cm)
    hand_path = []
    for index in range(0, len(hand_path_pre)):
        hand_path.append(
            100
            * math.sqrt(
                sum(
                    [
                        hand_path_pre[index][0] ** 2,
                        hand_path_pre[index][1] ** 2,
                        hand_path_pre[index][2] ** 2,
                    ]
                )
            )
        )

    fc = 5  # Hz
    N = 4  # Order of butterworth filter

    c, d = signal.butter(N, fc / (frame_rate / 2))

    #   % Compute distance between shoulders on the Z plane
    head_dist = []
    for index in range(0, len(ear_r)):
        head_dist.append(
            math.sqrt(
                (ear_r[index, 0] - ear_l[index, 0]) ** 2
                + (ear_r[index, 1] - ear_l[index, 1]) ** 2
            )
        )
    head_dist = np.array(head_dist)

    #  % Head turn (around Z axis)
    head_turn = []
    head_turn_deg_end = []

    for idx in range(0, len(ear_r)):
        head_turn.append(
            -math.atan2(
                nose[idx, 0] - skullbase[idx, 0], nose[idx, 1] - skullbase[idx, 1]
            )
        )

    head_turn_deg_end = 90 + np.array(head_turn) * (360 / (2 * np.pi))  # from rad 2 deg

    if downtheLine == 0:
        head_turn_deg_end = head_turn_deg_end.tolist()

    # % Head tilt (around X axis)
    head_tilt = []
    for index in range(0, len(ear_r)):
        head_tilt.append(
            math.atan2((ear_l[index, 2] - ear_r[index, 2]), head_dist[index])
        )
    head_tilt_deg = np.array(head_tilt) * 360 / (2 * np.pi)
    # from rad to deg
    head_tilt_deg = signal.filtfilt(c, d, head_tilt_deg)

    # % Head bend (around Y axis)
    head_bend = []
    for index in range(0, len(ear_r)):
        head_bend.append(
            math.atan2(
                (skull[index, 0] - skullbase[index, 0]),
                skull[index, 2] - skullbase[index, 2],
            )
        )
    head_bend_deg = np.array(head_bend) * 360 / (2 * np.pi)
    # from rad to deg
    head_bend_deg = signal.filtfilt(c, d, head_bend_deg)
    ################################################## PELVIS ANGLES (TURN, TILT and BEND) ############################################################
    x_hip_l = []
    y_hip_l = []
    z_hip_l = []
    for point in hip_l:
        x_hip_l.append(point[0])
        y_hip_l.append(point[1])
        z_hip_l.append(point[2])

    x_hip_r = []
    y_hip_r = []
    z_hip_r = []
    for point in hip_r:
        x_hip_r.append(point[0])
        y_hip_r.append(point[1])
        z_hip_r.append(point[2])

    #   % Compute distance between hips on the Z plane
    pelvis_dist = []
    for index in range(0, len(x_hip_r)):
        pelvis_dist.append(
            math.sqrt(
                (x_hip_r[index] - x_hip_l[index]) ** 2
                + (y_hip_r[index] - y_hip_l[index]) ** 2
            )
        )
    pelvis_dist = np.array(pelvis_dist)

    pelvis_turn = []
    pelvis_turn_deg_end = []
    #   % Pelvis turn (around Z axis)
    for idx in range(0, len(x_hip_l)):
        pelvis_turn.append(
            -math.atan2(x_hip_l[idx] - x_hip_r[idx], y_hip_l[idx] - y_hip_r[idx])
        )
    pelvis_turn_deg_end = np.array(pelvis_turn) * (360 / (2 * np.pi))

    #  % Pelvis tilt (around X axis)
    pelvis_tilt = []
    for index in range(0, len(z_hip_l)):
        pelvis_tilt.append(
            math.atan2((z_hip_l[index] - z_hip_r[index]), pelvis_dist[index])
        )
    pelvis_tilt_deg = np.array(pelvis_tilt) * 360 / (2 * np.pi)
    # from rad to deg
    pelvis_tilt_deg = signal.filtfilt(c, d, pelvis_tilt_deg)

    #   % Pelvis bend (triangle method to be used)
    hip_med = (hip_l + hip_r) / 2
    # mid point between hips (base of the triangle)
    x_hip_med = []
    z_hip_med = []
    for point in hip_med:
        x_hip_med.append(point[0])
        z_hip_med.append(point[2])

    x_pelvis = []
    z_pelvis = []
    for point in pelvis:
        x_pelvis.append(point[0])
        z_pelvis.append(point[2])

    pelvis_bend = []
    for index in range(0, len(x_hip_med)):
        pelvis_bend.append(
            math.atan2(
                x_pelvis[index] - x_hip_med[index], z_pelvis[index] - z_hip_med[index]
            )
        )  # inclination of the triangle with respect to the vertical axis﻿﻿

    pelvis_bend_deg = np.array(pelvis_bend) * 360 / (2 * np.pi)
    # from rad to deg
    pelvis_bend_deg = signal.filtfilt(c, d, pelvis_bend_deg)

    ################################################## UT ANGLES  (TURN, TILT and BEND) #######################################################
    x_shoulder_l = []
    y_shoulder_l = []
    z_shoulder_l = []
    for point in shoulder_l:
        x_shoulder_l.append(point[0])
        y_shoulder_l.append(point[1])
        z_shoulder_l.append(point[2])

    x_shoulder_r = []
    y_shoulder_r = []
    z_shoulder_r = []
    for point in shoulder_r:
        x_shoulder_r.append(point[0])
        y_shoulder_r.append(point[1])
        z_shoulder_r.append(point[2])

    # UT Angles (Turn, TILT, BEND)

    UT_dist = []
    for index in range(0, len(x_shoulder_r)):
        UT_dist.append(
            math.sqrt(
                (x_shoulder_r[index] - x_shoulder_l[index]) ** 2
                + (y_shoulder_r[index] - y_shoulder_l[index]) ** 2
            )
        )
    UT_dist = np.array(UT_dist)

    #  % UT turn (around Z axis)
    UT_turn = []
    UT_turn_deg_end = []

    for idx in range(0, len(z_shoulder_r)):
        UT_turn.append(
            -math.atan2(
                x_shoulder_l[idx] - x_shoulder_r[idx],
                y_shoulder_l[idx] - y_shoulder_r[idx],
            )
        )

    UT_turn_deg_end = np.array(UT_turn) * (
        360 / (2 * np.pi)
    )  # - target_line_deg # from rad 2 deg
    UT_turn_deg_end = signal.filtfilt(c, d, UT_turn_deg_end)

    # % UT tilt (around X axis)
    UT_tilt = []
    for index in range(0, len(z_shoulder_r)):
        UT_tilt.append(
            math.atan2((z_shoulder_l[index] - z_shoulder_r[index]), UT_dist[index])
        )
    UT_tilt_deg = np.array(UT_tilt) * 360 / (2 * np.pi)
    # from rad to deg
    UT_tilt_deg = signal.filtfilt(c, d, UT_tilt_deg)

    # % UT bend (around Y axis)
    UT_bend = []
    for index in range(0, len(y_shoulder_r)):
        UT_bend.append(
            math.atan2(
                (spine_high[index, 0] - spine_low[index, 0]),
                (spine_high[index, 2] - spine_low[index, 2]),
            )
        )
    UT_bend_deg = np.array(UT_bend) * 360 / (2 * np.pi)
    # from rad to deg
    UT_bend_deg = signal.filtfilt(c, d, UT_bend_deg)
    ## Knee angle right & ANGULAR VELOCITY
    knee_angle_r_deg = []
    for index in range(0, len(ankle_r)):
        A_r = math.sqrt(
            (ankle_r[index, 0] - knee_r[index, 0]) ** 2
            + (ankle_r[index, 1] - knee_r[index, 1]) ** 2
            + (ankle_r[index, 2] - knee_r[index, 2]) ** 2
        )
        B_r = math.sqrt(
            (hip_r[index, 0] - ankle_r[index, 0]) ** 2
            + (hip_r[index, 1] - ankle_r[index, 1]) ** 2
            + (hip_r[index, 2] - ankle_r[index, 2]) ** 2
        )
        C_r = math.sqrt(
            (hip_r[index, 0] - knee_r[index, 0]) ** 2
            + (hip_r[index, 1] - knee_r[index, 1]) ** 2
            + (hip_r[index, 2] - knee_r[index, 2]) ** 2
        )

        knee_angle_r = math.acos((A_r**2 + C_r**2 - B_r**2) / (2 * A_r * C_r))
        knee_angle_r_deg.append(knee_angle_r * 360 / (2 * np.pi))
    knee_angle_r_deg = np.array(knee_angle_r_deg)
    # knee_vel_r_deg = np.gradient(knee_angle_r_deg, dt)

    knee_angle_r_deg = np.around(knee_angle_r_deg, 2)
    knee_angle_r_deg = np.around(knee_angle_r_deg, 2)

    # Knee angle left & ANGULAR VELOCITY
    knee_angle_l_deg = []
    for index in range(0, len(ankle_l)):
        A_r = math.sqrt(
            (ankle_l[index, 0] - knee_l[index, 0]) ** 2
            + (ankle_l[index, 1] - knee_l[index, 1]) ** 2
            + (ankle_l[index, 2] - knee_l[index, 2]) ** 2
        )
        B_r = math.sqrt(
            (hip_l[index, 0] - ankle_l[index, 0]) ** 2
            + (hip_l[index, 1] - ankle_l[index, 1]) ** 2
            + (hip_l[index, 2] - ankle_l[index, 2]) ** 2
        )
        C_r = math.sqrt(
            (hip_l[index, 0] - knee_l[index, 0]) ** 2
            + (hip_l[index, 1] - knee_l[index, 1]) ** 2
            + (hip_l[index, 2] - knee_l[index, 2]) ** 2
        )

        knee_angle_l = math.acos((A_r**2 + C_r**2 - B_r**2) / (2 * A_r * C_r))
        knee_angle_l_deg.append(knee_angle_l * 360 / (2 * np.pi))
    knee_angle_l_deg = np.array(knee_angle_l_deg)
    # knee_vel_l_deg = np.gradient(knee_angle_l_deg, dt)

    knee_angle_l_deg = np.around(knee_angle_l_deg, 2)
    knee_angle_l_deg = np.around(knee_angle_l_deg, 2)

    ## RIGHT ELBOW ANGLE & ANGULAR VELOCITY
    elbow_angle_r_deg = []
    for index in range(0, len(wrist_r)):
        A_r = math.sqrt(
            (wrist_r[index, 0] - elbow_r[index, 0]) ** 2
            + (wrist_r[index, 1] - elbow_r[index, 1]) ** 2
            + (wrist_r[index, 2] - elbow_r[index, 2]) ** 2
        )
        B_r = math.sqrt(
            (shoulder_r[index, 0] - wrist_r[index, 0]) ** 2
            + (shoulder_r[index, 1] - wrist_r[index, 1]) ** 2
            + (shoulder_r[index, 2] - wrist_r[index, 2]) ** 2
        )
        C_r = math.sqrt(
            (shoulder_r[index, 0] - elbow_r[index, 0]) ** 2
            + (shoulder_r[index, 1] - elbow_r[index, 1]) ** 2
            + (shoulder_r[index, 2] - elbow_r[index, 2]) ** 2
        )

        elbow_angle_r = math.acos((A_r**2 + C_r**2 - B_r**2) / (2 * A_r * C_r))
        elbow_angle_r_deg.append(elbow_angle_r * 360 / (2 * np.pi))

    elbow_angle_r_deg = np.array(elbow_angle_r_deg)
    # elbow_vel_r_deg = np.gradient(elbow_angle_r_deg, dt)

    elbow_angle_r_deg = np.around(elbow_angle_r_deg, 2)
    # elbow_vel_r_deg = np.around(elbow_vel_r_deg,2)

    ## LEFT ELBOW ANGLE & ANGULAR VELOCITY
    elbow_angle_l_deg = []
    for index in range(0, len(wrist_l)):
        A_r = math.sqrt(
            (wrist_l[index, 0] - elbow_l[index, 0]) ** 2
            + (wrist_l[index, 1] - elbow_l[index, 1]) ** 2
            + (wrist_l[index, 2] - elbow_l[index, 2]) ** 2
        )
        B_r = math.sqrt(
            (shoulder_l[index, 0] - wrist_l[index, 0]) ** 2
            + (shoulder_l[index, 1] - wrist_l[index, 1]) ** 2
            + (shoulder_l[index, 2] - wrist_l[index, 2]) ** 2
        )
        C_r = math.sqrt(
            (shoulder_l[index, 0] - elbow_l[index, 0]) ** 2
            + (shoulder_l[index, 1] - elbow_l[index, 1]) ** 2
            + (shoulder_l[index, 2] - elbow_l[index, 2]) ** 2
        )

        elbow_angle_l = math.acos((A_r**2 + C_r**2 - B_r**2) / (2 * A_r * C_r))
        elbow_angle_l_deg.append(elbow_angle_l * 360 / (2 * np.pi))

    elbow_angle_l_deg = np.array(elbow_angle_l_deg)
    # elbow_vel_l_deg = np.gradient(elbow_angle_l_deg, dt)

    elbow_angle_l_deg = np.around(elbow_angle_l_deg, 2)
    # elbow_vel_l_deg = np.around(elbow_vel_l_deg,2)

    ########################################################### App offsets #######################################################

    ####################################################### MAX UT TURN AND MAX PELVIS TURN #################################################################

    # 26.09.2024: Adding the x factor metrics to the basic screen
    x_factor = UT_turn_deg_end - pelvis_turn_deg_end
    x_factor_min, x_factor_min_idx = min(x_factor), np.argmin(x_factor)

    # lead hand
    hand_speed_test = (hand_l[1:] - hand_l[0:-1]) * frame_rate
    hand_speed_vec = np.insert(hand_speed_test, 0, zero_array, axis=0)
    hand_speed = norm(hand_speed_vec, axis=-1)
    #
    # trail hand
    trail_hand_speed_test = (hand_r[1:] - hand_r[0:-1]) * frame_rate
    trail_hand_speed_vec = np.insert(trail_hand_speed_test, 0, zero_array, axis=0)
    trail_hand_speed = norm(trail_hand_speed_vec, axis=-1)

    dt = 1 / frame_rate

    ##################### Compute hand power ###############################
    dt = 1 / frame_rate
    hand_l_f = np.zeros(hand_l.shape)
    fc = 3  # Hz
    N = 4  # Order of butterworth filter

    c, d = signal.butter(N, fc / (frame_rate / 2))
    hand_l_f[:, 0] = signal.filtfilt(
        c, d, hand_l[:, 0]
    )  # Vfilt_array3(COM[:, 0], frame_rate, 3)
    hand_l_f[:, 1] = signal.filtfilt(
        c, d, hand_l[:, 1]
    )  # Vfilt_array3(COM[:, 1], frame_rate, 3)

    fc = 5  # Hz
    N = 4  # Order of butterworth filter
    c, d = signal.butter(N, fc / (frame_rate / 2))
    hand_l_f[:, 2] = signal.filtfilt(
        c, d, hand_l[:, 2]
    )  # Vfilt_array3(COM[:, 2], frame_rate, 5)

    # Left hand displacement
    hand_l_disp_x = hand_l_f[:, 0] - hand_l_f[0, 0]
    hand_l_disp_y = hand_l_f[:, 1] - hand_l_f[0, 1]
    hand_l_disp_z = hand_l_f[:, 2] - hand_l_f[0, 2]

    # Left hand speed (m/s)
    hand_l_speed_x = np.gradient(hand_l_disp_x, dt)
    hand_l_speed_y = np.gradient(hand_l_disp_y, dt)
    hand_l_speed_z = np.gradient(hand_l_disp_z, dt)

    # Left hand acceleration (m/s^2)
    hand_COM_acc_x = np.gradient(hand_l_speed_x, dt)
    hand_COM_acc_y = np.gradient(hand_l_speed_y, dt)
    hand_COM_acc_z = np.gradient(hand_l_speed_z, dt)

    # Estimate combined GRF for the Left hand (mass = W * hand_l_m)
    Fx_t_est_hand = W * SEGMENT_MASS_FRACTIONS["hand_l"] * hand_COM_acc_x
    Fy_t_est_hand = W * SEGMENT_MASS_FRACTIONS["hand_l"] * hand_COM_acc_y
    Fz_t_est_hand = W * SEGMENT_MASS_FRACTIONS["hand_l"] * hand_COM_acc_z

    # Combined force magnitude
    F_hand = []
    for i in range(len(Fx_t_est_hand)):
        F_hand.append(
            math.sqrt(
                Fx_t_est_hand[i] ** 2 + Fy_t_est_hand[i] ** 2 + Fz_t_est_hand[i] ** 2
            )
        )
    F_hand = np.array(F_hand)

    # Signed instantaneous power (dot product of force and velocity vectors)
    hand_power = (
        Fx_t_est_hand * hand_l_speed_x
        + Fy_t_est_hand * hand_l_speed_y
        + Fz_t_est_hand * hand_l_speed_z
    )

    # index_MAX_hand_power = np.argmax(hand_power[p4 : p7 + 1]) + p4
    ##################### Compute maximum hand speed in mph ###############################
    factor = 2.23693629
    hand_speed_mph = hand_speed * factor  # mph
    trail_hand_speed_mph = trail_hand_speed * factor

    fc = 20  # Hz
    N = 2  # Order of butterworth filter

    c, d = signal.butter(N, fc / (frame_rate / 2))
    hand_speed_mph = signal.filtfilt(c, d, hand_speed_mph)
    trail_hand_speed_mph = signal.filtfilt(c, d, trail_hand_speed_mph)

    # Find max value for hand speed
    _, index_MAX_hand_speed_mph = hand_speed_mph.max(), np.argmax(hand_speed_mph)  # mph
    index_MAX_hand_speed_mph = index_MAX_hand_speed_mph

    #################################################################################################

    fc = 5  # Hz
    N = 4  # Order of butterworth filter

    c, d = signal.butter(N, fc / (frame_rate / 2))
    COM_f = signal.filtfilt(c, d, COM[:, 2])

    # COM displacement in z direction
    COM_f_disp_z = COM_f[:] - COM_f[0]

    # Calculate velocity and acceleration
    COM_f_speed_z = np.gradient(COM_f_disp_z, dt)
    COM_f_acc_z = np.gradient(COM_f_speed_z, dt)

    Fz_t_est = (W) * (COM_f_acc_z + 9.81)
    Fz_t_est[Fz_t_est < 0] = 0

    weight_perc = Fz_t_est / (W * 9.81) * 100  # 11.06.2024: Update to percentage value
    # 13.08.24: Changed the output metrics, which can be plotted to be only having in total 2 decimals
    head_turn_deg_end = np.around(head_turn_deg_end, 2)
    head_tilt_deg = np.around(head_tilt_deg, 2)
    head_bend_deg = np.around(head_bend_deg, 2)
    UT_turn_deg_end = np.around(UT_turn_deg_end, 2)
    UT_tilt_deg = np.around(UT_tilt_deg, 2)
    UT_bend_deg = np.around(UT_bend_deg, 2)
    pelvis_turn_deg_end = np.around(pelvis_turn_deg_end, 2)
    pelvis_tilt_deg = np.around(pelvis_tilt_deg, 2)
    pelvis_bend_deg = np.around(pelvis_bend_deg, 2)
    knee_angle_r_deg = np.around(knee_angle_r_deg, 2)
    knee_angle_l_deg = np.around(knee_angle_l_deg, 2)
    elbow_angle_r_deg = np.around(elbow_angle_r_deg, 2)
    elbow_angle_l_deg = np.around(elbow_angle_l_deg, 2)
    weight_perc = np.around(weight_perc, 2)
    hand_speed_mph = np.around(hand_speed_mph, 2)
    segment_angles = {
        "head_turn_deg": head_turn_deg_end.tolist(),
        "head_tilt_deg": head_tilt_deg.tolist(),
        "head_bend_deg": head_bend_deg.tolist(),
        "UT_turn_deg": UT_turn_deg_end.tolist(),
        "UT_tilt_deg": UT_tilt_deg.tolist(),
        "UT_bend_deg": UT_bend_deg.tolist(),
        "pelvis_turn_deg": pelvis_turn_deg_end.tolist(),
        "pelvis_tilt_deg": pelvis_tilt_deg.tolist(),
        "pelvis_bend_deg": pelvis_bend_deg.tolist(),
        "knee_angle_r_deg": knee_angle_r_deg.tolist(),
        "knee_angle_l_deg": knee_angle_l_deg.tolist(),
        "elbow_angle_r_deg": elbow_angle_r_deg.tolist(),
        "elbow_angle_l_deg": elbow_angle_l_deg.tolist(),
        "weight_perc": weight_perc.tolist(),
        "hand_speed_mph": hand_speed_mph.tolist(),
        "index_MAX_hand_speed_mph": index_MAX_hand_speed_mph,
        "x_factor": x_factor.tolist(),  # 26.09.2024: Adding the x factor metrics to the basic screen
        "x_factor_min": x_factor_min,
        "x_factor_min_idx": x_factor_min_idx,
    }

    return segment_angles
