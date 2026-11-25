import math

import numpy as np
from scipy import signal

from .common import fit_plane
from .math_helpers import rms


def _fill_degenerate(vecs, valid_mask):
    # Forward-fill then back-fill frames where a unit vector could not be reliably defined
    # (near-zero MH speed / near-zero path curvature - e.g. near the top of the backswing).
    out = vecs.copy()
    last_valid = None
    for i in range(len(out)):
        if valid_mask[i]:
            last_valid = out[i].copy()
        elif last_valid is not None:
            out[i] = last_valid
    last_valid = None
    for i in range(len(out) - 1, -1, -1):
        if valid_mask[i]:
            last_valid = out[i].copy()
        elif not valid_mask[i] and last_valid is not None:
            out[i] = last_valid
    return out


def _butter_filt_cols(x_2d, fc_hz, frame_rate):
    Nb = 4
    cb, db = signal.butter(Nb, fc_hz / (frame_rate / 2))
    out = np.zeros_like(x_2d)
    for col in range(x_2d.shape[1]):
        out[:, col] = signal.filtfilt(cb, db, x_2d[:, col])
    return out


def raw_midhand(lefty, pose3d):
    hand_l_3d = pose3d[17 if lefty else 21]
    hand_r_3d = pose3d[21 if lefty else 17]
    MH_point = (hand_l_3d + hand_r_3d) / 2
    return MH_point


def shaft_angle(club_mid_2d, lefty, pose3d):
    elbow_l_3d = pose3d[15 if lefty else 19]
    wrist_l_3d = pose3d[16 if lefty else 20]
    lead_forearm_shaft_angle_2d = []
    for idx in range(0, len(elbow_l_3d)):
        # Wrist -> club midpoint
        A_l = np.sqrt(
            (club_mid_2d[idx, 0] - wrist_l_3d[idx, 0]) ** 2
            + (club_mid_2d[idx, 1] - wrist_l_3d[idx, 1]) ** 2
            + (club_mid_2d[idx, 2] - wrist_l_3d[idx, 2]) ** 2
        )

        # Elbow -> club midpoint
        C_l = np.sqrt(
            (elbow_l_3d[idx, 0] - club_mid_2d[idx, 0]) ** 2
            + (elbow_l_3d[idx, 1] - club_mid_2d[idx, 1]) ** 2
            + (elbow_l_3d[idx, 2] - club_mid_2d[idx, 2]) ** 2
        )

        # Elbow -> wrist
        B_l = np.sqrt(
            (elbow_l_3d[idx, 0] - wrist_l_3d[idx, 0]) ** 2
            + (elbow_l_3d[idx, 1] - wrist_l_3d[idx, 1]) ** 2
            + (elbow_l_3d[idx, 2] - wrist_l_3d[idx, 2]) ** 2
        )

        # Numerical protection for acos
        cos_angle = (A_l**2 + B_l**2 - C_l**2) / (2 * A_l * B_l)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        lead_forearm_shaft_angle_2d.append(math.acos(cos_angle))
    lead_forearm_shaft_angle_deg_2d = np.array(lead_forearm_shaft_angle_2d) * 360 / (2 * np.pi)
    return lead_forearm_shaft_angle_deg_2d


def ground_power(
    W,
    ankle_l,
    ankle_r,
    elbow_l,
    elbow_r,
    foot_l,
    foot_r,
    frame_rate,
    hand_l,
    hand_r,
    hip_l,
    hip_r,
    knee_l,
    knee_r,
    neck,
    pelvis,
    shoulder_l,
    shoulder_r,
    skull,
    skullbase,
    wrist_l,
    wrist_r,
):
    V = skull - skullbase
    factor = 0.5976
    head_COM = skull - V * factor
    V = neck - pelvis
    factor = 0.4486
    trunk_COM = neck - V * factor
    V = shoulder_r - elbow_r
    factor = 0.5772
    arm_r_COM = shoulder_r - V * factor
    V = elbow_r - wrist_r
    factor = 0.4574
    forearm_r_COM = elbow_r - V * factor
    V = wrist_r - hand_r
    factor = 0.79
    hand_r_COM = wrist_r - V * factor
    V = shoulder_l - elbow_l
    factor = 0.5772
    arm_l_COM = shoulder_l - V * factor
    V = elbow_l - wrist_l
    factor = 0.4574
    forearm_l_COM = elbow_l - V * factor
    V = wrist_l - hand_l
    factor = 0.79
    hand_l_COM = wrist_l - V * factor
    V = hip_r - knee_r
    factor = 0.4095
    thigh_r_COM = hip_r - V * factor
    V = knee_r - ankle_r
    factor = 0.4459
    shank_r_COM = knee_r - V * factor
    V = ankle_r - foot_r
    factor = 0.4415
    foot_r_COM = ankle_r - V * factor
    V = hip_l - knee_l
    factor = 0.4095
    thigh_l_COM = hip_l - V * factor
    V = knee_l - ankle_l
    factor = 0.4459
    shank_l_COM = knee_l - V * factor
    V = ankle_l - foot_l
    factor = 0.4415
    foot_l_COM = ankle_l - V * factor
    head_m = 6.94 / 100
    trunk_m = 43.36 / 100
    arm_r_m = 2.71 / 100
    forearm_r_m = 1.62 / 100
    hand_r_m = 0.61 / 100
    thigh_r_m = 14.16 / 100
    shank_r_m = 4.33 / 100
    foot_r_m = 1.37 / 100
    arm_l_m = 2.71 / 100
    forearm_l_m = 1.62 / 100
    hand_l_m = 0.61 / 100
    thigh_l_m = 14.16 / 100
    shank_l_m = 4.33 / 100
    foot_l_m = 1.37 / 100
    COM = (
        head_COM * head_m
        + trunk_COM * trunk_m
        + arm_r_COM * arm_r_m
        + forearm_r_COM * forearm_r_m
        + hand_r_COM * hand_r_m
        + thigh_r_COM * thigh_r_m
        + shank_r_COM * shank_r_m
        + foot_r_COM * foot_r_m
        + arm_l_COM * arm_l_m
        + forearm_l_COM * forearm_l_m
        + hand_l_COM * hand_l_m
        + thigh_l_COM * thigh_l_m
        + shank_l_COM * shank_l_m
        + foot_l_COM * foot_l_m
    )
    dt = 1 / frame_rate
    COM_f = np.zeros(COM.shape)
    fc = 3  # Hz
    N = 4  # Order of butterworth filter
    c, d = signal.butter(N, fc / (frame_rate / 2))
    COM_f[:, 0] = signal.filtfilt(c, d, COM[:, 0])  # Vfilt_array3(COM[:, 0], frame_rate, 3)
    COM_f[:, 1] = signal.filtfilt(c, d, COM[:, 1])  # Vfilt_array3(COM[:, 1], frame_rate, 3)
    fc = 5  # Hz
    N = 4  # Order of butterworth filter
    c, d = signal.butter(N, fc / (frame_rate / 2))
    COM_f[:, 2] = signal.filtfilt(c, d, COM[:, 2])  # Vfilt_array3(COM[:, 2], frame_rate, 5)
    COM_f_disp_x = COM_f[:, 0] - COM_f[0, 0]
    COM_f_disp_y = COM_f[:, 1] - COM_f[0, 1]
    COM_f_disp_z = COM_f[:, 2] - COM_f[0, 2]
    COM_f_speed_x = np.gradient(COM_f_disp_x, dt)
    COM_f_speed_y = np.gradient(COM_f_disp_y, dt)
    COM_f_speed_z = np.gradient(COM_f_disp_z, dt)
    COM_f_acc_x = np.gradient(COM_f_speed_x, dt)
    COM_f_acc_y = np.gradient(COM_f_speed_y, dt)
    COM_f_acc_z = np.gradient(COM_f_speed_z, dt)
    Fx_t_est = (W) * COM_f_acc_x
    Fy_t_est = (W) * COM_f_acc_y
    Fz_t_est = (W) * (COM_f_acc_z + 9.81)
    Fz_t_est[Fz_t_est < 0] = 0
    vertical_power = Fx_t_est * COM_f_speed_x + Fy_t_est * COM_f_speed_y + Fz_t_est * COM_f_speed_z
    return vertical_power


def lead_hand_power(W, dt, frame_rate, hand_l):
    hand_l_m = 0.61 / 100
    hand_l_f = np.zeros(hand_l.shape)
    fc = 3  # Hz
    N = 4  # Order of butterworth filter
    c, d = signal.butter(N, fc / (frame_rate / 2))
    hand_l_f[:, 0] = signal.filtfilt(c, d, hand_l[:, 0])  # Vfilt_array3(COM[:, 0], frame_rate, 3)
    hand_l_f[:, 1] = signal.filtfilt(c, d, hand_l[:, 1])  # Vfilt_array3(COM[:, 1], frame_rate, 3)
    fc = 5  # Hz
    N = 4  # Order of butterworth filter
    c, d = signal.butter(N, fc / (frame_rate / 2))
    hand_l_f[:, 2] = signal.filtfilt(c, d, hand_l[:, 2])  # Vfilt_array3(COM[:, 2], frame_rate, 5)
    hand_l_disp_x = hand_l_f[:, 0] - hand_l_f[0, 0]
    hand_l_disp_y = hand_l_f[:, 1] - hand_l_f[0, 1]
    hand_l_disp_z = hand_l_f[:, 2] - hand_l_f[0, 2]
    hand_l_speed_x = np.gradient(hand_l_disp_x, dt)
    hand_l_speed_y = np.gradient(hand_l_disp_y, dt)
    hand_l_speed_z = np.gradient(hand_l_disp_z, dt)
    hand_COM_acc_x = np.gradient(hand_l_speed_x, dt)
    hand_COM_acc_y = np.gradient(hand_l_speed_y, dt)
    hand_COM_acc_z = np.gradient(hand_l_speed_z, dt)
    Fx_t_est_hand = W * hand_l_m * hand_COM_acc_x
    Fy_t_est_hand = W * hand_l_m * hand_COM_acc_y
    Fz_t_est_hand = W * hand_l_m * hand_COM_acc_z
    hand_speed_X = np.diff(hand_l[:, 0]) * frame_rate  # m/s
    hand_speed_X = np.insert(hand_speed_X, 0, 0)
    hand_speed_Y = np.diff(hand_l[:, 1]) * frame_rate
    hand_speed_Y = np.insert(hand_speed_Y, 0, 0)
    hand_speed_Z = np.diff(hand_l[:, 2]) * frame_rate
    hand_speed_Z = np.insert(hand_speed_Z, 0, 0)
    hand_power_x = Fx_t_est_hand * hand_speed_X
    hand_power_y = Fy_t_est_hand * hand_speed_Y
    hand_power_z = Fz_t_est_hand * hand_speed_Z
    hand_power = hand_power_x + hand_power_y + hand_power_z
    return hand_power


def equipment_mass(bat_type):
    if bat_type == "driver":  # == 1:
        shaft_mass = 0.059  # in kg
        club_head_mass = 0.192  # in kg
    else:
        shaft_mass = 0.120  # in kg
        club_head_mass = 0.256  # in kg
    club_mass = shaft_mass + club_head_mass  # in kg
    return shaft_mass, club_head_mass, club_mass


def equipment_com(club_head, club_head_mass, club_mass, club_mid, shaft_mass):
    shaft_COM = club_mid
    club_COM = (shaft_COM * shaft_mass + club_head * club_head_mass) / club_mass
    return club_COM


def head_motion(club_head, club_head_mass, dt, frame_rate):
    club_speed_X = np.gradient(club_head[:, 0], dt)
    club_speed_Y = np.gradient(club_head[:, 1], dt)
    club_speed_Z = np.gradient(club_head[:, 2], dt)
    club_speed = np.sqrt(club_speed_X**2 + club_speed_Y**2 + club_speed_Z**2)  # m/s
    fc = 5  # Hz
    N = 4  # Order of butterworth filter
    c, d = signal.butter(N, fc / (frame_rate / 2))
    club_speed = np.array(club_speed) / 2.23693629
    club_speed = signal.filtfilt(c, d, club_speed)
    club_head_acc = np.gradient(club_speed, dt)
    F_club_head = club_head_mass * club_head_acc  # N
    return club_speed, F_club_head


def head_power(F_club_head, club_speed):
    club_head_power = F_club_head * club_speed
    return club_head_power


def power_index(bat_type, club_head_power, p4, p7):
    if p4 > p7:
        raise ValueError("power window requires p4 <= p7")
    peak_club_power = max(club_head_power[p4 : p7 + 1])
    if bat_type == "driver":
        mean_club_power1 = 4457
        std_club_power1 = 497
    else:
        mean_club_power1 = 7342
        std_club_power1 = 2863
    power_index2 = peak_club_power / (mean_club_power1 + std_club_power1 * 4) * 100
    return power_index2


def com_motion(club_COM, dt):
    club_COM_speed_X = np.gradient(club_COM[:, 0], dt)
    club_COM_speed_Y = np.gradient(club_COM[:, 1], dt)
    club_COM_speed_Z = np.gradient(club_COM[:, 2], dt)
    return club_COM_speed_X, club_COM_speed_Y, club_COM_speed_Z


def com_velocity(club_COM_speed_X, club_COM_speed_Y, club_COM_speed_Z, frame_rate):
    fc = 4  # Hz
    N = 4  # Order of butterworth filter
    c, d = signal.butter(N, fc / (frame_rate / 2))
    club_COM_speed_X_filt = signal.filtfilt(c, d, club_COM_speed_X)
    club_COM_speed_Y_filt = signal.filtfilt(c, d, club_COM_speed_Y)
    club_COM_speed_Z_filt = signal.filtfilt(c, d, club_COM_speed_Z)
    return club_COM_speed_X_filt, club_COM_speed_Y_filt, club_COM_speed_Z_filt


def com_acceleration(club_COM_speed_X_filt, club_COM_speed_Y_filt, club_COM_speed_Z_filt, dt):
    club_COM_acc_X = np.gradient(club_COM_speed_X_filt, dt)
    club_COM_acc_Y = np.gradient(club_COM_speed_Y_filt, dt)
    club_COM_acc_Z = np.gradient(club_COM_speed_Z_filt, dt)
    return club_COM_acc_X, club_COM_acc_Y, club_COM_acc_Z


def shaft_power(
    club_COM_acc_X,
    club_COM_acc_Y,
    club_COM_acc_Z,
    club_COM_speed_X_filt,
    club_COM_speed_Y_filt,
    club_COM_speed_Z_filt,
    club_mass,
):
    F_shaft_X = club_mass * club_COM_acc_X
    F_shaft_Y = club_mass * club_COM_acc_Y
    F_shaft_Z = club_mass * club_COM_acc_Z
    P_shaft = (
        F_shaft_X * club_COM_speed_X_filt
        + F_shaft_Y * club_COM_speed_Y_filt
        + F_shaft_Z * club_COM_speed_Z_filt
    )
    return P_shaft


def downswing_start(club_COM_speed_X, club_COM_speed_Y, club_COM_speed_Z, p1, p7):
    club_COM_speed = np.sqrt(club_COM_speed_X**2 + club_COM_speed_Y**2 + club_COM_speed_Z**2)  # m/s
    if p1 + 10 >= p7:
        raise ValueError("downswing search requires p1 + 10 < p7")
    index_min_club_COM_speed = np.argmin(club_COM_speed[p1 + 10 : p7]) + p1 + 10
    return index_min_club_COM_speed


def midhand_force(club_COM, club_COM_acc_X, club_COM_acc_Y, club_COM_acc_Z, club_mass):
    club_COM_acc_vec = np.column_stack([club_COM_acc_X, club_COM_acc_Y, club_COM_acc_Z])
    W_C_vec = np.tile(np.array([0.0, 0.0, -9.81 * club_mass]), (len(club_COM), 1))
    F_MH_vec = club_mass * club_COM_acc_vec - W_C_vec
    return F_MH_vec


def shaft_moment(club_head, club_main, club_mass, dt, frame_rate):
    shaft_vec = club_head - club_main
    shaft_len = np.linalg.norm(shaft_vec, axis=1)
    s_hat = shaft_vec / shaft_len[:, np.newaxis]
    s_hat_filt = _butter_filt_cols(s_hat, 10, frame_rate)
    s_hat_filt = s_hat_filt / np.linalg.norm(s_hat_filt, axis=1)[:, np.newaxis]
    ds_dt = np.gradient(s_hat_filt, dt, axis=0)
    omega_perp = np.cross(s_hat_filt, ds_dt)  # club angular velocity, perpendicular to the shaft
    I_trans = club_mass * shaft_len**2 / 12  # transverse MOI (slender rod about its own COM)
    h_C_vec = I_trans[:, np.newaxis] * omega_perp
    h_C_vec_filt = _butter_filt_cols(h_C_vec, 8, frame_rate)
    h_dot_C_vec = np.gradient(h_C_vec_filt, dt, axis=0)
    return h_dot_C_vec


def midhand_moments(F_MH_vec, MH_point, club_COM, h_dot_C_vec):
    r_CH_vec = MH_point - club_COM
    MH_force_moment_vec = np.cross(r_CH_vec, F_MH_vec)
    M_MH_vec = h_dot_C_vec - MH_force_moment_vec
    return MH_force_moment_vec, M_MH_vec


def net_moment(MH_force_moment_vec, M_MH_vec):
    T_net_vec = M_MH_vec + MH_force_moment_vec
    return T_net_vec


def plane_axis(MH_point, frame_rate, lefty, p4, p7):
    """Fit and orient the moving swing-plane normal in capture coordinates."""
    frame_window_ms = 100.0  # +/- half-window in ms - tunable
    half_win = max(3, int(round(frame_window_ms / 1000.0 * frame_rate)))

    B_hat = np.zeros((len(MH_point), 3))
    for _i in range(len(MH_point)):
        _lo = max(0, _i - half_win)
        _hi = min(len(MH_point), _i + half_win + 1)
        if _hi - _lo < 3:
            B_hat[_i] = np.array([0.0, 0.0, 1.0])
            continue
        _plane_v, _plane_com = fit_plane(MH_point[_lo:_hi, :])
        B_hat[_i] = _plane_v

    # Eigenvector sign is arbitrary per-window - enforce frame-to-frame continuity first.
    for _i in range(1, len(B_hat)):
        if np.dot(B_hat[_i], B_hat[_i - 1]) < 0:
            B_hat[_i] = -B_hat[_i]

    # Anchor the OVERALL sign using the POSITION geometry of the downswing sweep itself (P4:P7) - the
    # signed area swept by the MH path around its own local centroid, projected onto B_hat. This uses
    # only positions (no velocity/acceleration/angular-velocity derivatives at all), which proved more
    # reliable than the earlier omega_perp-based anchor (that one is contaminated by wrist cock/release,
    # which can rotate the club shaft in a way that doesn't track the overall arm-swing rotation, and
    # produced a regression on real data). Every downswing sweeps the hands in the same physical
    # rotational sense, so this anchor is swing-independent.
    _p4c, _p7c = min(p4, p7), max(p4, p7)
    _pts = MH_point[_p4c : _p7c + 1]
    _local_com = np.mean(_pts, axis=0)
    _rel = _pts - _local_com
    _cross_terms = np.cross(_rel[:-1], _rel[1:])
    _signed_area_vec = np.sum(_cross_terms, axis=0)
    _mean_B_over_window = np.mean(B_hat[_p4c : _p7c + 1], axis=0)
    _sweep_sign_val = np.dot(_signed_area_vec, _mean_B_over_window)
    B_hat_flipped = bool(_sweep_sign_val < 0)
    if B_hat_flipped:
        B_hat = -B_hat

    # Handedness correction: MH_point (from hand_l_3d/hand_r_3d) and the club data are deliberately kept
    # in the RAW, untransformed capture frame (paired with each other, per earlier agreement) - meaning
    # this frame is NOT mirrored for lefty golfers the way the rest of the script's pose-derived metrics
    # are. A lefty and a righty swing sweep the hands in opposite rotational senses within that same raw
    # frame, so the position-based sweep anchor above is only self-consistent WITHIN one handedness.
    # This additional correction makes "positive" consistently mean "promotes the golfer's own swing
    # direction" for both - confirmed as the likely cause of Fig.4-7 flipping between two real swings.
    if lefty == 1:
        B_hat = -B_hat
    return B_hat


def tangent_axis(B_hat, MH_point, dt, frame_rate):
    v_MH = _butter_filt_cols(np.gradient(MH_point, dt, axis=0), 6, frame_rate)
    v_proj = (
        v_MH - np.sum(v_MH * B_hat, axis=1)[:, np.newaxis] * B_hat
    )  # velocity projected into the plane
    v_proj_norm = np.linalg.norm(v_proj, axis=1)
    T_hat_valid = v_proj_norm > 1e-4
    T_hat = np.zeros_like(v_MH)
    T_hat[T_hat_valid] = v_proj[T_hat_valid] / v_proj_norm[T_hat_valid, np.newaxis]
    T_hat = _fill_degenerate(T_hat, T_hat_valid)
    return T_hat


def normal_axis(B_hat, T_hat):
    N_hat = np.cross(B_hat, T_hat)  # completes the right-handed frame (T x N = B)
    N_norm = np.linalg.norm(N_hat, axis=1)
    N_valid = N_norm > 1e-6
    N_hat_safe = np.zeros_like(N_hat)
    N_hat_safe[N_valid] = N_hat[N_valid] / N_norm[N_valid, np.newaxis]
    N_hat = _fill_degenerate(N_hat_safe, N_valid)
    return N_hat


def tangential_force(F_MH_vec, T_hat):
    F_MH_tangential = np.sum(F_MH_vec * T_hat, axis=1)
    F_tan_shaft = F_MH_tangential
    return F_tan_shaft


def normal_force(F_MH_vec, N_hat):
    F_MH_normal = np.sum(F_MH_vec * N_hat, axis=1)
    F_rad_shaft = np.abs(
        F_MH_normal
    )  # magnitude - grip must resist the radial component regardless of sign
    return F_MH_normal, F_rad_shaft


def grip_force(F_rad_shaft):
    taper_deg = 4
    taper_rad = taper_deg * 2 * np.pi / 360
    friction = 0.7
    Grip = F_rad_shaft / friction * (1 - np.sin(taper_rad))
    return Grip


def couple_projection(B_hat, M_MH_vec):
    torque_axis = -B_hat
    M_MH_swing_axis = np.sum(
        M_MH_vec * torque_axis, axis=1
    )  # in-plane driving torque (Kwon Fig. 4)
    Wrist_torque = M_MH_swing_axis
    C_in_plane = Wrist_torque
    return C_in_plane


def force_moment_projection(B_hat, MH_force_moment_vec):
    torque_axis = -B_hat
    MHforce_moment_swing_axis = np.sum(
        MH_force_moment_vec * torque_axis, axis=1
    )  # in-plane (Kwon Fig. 5)
    M_in_plane = MHforce_moment_swing_axis
    return M_in_plane


def net_moment_projection(B_hat, T_net_vec):
    T_net_swing_axis = np.sum(T_net_vec * B_hat, axis=1)  # in-plane (Kwon Fig. 6)
    T_in_plane = T_net_swing_axis
    return T_in_plane


def out_of_plane_moments(B_hat, F_MH_vec, MH_force_moment_vec, M_MH_vec, N_hat, T_hat):
    F_MH_swing_axis = np.sum(F_MH_vec * B_hat, axis=1)  # out-of-plane force component (new)
    torque_T = -T_hat
    torque_N = -N_hat
    M_MH_tangential = np.sum(M_MH_vec * torque_T, axis=1)  # out-of-plane (new)
    M_MH_normal = np.sum(M_MH_vec * torque_N, axis=1)  # out-of-plane (new)
    MHforce_moment_tangential = np.sum(MH_force_moment_vec * torque_T, axis=1)  # out-of-plane (new)
    MHforce_moment_normal = np.sum(MH_force_moment_vec * torque_N, axis=1)  # out-of-plane (new)
    return (
        F_MH_swing_axis,
        M_MH_tangential,
        M_MH_normal,
        MHforce_moment_tangential,
        MHforce_moment_normal,
    )


def out_of_plane_net_moment(N_hat, T_hat, T_net_vec):
    T_net_tangential = np.sum(T_net_vec * T_hat, axis=1)  # out-of-plane (new)
    T_net_normal = np.sum(T_net_vec * N_hat, axis=1)  # out-of-plane (new)
    return T_net_tangential, T_net_normal


def shaft_force_vectors(F_rad_shaft, F_tan_shaft, club_COM, club_head, club_main):
    F_tan_vec_norm = club_head[1:] - club_head[:-1]

    # Add a check for zero norm to avoid division by zero
    norms = np.linalg.norm(F_tan_vec_norm, axis=1)
    nonzero_norms = norms != 0  # Find elements with non-zero norms
    F_tan_vec_norm[nonzero_norms] /= norms[nonzero_norms, np.newaxis]

    F_tan_vec_norm = np.vstack([[0, 0, 0], F_tan_vec_norm])

    F_rad_vec_norm = club_main - club_COM
    F_rad_vec_norm = F_rad_vec_norm / np.linalg.norm(F_rad_vec_norm, axis=1)[:, np.newaxis]

    # Shaft force vectors (3D)
    # Shaft force vectors (3D)
    F_tan_vec = F_tan_vec_norm * F_tan_shaft[:, np.newaxis]
    F_rad_vec = F_rad_vec_norm * F_rad_shaft[:, np.newaxis]
    return F_tan_vec, F_rad_vec


def point_to_line_projection(x, u, v):
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    th = np.dot(x - v, u) / np.dot(u, u)

    h = v + th * u

    return h


def projected_com(
    F_rad_vec,
    F_tan_vec,
    club_head,
    club_head_mass,
    club_main,
    club_mass,
    index_min_club_COM_speed,
    shaft_mass,
):
    shaft_COM = (club_head + club_main) / 2
    club_COM_2d = (shaft_COM * shaft_mass + club_head * club_head_mass) / club_mass
    if len(club_COM_2d) != len(F_rad_vec):
        raise ValueError("club_3d frame count must match pose3d for projection")
    projection = []
    for frame, center in enumerate(club_COM_2d):
        if frame < index_min_club_COM_speed:
            direction = F_rad_vec[frame] - F_tan_vec[frame]
        else:
            direction = F_rad_vec[frame] + F_tan_vec[frame]
        projection.append(point_to_line_projection(center, direction, club_main[frame]))
    proj_club_COM = np.array(projection)
    return proj_club_COM, club_COM_2d


def highlights(C_in_plane, F_club_head, F_rad_shaft, F_tan_shaft, Grip, frame_rate, p1, p7, p8):
    Wrist_torque = C_in_plane
    max_F_tan_shaft, index_max_F_tan_shaft = (
        np.max(F_tan_shaft[p1 : p8 + 1]),
        np.argmax(F_tan_shaft[p1 : p8 + 1]),
    )
    time_to_impact_F_tan_shaft = 1000 * (index_max_F_tan_shaft - p7 + p1) / frame_rate
    max_F_rad_shaft, index_max_F_rad_shaft = (
        np.max(F_rad_shaft[p1 : p8 + 1]),
        np.argmax(F_rad_shaft[p1 : p8 + 1]),
    )
    time_to_impact_F_rad_shaft = 1000 * (index_max_F_rad_shaft - p7 + p1) / frame_rate
    max_Wrist_torque, index_max_Wrist_torque = (
        np.max(Wrist_torque[p1 : p8 + 1]),
        np.argmax(Wrist_torque[p1 : p8 + 1]),
    )
    time_to_impact_Wrist_torque = 1000 * (index_max_Wrist_torque - p7 + p1) / frame_rate
    max_F_club_head, index_max_F_club_head = (
        np.max(F_club_head[p1 : p8 + 1]),
        np.argmax(F_club_head[p1 : p8 + 1]),
    )
    time_to_impact_F_club_head = 1000 * (index_max_F_club_head - p7 + p1) / frame_rate
    max_Grip, index_max_Grip = np.max(Grip[p1 : p8 + 1]), np.argmax(Grip[p1 : p8 + 1])
    time_to_impact_Grip = 1000 * (index_max_Grip - p7 + p1) / frame_rate
    peak_force_order = [
        index_max_F_tan_shaft,
        index_max_F_rad_shaft,
        index_max_Wrist_torque,
    ]
    index_sorted = np.argsort(peak_force_order)
    peak_force_order_F_tan_shaft = np.where(index_sorted == 0)[0]
    peak_force_order_F_rad_shaft = np.where(index_sorted == 1)[0]
    peak_force_order_Wrist_torque = np.where(index_sorted == 2)[0]
    kinetic_order_frontend = {
        "peak_force_order_F_tangential_force": peak_force_order_F_tan_shaft[0] + 1,
        "peak_force_order_F_centripetal_force": peak_force_order_F_rad_shaft[0] + 1,
        "peak_force_order_Wrist_torque": peak_force_order_Wrist_torque[0] + 1,
    }
    time_to_impact = {
        "time_to_impact_tangential_force": time_to_impact_F_tan_shaft,
        "time_to_impact_centripetal_force": time_to_impact_F_rad_shaft,
        "time_to_impact_wrist_torque": time_to_impact_Wrist_torque,
        "time_to_impact_Grip_force": time_to_impact_Grip,
        "time_to_impact_F_club_head": time_to_impact_F_club_head,
    }
    peak_club_mechanics = {
        "peak_tangential_force": max_F_tan_shaft,
        "peak_centripetal_force": max_F_rad_shaft,
        "peak_wrist_torque": max_Wrist_torque,
        "peak_Grip_force": max_Grip,
        "peak_F_club_head": max_F_club_head,
    }
    M2C_club_mechanics_highlights = {
        "peak_club_mechanics": peak_club_mechanics,
        "time_to_impact": time_to_impact,
        "kinetic_order": kinetic_order_frontend,
    }
    return M2C_club_mechanics_highlights


def insights(
    arc_width,
    arc_width_inch,
    lead_forearm_shaft_angle_deg_2d,
    x_factor,
    x_factor_stretch,
):
    M2C_insights = {
        "arc_width": arc_width.tolist(),
        "arc_width_inch": arc_width_inch.tolist(),
        "lead_forearm_shaft_angle": lead_forearm_shaft_angle_deg_2d.tolist(),
        "x_factor": x_factor.tolist(),
        "x_factor_stretch": x_factor_stretch,
    }
    return M2C_insights


def power(P_shaft, club_head_power, hand_power, power_index2, vertical_power):
    M2C_power = {
        "vertical_power": vertical_power.tolist(),
        "hand_power": hand_power.tolist(),
        "Shaft_power": P_shaft.tolist(),
        "club_head_power": club_head_power.tolist(),
        "power_index": power_index2,
    }
    return M2C_power


def mechanics(C_in_plane, F_club_head, F_rad_shaft, F_tan_shaft, Grip):
    M2C_club_mechanics = {
        "Tangential_Force": F_tan_shaft.tolist(),
        "Centripetal_Force": F_rad_shaft.tolist(),
        "Grip_Force": Grip.tolist(),
        "F_club_head": F_club_head.tolist(),
        "Wrist_torque": C_in_plane.tolist(),
    }
    return M2C_club_mechanics


def bat_force_vectors(F_rad_vec, F_tan_vec, lefty, pose3d):
    hand = pose3d[21 if lefty else 17]
    # The legacy loop overwrote the entire tangential vector on each
    # iteration. Its final iteration always selected the positive vector.
    return {
        "start_3d": hand.tolist(),
        "end_rad_force_3d": (hand + F_rad_vec * 0.01).tolist(),
        "end_tan_force_3d": (hand + F_tan_vec * 0.01).tolist(),
    }


def wrist(dt, elbow_l, frame_rate, hand_l, lead_forearm_shaft_angle_deg_2d, wrist_l):
    wrist_sway = 100 * (wrist_l[:, 1] - wrist_l[0, 1])
    wrist_thrust = 100 * (wrist_l[:, 0] - wrist_l[0, 0])
    wrist_lift = 100 * (wrist_l[:, 2] - wrist_l[0, 2])
    A_l = np.sqrt(
        (hand_l[:, 0] - wrist_l[:, 0]) ** 2
        + (hand_l[:, 1] - wrist_l[:, 1]) ** 2
        + (hand_l[:, 2] - wrist_l[:, 2]) ** 2
    )
    B_l = np.sqrt(
        (elbow_l[:, 0] - hand_l[:, 0]) ** 2
        + (elbow_l[:, 1] - hand_l[:, 1]) ** 2
        + (elbow_l[:, 2] - hand_l[:, 2]) ** 2
    )
    C_l = np.sqrt(
        (elbow_l[:, 0] - wrist_l[:, 0]) ** 2
        + (elbow_l[:, 1] - wrist_l[:, 1]) ** 2
        + (elbow_l[:, 2] - wrist_l[:, 2]) ** 2
    )
    wrist_angle_l = np.arccos((A_l**2 + C_l**2 - B_l**2) / (2 * A_l * C_l))
    wrist_angle_l_deg = np.degrees(wrist_angle_l)
    wrist_vel_l_deg = np.gradient(wrist_angle_l_deg, dt)
    wrist_speed_X = (wrist_l[1:, 0] - wrist_l[:-1, 0]) * frame_rate  # m/s
    wrist_speed_Y = (wrist_l[1:, 1] - wrist_l[:-1, 1]) * frame_rate
    wrist_speed_Z = (wrist_l[1:, 2] - wrist_l[:-1, 2]) * frame_rate
    wrist_speed = np.concatenate(
        [[0], np.sqrt(wrist_speed_X**2 + wrist_speed_Y**2 + wrist_speed_Z**2)]
    )  # m/s
    factor = 2.23693629
    wrist_speed_mph = wrist_speed * factor
    M2C_wrist = {
        "lead_forearm_shaft_angle": lead_forearm_shaft_angle_deg_2d.tolist(),
        "lead_wrist_extension_flexion": wrist_angle_l_deg.tolist(),
        "lead_wrist_extension_flexion_vel": wrist_vel_l_deg.tolist(),
        "lead_wrist_speed": wrist_speed_mph.tolist(),
        "lead_wrist_thrust": wrist_thrust.tolist(),
        "lead_wrist_sway": wrist_sway.tolist(),
        "lead_wrist_lift": wrist_lift.tolist(),
    }
    return M2C_wrist


def in_plane_metrics(C_in_plane, M_in_plane, T_in_plane):
    inPlaneMetrics = {
        "M_in_plane": M_in_plane.tolist(),
        "C_in_plane": C_in_plane.tolist(),
        "T_in_plane": T_in_plane.tolist(),
    }
    return inPlaneMetrics


def out_of_plane_metrics(
    F_MH_swing_axis,
    MHforce_moment_normal,
    MHforce_moment_tangential,
    M_MH_normal,
    M_MH_tangential,
    T_net_normal,
    T_net_tangential,
):
    outOfPlaneMetrics = {
        "F_out_of_plane": F_MH_swing_axis.tolist(),  # net MH force, out-of-plane component
        "M_out_of_plane_tangential": MHforce_moment_tangential.tolist(),
        "M_out_of_plane_normal": MHforce_moment_normal.tolist(),
        "C_out_of_plane_tangential": M_MH_tangential.tolist(),
        "C_out_of_plane_normal": M_MH_normal.tolist(),
        "T_out_of_plane_tangential": T_net_tangential.tolist(),
        "T_out_of_plane_normal": T_net_normal.tolist(),
    }
    return outOfPlaneMetrics


def kinetics(
    B_hat,
    F_MH_normal,
    F_MH_vec,
    F_tan_shaft,
    MH_force_moment_vec,
    MH_point,
    M_MH_vec,
    N_hat,
    T_hat,
    T_net_vec,
):
    M2C_kinetics_3d = {
        "MH_point_3d": MH_point.tolist(),
        "F_MH_vec_3d": F_MH_vec.tolist(),
        "M_MH_vec_3d": M_MH_vec.tolist(),
        "MH_force_moment_vec_3d": MH_force_moment_vec.tolist(),
        "T_net_vec_3d": T_net_vec.tolist(),
        "swing_frame_tangential_3d": T_hat.tolist(),
        "swing_frame_normal_3d": N_hat.tolist(),
        "swing_frame_axis_3d": B_hat.tolist(),
        "F_MH_tangential_signed": F_tan_shaft.tolist(),  # same as Tangential_Force, kept here for plotting
        "F_MH_normal_signed": F_MH_normal.tolist(),  # signed version (Centripetal_Force is abs())
    }
    return M2C_kinetics_3d


def in_plane_vectors(club_COM_2d, club_main, proj_club_COM):
    proj_club_COM_pixel = proj_club_COM.squeeze()[:, :2] * 1080  # 2D Coordinates of projected COM
    club_COM_2d_pixel = club_COM_2d * 1080
    club_main_2d_pixel = club_main * 1080
    M2C_inplane_vectors = {
        "proj_club_COM_pixel": proj_club_COM_pixel.tolist(),
        "club_COM_2d_pixel": club_COM_2d_pixel.tolist(),
        "club_main_2d_pixel": club_main_2d_pixel.tolist(),
    }
    return M2C_inplane_vectors


def in_plane_vectors_3d(club_COM_2d, club_main, proj_club_COM):
    proj_club_COM_3d = proj_club_COM.squeeze()
    club_COM_3d = club_COM_2d
    club_main_3d = club_main
    M2C_inplane_vectors_3d = {
        "proj_club_COM_3d": proj_club_COM_3d.tolist(),
        "club_COM_3d": club_COM_3d.tolist(),
        "club_main_3d": club_main_3d.tolist(),
    }
    return M2C_inplane_vectors_3d


def performance(
    C_in_plane,
    F_rad_shaft,
    F_tan_shaft,
    M_in_plane,
    T_in_plane,
    T_net_normal,
    T_net_tangential,
    frame_rate,
    p7,
):
    impact_frame = p7
    peak_tan_idx = np.argmax(np.abs(F_tan_shaft))
    peak_tan_force = float(F_tan_shaft[peak_tan_idx])
    peak_tan_time_ms = (peak_tan_idx - impact_frame) / frame_rate * 1000
    peak_norm_idx = np.argmax(np.abs(F_rad_shaft))
    peak_norm_force = float(F_rad_shaft[peak_norm_idx])
    peak_norm_time_ms = (peak_norm_idx - impact_frame) / frame_rate * 1000
    T_net = T_in_plane
    peak_torque_idx = np.argmax(np.abs(T_net))
    peak_torque = float(T_net[peak_torque_idx])
    peak_torque_time_ms = (peak_torque_idx - impact_frame) / frame_rate * 1000
    C_in = C_in_plane
    M_in = M_in_plane
    rms_net = rms(T_net)
    rms_sources = rms(C_in) + rms(M_in)
    torque_efficiency = rms_net / rms_sources if rms_sources > 0 else 0
    torque_cancellation = 1.0 - torque_efficiency
    force_direction_ratio = rms(F_tan_shaft) / rms(F_rad_shaft) if rms(F_rad_shaft) > 0 else 0
    if force_direction_ratio < 0.6:
        force_direction_interpretation = "Steering dominant"

    elif force_direction_ratio < 1.0:
        force_direction_interpretation = "Balanced"

    else:
        force_direction_interpretation = "Acceleration dominant"
    rms_oop = rms(np.concatenate([T_net_tangential, T_net_normal]))
    rms_inplane = rms(T_net)
    barrel_stability_index = rms_oop / rms_inplane if rms_inplane > 0 else 0
    t_ms = (np.arange(len(T_net)) - impact_frame) / frame_rate * 1000
    late_mask = (t_ms >= -50) & (t_ms <= 20)
    oop_total = np.sqrt(np.array(T_net_tangential) ** 2 + np.array(T_net_normal) ** 2)
    late_oop = oop_total[late_mask]
    peak_oop = np.max(late_oop)
    peak_oop_idx = np.where(oop_total == peak_oop)[0][0]
    peak_oop_time = t_ms[peak_oop_idx]
    Performance_metrics = {
        "peak_tangential_force": {
            "value_N": peak_tan_force,
            "time_ms": peak_tan_time_ms,
        },
        "peak_normal_force": {
            "value_N": peak_norm_force,
            "time_ms": peak_norm_time_ms,
        },
        "peak_net_torque": {
            "value_Nm": peak_torque,
            "time_ms": peak_torque_time_ms,
        },
        "torque_efficiency": {
            "efficiency_percent": 100 * torque_efficiency,
            "cancellation_percent": 100 * torque_cancellation,
        },
        "force_direction_ratio": {
            "ratio": force_direction_ratio,
            "interpretation": force_direction_interpretation,
        },
        "barrel_stability_index": {
            "ratio": barrel_stability_index,
            "percent": 100 * barrel_stability_index,
        },
        "peak_out_of_plane_torque": {
            "value_Nm": peak_oop,
            "time_ms": peak_oop_time,
        },
    }
    return Performance_metrics
