import numpy as np

import json
from scipy.linalg import svd
import math
from scipy import signal
#import scipy as sp
#import os
#import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3d projection
# plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\Haider Haseeb\ffmpeg-8.1.2-essentials_build\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe"
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.signal import savgol_filter


def safe_normalize(vectors): #HH 10_06_2026
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    return vectors / norms

def numpy_int_parser(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError


# V7: low-pass cut-off (Hz) used for pitcher arm power / elbow kinetics. Clipped to 0.4 x Nyquist at runtime.
PITCH_KIN_FC_HZ = 10


def rms(x):
    x = np.asarray(x)
    return np.sqrt(np.mean(x**2))


def clubFitting(json3d):
    def fit_plane(point_cloud):
        com = np.sum(point_cloud, axis=0) / len(point_cloud)
        # calculate the center of mass
        q = point_cloud - com
        # move the com to the origin and translate all the points (use numpy broadcasting)
        Q = np.dot(q.T, q)
        # calculate 3x3 matrix. The inner product returns total sum of 3x3 matrix
        # Q is symmetric by construction (q.T @ q), so eigh (not eig) guarantees real eigenvalues/
        # eigenvectors and avoids the ComplexWarning from tiny floating-point asymmetry noise.
        la, vectors = np.linalg.eigh(Q)
        # Calculate eigenvalues and eigenvectors
        plane_v = vectors.T[np.argmin(la)]
        # Extract the eigenvector of the minimum eigenvalue

        return plane_v, com

    def point_to_plane_distance(M, n, I):
        nb_pts = len(M)
        ## Body
        d_I = -(n[0] * I[0] + n[1] * I[1] + n[2] * I[2])
        t_H = -(
            np.tile(d_I, [nb_pts, 1]) + n[0] * M[0] + n[1] * M[1] + n[2] * M[2]
        ) / sum(n**2)

        H = np.zeros([3, 1])

        H[0] = M[0] + t_H[0] * n[0]
        H[1] = M[1] + t_H[1] * n[1]
        H[2] = M[2] + t_H[2] * n[2]
        H = H.reshape(1, 3)

        d2H = math.sqrt(np.sum((M - H) ** 2))
        return d2H, H

    # V7: movement type. "hitting" / "pitching" behave exactly as before; anything else
    # (squat, golf, jump, ... or missing) is "generic": no p-positions / club / bat required.
    _cfg_swing = json3d["data_3d_bio"]["configuration"].get("swing", {})
    _swing_type_str = str(_cfg_swing.get("swingType", "")).lower()
    if _swing_type_str == "hitting":
        movement_type = "hitting"
    elif _swing_type_str == "pitching":
        movement_type = "pitching"
    else:
        movement_type = "generic"

    if str(_cfg_swing.get("handedness", "righthanded")).lower() == "lefthanded":
        lefty = 1
    else:
        lefty = 0

    frame_rate = json3d["data_3d_bio"]["configuration"]["video"]["frame_rate"]
    _W_cfg = json3d["data_3d_bio"]["configuration"].get("user", {}).get("weight")
    try:
        mass_assumed = not (_W_cfg is not None and float(_W_cfg) > 0)
    except (TypeError, ValueError):
        mass_assumed = True
    W = 1.0 if mass_assumed else _W_cfg   # no weight given -> results are per kg

    #M = json3d["data"]["pose"]
    M = json3d["data_3d_bio"]["data"]["pose"]

    if _cfg_swing.get("orientation", "FaceOn") == "FaceOn":
        downtheLine = 0
    else:
        downtheLine = 1

    if downtheLine == 1:
        if lefty == 0:
            temp = np.squeeze(M)
            new_x = temp[:, :, 1]
            new_y = -temp[:, :, 0]
            temp[:, :, 0] = new_x  # change x to -y
            temp[:, :, 1] = new_y  # change y to -x
            M = temp
        else:
            temp = np.squeeze(M)
            new_x = -temp[:, :, 1]
            new_y = -temp[:, :, 0]
            temp[:, :, 0] = new_x  # change x to y
            temp[:, :, 1] = new_y  # change y to -x
            M = temp
    if lefty == 1:
        if downtheLine == 0:
            temp = np.squeeze(M)
            new_y = -temp[:, :, 1]
            temp[:, :, 1] = new_y  # change y to -y    # mirroring
            M = temp
    else:
        if downtheLine == 0:
            M = np.squeeze(json3d["data_3d_bio"]["data"]["pose"])
    M_new = M
    M = M_new[:, :, :]

    # P-Positions
    if movement_type == "generic":
        # V7: generic movements have no critical events; whole clip is used where a window is needed.
        p1 = 0
        p8 = M.shape[1] - 1
        p2 = p3 = p4 = p5 = p6 = p7 = None
    elif downtheLine == 0:
        if json3d["data_3d_bio"]["configuration"]["swing"]["swingType"] == "pitching":
            p1 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p1"]
            p2 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p2"]
            p3 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p3"]
            p4 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p4"]
            p5 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p5"]
            p6 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p6"]
            p7 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p7"]
            p8 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p8"]
        else:  # if this is a short game swing
            p1 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p1"]
            p4 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p4"]
            p7 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p7"]
            p8 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p8"]
    else:  # if this is a down the line swing
        if json3d["data_3d_bio"]["configuration"]["swing"]["swingType"] == "pitching":
            p1 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p1"]
            p2 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p2"]
            p3 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p3"]
            p4 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p4"]
            p5 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p5"]
            p6 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p6"]
            p7 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p7"]
            p8 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p8"]
        else:  # if this is a short game swing
            p1 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p1"]
            p4 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p4"]
            p7 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p7"]
            p8 = json3d["data_3d_bio"]["configuration"]["swing"]["critical_events"]["p8"]

    # Loading 3d data tracking
    if lefty == 1:

        pelvis = np.squeeze(M[0])
        hip_l = np.squeeze(M[1])
        knee_l = np.squeeze(M[2])
        ankle_l = np.squeeze(M[3])
        foot_l = np.squeeze(M[4])  # changed
        midfoot_l = np.squeeze(M[5])
        hip_r = np.squeeze(M[6])
        knee_r = np.squeeze(M[7])
        ankle_r = np.squeeze(M[8])
        foot_r = np.squeeze(M[9])  # changed
        midfoot_r = np.squeeze(M[10])
        nose = np.squeeze(M[11])  # only for calculations
        spine_low = np.squeeze(M[12])
        spine_high = np.squeeze(M[13])
        shoulder_l = np.squeeze(M[14])
        elbow_l = np.squeeze(M[15])
        wrist_l = np.squeeze(M[16])
        hand_l = np.squeeze(M[17])
        shoulder_r = np.squeeze(M[18])
        elbow_r = np.squeeze(M[19])
        wrist_r = np.squeeze(M[20])
        hand_r = np.squeeze(M[21])
        neck = np.squeeze(M[22])
        skullbase = np.squeeze(M[23])
        skull = np.squeeze(M[24])
        club_main = np.squeeze(M[25])  # NaN
        club_head = np.squeeze(M[26])  # NaN
        ear_l = np.squeeze(M[27])  # only for calculations       # check if for leftys
        ear_r = np.squeeze(M[28])  # only for calculations

    else:
        pelvis = np.squeeze(M[0])
        hip_r = np.squeeze(M[1])
        knee_r = np.squeeze(M[2])
        ankle_r = np.squeeze(M[3])
        foot_r = np.squeeze(M[4])  # changed
        midfoot_r = np.squeeze(M[5])
        hip_l = np.squeeze(M[6])
        knee_l = np.squeeze(M[7])
        ankle_l = np.squeeze(M[8])
        foot_l = np.squeeze(M[9])  # changed
        midfoot_l = np.squeeze(M[10])
        nose = np.squeeze(M[11])  # only for calculations
        spine_low = np.squeeze(M[12])
        spine_high = np.squeeze(M[13])
        shoulder_r = np.squeeze(M[14])
        elbow_r = np.squeeze(M[15])
        wrist_r = np.squeeze(M[16])
        hand_r = np.squeeze(M[17])
        shoulder_l = np.squeeze(M[18])
        elbow_l = np.squeeze(M[19])
        wrist_l = np.squeeze(M[20])
        hand_l = np.squeeze(M[21])
        neck = np.squeeze(M[22])
        skullbase = np.squeeze(M[23])
        skull = np.squeeze(M[24])
        club_main = np.squeeze(M[25])  # NaN
        club_head = np.squeeze(M[26])  # NaN
        ear_r = np.squeeze(M[27])  # only for calculations
        ear_l = np.squeeze(M[28])  # only for calculations

    # V7: generic pose data may contain NaN keypoints -> linearly interpolate so every downstream
    # calculation (incl. the kinematic sequence) stays finite. Affected joints are listed in generic.meta.
    def _interp_nan(a):
        a = np.array(a, dtype=float)
        idx = np.arange(a.shape[0])
        for c in range(a.shape[1]):
            bad = ~np.isfinite(a[:, c])
            if bad.any() and not bad.all():
                a[bad, c] = np.interp(idx[bad], idx[~bad], a[~bad, c])
        return a

    generic_missing_joints = []
    if movement_type == "generic":
        if not np.isfinite(pelvis).all():
            generic_missing_joints.append("pelvis")
            pelvis = _interp_nan(pelvis)
        if not np.isfinite(hip_l).all():
            generic_missing_joints.append("hip_l")
            hip_l = _interp_nan(hip_l)
        if not np.isfinite(hip_r).all():
            generic_missing_joints.append("hip_r")
            hip_r = _interp_nan(hip_r)
        if not np.isfinite(knee_l).all():
            generic_missing_joints.append("knee_l")
            knee_l = _interp_nan(knee_l)
        if not np.isfinite(knee_r).all():
            generic_missing_joints.append("knee_r")
            knee_r = _interp_nan(knee_r)
        if not np.isfinite(ankle_l).all():
            generic_missing_joints.append("ankle_l")
            ankle_l = _interp_nan(ankle_l)
        if not np.isfinite(ankle_r).all():
            generic_missing_joints.append("ankle_r")
            ankle_r = _interp_nan(ankle_r)
        if not np.isfinite(foot_l).all():
            generic_missing_joints.append("foot_l")
            foot_l = _interp_nan(foot_l)
        if not np.isfinite(foot_r).all():
            generic_missing_joints.append("foot_r")
            foot_r = _interp_nan(foot_r)
        if not np.isfinite(midfoot_l).all():
            generic_missing_joints.append("midfoot_l")
            midfoot_l = _interp_nan(midfoot_l)
        if not np.isfinite(midfoot_r).all():
            generic_missing_joints.append("midfoot_r")
            midfoot_r = _interp_nan(midfoot_r)
        if not np.isfinite(nose).all():
            generic_missing_joints.append("nose")
            nose = _interp_nan(nose)
        if not np.isfinite(spine_low).all():
            generic_missing_joints.append("spine_low")
            spine_low = _interp_nan(spine_low)
        if not np.isfinite(spine_high).all():
            generic_missing_joints.append("spine_high")
            spine_high = _interp_nan(spine_high)
        if not np.isfinite(shoulder_l).all():
            generic_missing_joints.append("shoulder_l")
            shoulder_l = _interp_nan(shoulder_l)
        if not np.isfinite(shoulder_r).all():
            generic_missing_joints.append("shoulder_r")
            shoulder_r = _interp_nan(shoulder_r)
        if not np.isfinite(elbow_l).all():
            generic_missing_joints.append("elbow_l")
            elbow_l = _interp_nan(elbow_l)
        if not np.isfinite(elbow_r).all():
            generic_missing_joints.append("elbow_r")
            elbow_r = _interp_nan(elbow_r)
        if not np.isfinite(wrist_l).all():
            generic_missing_joints.append("wrist_l")
            wrist_l = _interp_nan(wrist_l)
        if not np.isfinite(wrist_r).all():
            generic_missing_joints.append("wrist_r")
            wrist_r = _interp_nan(wrist_r)
        if not np.isfinite(hand_l).all():
            generic_missing_joints.append("hand_l")
            hand_l = _interp_nan(hand_l)
        if not np.isfinite(hand_r).all():
            generic_missing_joints.append("hand_r")
            hand_r = _interp_nan(hand_r)
        if not np.isfinite(neck).all():
            generic_missing_joints.append("neck")
            neck = _interp_nan(neck)
        if not np.isfinite(skullbase).all():
            generic_missing_joints.append("skullbase")
            skullbase = _interp_nan(skullbase)
        if not np.isfinite(skull).all():
            generic_missing_joints.append("skull")
            skull = _interp_nan(skull)
        if not np.isfinite(ear_l).all():
            generic_missing_joints.append("ear_l")
            ear_l = _interp_nan(ear_l)
        if not np.isfinite(ear_r).all():
            generic_missing_joints.append("ear_r")
            ear_r = _interp_nan(ear_r)

    ## --------------------------------------- x-factor / x-factor stretch -----------------------------------------------
    pelvis_turn = []
    #   % Pelvis turn (around Z axis)
    for idx in range(0, len(hip_l)):
        pelvis_turn.append(
            -math.atan2(hip_l[idx, 0] - hip_r[idx, 0], hip_l[idx, 1] - hip_r[idx, 1])
        )
    pelvis_turn_deg = np.array(pelvis_turn) * (360 / (2 * np.pi))

    #  % UT turn (around Z axis)
    UT_turn = []
    UT_turn_deg = []

    for idx in range(0, len(shoulder_r)):
        UT_turn.append(
            -math.atan2(
                shoulder_l[idx, 0] - shoulder_r[idx, 0],
                shoulder_l[idx, 1] - shoulder_r[idx, 1],
            )
        )

    UT_turn_deg = np.array(UT_turn) * (
        360 / (2 * np.pi)
    )  # - target_line_deg # from rad 2 deg

    x_factor = UT_turn_deg - pelvis_turn_deg

    if movement_type == "generic":
        x_factor_stretch = None
    else:
        x_factor_stretch = min(x_factor[p4 : p7 + 1]) - x_factor[p4]
    x_factor = abs(x_factor)
    ## ---------------------------------------Transition (Steepening/Shallowing)--------------------------------------------------



    ## ----------------------------------Initial Downswing Direction (IDD)-----------------------------------------------------

    ##-------------------------------------------- Slice ------------------------------------------------------------------------


    ## ------------------------BACKSWING PLANE ANGLE (Angle of the wrist at impact in relation to the horizon)-----------------------
    

    # -------------------------------BACKSWING ARC ANGLE (Angle of the arc hand takes from P1 to P4)-----------------------------------
    

    ## ---------------------------------------------------- ARC WIDTH (cm) ------------------------------------------------------
    arc_width = []
    mid_hand = (hand_l + hand_r) * 0.5
    for i in range(0, len(hand_l)):
        arc_width.append(
            100
            * math.sqrt(
                (mid_hand[i, 0] - spine_high[i, 0]) ** 2
                + (mid_hand[i, 1] - spine_high[i, 1]) ** 2
                + (mid_hand[i, 2] - spine_high[i, 2]) ** 2
            )
        )
    arc_width = np.array(arc_width)
    cm_to_inch_factor = 0.3937007874
    arc_width_inch = arc_width * cm_to_inch_factor

    ## ----------------------------------------------- L. Wrist - shaft angle -------------------------------------------------
    # V7: 0 = hitting, 1 = pitching, 2 = generic
    swing_type = {"hitting": 0, "pitching": 1, "generic": 2}[movement_type]

    if swing_type == 0:
        C = json3d["data_3d_bio"]["data"]["club"]


        ## Loading club data (3d input in meters, keeping old "_2d"-suffixed variable naming for compatibility)
        club_head_2d = []
        club_start_2d = []
        club_mid_2d = []

        for element in C:
            club_head_2d.append(C[element]["end"])
            club_start_2d.append(C[element]["start"])
            club_mid_2d.append(C[element]["mid"])

        club_head_2d = np.array(club_head_2d)
        club_start_2d = np.array(club_start_2d)
        club_mid_2d = np.array(club_mid_2d)

        club_start_2d_raw = club_start_2d

        # No PPMM multiplication needed - club input is already in meters:
        # club_start_2d = club_start_2d * json3d["configuration"]["swing"]["PPMM"]
        # club_mid_2d = club_mid_2d * json3d["configuration"]["swing"]["PPMM"]

        ## --- PIXEL-SPACE hand/wrist/elbow (unchanged from Script A) ---
        ## Kept for anything still consuming genuine 2D pixel coordinates, e.g. M2C_force_vectors_shaft
        ## used for frontend video overlay.
        # elbow_l_2d = []
        # hand_l_2d = []
        # hand_r_2d = []
        # wrist_l_2d = []
        # if lefty == 0:
        #     for element in json2d["data"]:
        #         elbow_l_2d.append(element[19])
        #         wrist_l_2d.append(element[20])
        #         hand_l_2d.append(element[21])
        #         hand_r_2d.append(element[17])
        # else:
        #     for element in json2d["data"]:
        #         elbow_l_2d.append(element[15])
        #         wrist_l_2d.append(element[16])
        #         hand_l_2d.append(element[17])
        #         hand_r_2d.append(element[21])
        # elbow_l_2d = np.array(elbow_l_2d)
        # wrist_l_2d = np.array(wrist_l_2d)
        # hand_l_2d = np.array(hand_l_2d)
        # hand_r_2d = np.array(hand_r_2d)

        ## --- 3D-METERS hand/wrist/elbow (new source: json3d pose) ---
        ## Used for lead_forearm_shaft_angle (needs to be paired with the 3D club data) and for
        ## any new 3D-only outputs (e.g. future stick-figure / avatar integration).
        elbow_l_3d = []
        hand_l_3d = []
        hand_r_3d = []
        wrist_l_3d = []

        pose = np.squeeze(np.array(json3d["data_3d_bio"]["data"]["pose"]))

        if lefty == 0:
            for idx in range(pose.shape[1]):
                elbow_l_3d.append(pose[19, idx])
                wrist_l_3d.append(pose[20, idx])
                hand_l_3d.append(pose[21, idx])
                hand_r_3d.append(pose[17, idx])
        else:
            for idx in range(pose.shape[1]):
                elbow_l_3d.append(pose[15, idx])
                wrist_l_3d.append(pose[16, idx])
                hand_l_3d.append(pose[17, idx])
                hand_r_3d.append(pose[21, idx])

        elbow_l_3d = np.array(elbow_l_3d)
        wrist_l_3d = np.array(wrist_l_3d)
        hand_l_3d = np.array(hand_l_3d)
        hand_r_3d = np.array(hand_r_3d)

        # Estimating the lead wrist shaft angle
        # Uses the 3D-meters wrist/elbow (paired with the 3D-meters club data) - intentional per 3D club update.
        # 03.06.24: Updated the lead forearm shaft angle with the wrist marker instead of the club start marker

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
            cos_angle = (A_l ** 2 + B_l ** 2 - C_l ** 2) / (2 * A_l * B_l)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            lead_forearm_shaft_angle_2d.append(
                math.acos(cos_angle)
            )

        lead_forearm_shaft_angle_deg_2d = (
                np.array(lead_forearm_shaft_angle_2d) * 360 / (2 * np.pi)
        )

        ## --------------------------------------------- Club - Force & Power ------------------------------------------------

        # % Compute center of mass of each segment
        hide_COM = 1
        if hide_COM == 1:
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

            # Compute mass (%) of each segment
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

            # COM filtering
            dt = 1 / frame_rate
            COM_f = np.zeros(COM.shape)
            fc = 3  # Hz
            N = 4  # Order of butterworth filter

            c, d = signal.butter(N, fc / (frame_rate / 2))
            COM_f[:, 0] = signal.filtfilt(
                c, d, COM[:, 0]
            )  # Vfilt_array3(COM[:, 0], frame_rate, 3)
            COM_f[:, 1] = signal.filtfilt(
                c, d, COM[:, 1]
            )  # Vfilt_array3(COM[:, 1], frame_rate, 3)

            fc = 5  # Hz
            N = 4  # Order of butterworth filter
            c, d = signal.butter(N, fc / (frame_rate / 2))
            COM_f[:, 2] = signal.filtfilt(
                c, d, COM[:, 2]
            )  # Vfilt_array3(COM[:, 2], frame_rate, 5)

            # COM filtered displacements (m)
            COM_f_disp_x = COM_f[:, 0] - COM_f[p1, 0]
            COM_f_disp_y = COM_f[:, 1] - COM_f[p1, 1]
            COM_f_disp_z = COM_f[:, 2] - COM_f[p1, 2]

            # COM filtered speeds (m/s)
            COM_f_speed_x = np.gradient(COM_f_disp_x, dt)
            COM_f_speed_y = np.gradient(COM_f_disp_y, dt)
            COM_f_speed_z = np.gradient(COM_f_disp_z, dt)
            COM_f_speed = np.sqrt(COM_f_speed_x**2 + COM_f_speed_y**2 + COM_f_speed_z**2)

            # COM filtered accelerations (m/s^2)
            COM_f_acc_x = np.gradient(COM_f_speed_x, dt)
            COM_f_acc_y = np.gradient(COM_f_speed_y, dt)
            COM_f_acc_z = np.gradient(COM_f_speed_z, dt)

            # Estimated combined GRF
            Fx_t_est = (W) * COM_f_acc_x
            Fy_t_est = (W) * COM_f_acc_y
            Fz_t_est = (W) * (COM_f_acc_z + 9.81)
            Fz_t_est[Fz_t_est < 0] = 0

        F_comb = []
        for i in range(0, len(Fx_t_est)):
            F_comb.append(math.sqrt(Fx_t_est[i] ** 2 + Fy_t_est[i] ** 2 + Fz_t_est[i] ** 2))
        F_comb = np.array(F_comb)

        # vertical power / ground power
        # 06.07.2026: Updated to true instantaneous power (dot product of force and velocity vectors)
        # to introduce direction, replacing the magnitude-product formulation. Intentional per update.
        vertical_power = (
                Fx_t_est * COM_f_speed_x
                + Fy_t_est * COM_f_speed_y
                + Fz_t_est * COM_f_speed_z
        )

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
        hand_l_disp_x = hand_l_f[:, 0] - hand_l_f[p1, 0]
        hand_l_disp_y = hand_l_f[:, 1] - hand_l_f[p1, 1]
        hand_l_disp_z = hand_l_f[:, 2] - hand_l_f[p1, 2]

        # Left hand speed (m/s)
        hand_l_speed_x = np.gradient(hand_l_disp_x, dt)
        hand_l_speed_y = np.gradient(hand_l_disp_y, dt)
        hand_l_speed_z = np.gradient(hand_l_disp_z, dt)

        # Left hand acceleration (m/s^2)
        hand_COM_acc_x = np.gradient(hand_l_speed_x, dt)
        hand_COM_acc_y = np.gradient(hand_l_speed_y, dt)
        hand_COM_acc_z = np.gradient(hand_l_speed_z, dt)

        # Estimate the Left hand FORCE COMPONENTS
        # 07.11.23: Adding: W* --> to actually using the weight for the hand and not only the weight percent
        Fx_t_est_hand = W * hand_l_m * hand_COM_acc_x
        Fy_t_est_hand = W * hand_l_m * hand_COM_acc_y
        Fz_t_est_hand = W * hand_l_m * hand_COM_acc_z

        # 03.06.2024: Updated the combination formula for the combined hand force
        F_hand = []
        for i in range(0, len(Fx_t_est_hand)):
            F_hand.append(
                math.sqrt(
                    Fx_t_est_hand[i] ** 2 + Fy_t_est_hand[i] ** 2 + Fz_t_est_hand[i] ** 2
                )
            )
        F_hand = np.array(F_hand)

        # Three components of hand speed
        hand_speed_X = np.diff(hand_l[:, 0]) * frame_rate  # m/s
        hand_speed_X = np.insert(hand_speed_X, 0, 0)
        hand_speed_Y = np.diff(hand_l[:, 1]) * frame_rate
        hand_speed_Y = np.insert(hand_speed_Y, 0, 0)
        hand_speed_Z = np.diff(hand_l[:, 2]) * frame_rate
        hand_speed_Z = np.insert(hand_speed_Z, 0, 0)

        # Calculate hand power
        hand_power_x = Fx_t_est_hand * hand_speed_X
        hand_power_y = Fy_t_est_hand * hand_speed_Y
        hand_power_z = Fz_t_est_hand * hand_speed_Z  

        # 03.06.2024: Updated the combination formula for the combined hand speed
        hand_speed = []
        for i in range(0, len(hand_speed_X)):
            hand_speed.append(
                math.sqrt(
                    hand_speed_X[i] ** 2 + hand_speed_Y[i] ** 2 + hand_speed_Z[i] ** 2
                )
            )
        hand_speed = np.array(hand_speed)

        # 06.07.2026: Updated to true instantaneous power (dot product of force and velocity components),
        # to introduce direction. Intentional per update.
        hand_power = (
                hand_power_x
                + hand_power_y
                + hand_power_z
        )


        # ------------------------------------------------- club mechanics ------------------------------------------------------
        # Club coordinates are now supplied directly in meters (3D), so no PPMM pixel-to-mm conversion is needed.
        nDigits = club_head_2d.shape
        nDigits = nDigits[1]
        # ballSizePixel = []
        # if len(json3d['data'].keys()) == 3:
        #     ballData = json3d['data']['ball']
        #     for element in ballData:
        #         if ballSizePixel == []:
        #             ballSizePixel = (ballData[element][2] + ballData[element][3]) / 2  # pix
        #     ballSizeReal = 42.67  # mm
        #     PPMM = ballSizeReal / ballSizePixel
        # else:
        #     PPMM = 1
        # PPMM = 1
        # nDigits = 3
        club_head = np.zeros((len(hand_power_z), nDigits))
        club_mid = np.zeros((len(hand_power_z), nDigits))
        club_main = np.zeros((len(hand_power_z), nDigits))

        # Loading club marker data
        other_index = 0
        for element in C:  # i in range(0, len(hand_power_z)):
            if other_index < len(hand_power_z):
                club_main[other_index] = C[element]["start"]  # * PPMM / 1000
                club_head[other_index] = C[element]["end"]  # * PPMM / 1000
                club_mid[other_index] = C[element]["mid"]  # * PPMM / 1000

            other_index += 1

        # Club coordinates already in meters - no pixel scaling needed
        club_head = np.array(club_head)  # * PPMM / 1080
        club_main = np.array(club_main)  # * PPMM / 1080
        club_mid = np.array(club_mid)  # * PPMM / 1080
        club_main_2d = np.array(club_main)  # * json3d["configuration"]["swing"]["PPMM"]

        # Three different methods to determine the hand club speed
        club_disp_X = club_head_2d[1:, 0] - club_head_2d[:-1, 0]
        club_disp_Y = club_head_2d[1:, 1] - club_head_2d[:-1, 1]
        club_disp_Z = club_head_2d[1:, 2] - club_head_2d[:-1, 2]  # 06.07.2026: 3D club input now available
        club_disp_X = np.insert(club_disp_X, 0, 0)
        club_disp_Y = np.insert(club_disp_Y, 0, 0)
        club_disp_Z = np.insert(club_disp_Z, 0, 0)  # 06.07.2026: 3D club input now available

        club_speed_X = np.gradient(club_head[:, 0], dt)
        club_speed_Y = np.gradient(club_head[:, 1], dt)
        club_speed_Z = np.gradient(club_head[:, 2], dt)
        club_speed = np.sqrt(club_speed_X**2 + club_speed_Y**2 + club_speed_Z**2)  # m/s

        if json3d["data_3d_bio"]["configuration"]["swing"]["BatType"] == "driver":
            fc = 20
        else:
            fc = 5

        # Defining shaft and club head masses, based on values from the real world
        if json3d["data_3d_bio"]["configuration"]["swing"]["BatType"] == "driver":  # == 1:
            shaft_mass = 0.059  # in kg
            club_head_mass = 0.192  # in kg
        else:
            shaft_mass = 0.120  # in kg
            club_head_mass = 0.256  # in kg

        club_mass = shaft_mass + club_head_mass  # in kg
        shaft_COM = club_mid

        # Compute club COM
        club_COM = (shaft_COM * shaft_mass + club_head * club_head_mass) / club_mass

        # Compute club COM speed
        club_COM_speed_X = np.gradient(club_COM[:, 0], dt)
        club_COM_speed_Y = np.gradient(club_COM[:, 1], dt)
        club_COM_speed_Z = np.gradient(club_COM[:, 2], dt)
        club_COM_speed = np.sqrt(club_COM_speed_X**2 + club_COM_speed_Y**2 + club_COM_speed_Z**2)  # m/s

        if json3d["data_3d_bio"]["configuration"]["swing"]["swingType"] == "pitching":
            index_min_club_COM_speed = np.argmin(club_COM_speed[p3:p5]) + p3
        else:
            index_min_club_COM_speed = np.argmin(club_COM_speed[p1 + 10 : p7]) + p1 + 10
        # Invert values up to index_min_club_COM_speed
        club_COM_speed[:index_min_club_COM_speed] = -club_COM_speed[
            :index_min_club_COM_speed
        ]

        # Apply filtering to club_COM_speed
        fc = 5  # Hz
        N = 4  # Order of butterworth filter
        c, d = signal.butter(N, fc / (frame_rate / 2))
        club_COM_speed = signal.filtfilt(c, d, club_COM_speed)

        club_COM_acc = np.gradient(club_COM_speed, dt)

        club_speed = np.array(club_speed)/ 2.23693629
        club_speed = signal.filtfilt(c, d, club_speed)
        club_head_acc = np.gradient(club_speed, dt)

        # Calculating the club head force
        F_club_head = club_head_mass * club_head_acc  # N

        # Calculating the club head power
        club_head_power = F_club_head * club_speed

        peak_club_power = max(club_head_power[p4 : p7 + 1])
        mean_club_power = np.mean(club_head_power[p4 : p7 + 1])
        std_club_power = np.std(club_head_power[p4 : p7 + 1])

        # third method to calculate the power index cv speed
        if json3d["data_3d_bio"]["configuration"]["swing"]["BatType"] == "driver":
            mean_club_power1 = 4457
            std_club_power1 = 497
        else:
            mean_club_power1 = 7342
            std_club_power1 = 2863

        # 29.01.2024: this is the new calculations for the power index
        power_index2 = peak_club_power / (mean_club_power1 + std_club_power1 * 4) * 100

        # Defining over all club COM
        club_COM = (shaft_COM * shaft_mass + club_head * club_head_mass) / club_mass

        # Calculating Club COM velocity
        club_COM_speed_X = np.gradient(club_COM[:, 0], dt)
        club_COM_speed_Y = np.gradient(club_COM[:, 1], dt)
        club_COM_speed_Z = np.gradient(club_COM[:, 2], dt)

        fc = 4  # Hz
        N = 4  # Order of butterworth filter
        c, d = signal.butter(N, fc / (frame_rate / 2))

        # Filtering Club COM velocity
        club_COM_speed_X_filt = signal.filtfilt(c, d, club_COM_speed_X)
        club_COM_speed_Y_filt = signal.filtfilt(c, d, club_COM_speed_Y)
        club_COM_speed_Z_filt = signal.filtfilt(c, d, club_COM_speed_Z)
        club_COM_speed1 = np.sqrt(club_COM_speed_X_filt**2 + club_COM_speed_Y_filt**2 + club_COM_speed_Z_filt**2)

        club_COM_acc_X = np.gradient(club_COM_speed_X_filt, dt)
        club_COM_acc_Y = np.gradient(club_COM_speed_Y_filt, dt)
        club_COM_acc_Z = np.gradient(club_COM_speed_Z_filt, dt)

        # Calculating the Forces at the shaft in 3D
        F_shaft_X = club_mass * club_COM_acc_X
        F_shaft_Y = club_mass * club_COM_acc_Y
        F_shaft_Z = club_mass * club_COM_acc_Z
        F_shaft1 = np.sqrt(F_shaft_X**2 + F_shaft_Y**2 + F_shaft_Z**2)

        # 06.07.2026: Updated Shaft power to true instantaneous power (dot product), to introduce direction.
        # Intentional per update.
        P_shaft = (
                F_shaft_X * club_COM_speed_X_filt
                + F_shaft_Y * club_COM_speed_Y_filt
                + F_shaft_Z * club_COM_speed_Z_filt
        )

        # Distance between lead hand and club COM
        club_COM_hand_dist = []
        for i in range(0, len(club_main)):
            club_COM_hand_dist.append(
                np.sqrt(
                    (club_main[i, 0] - club_COM[i, 0]) ** 2 +
                    (club_main[i, 1] - club_COM[i, 1]) ** 2 +
                    (club_main[i, 2] - club_COM[i, 2]) ** 2
                )
            )
        club_COM_hand_dist = np.array(club_COM_hand_dist)

        ## ====================================================
        ## KWON HAND-CLUB INVERSE DYNAMICS
        ## Y.-H. Kwon, "Hand-Club Interaction: 1. Inverse Dynamics" (Dec 2017 / rev. Feb 2018)
        ## Point of interaction (MH) = actual mid-hand landmark (hand_l_3d/hand_r_3d)
        ## Swing plane = continuously moving frame, re-fit every frame from the MH path (Frenet-Serret
        ## tangential/normal/binormal) - matches Kwon's Figs. 3-6 "moving MH frame".
        ## ============================================================================================

        DEBUG = False

        def _debug_print(label, arr):
            if not DEBUG:
                return
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0:
                print(f"[DEBUG] {label:40s} EMPTY ARRAY")
                return
            nan_count = int(np.isnan(arr).sum())
            print(
                f"[DEBUG] {label:40s} shape={str(arr.shape):14s} "
                f"min={np.nanmin(arr):10.4g} max={np.nanmax(arr):10.4g} "
                f"mean={np.nanmean(arr):10.4g} nan_count={nan_count}"
            )

        MH_point = (hand_l_3d + hand_r_3d) / 2  # Kwon's mid-hand (MH) point - actual hand landmarks
        _debug_print("MH_point", MH_point)

        def _butter_filt_cols(x_2d, fc_hz):
            Nb = 4
            cb, db = signal.butter(Nb, fc_hz / (frame_rate / 2))
            out = np.zeros_like(x_2d)
            for col in range(x_2d.shape[1]):
                out[:, col] = signal.filtfilt(cb, db, x_2d[:, col])
            return out

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

        # --- Net MH force (Eq. 12): F_MH = m_C * a_C - W_C (full 3D vector, gravity handled properly) ---
        club_COM_acc_vec = np.column_stack([club_COM_acc_X, club_COM_acc_Y, club_COM_acc_Z])
        W_C_vec = np.tile(np.array([0.0, 0.0, -9.81 * club_mass]), (len(club_COM), 1))
        F_MH_vec = club_mass * club_COM_acc_vec - W_C_vec
        _debug_print("F_MH_vec", F_MH_vec)

        # --- Club angular momentum about its own COM, via shaft-direction vector ---
        # Slender axisymmetric-rod assumption (same assumption already implicit in the legacy I=mL^2/12
        # formula): transverse MOI is identical about any axis perpendicular to the shaft, so the shaft's
        # own roll/twist about its long axis - unmeasurable from a 3-collinear-marker club track - is not
        # required to compute h_C this way.
        shaft_vec = club_head - club_main
        shaft_len = np.linalg.norm(shaft_vec, axis=1)
        s_hat = shaft_vec / shaft_len[:, np.newaxis]
        s_hat_filt = _butter_filt_cols(s_hat, fc_hz=10)
        s_hat_filt = s_hat_filt / np.linalg.norm(s_hat_filt, axis=1)[:, np.newaxis]
        ds_dt = np.gradient(s_hat_filt, dt, axis=0)
        omega_perp = np.cross(s_hat_filt, ds_dt)  # club angular velocity, perpendicular to the shaft

        I_trans = club_mass * shaft_len**2 / 12  # transverse MOI (slender rod about its own COM)
        h_C_vec = I_trans[:, np.newaxis] * omega_perp
        h_C_vec_filt = _butter_filt_cols(h_C_vec, fc_hz=8)
        h_dot_C_vec = np.gradient(h_C_vec_filt, dt, axis=0)
        _debug_print("shaft_len", shaft_len)
        _debug_print("omega_perp", omega_perp)
        _debug_print("h_C_vec", h_C_vec)
        _debug_print("h_dot_C_vec", h_dot_C_vec)

        # --- Point of interaction: r_CH = MH position relative to club COM ---
        r_CH_vec = MH_point - club_COM
        _debug_print("r_CH_vec", r_CH_vec)

        # --- Moment produced by the net MH force about club's COM (Kwon Fig. 5 quantity) ---
        MH_force_moment_vec = np.cross(r_CH_vec, F_MH_vec)
        _debug_print("MH_force_moment_vec", MH_force_moment_vec)

        # --- Net MH moment (Eq. 13) - couple action of the hands (Kwon Fig. 4 quantity) ---
        M_MH_vec = h_dot_C_vec - MH_force_moment_vec
        _debug_print("M_MH_vec", M_MH_vec)

        # --- Net club moment (Kwon Fig. 6 quantity) - equals h_dot_C_vec by construction ---
        T_net_vec = M_MH_vec + MH_force_moment_vec
        _debug_print("T_net_vec", T_net_vec)

        # --- Moving MH reference frame - windowed local plane fit ---
        # The swing axis (B_hat) is now derived from a small spatial window of nearby MH positions via
        # fit_plane() (same SVD approach already used for BS_plane/DS_plane), rather than from instantaneous
        # acceleration (2nd derivative of noisy marker position). This is far more stable exactly where hand
        # speed passes through ~zero (e.g. top of backswing), which is where the previous acceleration-based
        # curvature estimate produced spurious high-frequency spikes once tested against real data.
        frame_window_ms = 100.0  # +/- half-window in ms - tunable
        half_win = max(3, int(round(frame_window_ms / 1000.0 * frame_rate)))

        B_hat = np.zeros((len(MH_point), 3))
        B_center = np.zeros((len(MH_point), 3))  # local windowed centroid, reused below for N_hat's sign
        for _i in range(len(MH_point)):
            _lo = max(0, _i - half_win)
            _hi = min(len(MH_point), _i + half_win + 1)
            if _hi - _lo < 3:
                B_hat[_i] = np.array([0.0, 0.0, 1.0])
                B_center[_i] = MH_point[_i]
                continue
            _plane_v, _plane_com = fit_plane(MH_point[_lo:_hi, :])
            B_hat[_i] = _plane_v
            B_center[_i] = _plane_com

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
        _pts = MH_point[_p4c:_p7c + 1]
        _local_com = np.mean(_pts, axis=0)
        _rel = _pts - _local_com
        _cross_terms = np.cross(_rel[:-1], _rel[1:])
        _signed_area_vec = np.sum(_cross_terms, axis=0)
        _mean_B_over_window = np.mean(B_hat[_p4c:_p7c + 1], axis=0)
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
        if DEBUG:
            print(
                f"[DEBUG] B_hat swing-direction sign correction applied: {B_hat_flipped} "
                f"(anchored to P4:P7 MH-path signed-area sweep, position-only, no derivatives) | "
                f"lefty={lefty} (additional handedness correction {'APPLIED' if lefty == 1 else 'not needed'})"
            )

        v_MH = _butter_filt_cols(np.gradient(MH_point, dt, axis=0), fc_hz=6)
        v_proj = v_MH - np.sum(v_MH * B_hat, axis=1)[:, np.newaxis] * B_hat  # velocity projected into the plane
        v_proj_norm = np.linalg.norm(v_proj, axis=1)
        T_hat_valid = v_proj_norm > 1e-4
        T_hat = np.zeros_like(v_MH)
        T_hat[T_hat_valid] = v_proj[T_hat_valid] / v_proj_norm[T_hat_valid, np.newaxis]
        T_hat = _fill_degenerate(T_hat, T_hat_valid)

        N_hat = np.cross(B_hat, T_hat)  # completes the right-handed frame (T x N = B)
        N_norm = np.linalg.norm(N_hat, axis=1)
        N_valid = N_norm > 1e-6
        N_hat_safe = np.zeros_like(N_hat)
        N_hat_safe[N_valid] = N_hat[N_valid] / N_norm[N_valid, np.newaxis]
        N_hat = _fill_degenerate(N_hat_safe, N_valid)

        # 15.07.2026: The per-frame "outward-positive" N_hat correction that was here has been REMOVED.
        # It assumed Kwon's normal axis was dynamically re-oriented to always mean "outward," but empirical
        # testing against real swings (both players) showed the NATURAL N_hat = B_hat x T_hat triad (no
        # extra correction) reproduces Kwon's Fig. 3 pattern correctly on its own. N_hat is left as the
        # plain cross product below.

        if DEBUG:
            n_frames_total = len(T_hat_valid)
            print(
                f"[DEBUG] windowed plane fit: half_win={half_win} frames (~{frame_window_ms:.0f}ms), "
                f"T_hat valid {T_hat_valid.sum()}/{n_frames_total} (degenerate frames forward/back-filled)"
            )
            dotTN = np.abs(np.sum(T_hat * N_hat, axis=1))
            dotTB = np.abs(np.sum(T_hat * B_hat, axis=1))
            dotNB = np.abs(np.sum(N_hat * B_hat, axis=1))
            print(
                f"[DEBUG] frame orthonormality check (should be ~0): "
                f"max|T.N|={np.nanmax(dotTN):.4g} max|T.B|={np.nanmax(dotTB):.4g} max|N.B|={np.nanmax(dotNB):.4g}"
            )

        # NOTE (15.07.2026): validated against two real (righthanded) swings - Fig.3-6 now match Kwon's
        # qualitative pattern and are cross-swing consistent. Not yet validated for lefty or down-the-line
        # swings, since hand_l_3d/hand_r_3d/MH_point come from the RAW, untransformed pose data.

        # --- Decompose net MH force: in-plane (tangential/normal) + out-of-plane (swing-axis) ---
        F_MH_tangential = np.sum(F_MH_vec * T_hat, axis=1)
        F_MH_normal = np.sum(F_MH_vec * N_hat, axis=1)
        F_MH_swing_axis = np.sum(F_MH_vec * B_hat, axis=1)  # out-of-plane force component (new)

        # Moments are pseudovectors (defined via a cross product / right-hand rule), so their positive-
        # rotation-direction convention is inherently tied to "which side you view the rotation from" - a
        # convention the force-decomposition frame (T_hat, N_hat, B_hat), built purely from position
        # geometry, has no way to know on its own. Empirically, Kwon's positive in-plane (swing-axis) torque
        # convention comes out opposite to B_hat's geometric orientation. Since this is a property of moments
        # in general (not specific to the swing axis), the SAME negation is applied to all three torque axes
        # uniformly - one consistent rule for every moment quantity (in-plane AND out-of-plane), rather than
        # a special case for B_hat alone. This is what keeps Fig.3 (force) and Figs.4-7 (torque) internally
        # consistent with each other.
        torque_T = -T_hat
        torque_N = -N_hat
        torque_axis = -B_hat

        # --- Decompose net MH moment (the "couple") ---
        M_MH_swing_axis = np.sum(M_MH_vec * torque_axis, axis=1)  # in-plane driving torque (Kwon Fig. 4)
        M_MH_tangential = np.sum(M_MH_vec * torque_T, axis=1)  # out-of-plane (new)
        M_MH_normal = np.sum(M_MH_vec * torque_N, axis=1)      # out-of-plane (new)

        # --- Decompose MH force moment (r_CH x F_MH) ---
        MHforce_moment_swing_axis = np.sum(MH_force_moment_vec * torque_axis, axis=1)  # in-plane (Kwon Fig. 5)
        MHforce_moment_tangential = np.sum(MH_force_moment_vec * torque_T, axis=1)  # out-of-plane (new)
        MHforce_moment_normal = np.sum(MH_force_moment_vec * torque_N, axis=1)      # out-of-plane (new)

        # --- Decompose net club moment (total) ---
        # T_net = h_dot_C_vec is the club's own kinematic response (not an externally-applied moment from
        # the hands like M_MH/MH_force_moment), so - per empirical validation against real swing data -
        # it stays on the plain geometric axes (B_hat/T_hat/N_hat), NOT the flipped torque_axis/torque_T/
        # torque_N used for the two hand-applied moment components above.
        T_net_swing_axis = np.sum(T_net_vec * B_hat, axis=1)  # in-plane (Kwon Fig. 6)
        T_net_tangential = np.sum(T_net_vec * T_hat, axis=1)  # out-of-plane (new)
        T_net_normal = np.sum(T_net_vec * N_hat, axis=1)      # out-of-plane (new)


        _debug_print("F_MH_tangential (Fig.3 green)", F_MH_tangential)
        _debug_print("F_MH_normal (Fig.3 blue, signed)", F_MH_normal)
        _debug_print("F_MH_swing_axis (out-of-plane force)", F_MH_swing_axis)
        _debug_print("M_MH_swing_axis (Fig.4 red)", M_MH_swing_axis)
        _debug_print("MHforce_moment_swing_axis (Fig.5 green)", MHforce_moment_swing_axis)
        _debug_print("T_net_swing_axis (Fig.6 blue)", T_net_swing_axis)
        _debug_print("M_MH_tangential/normal (out-of-plane)", np.column_stack([M_MH_tangential, M_MH_normal]))
        _debug_print("MHforce_moment_tan/normal (out-of-plane)", np.column_stack([MHforce_moment_tangential, MHforce_moment_normal]))
        ## ============================================================================================

        # 06.07.2026: Tangential_Force / Centripetal_Force now sourced from the Kwon net MH force
        # decomposition (F_MH = m*a_C - W_C, projected onto the moving swing-plane frame), replacing the
        # previous scalar-speed-based approximation. Gravity is now embedded properly via W_C rather than
        # added ad hoc to the radial term.
        F_tan_shaft = F_MH_tangential
        F_rad_shaft = np.abs(F_MH_normal)  # magnitude - grip must resist the radial component regardless of sign

        # Grip force
        taper_deg = 4
        # degrees
        taper_rad = taper_deg * 2 * np.pi / 360
        # rad
        friction = 0.7
        # coefficient
        Grip = F_rad_shaft / friction * (1 - np.sin(taper_rad))
        _debug_print("F_tan_shaft (reassigned)", F_tan_shaft)
        _debug_print("F_rad_shaft (reassigned, abs)", F_rad_shaft)
        _debug_print("Grip", Grip)

        # Build shaft force vectors normalized (direction) - now full 3D
        F_tan_vec_norm = club_head[1:] - club_head[:-1]

        # Add a check for zero norm to avoid division by zero
        norms = np.linalg.norm(F_tan_vec_norm, axis=1)
        nonzero_norms = norms != 0  # Find elements with non-zero norms
        F_tan_vec_norm[nonzero_norms] /= norms[nonzero_norms, np.newaxis]

        F_tan_vec_norm = np.vstack([[0, 0, 0], F_tan_vec_norm])

        shaft_COM = (club_head + club_main) / 2

        F_rad_vec_norm = club_main - club_COM
        F_rad_vec_norm = (
            F_rad_vec_norm / np.linalg.norm(F_rad_vec_norm, axis=1)[:, np.newaxis]
        )

        # Shaft force vectors (3D)
        # Shaft force vectors (3D)
        F_tan_vec = F_tan_vec_norm * F_tan_shaft[:, np.newaxis]
        F_rad_vec = F_rad_vec_norm * F_rad_shaft[:, np.newaxis]

        # ---------------------------------------------------------
        # Scale vectors for frontend visualization only
        # (does NOT affect biomechanics calculations)
        # ---------------------------------------------------------
        shaft_force_arrow_scale = 0.01      # Adjust as needed

        F_tan_vec_vis = F_tan_vec * shaft_force_arrow_scale
        F_rad_vec_vis = F_rad_vec * shaft_force_arrow_scale

        # 20.02.25 Resultant force from Radial and Tangential forces
        F_res_vec = (
            F_rad_vec + F_tan_vec
        )  # Adding F_rad_vec and F_tan_vec to get the resultant force vector

        ## --- PIXEL-SPACE shaft force vectors (unchanged from Script A) ---
        ## Uses the restored pixel-space hand_l_2d/hand_r_2d. Only the X/Y components of the (now 3D)
        ## force vectors are used, matching the original 2D visualization logic exactly.
        # midhand_2d = (np.array(hand_l_2d) + np.array(hand_r_2d)) / 2
        # start_x = midhand_2d[:, 0]
        # start_y = midhand_2d[:, 1]
        # end_x_rad_force = midhand_2d[:, 0] + F_rad_vec[:, 0]
        # end_y_rad_force = midhand_2d[:, 1] + F_rad_vec[:, 1]
        # end_x_tan_force = []
        # end_y_tan_force = []
        # for i in range(0, len(midhand_2d)):
        #     if i < index_min_club_COM_speed:
        #         end_x_tan_force = midhand_2d[:, 0] - F_tan_vec[:, 0]
        #         end_y_tan_force = midhand_2d[:, 1] - F_tan_vec[:, 1]
        #     else:
        #         end_x_tan_force = midhand_2d[:, 0] + F_tan_vec[:, 0]
        #         end_y_tan_force = midhand_2d[:, 1] + F_tan_vec[:, 1]

        ## --- NEW: 3D shaft force vectors (for future stick-figure/avatar integration) ---
        midhand_3d = (hand_l_3d + hand_r_3d) / 2
        start_x_3d = hand_r_3d[:, 0]
        start_y_3d = hand_r_3d[:, 1]
        start_z_3d = hand_r_3d[:, 2]
        end_x_rad_force_3d = hand_r_3d[:, 0] + F_rad_vec_vis[:, 0]
        end_y_rad_force_3d = hand_r_3d[:, 1] + F_rad_vec_vis[:, 1]
        end_z_rad_force_3d = hand_r_3d[:, 2] + F_rad_vec_vis[:, 2]
        end_x_tan_force_3d = []
        end_y_tan_force_3d = []
        end_z_tan_force_3d = []
        for i in range(len(midhand_3d)):
            if i < index_min_club_COM_speed:
                end_x_tan_force_3d = hand_r_3d[:, 0] - F_tan_vec_vis[:, 0]
                end_y_tan_force_3d = hand_r_3d[:, 1] - F_tan_vec_vis[:, 1]
                end_z_tan_force_3d = hand_r_3d[:, 2] - F_tan_vec_vis[:, 2]
            else:
                end_x_tan_force_3d = hand_r_3d[:, 0] + F_tan_vec_vis[:, 0]
                end_y_tan_force_3d = hand_r_3d[:, 1] + F_tan_vec_vis[:, 1]
                end_z_tan_force_3d = hand_r_3d[:, 2] + F_tan_vec_vis[:, 2]

        # Compute club COM (pixel-labeled dict, computed from the 3D-meters club data - see note below)
        club_COM_2d = (shaft_COM * shaft_mass + club_head_2d * club_head_mass) / club_mass
        club_COM_2d = club_COM_2d

        def point_to_line_projection(x, u, v):
            x = np.asarray(x, dtype=float)
            u = np.asarray(u, dtype=float)
            v = np.asarray(v, dtype=float)

            th = np.dot(x - v, u) / np.dot(u, u)

            h = v + th * u

            return h

        def point_to_line_distance(p, a, b):

            p = np.asarray(p, dtype=float).reshape(-1)
            a = np.asarray(a, dtype=float).reshape(-1)
            b = np.asarray(b, dtype=float).reshape(-1)

            if p.size == 2:
                p = np.append(p, 0.0)

            if a.size == 2:
                a = np.append(a, 0.0)

            if b.size == 2:
                b = np.append(b, 0.0)

            denom = np.dot(a, a)
            if denom < 1e-12:
                return np.linalg.norm(p - b)

            th = np.dot(p - b, a) / denom
            h_test = b + th * a

            return np.linalg.norm(p - h_test)

        club_main_end_x = []
        club_main_end_y = []

        proj_club_COM = []
        for i in range(0, len(club_COM_2d)):
            if i < index_min_club_COM_speed:
                proj_club_COM.append(
                    point_to_line_projection(
                        club_COM_2d[i, :],
                        F_rad_vec[i] - F_tan_vec[i],
                        club_main_2d[i, :],
                    )
                )
                club_main_end_x.append(
                    club_main_2d[i, 0] + (F_rad_vec[i, 0] - F_tan_vec[i, 0])
                )
                club_main_end_y.append(
                    club_main_2d[i, 1] + (F_rad_vec[i, 1] - F_tan_vec[i, 1])
                )
            else:
                proj_club_COM.append(
                    point_to_line_projection(
                        club_COM_2d[i, :],
                        F_rad_vec[i] + F_tan_vec[i],
                        club_main_2d[i, :],
                    )
                )
                club_main_end_x.append(
                    club_main_2d[i, 0] + (F_rad_vec[i, 0] + F_tan_vec[i, 0])
                )
                club_main_end_y.append(
                    club_main_2d[i, 1] + (F_rad_vec[i, 1] + F_tan_vec[i, 1])
                )

        lever_arm_res = []
        for i in range(0, len(club_COM_2d)):
            if i < index_min_club_COM_speed:
                # 06.07.2026: fixed to pass the full 3D force-direction vector (was previously only
                # X/Y, inconsistent with proj_club_COM directly above which already uses the full vector).
                lever_arm_res.append(
                    point_to_line_distance(
                        club_COM_2d[i, :],
                        F_rad_vec[i] - F_tan_vec[i],
                        club_main_2d[i, :],
                    )
                )
                club_main_end_x.append(
                    club_main_2d[i, 0] + (F_rad_vec[i, 0] - F_tan_vec[i, 0])
                )
                club_main_end_y.append(
                    club_main_2d[i, 1] - F_tan_vec[i, 1] + F_rad_vec[i, 1]
                )
            else:
                lever_arm_res.append(
                    point_to_line_distance(
                        club_COM_2d[i, :],
                        F_rad_vec[i] + F_tan_vec[i],
                        club_main_2d[i, :],
                    )
                )
                club_main_end_x.append(
                    club_main_2d[i, 0] + (F_rad_vec[i, 0] + F_tan_vec[i, 0])
                )
                club_main_end_y.append(
                    club_main_2d[i, 1] + (F_rad_vec[i, 1] + F_tan_vec[i, 1])
                )
        # Lever arms from point_to_line_project and point_to_line_distance
        proj_club_COM = np.array(proj_club_COM)  # point_to_line_project
        lever_arm_res = np.array(lever_arm_res)  # point_to_line_distance

        # 06.07.2026: M_in_plane now sourced from the Kwon "MH force moment" (r_CH x F_MH), swing-axis
        # component - replacing the previous world-frame-X-component * 3D-lever-arm approximation.
        M_in_plane = MHforce_moment_swing_axis

        # 06.07.2026: Wrist_torque (the couple) now sourced from the Kwon net MH moment (Eq. 13),
        # swing-axis component - replacing the previous planar-circular-motion approximation
        # (I=mL^2/12 * a/r), which assumed a fixed pivot and discarded genuine 3D angular momentum.
        Wrist_torque = M_MH_swing_axis
        _debug_print("M_in_plane (reassigned)", M_in_plane)
        _debug_print("Wrist_torque / C_in_plane (reassigned)", Wrist_torque)

        # Compute MAX values and time to impact
        max_F_tan_shaft, index_max_F_tan_shaft = np.max(
            F_tan_shaft[p1 : p8 + 1]
        ), np.argmax(F_tan_shaft[p1 : p8 + 1])
        time_to_impact_F_tan_shaft = 1000 * (index_max_F_tan_shaft - p7 + p1) / frame_rate

        max_F_rad_shaft, index_max_F_rad_shaft = np.max(
            F_rad_shaft[p1 : p8 + 1]
        ), np.argmax(F_rad_shaft[p1 : p8 + 1])
        time_to_impact_F_rad_shaft = 1000 * (index_max_F_rad_shaft - p7 + p1) / frame_rate

        max_Wrist_torque, index_max_Wrist_torque = np.max(
            Wrist_torque[p1 : p8 + 1]
        ), np.argmax(Wrist_torque[p1 : p8 + 1])
        time_to_impact_Wrist_torque = 1000 * (index_max_Wrist_torque - p7 + p1) / frame_rate

        max_F_club_head, index_max_F_club_head = np.max(
            F_club_head[p1 : p8 + 1]
        ), np.argmax(F_club_head[p1 : p8 + 1])
        time_to_impact_F_club_head = 1000 * (index_max_F_club_head - p7 + p1) / frame_rate

        max_Grip, index_max_Grip = np.max(Grip[p1 : p8 + 1]), np.argmax(Grip[p1 : p8 + 1])
        time_to_impact_Grip = 1000 * (index_max_Grip - p7 + p1) / frame_rate

        # Create a list of maximum values in the desired order
        peak_force_order = [
            index_max_F_tan_shaft,
            index_max_F_rad_shaft,
            index_max_Wrist_torque,
        ]

        # Sort the list and get the indices that represent the original order
        index_sorted = np.argsort(peak_force_order)

        # Find the positions of each measurement in the sorted order
        peak_force_order_F_tan_shaft = np.where(index_sorted == 0)[0]
        peak_force_order_F_rad_shaft = np.where(index_sorted == 1)[0]
        peak_force_order_Wrist_torque = np.where(index_sorted == 2)[0]

        ##################################### In plane metrics ######################################################
        # In plane couple
        C_in_plane = Wrist_torque

        # In plane torque (total torque)
        # T_in_plane now uses T_net_swing_axis directly (T_net_vec projected onto the plain B_hat, not
        # torque_axis) - NOT C_in_plane + M_in_plane. That old sum was a leftover from the original script
        # and never actually reflected the T_net/B_hat sign-convention fix, even after T_net_swing_axis
        # itself was corrected - this was the real, final bug in the chain.
        T_in_plane = T_net_swing_axis

        inPlaneMetrics = {
            "M_in_plane": M_in_plane.tolist(),
            "C_in_plane": C_in_plane.tolist(),
            "T_in_plane": T_in_plane.tolist(),
        }

        # NEW: out-of-plane counterparts, from the same Kwon vector decomposition above.
        # Each in-plane moment (swing-axis component) has two out-of-plane siblings (tangential-axis and
        # normal-axis components, i.e. the two axes that lie IN the swing plane) representing torque that
        # would push the club above/below the swing plane rather than driving its rotation within it.
        outOfPlaneMetrics = {
            "F_out_of_plane": F_MH_swing_axis.tolist(),                     # net MH force, out-of-plane component
            "M_out_of_plane_tangential": MHforce_moment_tangential.tolist(),
            "M_out_of_plane_normal": MHforce_moment_normal.tolist(),
            "C_out_of_plane_tangential": M_MH_tangential.tolist(),
            "C_out_of_plane_normal": M_MH_normal.tolist(),
            "T_out_of_plane_tangential": T_net_tangential.tolist(),
            "T_out_of_plane_normal": T_net_normal.tolist(),
        }

        # NEW: supplementary raw 3D kinetics vectors + moving swing-plane frame axes, for future
        # stick-figure/avatar visualization and debugging. Not required for the scalar metrics above.
        M2C_kinetics_3d = {
            "MH_point_3d": MH_point.tolist(),
            "F_MH_vec_3d": F_MH_vec.tolist(),
            "M_MH_vec_3d": M_MH_vec.tolist(),
            "MH_force_moment_vec_3d": MH_force_moment_vec.tolist(),
            "T_net_vec_3d": T_net_vec.tolist(),
            "swing_frame_tangential_3d": T_hat.tolist(),
            "swing_frame_normal_3d": N_hat.tolist(),
            "swing_frame_axis_3d": B_hat.tolist(),
            "F_MH_tangential_signed": F_MH_tangential.tolist(),  # same as Tangential_Force, kept here for plotting
            "F_MH_normal_signed": F_MH_normal.tolist(),          # signed version (Centripetal_Force is abs())
        }

        if DEBUG:
            print("[DEBUG] ---- Final in-plane / out-of-plane metric summary ----")
            _debug_print("inPlaneMetrics.C_in_plane", inPlaneMetrics["C_in_plane"])
            _debug_print("inPlaneMetrics.M_in_plane", inPlaneMetrics["M_in_plane"])
            _debug_print("inPlaneMetrics.T_in_plane", inPlaneMetrics["T_in_plane"])
            _debug_print("outOfPlaneMetrics.C_out_of_plane_tangential", outOfPlaneMetrics["C_out_of_plane_tangential"])
            _debug_print("outOfPlaneMetrics.C_out_of_plane_normal", outOfPlaneMetrics["C_out_of_plane_normal"])
            _debug_print("outOfPlaneMetrics.M_out_of_plane_tangential", outOfPlaneMetrics["M_out_of_plane_tangential"])
            _debug_print("outOfPlaneMetrics.M_out_of_plane_normal", outOfPlaneMetrics["M_out_of_plane_normal"])
            _debug_print("outOfPlaneMetrics.T_out_of_plane_tangential", outOfPlaneMetrics["T_out_of_plane_tangential"])
            _debug_print("outOfPlaneMetrics.T_out_of_plane_normal", outOfPlaneMetrics["T_out_of_plane_normal"])
            # sanity check: out-of-plane should generally be small relative to in-plane for a well-planed swing
            in_plane_rms = np.sqrt(np.mean(np.array(inPlaneMetrics["T_in_plane"])**2))
            out_tan_rms = np.sqrt(np.mean(np.array(outOfPlaneMetrics["T_out_of_plane_tangential"])**2))
            out_norm_rms = np.sqrt(np.mean(np.array(outOfPlaneMetrics["T_out_of_plane_normal"])**2))
            print(
                f"[DEBUG] RMS check - in-plane T: {in_plane_rms:.4g} Nm | "
                f"out-of-plane T tangential: {out_tan_rms:.4g} Nm ({100*out_tan_rms/max(in_plane_rms,1e-9):.1f}% of in-plane) | "
                f"out-of-plane T normal: {out_norm_rms:.4g} Nm ({100*out_norm_rms/max(in_plane_rms,1e-9):.1f}% of in-plane)"
            )
            # Kwon: T-Net "largely remains positive... but turns to negative right before the impact."
            # Find the positive->negative crossing nearest P7 to check this precisely, rather than by eye.
            T_arr = np.array(inPlaneMetrics["T_in_plane"])
            search_lo, search_hi = max(0, p4), min(len(T_arr), p8 + 1)
            crossings = [
                _i for _i in range(search_lo + 1, search_hi)
                if T_arr[_i - 1] >= 0 and T_arr[_i] < 0
            ]
            if crossings:
                _closest = min(crossings, key=lambda idx: abs(idx - p7))
                _t_cross_ms = (_closest - p7) / frame_rate * 1000.0
                print(
                    f"[DEBUG] T_in_plane +->- crossing nearest P7: frame {_closest}, {_t_cross_ms:+.1f} ms "
                    f"relative to P7 (negative ms = before impact, matches Kwon's 'turns negative right "
                    f"before impact' if small and negative)"
                )
            else:
                print("[DEBUG] T_in_plane: no positive->negative crossing found in the P4:P8 window")

        # In Plane Metrics "Pixel" Values for frontend - UNCHANGED FROM PRIOR CODE.
        # NOTE (06.07.2026): club_COM_2d / club_main_2d / proj_club_COM are now derived from the
        # 3D-meters club input (Script B no longer receives a 2D pixel club track), so the *1080
        # scaling below no longer converts to genuine pixel coordinates - it is left as-is pending
        # confirmation of whether a separate 2D club track is still available for this dict.
        proj_club_COM_pixel = (
            proj_club_COM.squeeze()[:, :2] * 1080
        )  # 2D Coordinates of projected COM
        club_COM_2d_pixel = club_COM_2d * 1080
        club_main_2d_pixel = club_main_2d * 1080

        M2C_inplane_vectors = {
            "proj_club_COM_pixel": proj_club_COM_pixel.tolist(),
            "club_COM_2d_pixel": club_COM_2d_pixel.tolist(),
            "club_main_2d_pixel": club_main_2d_pixel.tolist(),
        }

        # NEW: 3D (meters) in-plane vectors, unscaled, for future stick-figure/avatar integration.
        proj_club_COM_3d = proj_club_COM.squeeze()
        club_COM_3d = club_COM_2d
        club_main_3d = club_main_2d

        M2C_inplane_vectors_3d = {
            "proj_club_COM_3d": proj_club_COM_3d.tolist(),
            "club_COM_3d": club_COM_3d.tolist(),
            "club_main_3d": club_main_3d.tolist(),
        }
        ####################################### lead wrist metrics ####################################################################

        # Wrist displacements
        wrist_sway = 100 * (wrist_l[:, 1] - wrist_l[0, 1])
        wrist_thrust = 100 * (wrist_l[:, 0] - wrist_l[0, 0])
        wrist_lift = 100 * (wrist_l[:, 2] - wrist_l[0, 2])

        # Imperial


        # LEFT WRIST ANGLE
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

        # Wrist velocity in degrees per second
        wrist_vel_l_deg = np.gradient(wrist_angle_l_deg, dt)

        # Three components of hand speed
        wrist_speed_X = (wrist_l[1:, 0] - wrist_l[:-1, 0]) * frame_rate  # m/s
        wrist_speed_Y = (wrist_l[1:, 1] - wrist_l[:-1, 1]) * frame_rate
        wrist_speed_Z = (wrist_l[1:, 2] - wrist_l[:-1, 2]) * frame_rate

        # Resultant wrist speed
        wrist_speed = np.concatenate(
            [[0], np.sqrt(wrist_speed_X**2 + wrist_speed_Y**2 + wrist_speed_Z**2)]
        )  # m/s

        # Convert to MPH if requested
        factor = 2.23693629
        wrist_speed_mph = wrist_speed * factor


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

        M2C_insights = {
            "arc_width": arc_width.tolist(),
            "arc_width_inch": arc_width_inch.tolist(),
            "lead_forearm_shaft_angle": lead_forearm_shaft_angle_deg_2d.tolist(),
            "x_factor": x_factor.tolist(),
            "x_factor_stretch": x_factor_stretch,
        }

        

        M2C_power = {
            "vertical_power": vertical_power.tolist(),
            "hand_power": hand_power.tolist(),
            "Shaft_power": P_shaft.tolist(),
            "club_head_power": club_head_power.tolist(),
            "power_index": power_index2,
        }

        

        M2C_club_mechanics = {
            "Tangential_Force": F_tan_shaft.tolist(),
            "Centripetal_Force": F_rad_shaft.tolist(),
            "Grip_Force": Grip.tolist(),
            "F_club_head": F_club_head.tolist(),
            "Wrist_torque": Wrist_torque.tolist(),
        }

        
        
        M2C_club_mechanics_highlights = {
            "peak_club_mechanics": peak_club_mechanics,
            "time_to_impact": time_to_impact,
            "kinetic_order": kinetic_order_frontend,
        }

        # UNCHANGED from Script A - pixel-space, for video overlay
        # M2C_force_vectors_shaft = {
        #     "start_x": start_x.tolist(),
        #     "start_y": start_y.tolist(),
        #     "end_x_rad_force": end_x_rad_force.tolist(),
        #     "end_y_rad_force": end_y_rad_force.tolist(),
        #     "end_x_tan_force": end_x_tan_force.tolist(),
        #     "end_y_tan_force": end_y_tan_force.tolist(),
        # }

        # NEW: 3D (meters) shaft force vectors, for future stick-figure/avatar integration
        # M2C_force_vectors_shaft_3d = {
        #     "start_x_3d": start_x_3d.tolist(),
        #     "start_y_3d": start_y_3d.tolist(),
        #     "start_z_3d": start_z_3d.tolist(),
        #     "end_x_rad_force_3d": end_x_rad_force_3d.tolist(),
        #     "end_y_rad_force_3d": end_y_rad_force_3d.tolist(),
        #     "end_z_rad_force_3d": end_z_rad_force_3d.tolist(),
        #     "end_x_tan_force_3d": end_x_tan_force_3d.tolist(),
        #     "end_y_tan_force_3d": end_y_tan_force_3d.tolist(),
        #     "end_z_tan_force_3d": end_z_tan_force_3d.tolist(),
        # }

        M2C_force_vectors_shaft_3d = {
            "start_3d": [
                [x, y, z]
                for x, y, z in zip(
                    start_x_3d.tolist(),
                    start_y_3d.tolist(),
                    start_z_3d.tolist()
                )
            ],
            "end_rad_force_3d": [
                [x, y, z]
                for x, y, z in zip(
                    end_x_rad_force_3d.tolist(),
                    end_y_rad_force_3d.tolist(),
                    end_z_rad_force_3d.tolist()
                )
            ],
            "end_tan_force_3d": [
                [x, y, z]
                for x, y, z in zip(
                    end_x_tan_force_3d.tolist(),
                    end_y_tan_force_3d.tolist(),
                    end_z_tan_force_3d.tolist()
                )
            ],
        }

        

        M2C_wrist = {
            "lead_forearm_shaft_angle": lead_forearm_shaft_angle_deg_2d.tolist(),
            "lead_wrist_extension_flexion": wrist_angle_l_deg.tolist(),
            "lead_wrist_extension_flexion_vel": wrist_vel_l_deg.tolist(),
            "lead_wrist_speed": wrist_speed_mph.tolist(),
            "lead_wrist_thrust": wrist_thrust.tolist(),
            "lead_wrist_sway": wrist_sway.tolist(),
            "lead_wrist_lift": wrist_lift.tolist(),

        }


            ############################################################
        # PERFORMANCE METRICS
        ############################################################

        impact_frame = p7  

        # ----------------------------------------------------------
        # 1. Peak Tangential Force
        # ----------------------------------------------------------
        peak_tan_idx = np.argmax(np.abs(F_tan_shaft))
        peak_tan_force = float(F_tan_shaft[peak_tan_idx])

        peak_tan_time_ms = (
            (peak_tan_idx - impact_frame)
            / frame_rate
            * 1000
        )

        # ----------------------------------------------------------
        # 2. Peak Normal Force
        # ----------------------------------------------------------
        peak_norm_idx = np.argmax(np.abs(F_rad_shaft))
        peak_norm_force = float(F_rad_shaft[peak_norm_idx])

        peak_norm_time_ms = (
            (peak_norm_idx - impact_frame)
            / frame_rate
            * 1000
        )

        # ----------------------------------------------------------
        # 3. Peak Net Torque
        # ----------------------------------------------------------
        T_net = np.array(inPlaneMetrics["T_in_plane"])

        peak_torque_idx = np.argmax(np.abs(T_net))
        peak_torque = float(T_net[peak_torque_idx])

        peak_torque_time_ms = (
            (peak_torque_idx - impact_frame)
            / frame_rate
            * 1000
        )

        # ----------------------------------------------------------
        # 4. Torque Efficiency
        # ----------------------------------------------------------
        C_in = np.array(inPlaneMetrics["C_in_plane"])
        M_in = np.array(inPlaneMetrics["M_in_plane"])

        rms_net = rms(T_net)

        rms_sources = (
            rms(C_in) +
            rms(M_in)
        )

        torque_efficiency = (
            rms_net /
            rms_sources
            if rms_sources > 0 else 0
        )

        torque_cancellation = 1.0 - torque_efficiency

        # ----------------------------------------------------------
        # 5. Force Direction Ratio
        # ----------------------------------------------------------
        force_direction_ratio = (
            rms(F_tan_shaft) /
            rms(F_rad_shaft)
            if rms(F_rad_shaft) > 0 else 0
        )

        if force_direction_ratio < 0.6:
            force_direction_interpretation = "Steering dominant"

        elif force_direction_ratio < 1.0:
            force_direction_interpretation = "Balanced"

        else:
            force_direction_interpretation = "Acceleration dominant"

        # ----------------------------------------------------------
        # 6. Barrel Stability Index
        # ----------------------------------------------------------

        rms_oop = rms(
            np.concatenate([
                outOfPlaneMetrics["T_out_of_plane_tangential"],
                outOfPlaneMetrics["T_out_of_plane_normal"]
            ])
        )

        rms_inplane = rms(T_net)

        barrel_stability_index = (
            rms_oop /
            rms_inplane
            if rms_inplane > 0 else 0
        )

        # ----------------------------------------------------------
        # 7. Peak Out-of-Plane Torque
        # Late swing only
        # ----------------------------------------------------------

        t_ms = (
            (np.arange(len(T_net)) - impact_frame)
            / frame_rate
            * 1000
        )

        late_mask = (
            (t_ms >= -50) &
            (t_ms <= 20)
        )

        oop_total = np.sqrt(
            np.array(outOfPlaneMetrics["T_out_of_plane_tangential"])**2 +
            np.array(outOfPlaneMetrics["T_out_of_plane_normal"])**2
        )

        late_oop = oop_total[late_mask]

        peak_oop = np.max(late_oop)

        peak_oop_idx = np.where(
            oop_total == peak_oop
        )[0][0]

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
                "efficiency_percent":
                    100 * torque_efficiency,

                "cancellation_percent":
                    100 * torque_cancellation,
            },

            "force_direction_ratio": {
                "ratio":
                    force_direction_ratio,

                "interpretation":
                    force_direction_interpretation,
            },

            "barrel_stability_index": {
                "ratio":
                    barrel_stability_index,

                "percent":
                    100 * barrel_stability_index,
            },

            "peak_out_of_plane_torque": {
                "value_Nm":
                    peak_oop,

                "time_ms":
                    peak_oop_time,
            }
        }



    # NEW: full-body skeleton + club positions (3D meters), for the dynamic swing visualization.
    if swing_type == 0:
        M2C_skeleton_3d = {
            "pelvis": pelvis.tolist(), "hip_l": hip_l.tolist(), "hip_r": hip_r.tolist(),
            "knee_l": knee_l.tolist(), "knee_r": knee_r.tolist(),
            "ankle_l": ankle_l.tolist(), "ankle_r": ankle_r.tolist(),
            "foot_l": foot_l.tolist(), "foot_r": foot_r.tolist(),
            "spine_low": spine_low.tolist(), "spine_high": spine_high.tolist(),
            "shoulder_l": shoulder_l.tolist(), "shoulder_r": shoulder_r.tolist(),
            "elbow_l": elbow_l.tolist(), "elbow_r": elbow_r.tolist(),
            "wrist_l": wrist_l.tolist(), "wrist_r": wrist_r.tolist(),
            "hand_l": hand_l.tolist(), "hand_r": hand_r.tolist(),
            "neck": neck.tolist(), "skull": skull.tolist(),
            "club_main": club_main.tolist(), "club_mid": club_mid.tolist(), "club_head": club_head.tolist(),
        }

    else:
        M2C_skeleton_3d = {
            "pelvis": pelvis.tolist(), "hip_l": hip_l.tolist(), "hip_r": hip_r.tolist(),
            "knee_l": knee_l.tolist(), "knee_r": knee_r.tolist(),
            "ankle_l": ankle_l.tolist(), "ankle_r": ankle_r.tolist(),
            "foot_l": foot_l.tolist(), "foot_r": foot_r.tolist(),
            "spine_low": spine_low.tolist(), "spine_high": spine_high.tolist(),
            "shoulder_l": shoulder_l.tolist(), "shoulder_r": shoulder_r.tolist(),
            "elbow_l": elbow_l.tolist(), "elbow_r": elbow_r.tolist(),
            "wrist_l": wrist_l.tolist(), "wrist_r": wrist_r.tolist(),
            "hand_l": hand_l.tolist(), "hand_r": hand_r.tolist(),
            "neck": neck.tolist(), "skull": skull.tolist(),
        }

    

    #################################################################### KINEMATIC SEQUENCE ##################################################################

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
    #Lead forearm rotational velocity
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

    #####################################################################
    # PITCHER PERFORMANCE DASHBOARD INITIALIZATION
    #####################################################################

    Pitcher_performance_metrics = {}


    # Existing convention:
    # swing_type == 0 : hitter
    # swing_type == 1 : pitcher

    swing_type = {"hitting": 0, "pitching": 1, "generic": 2}[movement_type]

    is_pitcher = swing_type == 1

    #####################################################################
    # PITCHER HAND DOMINANCE SETUP
    #####################################################################

    if is_pitcher:

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

    #####################################################################
    # PEAK HAND SPEED
    #####################################################################

    if is_pitcher:


        hand_velocity = np.gradient(
            throwing_wrist,
            dt,
            axis=0
        )


        hand_speed = np.linalg.norm(
            hand_velocity,
            axis=1
        )


        peak_hand_frame = np.argmax(
            hand_speed[p4:p7]
        ) + p4


        Pitcher_performance_metrics["peak_hand_speed"] = {

            "value_mps":
                float(
                    hand_speed[peak_hand_frame]
                ),

            "frame":
                int(
                    peak_hand_frame
                ),

            "time_to_release_ms":
                float(
                    (peak_hand_frame-p7)
                    /
                    frame_rate
                    *
                    1000
                )
        }

    #####################################################################
    # RELEASE HEIGHT
    #####################################################################

    if is_pitcher:


        release_position = throwing_wrist[p7]


        Pitcher_performance_metrics["release_height"] = {

            "value_m":
                float(
                    release_position[2]
                ),

            "frame":
                int(p7)

        }

    #####################################################################
    # RELEASE EXTENSION
    #####################################################################

        #####################################################################
    # RELEASE EXTENSION
    #####################################################################

    if is_pitcher:


        extension_vector = (
            throwing_wrist[p7]
            -
            lead_leg[p7]
        )


        release_extension = abs(
            extension_vector[0]
        )


        Pitcher_performance_metrics["release_extension"]={

            "value_m":
                float(
                    release_extension
                ),

            "frame":
                int(p7)

        }


    # ARM WHIP INDEX
    #####################################################################

    if is_pitcher:


        #################################################################
        # Throwing forearm rotational velocity (handedness corrected)
        #################################################################

        throwing_forearm_rot_speed = rotational_velocity_about_spine(

            throwing_elbow,

            throwing_wrist,

            spine_high,

            spine_low,

            dt,

            filter_win=41

        )


        #################################################################
        # Peak forearm rotational velocity before release
        #################################################################

        forearm_peak_frame = np.argmax(

            abs(
                throwing_forearm_rot_speed[p4:p7]
            )

        ) + p4


        peak_forearm_velocity = (

            throwing_forearm_rot_speed[
                forearm_peak_frame
            ]

        )


        # #################################################################
        # # Arm whip ratio
        # #################################################################

        # whip = (

        #     abs(
        #         peak_forearm_velocity
        #     )

        #     /

        #     max(

        #         abs(
        #             trunk_peak["peak_velocity_deg_s"]
        #         ),

        #         1e-6

        #     )

        # )


        # Pitcher_performance_metrics["arm_whip_index"] = {


        #     "value":

        #         float(
        #             whip
        #         ),


        #     "peak_forearm_velocity_deg_s":

        #         float(
        #             peak_forearm_velocity
        #         ),


        #     "peak_forearm_velocity_frame":

        #         int(
        #             forearm_peak_frame
        #         ),


        #     "peak_forearm_velocity_time_to_release_ms":

        #         float(

        #             (forearm_peak_frame - p7)

        #             /

        #             frame_rate

        #             *

        #             1000

        #         ),


        #     "peak_trunk_velocity_deg_s":

        #         float(
        #             trunk_peak["peak_velocity_deg_s"]
        #         )

        # }


    #####################################################################
    # TRUNK ROTATION
    #####################################################################

    if is_pitcher:


        #################################################################
        # Trunk rotational velocity (handedness neutral)
        #################################################################

        trunk_rot_velocity = abs(
            UT_rot_speed
        )


        #################################################################
        # Peak trunk rotation velocity
        #################################################################

        trunk_peak_idx = np.argmax(

            trunk_rot_velocity[p1:p8 + 1]

        ) + p1


        trunk_peak = {


            "peak_velocity_deg_s":

                float(
                    trunk_rot_velocity[trunk_peak_idx]
                ),


            "time_to_release_ms":

                float(

                    1000
                    *
                    (trunk_peak_idx - p7)
                    /
                    frame_rate

                )

        }


        #################################################################
        # Trunk rotation range
        #################################################################

        trunk_rotation_angle = np.cumsum(

            UT_rot_speed * dt

        )


        trunk_rotation_window = (

            trunk_rotation_angle[p1:p8 + 1]

        )


        trunk_rotation_range = float(

            np.max(
                trunk_rotation_window
            )

            -

            np.min(
                trunk_rotation_window
            )

        )


        Pitcher_performance_metrics["trunk_rotation"] = {


            "peak_velocity_deg_s":

                trunk_peak["peak_velocity_deg_s"],


            "peak_time_ms":

                trunk_peak["time_to_release_ms"],


            "rotation_range_deg":

                trunk_rotation_range

        }


   #####################################################################
    # SHOULDER ROTATION
    #####################################################################

    if is_pitcher:


        #################################################################
        # Throwing shoulder rotational velocity
        #################################################################

        throwing_arm_rot_speed = rotational_velocity_about_spine(

            throwing_shoulder,

            throwing_elbow,

            spine_high,

            spine_low,

            dt,

            filter_win=41

        )


        #################################################################
        # Peak shoulder rotation velocity
        #################################################################

        shoulder_peak_idx = np.argmax(

            abs(
                throwing_arm_rot_speed[p1:p8+1]
            )

        ) + p1


        shoulder_peak_velocity = (

            throwing_arm_rot_speed[
                shoulder_peak_idx
            ]

        )


        Pitcher_performance_metrics["shoulder_rotation"] = {


            "peak_velocity_deg_s":

                float(

                    abs(
                        shoulder_peak_velocity
                    )

                ),


            "peak_time_ms":

                float(

                    1000
                    *
                    (shoulder_peak_idx-p7)
                    /
                    frame_rate

                ),


            "peak_frame":

                int(
                    shoulder_peak_idx
                )

        }

    #####################################################################
    # ELBOW EXTENSION VELOCITY
    #####################################################################

    if is_pitcher:

        upper_arm = safe_normalize(
            throwing_shoulder
            -
            throwing_elbow
        )

        forearm = safe_normalize(
            throwing_wrist
            -
            throwing_elbow
        )

        elbow_angle = np.degrees(

            np.arccos(

                np.clip(

                    np.sum(
                        upper_arm *
                        forearm,
                        axis=1
                    ),

                    -1,
                    1

                )

            )

        )

        elbow_velocity = np.gradient(
            elbow_angle,
            dt
        )

        idx = np.argmax(
            np.abs(
                elbow_velocity[p4:p7]
            )
        ) + p4

        Pitcher_performance_metrics["elbow_extension_velocity"] = {

            "peak_velocity_deg_s":
                float(
                    elbow_velocity[idx]
                ),

            "peak_time_ms":
                float(
                    (idx-p7)
                    /
                    frame_rate
                    *
                    1000
                ),

            "extension_range_deg":
                float(
                    np.max(elbow_angle)
                    -
                    np.min(elbow_angle)
                )

        }

    ######################################################################
    #####################################################################
    # LEAD LEG BRACING
    #####################################################################

    if is_pitcher:

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

        thigh = thigh / (
            np.linalg.norm(thigh,axis=1,keepdims=True)+1e-8
        )

        shank = shank / (
            np.linalg.norm(shank,axis=1,keepdims=True)+1e-8
        )

        knee_angle = np.degrees(

            np.arccos(

                np.clip(

                    np.sum(

                        thigh*shank,

                        axis=1

                    ),

                    -1,

                    1

                )

            )

        )

        knee_extension_velocity = np.gradient(
            knee_angle,
            dt
        )

        knee_extension_acceleration = np.gradient(
            knee_extension_velocity,
            dt
        )

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

            lead_leg_bracing = np.clip(

                100
                *
                peak_bracing
                /
                15000,

                0,

                100

            )

        Pitcher_performance_metrics["lead_leg_bracing"] = {

            "score":

                float(lead_leg_bracing),

            "peak_extension_acceleration_deg_s2":

                float(peak_bracing)

        }


        # knee_extension_velocity = np.gradient(
        #     knee_angle,
        #     dt
        # )

        # knee_extension_acceleration = np.gradient(
        #     knee_extension_velocity,
        #     dt
        # )

        # release_frame = p7

        # window = slice(max(p5,0),release_frame+1)

        # peak_bracing = np.max(
        #     knee_extension_acceleration[window]
        # )

        # lead_leg_bracing = np.clip(

        #     100
        #     *
        #     peak_bracing
        #     /
        #     15000,

        #     0,

        #     100

        # )

        # Pitcher_performance_metrics["lead_leg_bracing"]={

        #     "score":

        #         float(lead_leg_bracing),

        #     "peak_extension_acceleration_deg_s2":

        #         float(peak_bracing)

        # }



    # LATE ARM ACCELERATION
    #####################################################################

    if is_pitcher:


        #################################################################
        # Throwing forearm rotational velocity (handedness corrected)
        #################################################################

        throwing_hand_rot_speed = rotational_velocity_about_spine(

            throwing_elbow,

            throwing_wrist,

            spine_high,

            spine_low,

            dt,

            filter_win=41

        )


        #################################################################
        # Angular acceleration
        #################################################################

        arm_acceleration = np.gradient(

            throwing_hand_rot_speed,

            dt

        )


        #################################################################
        # Late acceleration window before release
        #################################################################

        late_window = slice(

            max(
                release_frame - 12,
                p1
            ),

            release_frame + 1

        )


        #################################################################
        # Peak acceleration magnitude
        #################################################################

        peak_idx = np.argmax(

            abs(
                arm_acceleration[late_window]
            )

        ) + late_window.start


        peak_acceleration = (

            arm_acceleration[
                peak_idx
            ]

        )


        Pitcher_performance_metrics["late_arm_acceleration"] = {


            "peak_deg_s2":

                float(

                    abs(
                        peak_acceleration
                    )

                ),


            "time_to_release_ms":

                float(

                    1000
                    *
                    (peak_idx-release_frame)
                    /
                    frame_rate

                ),


            "peak_frame":

                int(
                    peak_idx
                )

        }

    ###
    #####################################################################
    # ENERGY TRANSFER
    #####################################################################

    if is_pitcher:

        pelvis_peak=np.max(
            np.abs(
                pelvis_rot_speed[p1:p8+1]
            )
        )

        trunk_peak=np.max(
            np.abs(
                UT_rot_speed[p1:p8+1]
            )
        )

        arm_peak=np.max(
            np.abs(
                arm_rot_speed[p1:p8+1]
            )
        )

        hand_peak=np.max(
            np.abs(
                hand_rot_speed[p1:p8+1]
            )
        )

        transfer=np.mean([

            trunk_peak/pelvis_peak,

            arm_peak/trunk_peak,

            hand_peak/arm_peak

        ])

        energy_transfer=np.clip(

            100*transfer,

            0,

            100

        )

        Pitcher_performance_metrics["energy_transfer"]={

            "score":

                float(energy_transfer)

        }

    ####

    #####################################################################
    # HAND PATH EFFICIENCY
    #####################################################################

    if is_pitcher:


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

        hand_segment = hand_path[p4:p7+1]


        #################################################################
        # Actual hand path travelled
        #################################################################

        segment_lengths = np.linalg.norm(
            np.diff(
                hand_segment,
                axis=0
            ),
            axis=1
        )


        actual_path = np.sum(
            segment_lengths
        )


        #################################################################
        # Direct displacement
        #################################################################

        direct_path = np.linalg.norm(

            hand_segment[-1]
            -
            hand_segment[0]

        )


        #################################################################
        # Efficiency calculation
        #################################################################

        if actual_path > 1e-6:

            efficiency = (
                direct_path
                /
                actual_path
            )

        else:

            efficiency = 0.0



        #################################################################
        # Additional interpretation values
        #################################################################

        path_waste = (

            1.0
            -
            efficiency

        )


        Pitcher_performance_metrics["hand_path_efficiency"] = {


            "value":

                float(
                    efficiency
                ),


            "percent":

                float(
                    efficiency * 100
                ),


            "actual_path_length_m":

                float(
                    actual_path
                ),


            "direct_displacement_m":

                float(
                    direct_path
                ),


            "path_waste_percent":

                float(
                    path_waste * 100
                ),


            "window":

                {

                    "start_frame":
                        int(p4),

                    "release_frame":
                        int(p7)

                }

        }

    ###

    #####################################################################
    # RELEASE STABILITY
    #####################################################################

    if is_pitcher:


        #################################################################
        # Throwing forearm rotational velocity (handedness corrected)
        #################################################################

        throwing_hand_rot_speed = rotational_velocity_about_spine(

            throwing_elbow,

            throwing_wrist,

            spine_high,

            spine_low,

            dt,

            filter_win=41

        )


        #################################################################
        # Release window stability
        #################################################################

        release_window = slice(

            max(release_frame - 3, p1),

            min(release_frame + 4, p8)

        )


        release_velocity_variability = np.std(

            throwing_hand_rot_speed[
                release_window
            ]

        )


        #################################################################
        # Stability score
        #################################################################

        stability = np.clip(

            100

            -

            release_velocity_variability / 10,

            0,

            100

        )


        if stability > 90:

            label = "Excellent"

        elif stability > 75:

            label = "Good"

        elif stability > 60:

            label = "Average"

        else:

            label = "Poor"


        Pitcher_performance_metrics["release_stability"] = {


            "score":

                float(
                    stability
                ),


            "classification":

                label,


            "release_velocity_sd_deg_s":

                float(
                    release_velocity_variability
                ),


            "release_window_frames":

                [

                    int(max(release_frame - 3, p1)),

                    int(min(release_frame + 4, p8))

                ]

        }

    if is_pitcher:



                # =========================================================
        # RIGHT HAND FORCE
        # =========================================================

        hand_r_f = np.zeros(hand_r.shape)
        hand_l_f = np.zeros(hand_l.shape)

        fc = 3  # Hz
        N = 4  # Order of Butterworth filter

        c, d = signal.butter(N, fc / (frame_rate / 2))

        hand_r_f[:, 0] = signal.filtfilt(
            c, d, hand_r[:, 0]
        )

        hand_r_f[:, 1] = signal.filtfilt(
            c, d, hand_r[:, 1]
        )

        hand_l_f[:, 0] = signal.filtfilt(
            c, d, hand_l[:, 0]
        )

        hand_l_f[:, 1] = signal.filtfilt(
            c, d, hand_l[:, 1]
        )


        fc = 5  # Hz
        N = 4

        c, d = signal.butter(N, fc / (frame_rate / 2))

        hand_r_f[:, 2] = signal.filtfilt(
            c, d, hand_r[:, 2]
        )

        hand_l_f[:, 2] = signal.filtfilt(
            c, d, hand_l[:, 2]
        )

        # ---------------------------------------------------------
        # Right hand displacement
        # ---------------------------------------------------------

        hand_r_disp_x = hand_r_f[:, 0] - hand_r_f[p1, 0]
        hand_r_disp_y = hand_r_f[:, 1] - hand_r_f[p1, 1]
        hand_r_disp_z = hand_r_f[:, 2] - hand_r_f[p1, 2]

        # ---------------------------------------------------------
        # Left hand displacement
        # ---------------------------------------------------------

        hand_l_disp_x = hand_l_f[:, 0] - hand_l_f[p1, 0]
        hand_l_disp_y = hand_l_f[:, 1] - hand_l_f[p1, 1]
        hand_l_disp_z = hand_l_f[:, 2] - hand_l_f[p1, 2]


        # ---------------------------------------------------------
        # Right hand speed
        # ---------------------------------------------------------

        hand_r_speed_x = np.gradient(hand_r_disp_x, dt)
        hand_r_speed_y = np.gradient(hand_r_disp_y, dt)
        hand_r_speed_z = np.gradient(hand_r_disp_z, dt)

        # ---------------------------------------------------------
        # Left hand speed
        # ---------------------------------------------------------

        hand_l_speed_x = np.gradient(hand_l_disp_x, dt)
        hand_l_speed_y = np.gradient(hand_l_disp_y, dt)
        hand_l_speed_z = np.gradient(hand_l_disp_z, dt)
        hand_r_speed_z = np.gradient(hand_r_disp_z, dt)


        # ---------------------------------------------------------
        # Right hand acceleration
        # ---------------------------------------------------------

        hand_r_acc_x = np.gradient(hand_r_speed_x, dt)
        hand_r_acc_y = np.gradient(hand_r_speed_y, dt)
        hand_r_acc_z = np.gradient(hand_r_speed_z, dt)

        # ---------------------------------------------------------
        # Left hand acceleration
        # ---------------------------------------------------------

        hand_l_acc_x = np.gradient(hand_l_speed_x, dt)
        hand_l_acc_y = np.gradient(hand_l_speed_y, dt)
        hand_l_acc_z = np.gradient(hand_l_speed_z, dt)


        # ---------------------------------------------------------
        # Right hand FORCE COMPONENTS
        # ---------------------------------------------------------
        hand_r_m = 0.61 / 100
        hand_l_m = 0.61 / 100

        Fx_t_est_hand_r = W * hand_r_m * hand_r_acc_x
        Fy_t_est_hand_r = W * hand_r_m * hand_r_acc_y
        Fz_t_est_hand_r = W * hand_r_m * hand_r_acc_z

        Fx_t_est_hand_l = W * hand_l_m * hand_l_acc_x
        Fy_t_est_hand_l = W * hand_l_m * hand_l_acc_y
        Fz_t_est_hand_l = W * hand_l_m * hand_l_acc_z


        # ---------------------------------------------------------
        # Combined right hand force magnitude
        # ---------------------------------------------------------

        F_hand_r = np.sqrt(
            Fx_t_est_hand_r ** 2
            + Fy_t_est_hand_r ** 2
            + Fz_t_est_hand_r ** 2
        )


        # ---------------------------------------------------------
        # Right hand velocity components
        # ---------------------------------------------------------

        hand_r_speed_X = np.diff(hand_r[:, 0]) * frame_rate
        hand_r_speed_X = np.insert(hand_r_speed_X, 0, 0)

        hand_r_speed_Y = np.diff(hand_r[:, 1]) * frame_rate
        hand_r_speed_Y = np.insert(hand_r_speed_Y, 0, 0)

        hand_r_speed_Z = np.diff(hand_r[:, 2]) * frame_rate
        hand_r_speed_Z = np.insert(hand_r_speed_Z, 0, 0)


        # ---------------------------------------------------------
        # Right hand power
        # ---------------------------------------------------------

        hand_r_power_x = Fx_t_est_hand_r * hand_r_speed_X
        hand_r_power_y = Fy_t_est_hand_r * hand_r_speed_Y
        hand_r_power_z = Fz_t_est_hand_r * hand_r_speed_Z

        hand_r_power = (
            hand_r_power_x
            + hand_r_power_y
            + hand_r_power_z
        )


        # ---------------------------------------------------------
        # Combined right hand speed
        # ---------------------------------------------------------

        hand_r_speed = np.sqrt(
            hand_r_speed_X ** 2
            + hand_r_speed_Y ** 2
            + hand_r_speed_Z ** 2
        )

        # =========================================================
        # THROWING HAND FORCE VECTORS - 3D
        # Handles right-handed and left-handed pitchers
        # =========================================================

        if lefty == 1:

            # =====================================================
            # LEFT HAND / LEFT ELBOW
            # =====================================================

            # -----------------------------------------------------
            # 1. Tangential direction
            #    Direction of instantaneous left-hand movement
            # -----------------------------------------------------

            F_tan_hand_vec_norm = hand_l_f[1:] - hand_l_f[:-1]

            norms = np.linalg.norm(F_tan_hand_vec_norm, axis=1)

            nonzero = norms > 0

            F_tan_hand_vec_norm[nonzero] /= (
                norms[nonzero, np.newaxis]
            )

            # Keep same number of frames
            F_tan_hand_vec_norm = np.vstack([
                np.zeros((1, 3)),
                F_tan_hand_vec_norm
            ])


            # -----------------------------------------------------
            # 2. Axial direction
            #    Left forearm axis: elbow -> hand
            # -----------------------------------------------------

            F_axial_hand_vec_norm = hand_l_f - elbow_l

            norms = np.linalg.norm(
                F_axial_hand_vec_norm,
                axis=1
            )

            nonzero = norms > 0

            F_axial_hand_vec_norm[nonzero] /= (
                norms[nonzero, np.newaxis]
            )


            # -----------------------------------------------------
            # 3. Force magnitudes
            # -----------------------------------------------------

            F_hand_vec = np.column_stack([
                Fx_t_est_hand_l,
                Fy_t_est_hand_l,
                Fz_t_est_hand_l
            ])


            # Tangential force magnitude
            F_tan_hand = np.sum(
                F_hand_vec * F_tan_hand_vec_norm,
                axis=1
            )


            # Axial force magnitude
            F_axial_hand = np.sum(
                F_hand_vec * F_axial_hand_vec_norm,
                axis=1
            )


            # -----------------------------------------------------
            # 4. Construct the three actual force vectors
            # -----------------------------------------------------

            # Tangential
            F_tan_hand_vec = (
                F_tan_hand_vec_norm
                * F_tan_hand[:, np.newaxis]
            )


            # Axial
            F_axial_hand_vec = (
                F_axial_hand_vec_norm
                * F_axial_hand[:, np.newaxis]
            )


            # Combined / resultant left-hand force
            F_combined_hand_vec = F_hand_vec


            # =====================================================
            # FRONTEND VISUALIZATION SCALING ONLY
            # =====================================================

            hand_force_arrow_scale = 0.01

            F_tan_hand_vec_vis = (
                F_tan_hand_vec
                * hand_force_arrow_scale
            )

            F_axial_hand_vec_vis = (
                F_axial_hand_vec
                * hand_force_arrow_scale
            )

            F_combined_hand_vec_vis = (
                F_combined_hand_vec
                * hand_force_arrow_scale
            )


            # =====================================================
            # LEFT HAND FORCE VECTOR START / END POINTS
            # =====================================================

            # Start all three arrows at the left hand
            start_x_hand_3d = hand_l[:, 0]
            start_y_hand_3d = hand_l[:, 1]
            start_z_hand_3d = hand_l[:, 2]


            # -----------------------------------------------------
            # Tangential force arrow end point
            # -----------------------------------------------------

            end_x_tan_force_hand_3d = (
                start_x_hand_3d + F_tan_hand_vec_vis[:, 0]
            )

            end_y_tan_force_hand_3d = (
                start_y_hand_3d + F_tan_hand_vec_vis[:, 1]
            )

            end_z_tan_force_hand_3d = (
                start_z_hand_3d + F_tan_hand_vec_vis[:, 2]
            )


            # -----------------------------------------------------
            # Axial force arrow end point
            # -----------------------------------------------------

            end_x_axial_force_hand_3d = (
                start_x_hand_3d + F_axial_hand_vec_vis[:, 0]
            )

            end_y_axial_force_hand_3d = (
                start_y_hand_3d + F_axial_hand_vec_vis[:, 1]
            )

            end_z_axial_force_hand_3d = (
                start_z_hand_3d + F_axial_hand_vec_vis[:, 2]
            )


            # -----------------------------------------------------
            # Combined force arrow end point
            # -----------------------------------------------------

            end_x_combined_force_hand_3d = (
                start_x_hand_3d + F_combined_hand_vec_vis[:, 0]
            )

            end_y_combined_force_hand_3d = (
                start_y_hand_3d + F_combined_hand_vec_vis[:, 1]
            )

            end_z_combined_force_hand_3d = (
                start_z_hand_3d + F_combined_hand_vec_vis[:, 2]
            )


        else:

            # =====================================================
            # RIGHT HAND / RIGHT ELBOW
            # =====================================================

            # -----------------------------------------------------
            # 1. Tangential direction
            # -----------------------------------------------------

            F_tan_hand_vec_norm = hand_r_f[1:] - hand_r_f[:-1]

            norms = np.linalg.norm(F_tan_hand_vec_norm, axis=1)

            nonzero = norms > 0

            F_tan_hand_vec_norm[nonzero] /= (
                norms[nonzero, np.newaxis]
            )

            F_tan_hand_vec_norm = np.vstack([
                np.zeros((1, 3)),
                F_tan_hand_vec_norm
            ])


            # -----------------------------------------------------
            # 2. Axial direction
            #    Right forearm axis: elbow -> hand
            # -----------------------------------------------------

            F_axial_hand_vec_norm = hand_r_f - elbow_r

            norms = np.linalg.norm(
                F_axial_hand_vec_norm,
                axis=1
            )

            nonzero = norms > 0

            F_axial_hand_vec_norm[nonzero] /= (
                norms[nonzero, np.newaxis]
            )


            # -----------------------------------------------------
            # 3. Force magnitudes
            # -----------------------------------------------------

            F_hand_vec = np.column_stack([
                Fx_t_est_hand_r,
                Fy_t_est_hand_r,
                Fz_t_est_hand_r
            ])


            F_tan_hand = np.sum(
                F_hand_vec * F_tan_hand_vec_norm,
                axis=1
            )


            F_axial_hand = np.sum(
                F_hand_vec * F_axial_hand_vec_norm,
                axis=1
            )


            # -----------------------------------------------------
            # 4. Construct force vectors
            # -----------------------------------------------------

            F_tan_hand_vec = (
                F_tan_hand_vec_norm
                * F_tan_hand[:, np.newaxis]
            )

            F_axial_hand_vec = (
                F_axial_hand_vec_norm
                * F_axial_hand[:, np.newaxis]
            )

            F_combined_hand_vec = F_hand_vec


            # =====================================================
            # FRONTEND VISUALIZATION SCALING ONLY
            # =====================================================

            hand_force_arrow_scale = 0.01

            F_tan_hand_vec_vis = (
                F_tan_hand_vec
                * hand_force_arrow_scale
            )

            F_axial_hand_vec_vis = (
                F_axial_hand_vec
                * hand_force_arrow_scale
            )

            F_combined_hand_vec_vis = (
                F_combined_hand_vec
                * hand_force_arrow_scale
            )


            # =====================================================
            # RIGHT HAND FORCE VECTOR START / END POINTS
            # =====================================================

            start_x_hand_3d = hand_r[:, 0]
            start_y_hand_3d = hand_r[:, 1]
            start_z_hand_3d = hand_r[:, 2]


            # Tangential
            end_x_tan_force_hand_3d = (
                start_x_hand_3d + F_tan_hand_vec_vis[:, 0]
            )

            end_y_tan_force_hand_3d = (
                start_y_hand_3d + F_tan_hand_vec_vis[:, 1]
            )

            end_z_tan_force_hand_3d = (
                start_z_hand_3d + F_tan_hand_vec_vis[:, 2]
            )


            # Axial
            end_x_axial_force_hand_3d = (
                start_x_hand_3d + F_axial_hand_vec_vis[:, 0]
            )

            end_y_axial_force_hand_3d = (
                start_y_hand_3d + F_axial_hand_vec_vis[:, 1]
            )

            end_z_axial_force_hand_3d = (
                start_z_hand_3d + F_axial_hand_vec_vis[:, 2]
            )


            # Combined
            end_x_combined_force_hand_3d = (
                start_x_hand_3d + F_combined_hand_vec_vis[:, 0]
            )

            end_y_combined_force_hand_3d = (
                start_y_hand_3d + F_combined_hand_vec_vis[:, 1]
            )

            end_z_combined_force_hand_3d = (
                start_z_hand_3d + F_combined_hand_vec_vis[:, 2]
            )
        
        
        # =========================================================
        # =========================================================
        # JSON RESPONSE
        # =========================================================

        M2C_force_vectors_hand_3d = {
            "start_3d": [
                [x, y, z]
                for x, y, z in zip(
                    start_x_hand_3d.tolist(),
                    start_y_hand_3d.tolist(),
                    start_z_hand_3d.tolist()
                )
            ],

            "end_rad_force_3d": [
                [x, y, z]
                for x, y, z in zip(
                    end_x_axial_force_hand_3d.tolist(),
                    end_y_axial_force_hand_3d.tolist(),
                    end_z_axial_force_hand_3d.tolist()
                )
            ],

            "end_tan_force_3d": [
                [x, y, z]
                for x, y, z in zip(
                    end_x_tan_force_hand_3d.tolist(),
                    end_y_tan_force_hand_3d.tolist(),
                    end_z_tan_force_hand_3d.tolist()
                )
            ],

            "end_combined_force_3d": [
                [x, y, z]
                for x, y, z in zip(
                    end_x_combined_force_hand_3d.tolist(),
                    end_y_combined_force_hand_3d.tolist(),
                    end_z_combined_force_hand_3d.tolist()
                )
            ],
        }
    



    #####################################################################
    # V7: SHARED HELPERS (used by pitcher M2C_* equivalents and "generic")
    #####################################################################
    # Pose-only helpers. They never touch the hitter calculations above.

    def _lp(x, fc, order=4):
        """Zero-lag Butterworth low-pass. Returns x unchanged if the clip is too short / fc >= Nyquist."""
        x = np.asarray(x, dtype=float)
        wn = fc / (frame_rate / 2.0)
        if wn >= 1.0 or x.shape[0] <= 3 * (order + 1):
            return x
        b_, a_ = signal.butter(order, wn)
        return signal.filtfilt(b_, a_, x, axis=0)

    def _fill_nan(a):
        a = np.array(a, dtype=float)
        idx = np.arange(a.shape[0])
        for k in range(a.shape[1]):
            col = a[:, k]
            bad = ~np.isfinite(col)
            if bad.any() and not bad.all():
                col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
        return a

    def _deriv(x, order=1):
        """Savitzky-Golay derivative with a ~0.15 s window (falls back to np.gradient on very short clips)."""
        x = np.asarray(x, dtype=float)
        n_ = x.shape[0]
        if n_ < 2:
            return np.zeros_like(x)
        win = max(int(round(0.15 * frame_rate)), 5)
        if win % 2 == 0:
            win += 1
        if win > n_:
            win = n_ if n_ % 2 == 1 else n_ - 1
        poly = max(order + 1, 3)
        if win <= poly or not np.all(np.isfinite(x)):
            out_ = x
            for _ in range(order):
                out_ = np.gradient(out_, dt, axis=0)
            return out_
        return savgol_filter(x, win, poly, deriv=order, delta=dt, axis=0, mode="interp")

    def _ang_between(u, v):
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        c_ = np.einsum("ij,ij->i", u, v) / np.maximum(nu * nv, 1e-9)
        return np.degrees(np.arccos(np.clip(c_, -1.0, 1.0)))

    def _ang3(a, b, c):
        """Interior angle (deg) at b between b->a and b->c."""
        return _ang_between(a - b, c - b)

    def _segments(J):
        """Dempster-style segments, identical COM fractions / mass fractions to the hitter COM code.
        (name, proximal, distal, COM fraction from proximal, mass fraction)"""
        return [
            ("head", J["skull"], J["skullbase"], 0.5976, 0.0694),
            ("trunk", J["neck"], J["pelvis"], 0.4486, 0.4336),
            ("arm_r", J["shoulder_r"], J["elbow_r"], 0.5772, 0.0271),
            ("forearm_r", J["elbow_r"], J["wrist_r"], 0.4574, 0.0162),
            ("hand_r", J["wrist_r"], J["hand_r"], 0.79, 0.0061),
            ("arm_l", J["shoulder_l"], J["elbow_l"], 0.5772, 0.0271),
            ("forearm_l", J["elbow_l"], J["wrist_l"], 0.4574, 0.0162),
            ("hand_l", J["wrist_l"], J["hand_l"], 0.79, 0.0061),
            ("thigh_r", J["hip_r"], J["knee_r"], 0.4095, 0.1416),
            ("shank_r", J["knee_r"], J["ankle_r"], 0.4459, 0.0433),
            ("foot_r", J["ankle_r"], J["foot_r"], 0.4415, 0.0137),
            ("thigh_l", J["hip_l"], J["knee_l"], 0.4095, 0.1416),
            ("shank_l", J["knee_l"], J["ankle_l"], 0.4459, 0.0433),
            ("foot_l", J["ankle_l"], J["foot_l"], 0.4415, 0.0137),
        ]

    def _seg_com(seg):
        return seg[1] + seg[3] * (seg[2] - seg[1])

    def _body_com(J):
        segs_ = _segments(J)
        return sum(_seg_com(s_) * s_[4] for s_ in segs_), segs_

    def _stats(arr, unit):
        a = np.asarray(arr, dtype=float)
        if a.size == 0 or not np.isfinite(a).any():
            return {"unit": unit}
        imax, imin = int(np.nanargmax(a)), int(np.nanargmin(a))
        return {
            "unit": unit,
            "min": float(a[imin]), "max": float(a[imax]),
            "mean": float(np.nanmean(a)), "range": float(a[imax] - a[imin]),
            "max_frame": imax, "max_time_s": imax * dt,
            "min_frame": imin, "min_time_s": imin * dt,
        }

    def _clean(obj):
        """Make everything JSON-safe: numpy -> python, NaN/inf -> None."""
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return _clean(obj.tolist())
        if isinstance(obj, (np.floating, float)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    #####################################################################
    # V7: PITCHER EQUIVALENTS OF M2C_power / M2C_club_mechanics / M2C_wrist
    #####################################################################
    # Same widget structure as the hitter. Anything that depends on the bat
    # (Shaft_power, club_head_power, Grip_Force, F_club_head, Wrist_torque,
    # lead_forearm_shaft_angle) is replaced by a throwing-arm equivalent.
    # See "pitcher_metric_equivalents" in the output for the exact mapping.

    def _limb_widgets(side, i_first, i_last, i_ref, i_win0, i_win1):
        """Builds the M2C_power / M2C_club_mechanics / M2C_club_mechanics_highlights / M2C_wrist
        widgets for ONE arm (the throwing arm for pitchers, the primary arm for generic movements).

        side    : "l" or "r"
        i_first : first frame of the analysis window (pitcher p1, generic 0) - also the displacement reference
        i_last  : last frame of the analysis window (pitcher p8, generic n-1)
        i_ref   : reference event frame for time_to_impact (pitcher release p7, generic peak wrist speed)
        i_win0/1: window used for power_index (pitcher p4..p7+1, generic whole clip)
        """
        sh_ = {"l": shoulder_l, "r": shoulder_r}[side]
        el_ = {"l": elbow_l, "r": elbow_r}[side]
        wr_ = {"l": wrist_l, "r": wrist_r}[side]
        hd_ = {"l": hand_l, "r": hand_r}[side]
        n_f = len(hd_)

        J_raw = {
            "pelvis": pelvis, "hip_l": hip_l, "hip_r": hip_r, "knee_l": knee_l, "knee_r": knee_r,
            "ankle_l": ankle_l, "ankle_r": ankle_r, "foot_l": foot_l, "foot_r": foot_r,
            "skull": skull, "skullbase": skullbase, "neck": neck,
            "shoulder_l": shoulder_l, "shoulder_r": shoulder_r, "elbow_l": elbow_l, "elbow_r": elbow_r,
            "wrist_l": wrist_l, "wrist_r": wrist_r, "hand_l": hand_l, "hand_r": hand_r,
        }

        # ---- vertical_power: same formulation as the hitter (COM force . COM velocity) ----
        COM_p, segs_p = _body_com(J_raw)
        COM_pf = np.column_stack([_lp(COM_p[:, 0], 3), _lp(COM_p[:, 1], 3), _lp(COM_p[:, 2], 5)])
        COM_p_vel = np.gradient(COM_pf - COM_pf[i_first], dt, axis=0)
        COM_p_acc = np.gradient(COM_p_vel, dt, axis=0)
        Fx_p, Fy_p = W * COM_p_acc[:, 0], W * COM_p_acc[:, 1]
        Fz_p = W * (COM_p_acc[:, 2] + 9.81)
        Fz_p[Fz_p < 0] = 0
        vertical_power_p = Fx_p * COM_p_vel[:, 0] + Fy_p * COM_p_vel[:, 1] + Fz_p * COM_p_vel[:, 2]

        # ---- hand force (same method as the hitter lead-hand / pitcher throwing-hand force) ----
        hand_f_ = np.column_stack([_lp(hd_[:, 0], 3), _lp(hd_[:, 1], 3), _lp(hd_[:, 2], 5)])
        hand_vel_f = np.gradient(hand_f_ - hand_f_[i_first], dt, axis=0)
        hand_acc_f = np.gradient(hand_vel_f, dt, axis=0)
        F_hand_vec_ = W * (0.61 / 100) * hand_acc_f
        tan_dir = hand_f_[1:] - hand_f_[:-1]
        nz = np.linalg.norm(tan_dir, axis=1)
        tan_dir[nz > 0] /= nz[nz > 0, None]
        tan_dir = np.vstack([np.zeros((1, 3)), tan_dir])
        ax_dir = hand_f_ - el_
        nz = np.linalg.norm(ax_dir, axis=1)
        ax_dir[nz > 0] /= nz[nz > 0, None]
        F_tan_ = np.sum(F_hand_vec_ * tan_dir, axis=1)
        F_axial_ = np.sum(F_hand_vec_ * ax_dir, axis=1)

        # ---- hand_power: same formulation as hitter hand_power ----
        hand_vel_raw = np.vstack([np.zeros((1, 3)), np.diff(hd_, axis=0) * frame_rate])
        hand_power_p = np.sum(F_hand_vec_ * hand_vel_raw, axis=1)

        # ---- arm_power (replaces Shaft_power) / distal_segment_power (replaces club_head_power) ----
        fc_arm = min(PITCH_KIN_FC_HZ, 0.4 * frame_rate / 2.0)
        arm_power_p = np.zeros(n_f)
        distal_power_p = np.zeros(n_f)
        for s_ in segs_p:
            if s_[0] in (f"arm_{side}", f"forearm_{side}", f"hand_{side}"):
                c_f = _lp(_seg_com(s_), fc_arm)
                v_ = np.gradient(c_f, dt, axis=0)
                a_ = np.gradient(v_, dt, axis=0)
                p_ = W * s_[4] * np.sum(a_ * v_, axis=1)
                arm_power_p += p_
                if s_[0] != f"arm_{side}":
                    distal_power_p += p_
        power_index_p = float(np.max(arm_power_p[i_win0:i_win1]) / max(W, 1e-6))

        # ---- elbow joint kinetics (forearm+hand rigid body, Newton-Euler, no external load) ----
        elb_f = _lp(el_, fc_arm)
        wri_f = _lp(wr_, fc_arm)
        han_f = _lp(hd_, fc_arm)
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
        Elbow_Force = np.linalg.norm(F_elbow_vec, axis=1)
        Elbow_torque = np.linalg.norm(M_elbow_vec, axis=1)
        F_hand_mag = np.linalg.norm(F_hand_vec_, axis=1)

        M2C_power_w = {
            "vertical_power": vertical_power_p.tolist(),
            "hand_power": hand_power_p.tolist(),
            "arm_power": arm_power_p.tolist(),                      # replaces Shaft_power
            "distal_segment_power": distal_power_p.tolist(),        # replaces club_head_power
            "power_index": power_index_p,                           # W/kg (not 0-100)
        }
        M2C_club_mechanics_w = {
            "Tangential_Force": F_tan_.tolist(),
            "Centripetal_Force": np.abs(F_axial_).tolist(),         # axial (along forearm) force
            "Elbow_Force": Elbow_Force.tolist(),                    # replaces Grip_Force
            "F_hand": F_hand_mag.tolist(),                          # replaces F_club_head
            "Elbow_torque": Elbow_torque.tolist(),                  # replaces Wrist_torque
        }

        def _peak_in_window(arr):
            seg_ = arr[i_first:i_last + 1]
            i_ = int(np.argmax(seg_))
            return float(seg_[i_]), i_, float(1000.0 * (i_ + i_first - i_ref) / frame_rate)

        pk_tan = _peak_in_window(F_tan_)
        pk_axl = _peak_in_window(np.abs(F_axial_))
        pk_elf = _peak_in_window(Elbow_Force)
        pk_hnd = _peak_in_window(F_hand_mag)
        pk_elt = _peak_in_window(Elbow_torque)
        order_idx = np.argsort([pk_tan[1], pk_axl[1], pk_elt[1]])
        highlights_w = {
            "peak_club_mechanics": {
                "peak_tangential_force": pk_tan[0],
                "peak_centripetal_force": pk_axl[0],
                "peak_Elbow_force": pk_elf[0],
                "peak_F_hand": pk_hnd[0],
                "peak_Elbow_torque": pk_elt[0],
            },
            "time_to_impact": {          # ms relative to the reference event (pitcher: release p7)
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

        wrist_angle_t = _ang3(hd_, wr_, el_)
        wrist_speed_t = np.concatenate(
            [[0], np.linalg.norm(np.diff(wr_, axis=0) * frame_rate, axis=1)]
        ) * 2.23693629
        elbow_flexion_t = 180.0 - _ang3(sh_, el_, wr_)
        M2C_wrist_w = {
            "throwing_elbow_flexion": elbow_flexion_t.tolist(),                 # replaces lead_forearm_shaft_angle
            "throwing_wrist_extension_flexion": wrist_angle_t.tolist(),
            "throwing_wrist_extension_flexion_vel": np.gradient(wrist_angle_t, dt).tolist(),
            "throwing_wrist_speed": wrist_speed_t.tolist(),                     # mph
            "throwing_wrist_thrust": (100 * (wr_[:, 0] - wr_[0, 0])).tolist(),
            "throwing_wrist_sway": (100 * (wr_[:, 1] - wr_[0, 1])).tolist(),
            "throwing_wrist_lift": (100 * (wr_[:, 2] - wr_[0, 2])).tolist(),
        }
        return {
            "M2C_power": M2C_power_w,
            "M2C_club_mechanics": M2C_club_mechanics_w,
            "M2C_club_mechanics_highlights": highlights_w,
            "M2C_wrist": M2C_wrist_w,
            "_elbow_flexion": elbow_flexion_t,
        }

    _WIDGET_EQUIVALENTS = {
        "M2C_power": {
            "vertical_power": "same as hitter (COM force . COM velocity)",
            "hand_power": "primary/throwing hand (hitter: lead hand)",
            "arm_power": "replaces Shaft_power (arm segment power, W)",
            "distal_segment_power": "replaces club_head_power (forearm+hand segment power, W)",
            "power_index": "peak arm_power / body mass over the analysis window, W/kg (hitter: normative 0-100 index)",
        },
        "M2C_club_mechanics": {
            "Tangential_Force": "hand force along hand path (N)",
            "Centripetal_Force": "hand force along forearm axis, abs (N)",
            "Elbow_Force": "replaces Grip_Force (elbow joint reaction force, N)",
            "F_hand": "replaces F_club_head (hand resultant force, N)",
            "Elbow_torque": "replaces Wrist_torque (net elbow joint torque, N*m)",
        },
        "M2C_wrist": {
            "throwing_elbow_flexion": "replaces lead_forearm_shaft_angle (deg, 0 = straight arm)",
            "throwing_wrist_*": "same metrics as lead_wrist_* but for the primary/throwing hand",
        },
    }

    pitcher_outputs = {}

    if is_pitcher:
        side = "l" if lefty == 1 else "r"
        _w = _limb_widgets(side, p1, p8, p7, p4, p7 + 1)
        pitcher_outputs = {
            "M2C_insights": {
                "arc_width": arc_width.tolist(),
                "arc_width_inch": arc_width_inch.tolist(),
                "throwing_elbow_flexion": _w["_elbow_flexion"].tolist(),
                "x_factor": x_factor.tolist(),
                "x_factor_stretch": x_factor_stretch,
            },
            "M2C_power": _w["M2C_power"],
            "M2C_club_mechanics": _w["M2C_club_mechanics"],
            "M2C_club_mechanics_highlights": _w["M2C_club_mechanics_highlights"],
            "M2C_wrist": _w["M2C_wrist"],
            "pitcher_metric_equivalents": dict(_WIDGET_EQUIVALENTS, time_to_impact="for pitchers = ms relative to release (p7)"),
        }

    # Generic movements: same three widgets (+ highlights) for the "primary" arm.
    generic_outputs = {}
    if movement_type == "generic":
        try:
            if "handedness" in _cfg_swing:
                g_side = "l" if lefty == 1 else "r"
                g_sel = "config handedness"
            else:
                _sp = {s_: np.max(np.linalg.norm(_deriv(wr_a, 1), axis=1))
                       for s_, wr_a in (("l", wrist_l), ("r", wrist_r))}
                g_side = max(_sp, key=_sp.get)
                g_sel = "arm with the higher peak wrist speed"
            g_n = len(wrist_l)
            _wr_by = {"l": wrist_l, "r": wrist_r}

            def _peak_speed_frame(s_):
                return int(np.argmax(np.linalg.norm(_deriv(_wr_by[s_], 1), axis=1)))

            g_other = "r" if g_side == "l" else "l"
            _name = {"l": "left", "r": "right"}
            g_ref = _peak_speed_frame(g_side)
            _w = _limb_widgets(g_side, 0, g_n - 1, g_ref, 0, g_n)
            generic_outputs = {
                "M2C_power": _w["M2C_power"],
                "M2C_club_mechanics": _w["M2C_club_mechanics"],
                "M2C_club_mechanics_highlights": _w["M2C_club_mechanics_highlights"],
                "M2C_wrist": _w["M2C_wrist"],
            }

            # Opposite side: identical metric set, suffixed with the side name (e.g. *_left when primary = right).
            sfx = "_" + _name[g_other]
            g_ref_o = None
            try:
                g_ref_o = _peak_speed_frame(g_other)
                _wo = _limb_widgets(g_other, 0, g_n - 1, g_ref_o, 0, g_n)
                for _k in ("M2C_power", "M2C_club_mechanics", "M2C_club_mechanics_highlights", "M2C_wrist"):
                    generic_outputs[_k + sfx] = _wo[_k]
            except Exception as ex_other:
                generic_outputs["M2C_other_side_error"] = f"{type(ex_other).__name__}: {ex_other}"

            generic_outputs["generic_metric_equivalents"] = dict(
                _WIDGET_EQUIVALENTS,
                primary_side={"side": _name[g_side], "selected_by": g_sel},
                other_side={
                    "side": _name[g_other],
                    "keys_suffix": sfx,
                    "note": "same metrics as the primary side; each arm is referenced to its own peak wrist speed",
                },
                time_to_impact={
                    "meaning": "ms relative to the reference event (peak wrist speed of that arm), negative = before it",
                    "reference_frame": {_name[g_side]: g_ref, _name[g_other]: g_ref_o},
                    "reference_time_s": {_name[g_side]: g_ref * dt,
                                         _name[g_other]: None if g_ref_o is None else g_ref_o * dt},
                },
                analysis_window="whole clip",
            )
        except Exception as ex_limb:
            generic_outputs = {"generic_widgets_error": f"{type(ex_limb).__name__}: {ex_limb}"}

    #####################################################################
    # V7: GENERIC (pose-only) METRICS - no p-positions, bat or club needed
    #####################################################################

    def _build_generic():
        G = {"meta": {}, "time_s": None, "metrics": {}, "summary": {}, "scalars": {}, "warnings": []}
        raw = {
            "pelvis": pelvis, "hip_l": hip_l, "hip_r": hip_r, "knee_l": knee_l, "knee_r": knee_r,
            "ankle_l": ankle_l, "ankle_r": ankle_r, "foot_l": foot_l, "foot_r": foot_r,
            "midfoot_l": midfoot_l, "midfoot_r": midfoot_r, "nose": nose,
            "spine_low": spine_low, "spine_high": spine_high,
            "shoulder_l": shoulder_l, "shoulder_r": shoulder_r, "elbow_l": elbow_l, "elbow_r": elbow_r,
            "wrist_l": wrist_l, "wrist_r": wrist_r, "hand_l": hand_l, "hand_r": hand_r,
            "neck": neck, "skullbase": skullbase, "skull": skull, "ear_l": ear_l, "ear_r": ear_r,
        }
        raw = {k: np.asarray(v, dtype=float) for k, v in raw.items()}
        n_ = raw["pelvis"].shape[0]
        missing = sorted(generic_missing_joints) if generic_missing_joints else sorted(k for k, v in raw.items() if not np.isfinite(v).all())
        J = {k: _fill_nan(v) for k, v in raw.items()}
        up = np.array([0.0, 0.0, 1.0])

        G["time_s"] = np.arange(n_) * dt
        G["meta"] = {
            "n_frames": int(n_), "frame_rate": float(frame_rate), "duration_s": float((n_ - 1) * dt),
            "movement_type": movement_type, "mass_kg": float(W), "mass_assumed": bool(mass_assumed),
            "joints_with_missing_values": missing,
            "coordinates": "metres, z = up (same transformed frame as the rest of the output)",
            "side_convention": "_l / _r follow the script's existing joint naming (right-first ordering unless handedness=lefthanded)",
            "notes": [
                "All derivatives use a Savitzky-Golay filter (~0.15 s window).",
                "Energy terms are translational segment-COM only (rotational KE ignored).",
                "GRF / power terms are COM-based estimates (no force plate).",
            ] + (["No body weight in config: forces/energies/power are PER KG (mass = 1)."] if mass_assumed else []),
        }
        if n_ < 2:
            G["warnings"].append("fewer than 2 frames - no metrics computed")
            return G

        def add(group, name, arr, unit):
            arr = np.asarray(arr, dtype=float)
            G["metrics"].setdefault(group, {})[name] = arr
            G["summary"].setdefault(group, {})[name] = _stats(arr, unit)

        def add_scalar(group, name, val, unit=None):
            G["scalars"].setdefault(group, {})[name] = {"value": val, "unit": unit}

        def run(group, fn):
            try:
                with np.errstate(all="ignore"):
                    fn()
            except Exception as ex:  # never let one group break the rest
                G["warnings"].append(f"{group}: {type(ex).__name__}: {ex}")

        trunk_up = J["neck"] - J["pelvis"]

        # ---------------- joint angles ----------------
        def g_angles():
            gname = "joint_angles_deg"
            for s_ in ("l", "r"):
                add(gname, f"elbow_flexion_{s_}", 180 - _ang3(J[f"shoulder_{s_}"], J[f"elbow_{s_}"], J[f"wrist_{s_}"]), "deg")
                add(gname, f"shoulder_elevation_{s_}", _ang_between(J[f"elbow_{s_}"] - J[f"shoulder_{s_}"], J["pelvis"] - J["neck"]), "deg")
                add(gname, f"wrist_angle_{s_}", 180 - _ang3(J[f"elbow_{s_}"], J[f"wrist_{s_}"], J[f"hand_{s_}"]), "deg")
                add(gname, f"hip_flexion_{s_}", 180 - _ang_between(trunk_up, J[f"knee_{s_}"] - J[f"hip_{s_}"]), "deg")
                add(gname, f"knee_flexion_{s_}", 180 - _ang3(J[f"hip_{s_}"], J[f"knee_{s_}"], J[f"ankle_{s_}"]), "deg")
                add(gname, f"ankle_angle_{s_}", _ang3(J[f"knee_{s_}"], J[f"ankle_{s_}"], J[f"foot_{s_}"]), "deg (interior)")
            add(gname, "spine_curvature", 180 - _ang3(J["spine_low"], J["spine_high"], J["neck"]), "deg")
            add(gname, "neck_flexion", 180 - _ang3(J["spine_high"], J["neck"], J["skullbase"]), "deg")
            add(gname, "head_tilt", 180 - _ang3(J["neck"], J["skullbase"], J["skull"]), "deg")
        run("joint_angles_deg", g_angles)

        # ---------------- segment orientation / rotations ----------------
        def g_orient():
            gname = "segment_orientation_deg"

            def turn(l_, r_):
                return np.degrees(np.unwrap(-np.arctan2(l_[:, 0] - r_[:, 0], l_[:, 1] - r_[:, 1])))

            pel = turn(J["hip_l"], J["hip_r"])
            tho = turn(J["shoulder_l"], J["shoulder_r"])
            add(gname, "pelvis_rotation", pel, "deg")
            add(gname, "thorax_rotation", tho, "deg")
            add(gname, "head_rotation", turn(J["ear_l"], J["ear_r"]), "deg")
            add(gname, "x_factor_signed", tho - pel, "deg")
            add(gname, "x_factor", np.abs(tho - pel), "deg")
            add(gname, "trunk_tilt_from_vertical", _ang_between(trunk_up, np.tile(up, (n_, 1))), "deg")
            lat = J["hip_l"] - J["hip_r"]
            lat[:, 2] = 0
            lat = safe_normalize(lat)
            fwd = np.cross(lat, up)
            add(gname, "trunk_forward_lean", np.degrees(np.arctan2(np.einsum("ij,ij->i", trunk_up, fwd), trunk_up[:, 2])), "deg (+ = forward of pelvis heading)")
            add(gname, "trunk_lateral_bend", np.degrees(np.arctan2(np.einsum("ij,ij->i", trunk_up, lat), trunk_up[:, 2])), "deg")
            for nm, a_, b_ in (("shoulder_line_tilt", "shoulder_l", "shoulder_r"), ("hip_line_tilt", "hip_l", "hip_r")):
                v_ = J[a_] - J[b_]
                add(gname, nm, np.degrees(np.arcsin(np.clip(v_[:, 2] / np.maximum(np.linalg.norm(v_, axis=1), 1e-9), -1, 1))), "deg")
        run("segment_orientation_deg", g_orient)

        # ---------------- angular velocities of the above ----------------
        def g_angvel():
            for grp in ("joint_angles_deg", "segment_orientation_deg"):
                for nm, ser in list(G["metrics"].get(grp, {}).items()):
                    add("angular_velocity_deg_s", f"{nm}_vel", _deriv(ser, 1), "deg/s")
        run("angular_velocity_deg_s", g_angvel)

        # ---------------- kinematic sequence (rotation about the spine axis) ----------------
        def g_kseq():
            pairs = {
                "pelvis": ("hip_l", "hip_r"), "thorax": ("shoulder_l", "shoulder_r"),
                "arm_l": ("shoulder_l", "elbow_l"), "forearm_l": ("elbow_l", "wrist_l"),
                "arm_r": ("shoulder_r", "elbow_r"), "forearm_r": ("elbow_r", "wrist_r"),
            }
            peaks = {}
            for nm, (pj, dj) in pairs.items():
                w_ = rotational_velocity_about_spine(J[pj], J[dj], J["spine_high"], J["spine_low"], dt, filter_win=41)
                add("kinematic_sequence_deg_s", nm, w_, "deg/s")
                ip = int(np.argmax(np.abs(w_)))
                peaks[nm] = ip
                add_scalar("kinematic_sequence", f"{nm}_peak_abs", float(abs(w_[ip])), "deg/s")
                add_scalar("kinematic_sequence", f"{nm}_peak_time", float(ip * dt), "s")
            add_scalar("kinematic_sequence", "peak_order", sorted(peaks, key=peaks.get), None)
        run("kinematic_sequence_deg_s", g_kseq)

        # ---------------- joint speeds / accelerations ----------------
        def g_speed():
            for k in ("pelvis", "spine_high", "neck", "skull", "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
                      "wrist_l", "wrist_r", "hand_l", "hand_r", "hip_l", "hip_r", "knee_l", "knee_r",
                      "ankle_l", "ankle_r", "foot_l", "foot_r"):
                add("joint_speed_mps", k, np.linalg.norm(_deriv(J[k], 1), axis=1), "m/s")
            for k in ("pelvis", "skull", "wrist_l", "wrist_r", "hand_l", "hand_r", "ankle_l", "ankle_r", "foot_l", "foot_r"):
                add("joint_acceleration_mps2", k, np.linalg.norm(_deriv(J[k], 2), axis=1), "m/s^2")
        run("joint_speed_mps", g_speed)

        # ---------------- centre of mass + GRF estimate + energy ----------------
        COM, segs_g = _body_com(J)
        COM_s = np.column_stack([_lp(COM[:, 0], 3), _lp(COM[:, 1], 3), _lp(COM[:, 2], 5)])

        def g_com():
            vel = np.gradient(COM_s - COM_s[0], dt, axis=0)
            acc = np.gradient(vel, dt, axis=0)
            d_ = 100 * (COM_s - COM_s[0])
            add("com", "com_thrust_x", d_[:, 0], "cm (rel. frame 0)")
            add("com", "com_sway_y", d_[:, 1], "cm (rel. frame 0)")
            add("com", "com_lift_z", d_[:, 2], "cm (rel. frame 0)")
            add("com", "com_speed", np.linalg.norm(vel, axis=1), "m/s")
            add("com", "com_horizontal_speed", np.linalg.norm(vel[:, :2], axis=1), "m/s")
            add("com", "com_vertical_velocity", vel[:, 2], "m/s")
            add("com", "com_acceleration", np.linalg.norm(acc, axis=1), "m/s^2")
            add_scalar("com", "path_length", float(np.sum(np.linalg.norm(np.diff(COM_s, axis=0), axis=1))), "m")
            for i_, ax_ in enumerate("xyz"):
                add_scalar("com", f"range_{ax_}", float(100 * np.ptp(COM_s[:, i_])), "cm")

            Fx_, Fy_ = W * acc[:, 0], W * acc[:, 1]
            Fz_ = W * (acc[:, 2] + 9.81)
            Fz_[Fz_ < 0] = 0
            Fr_ = np.sqrt(Fx_ ** 2 + Fy_ ** 2 + Fz_ ** 2)
            bw = W * 9.81
            add("grf_estimate", "Fx", Fx_, "N" if not mass_assumed else "N/kg")
            add("grf_estimate", "Fy", Fy_, "N" if not mass_assumed else "N/kg")
            add("grf_estimate", "Fz", Fz_, "N" if not mass_assumed else "N/kg")
            add("grf_estimate", "F_resultant", Fr_, "N" if not mass_assumed else "N/kg")
            add("grf_estimate", "Fz_body_weights", Fz_ / bw, "BW")
            add("grf_estimate", "horizontal_force_body_weights", np.sqrt(Fx_ ** 2 + Fy_ ** 2) / bw, "BW")
            add("grf_estimate", "vertical_power", Fx_ * vel[:, 0] + Fy_ * vel[:, 1] + Fz_ * vel[:, 2], "W")
            add("grf_estimate", "rate_of_force_development", _deriv(Fz_, 1), "N/s")
            add_scalar("grf_estimate", "vertical_impulse", float(np.sum(Fz_) * dt), "N*s")
        run("com", g_com)

        def g_energy():
            ke = {}
            for s_ in segs_g:
                v_ = _deriv(_seg_com(s_), 1)
                ke[s_[0]] = 0.5 * W * s_[4] * np.sum(v_ ** 2, axis=1)
            add("energy_J", "trunk_KE", ke["trunk"], "J")
            add("energy_J", "head_KE", ke["head"], "J")
            for s_ in ("l", "r"):
                add("energy_J", f"upper_limb_KE_{s_}", ke[f"arm_{s_}"] + ke[f"forearm_{s_}"] + ke[f"hand_{s_}"], "J")
                add("energy_J", f"lower_limb_KE_{s_}", ke[f"thigh_{s_}"] + ke[f"shank_{s_}"] + ke[f"foot_{s_}"], "J")
            tot = sum(ke.values())
            pe = W * 9.81 * (COM_s[:, 2] - np.min(COM_s[:, 2]))
            add("energy_J", "total_body_KE", tot, "J")
            add("energy_J", "com_PE", pe, "J")
            add("energy_J", "mechanical_energy", tot + pe, "J")
            add("energy_J", "mechanical_power", _deriv(tot + pe, 1), "W")
        run("energy_J", g_energy)

        # ---------------- balance / posture ----------------
        def g_balance():
            g = "balance_posture"
            ground = min(J["foot_l"][:, 2].min(), J["foot_r"][:, 2].min())
            hz = lambda a_, b_: np.linalg.norm((J[a_] - J[b_])[:, :2], axis=1)
            add(g, "stance_width", hz("ankle_l", "ankle_r"), "m")
            add(g, "foot_separation_3d", np.linalg.norm(J["midfoot_l"] - J["midfoot_r"], axis=1), "m")
            add(g, "shoulder_width", np.linalg.norm(J["shoulder_l"] - J["shoulder_r"], axis=1), "m")
            add(g, "hip_width", np.linalg.norm(J["hip_l"] - J["hip_r"], axis=1), "m")
            add(g, "hands_separation", np.linalg.norm(J["hand_l"] - J["hand_r"], axis=1), "m")
            mid_hand_g = (J["hand_l"] + J["hand_r"]) / 2
            add(g, "hands_to_chest_distance", np.linalg.norm(mid_hand_g - J["spine_high"], axis=1), "m")
            add(g, "hands_height_above_shoulders", mid_hand_g[:, 2] - (J["shoulder_l"][:, 2] + J["shoulder_r"][:, 2]) / 2, "m")
            add(g, "foot_height_l", J["foot_l"][:, 2] - ground, "m (above lowest foot point)")
            add(g, "foot_height_r", J["foot_r"][:, 2] - ground, "m (above lowest foot point)")
            add(g, "pelvis_height", J["pelvis"][:, 2] - ground, "m")
            add(g, "head_height", J["skull"][:, 2] - ground, "m")
            add(g, "com_height", COM_s[:, 2] - ground, "m")
            bos = (J["midfoot_l"] + J["midfoot_r"]) / 2
            off = COM_s[:, :2] - bos[:, :2]
            add(g, "com_bos_offset_x", off[:, 0], "m")
            add(g, "com_bos_offset_y", off[:, 1], "m")
            add(g, "com_bos_offset", np.linalg.norm(off, axis=1), "m")
            hd = (J["skull"] - J["pelvis"])
            hd = 100 * (hd - hd[0])
            add(g, "head_displacement_x", hd[:, 0], "cm (rel. pelvis, rel. frame 0)")
            add(g, "head_displacement_y", hd[:, 1], "cm (rel. pelvis, rel. frame 0)")
            add(g, "head_displacement_z", hd[:, 2], "cm (rel. pelvis, rel. frame 0)")
        run("balance_posture", g_balance)

        # ---------------- symmetry ----------------
        def g_sym():
            def si(l_, r_):
                return float(200 * (l_ - r_) / (l_ + r_)) if (l_ + r_) not in (0, 0.0) else None
            for grp, base in (("joint_angles_deg", "elbow_flexion"), ("joint_angles_deg", "knee_flexion"),
                              ("joint_angles_deg", "hip_flexion"), ("joint_angles_deg", "shoulder_elevation"),
                              ("joint_speed_mps", "wrist"), ("joint_speed_mps", "elbow"), ("joint_speed_mps", "ankle"),
                              ("joint_speed_mps", "knee"), ("joint_speed_mps", "hand")):
                gm = G["metrics"].get(grp, {})
                if f"{base}_l" not in gm or f"{base}_r" not in gm:
                    continue
                l_, r_ = float(np.nanmax(gm[f"{base}_l"])), float(np.nanmax(gm[f"{base}_r"]))
                rl, rr = float(np.ptp(gm[f"{base}_l"])), float(np.ptp(gm[f"{base}_r"]))
                add_scalar("symmetry", f"{base}_peak_l", l_)
                add_scalar("symmetry", f"{base}_peak_r", r_)
                add_scalar("symmetry", f"{base}_peak_symmetry_index", si(l_, r_), "% (+ = left larger)")
                add_scalar("symmetry", f"{base}_range_symmetry_index", si(rl, rr), "% (+ = left larger)")
        run("symmetry", g_sym)

        # ---------------- smoothness / path ----------------
        def g_smooth():
            T_ = (n_ - 1) * dt
            for nm, pos in (("hand_l", J["hand_l"]), ("hand_r", J["hand_r"]), ("wrist_l", J["wrist_l"]),
                            ("wrist_r", J["wrist_r"]), ("com", COM_s)):
                L_ = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
                sp = np.linalg.norm(_deriv(pos, 1), axis=1)
                jk = _deriv(pos, 3)
                dj = float(np.sum(np.sum(jk ** 2, axis=1)) * dt * T_ ** 5 / L_ ** 2) if L_ > 1e-6 else None
                add_scalar("smoothness", f"{nm}_path_length", L_, "m")
                add_scalar("smoothness", f"{nm}_mean_speed", float(np.mean(sp)), "m/s")
                add_scalar("smoothness", f"{nm}_peak_speed", float(np.max(sp)), "m/s")
                add_scalar("smoothness", f"{nm}_dimensionless_jerk", dj, "-  (lower = smoother; noise sensitive)")
            hands = np.vstack([J["hand_l"], J["hand_r"]])
            add_scalar("smoothness", "hand_workspace_bbox_volume", float(np.prod(np.ptp(hands, axis=0))), "m^3")
        run("smoothness", g_smooth)

        # ---------------- segment lengths (anthropometry + tracking QC) ----------------
        def g_lengths():
            defs = {
                "thigh_l": ("hip_l", "knee_l"), "thigh_r": ("hip_r", "knee_r"),
                "shank_l": ("knee_l", "ankle_l"), "shank_r": ("knee_r", "ankle_r"),
                "upper_arm_l": ("shoulder_l", "elbow_l"), "upper_arm_r": ("shoulder_r", "elbow_r"),
                "forearm_l": ("elbow_l", "wrist_l"), "forearm_r": ("elbow_r", "wrist_r"),
                "trunk": ("pelvis", "neck"), "shoulder_width": ("shoulder_l", "shoulder_r"),
                "hip_width": ("hip_l", "hip_r"),
            }
            for nm, (a_, b_) in defs.items():
                d_ = np.linalg.norm(J[a_] - J[b_], axis=1)
                add_scalar("segment_lengths", f"{nm}_mean", float(np.mean(d_)), "m")
                add_scalar("segment_lengths", f"{nm}_cv", float(100 * np.std(d_) / max(np.mean(d_), 1e-9)), "% (tracking consistency)")
        run("segment_lengths", g_lengths)

        return _clean(G)

    try:
        generic = _build_generic()
    except Exception as ex_generic:
        generic = {"error": f"{type(ex_generic).__name__}: {ex_generic}"}

    if swing_type == 0:

        return {
            "movement_type": movement_type,
            "M2C_insights": M2C_insights,
            "Performance_metrics": Performance_metrics,
            #"Pitcher_performance_metrics": Pitcher_performance_metrics,
            "M2C_power": M2C_power,
            "M2C_club_mechanics": M2C_club_mechanics,
            "M2C_club_mechanics_highlights": M2C_club_mechanics_highlights,
            "M2C_force_vectors_bat_3d": M2C_force_vectors_shaft_3d,
            "M2C_wrist": M2C_wrist,
            "inPlaneMetrics": inPlaneMetrics,
            "outOfPlaneMetrics": outOfPlaneMetrics,
            "M2C_inplane_vectors": M2C_inplane_vectors,
            "M2C_inplane_vectors_3d": M2C_inplane_vectors_3d,
            "M2C_kinetics_3d": M2C_kinetics_3d,
            "M2C_skeleton_3d": M2C_skeleton_3d,
            "generic": generic,
        }

    elif swing_type == 1:

        return {
            "movement_type": movement_type,
            "M2C_insights": pitcher_outputs["M2C_insights"],
            "Performance_metrics": Pitcher_performance_metrics,
            "M2C_power": pitcher_outputs["M2C_power"],
            "M2C_club_mechanics": pitcher_outputs["M2C_club_mechanics"],
            "M2C_club_mechanics_highlights": pitcher_outputs["M2C_club_mechanics_highlights"],
            "M2C_wrist": pitcher_outputs["M2C_wrist"],
            "M2C_force_vectors_hand_r_3d": M2C_force_vectors_hand_3d,  # title misleading but handles both left and right handed pitchers
            "M2C_skeleton_3d": M2C_skeleton_3d,
            "pitcher_metric_equivalents": pitcher_outputs["pitcher_metric_equivalents"],
            "generic": generic,
        }

    else:

        return {
            "movement_type": movement_type,
            **_clean(generic_outputs),
            "generic": generic,
            "M2C_skeleton_3d": _clean(M2C_skeleton_3d),
        }


json_path_inputs = "inputs_baseball/"
json_path_outputs = "outputs_baseball/"

json_file = 'm2c-dc1-11168-1779492668-pitcher'

# Opening TRACKING - JSON file6
f = open(json_path_inputs + json_file + '.json')

# returns JSON object as
# a dictionary
json3d = json.load(f)

# Closing file
f.close()



####################################### Generating output json file  ##########################################################
response = clubFitting(json3d)

f = open(json_path_outputs + json_file + '_performance_and_hand_vectors_v9.json', "w")
f.write(json.dumps(response, default=numpy_int_parser))
f.close()

