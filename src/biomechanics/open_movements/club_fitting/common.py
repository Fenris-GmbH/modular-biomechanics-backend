import math
import numpy as np
from biomechanics.constants import Conversions


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


def torso_turn(hip_l, hip_r, shoulder_l, shoulder_r):
    pelvis_turn = []
    for idx in range(0, len(hip_l)):
        pelvis_turn.append(
            -math.atan2(hip_l[idx, 0] - hip_r[idx, 0], hip_l[idx, 1] - hip_r[idx, 1])
        )
    pelvis_turn_deg = np.array(pelvis_turn) * (360 / (2 * np.pi))
    UT_turn = []
    for idx in range(0, len(shoulder_r)):
        UT_turn.append(
            -math.atan2(
                shoulder_l[idx, 0] - shoulder_r[idx, 0],
                shoulder_l[idx, 1] - shoulder_r[idx, 1],
            )
        )
    UT_turn_deg = np.array(UT_turn) * (360 / (2 * np.pi))  # - target_line_deg # from rad 2 deg
    x_factor_raw = UT_turn_deg - pelvis_turn_deg
    return x_factor_raw


def turn_insight(p4, p7, x_factor_raw):
    x_factor_stretch = min(x_factor_raw[p4 : p7 + 1]) - x_factor_raw[p4]
    x_factor = abs(x_factor_raw)
    return x_factor, x_factor_stretch


def hand_arc(hand_l, hand_r, spine_high):
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
    arc_width_inch = arc_width * Conversions.CM_TO_INCH
    return arc_width, arc_width_inch


RIGHT_MARKERS = (
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
    "pose_club_main",
    "pose_club_head",
    "ear_r",
    "ear_l",
)
LEFT_MARKERS = tuple(
    name[:-2] + ("_r" if name.endswith("_l") else "_l") if name.endswith(("_l", "_r")) else name
    for name in RIGHT_MARKERS
)


def raw_pose(pose3d):
    """Unwrap capture payload axes while retaining the frame axis."""
    pose = np.asarray(pose3d, dtype=float)
    if pose.ndim != 3:
        pose = np.squeeze(pose)
    if (
        pose.ndim != 3
        or pose.shape[0] < len(RIGHT_MARKERS)
        or pose.shape[1] < 1
        or pose.shape[2] != 3
    ):
        raise ValueError("pose3d must have shape (at least 29 markers, frames, 3)")
    return pose


def orient_pose(raw_pose, lefty, down_the_line):
    pose = raw_pose.copy()
    if down_the_line:
        pose[:, :, 0] = -raw_pose[:, :, 1] if lefty else raw_pose[:, :, 1]
        pose[:, :, 1] = -raw_pose[:, :, 0]
    elif lefty:
        pose[:, :, 1] = -raw_pose[:, :, 1]
    return pose


def interpolate_marker(marker):
    result = marker.copy()
    frames = np.arange(len(result))
    for column in result.T:
        bad = ~np.isfinite(column)
        if bad.any() and not bad.all():
            column[bad] = np.interp(frames[bad], frames[~bad], column[~bad])
    return result


def body_mass(weight):
    try:
        mass = float(weight)
    except (TypeError, ValueError):
        mass = 0.0
    if not np.isfinite(mass) or mass <= 0:
        return 1.0, True
    return mass, False
