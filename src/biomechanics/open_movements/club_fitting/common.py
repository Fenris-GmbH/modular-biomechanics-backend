from typing import Any
from .math_helpers import *

def calculate(state: dict[str, Any]) -> dict[str, Any]:
    """Compute shared movement classification, context, and common metrics."""
    H = state.get('H')
    I = state.get('I')
    M = state.get('M')
    M_new = state.get('M_new')
    Q = state.get('Q')
    UT_turn = state.get('UT_turn')
    UT_turn_deg = state.get('UT_turn_deg')
    _W_cfg = state.get('_W_cfg')
    _cfg_swing = state.get('_cfg_swing')
    _interp_nan = state.get('_interp_nan')
    _swing_type_str = state.get('_swing_type_str')
    a = state.get('a')
    ankle_l = state.get('ankle_l')
    ankle_r = state.get('ankle_r')
    arc_width = state.get('arc_width')
    bad = state.get('bad')
    c = state.get('c')
    cm_to_inch_factor = state.get('cm_to_inch_factor')
    com = state.get('com')
    d2H = state.get('d2H')
    d_I = state.get('d_I')
    downtheLine = state.get('downtheLine')
    ear_l = state.get('ear_l')
    ear_r = state.get('ear_r')
    elbow_l = state.get('elbow_l')
    elbow_r = state.get('elbow_r')
    foot_l = state.get('foot_l')
    foot_r = state.get('foot_r')
    generic_missing_joints = state.get('generic_missing_joints')
    hand_l = state.get('hand_l')
    hand_r = state.get('hand_r')
    hip_l = state.get('hip_l')
    hip_r = state.get('hip_r')
    i = state.get('i')
    idx = state.get('idx')
    knee_l = state.get('knee_l')
    knee_r = state.get('knee_r')
    la = state.get('la')
    lefty = state.get('lefty')
    mass_assumed = state.get('mass_assumed')
    mid_hand = state.get('mid_hand')
    midfoot_l = state.get('midfoot_l')
    midfoot_r = state.get('midfoot_r')
    movement_type = state.get('movement_type')
    n = state.get('n')
    nb_pts = state.get('nb_pts')
    neck = state.get('neck')
    new_x = state.get('new_x')
    new_y = state.get('new_y')
    nose = state.get('nose')
    p4 = state.get('p4')
    p7 = state.get('p7')
    pelvis = state.get('pelvis')
    pelvis_turn = state.get('pelvis_turn')
    pelvis_turn_deg = state.get('pelvis_turn_deg')
    plane_v = state.get('plane_v')
    point_cloud = state.get('point_cloud')
    q = state.get('q')
    shoulder_l = state.get('shoulder_l')
    shoulder_r = state.get('shoulder_r')
    skull = state.get('skull')
    skullbase = state.get('skullbase')
    spine_high = state.get('spine_high')
    spine_low = state.get('spine_low')
    t_H = state.get('t_H')
    temp = state.get('temp')
    vectors = state.get('vectors')
    wrist_l = state.get('wrist_l')
    wrist_r = state.get('wrist_r')
    x_factor = state.get('x_factor')
    json3d = state["json3d"]
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
    pelvis_turn = []
    for idx in range(0, len(hip_l)):
        pelvis_turn.append(
            -math.atan2(hip_l[idx, 0] - hip_r[idx, 0], hip_l[idx, 1] - hip_r[idx, 1])
        )
    pelvis_turn_deg = np.array(pelvis_turn) * (360 / (2 * np.pi))
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
    swing_type = {"hitting": 0, "pitching": 1, "generic": 2}[movement_type]
    out = locals().copy()
    out.pop("state", None)
    return out
