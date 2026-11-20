from typing import Any

from .math_helpers import *

def calculate(state: dict[str, Any]) -> dict[str, Any]:
    """Compute helpers stage with cached dependencies."""
    COM_p = state.get('COM_p')
    COM_p_acc = state.get('COM_p_acc')
    COM_p_vel = state.get('COM_p_vel')
    COM_pf = state.get('COM_pf')
    Elbow_Force = state.get('Elbow_Force')
    Elbow_torque = state.get('Elbow_torque')
    F_axial_ = state.get('F_axial_')
    F_elbow_vec = state.get('F_elbow_vec')
    F_hand_mag = state.get('F_hand_mag')
    F_hand_vec_ = state.get('F_hand_vec_')
    F_tan_ = state.get('F_tan_')
    Fx_p = state.get('Fx_p')
    Fy_p = state.get('Fy_p')
    Fz_p = state.get('Fz_p')
    I_comp = state.get('I_comp')
    J = state.get('J')
    J_raw = state.get('J_raw')
    M2C_club_mechanics_w = state.get('M2C_club_mechanics_w')
    M2C_power_w = state.get('M2C_power_w')
    M2C_wrist_w = state.get('M2C_wrist_w')
    M_elbow_vec = state.get('M_elbow_vec')
    W = state.get('W')
    _WIDGET_EQUIVALENTS = state.get('_WIDGET_EQUIVALENTS')
    _ang3 = state.get('_ang3')
    _ang_between = state.get('_ang_between')
    _body_com = state.get('_body_com')
    _clean = state.get('_clean')
    _limb_widgets = state.get('_limb_widgets')
    _lp = state.get('_lp')
    _peak_in_window = state.get('_peak_in_window')
    _seg_com = state.get('_seg_com')
    _segments = state.get('_segments')
    _w = state.get('_w')
    a = state.get('a')
    a_ = state.get('a_')
    a_comp = state.get('a_comp')
    ankle_l = state.get('ankle_l')
    ankle_r = state.get('ankle_r')
    arc_width = state.get('arc_width')
    arc_width_inch = state.get('arc_width_inch')
    arm_power_p = state.get('arm_power_p')
    arr = state.get('arr')
    ax_dir = state.get('ax_dir')
    b = state.get('b')
    b_ = state.get('b_')
    bad = state.get('bad')
    c = state.get('c')
    c_ = state.get('c_')
    c_comp = state.get('c_comp')
    c_f = state.get('c_f')
    c_fore = state.get('c_fore')
    c_hand = state.get('c_hand')
    col = state.get('col')
    distal_power_p = state.get('distal_power_p')
    dt = state.get('dt')
    e_ax = state.get('e_ax')
    e_dd = state.get('e_dd')
    el_ = state.get('el_')
    elb_f = state.get('elb_f')
    elbow_flexion_t = state.get('elbow_flexion_t')
    elbow_l = state.get('elbow_l')
    elbow_r = state.get('elbow_r')
    fc = state.get('fc')
    fc_arm = state.get('fc_arm')
    foot_l = state.get('foot_l')
    foot_r = state.get('foot_r')
    frame_rate = state.get('frame_rate')
    han_f = state.get('han_f')
    hand_acc_f = state.get('hand_acc_f')
    hand_f_ = state.get('hand_f_')
    hand_l = state.get('hand_l')
    hand_power_p = state.get('hand_power_p')
    hand_r = state.get('hand_r')
    hand_vel_f = state.get('hand_vel_f')
    hand_vel_raw = state.get('hand_vel_raw')
    hd_ = state.get('hd_')
    highlights_w = state.get('highlights_w')
    hip_l = state.get('hip_l')
    hip_r = state.get('hip_r')
    i_ = state.get('i_')
    i_first = state.get('i_first')
    i_last = state.get('i_last')
    i_ref = state.get('i_ref')
    i_win0 = state.get('i_win0')
    i_win1 = state.get('i_win1')
    idx = state.get('idx')
    imax = state.get('imax')
    imin = state.get('imin')
    is_pitcher = state.get('movement_type') == 'pitching'
    k = state.get('k')
    knee_l = state.get('knee_l')
    knee_r = state.get('knee_r')
    lefty = state.get('lefty')
    m_comp = state.get('m_comp')
    m_fore = state.get('m_fore')
    m_hand = state.get('m_hand')
    n_ = state.get('n_')
    n_f = state.get('n_f')
    neck = state.get('neck')
    nu = state.get('nu')
    nv = state.get('nv')
    nz = state.get('nz')
    obj = state.get('obj')
    order = state.get('order')
    order_idx = state.get('order_idx')
    out_ = state.get('out_')
    p1 = state.get('p1')
    p4 = state.get('p4')
    p7 = state.get('p7')
    p8 = state.get('p8')
    p_ = state.get('p_')
    pelvis = state.get('pelvis')
    pk_axl = state.get('pk_axl')
    pk_elf = state.get('pk_elf')
    pk_elt = state.get('pk_elt')
    pk_hnd = state.get('pk_hnd')
    pk_tan = state.get('pk_tan')
    poly = state.get('poly')
    power_index_p = state.get('power_index_p')
    r_el = state.get('r_el')
    s_ = state.get('s_')
    seg = state.get('seg')
    seg_ = state.get('seg_')
    seg_len = state.get('seg_len')
    segs_ = state.get('segs_')
    segs_p = state.get('segs_p')
    sh_ = state.get('sh_')
    shoulder_l = state.get('shoulder_l')
    shoulder_r = state.get('shoulder_r')
    side = state.get('side')
    skull = state.get('skull')
    skullbase = state.get('skullbase')
    tan_dir = state.get('tan_dir')
    u = state.get('u')
    unit = state.get('unit')
    v = state.get('v')
    v_ = state.get('v_')
    vertical_power_p = state.get('vertical_power_p')
    win = state.get('win')
    wn = state.get('wn')
    wr_ = state.get('wr_')
    wri_f = state.get('wri_f')
    wrist_angle_t = state.get('wrist_angle_t')
    wrist_l = state.get('wrist_l')
    wrist_r = state.get('wrist_r')
    wrist_speed_t = state.get('wrist_speed_t')
    x = state.get('x')
    x_factor = state.get('x_factor')
    x_factor_stretch = state.get('x_factor_stretch')
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
    out = locals().copy()
    out.pop("state", None)
    return out
