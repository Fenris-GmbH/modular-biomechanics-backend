from typing import Any

from .math_helpers import *

def calculate(state: dict[str, Any]) -> dict[str, Any]:
    """Compute generic_metrics stage with cached dependencies."""
    COM = state.get('COM')
    COM_s = state.get('COM_s')
    Fr_ = state.get('Fr_')
    Fx_ = state.get('Fx_')
    Fy_ = state.get('Fy_')
    Fz_ = state.get('Fz_')
    G = state.get('G')
    J = state.get('J')
    L_ = state.get('L_')
    T_ = state.get('T_')
    W = state.get('W')
    _ang3 = state.get('_ang3')
    _ang_between = state.get('_ang_between')
    _body_com = state.get('_body_com')
    _build_generic = state.get('_build_generic')
    _clean = state.get('_clean')
    _deriv = state.get('_deriv')
    _fill_nan = state.get('_fill_nan')
    _lp = state.get('_lp')
    _seg_com = state.get('_seg_com')
    _stats = state.get('_stats')
    a_ = state.get('a_')
    acc = state.get('acc')
    add = state.get('add')
    add_scalar = state.get('add_scalar')
    ankle_l = state.get('ankle_l')
    ankle_r = state.get('ankle_r')
    arr = state.get('arr')
    ax_ = state.get('ax_')
    b_ = state.get('b_')
    base = state.get('base')
    bos = state.get('bos')
    bw = state.get('bw')
    d_ = state.get('d_')
    defs = state.get('defs')
    dj = state.get('dj')
    dt = state.get('dt')
    ear_l = state.get('ear_l')
    ear_r = state.get('ear_r')
    elbow_l = state.get('elbow_l')
    elbow_r = state.get('elbow_r')
    ex = state.get('ex')
    ex_generic = state.get('ex_generic')
    fn = state.get('fn')
    foot_l = state.get('foot_l')
    foot_r = state.get('foot_r')
    frame_rate = state.get('frame_rate')
    fwd = state.get('fwd')
    g = state.get('g')
    g_angles = state.get('g_angles')
    g_angvel = state.get('g_angvel')
    g_balance = state.get('g_balance')
    g_com = state.get('g_com')
    g_energy = state.get('g_energy')
    g_kseq = state.get('g_kseq')
    g_lengths = state.get('g_lengths')
    g_orient = state.get('g_orient')
    g_smooth = state.get('g_smooth')
    g_speed = state.get('g_speed')
    g_sym = state.get('g_sym')
    generic_missing_joints = state.get('generic_missing_joints')
    gm = state.get('gm')
    gname = state.get('gname')
    ground = state.get('ground')
    group = state.get('group')
    grp = state.get('grp')
    hand_l = state.get('hand_l')
    hand_r = state.get('hand_r')
    hands = state.get('hands')
    hd = state.get('hd')
    hip_l = state.get('hip_l')
    hip_r = state.get('hip_r')
    hz = state.get('hz')
    i_ = state.get('i_')
    ip = state.get('ip')
    jk = state.get('jk')
    k = state.get('k')
    ke = state.get('ke')
    knee_l = state.get('knee_l')
    knee_r = state.get('knee_r')
    l_ = state.get('l_')
    lat = state.get('lat')
    mass_assumed = state.get('mass_assumed')
    mid_hand_g = state.get('mid_hand_g')
    midfoot_l = state.get('midfoot_l')
    midfoot_r = state.get('midfoot_r')
    missing = state.get('missing')
    movement_type = state.get('movement_type')
    n_ = state.get('n_')
    name = state.get('name')
    neck = state.get('neck')
    nm = state.get('nm')
    nose = state.get('nose')
    off = state.get('off')
    pairs = state.get('pairs')
    pe = state.get('pe')
    peaks = state.get('peaks')
    pel = state.get('pel')
    pelvis = state.get('pelvis')
    pj = state.get('pj')
    pos = state.get('pos')
    r_ = state.get('r_')
    raw = state.get('raw')
    rl = state.get('rl')
    rotational_velocity_about_spine = state.get('rotational_velocity_about_spine')
    rr = state.get('rr')
    run = state.get('run')
    s_ = state.get('s_')
    segs_g = state.get('segs_g')
    ser = state.get('ser')
    shoulder_l = state.get('shoulder_l')
    shoulder_r = state.get('shoulder_r')
    si = state.get('si')
    skull = state.get('skull')
    skullbase = state.get('skullbase')
    sp = state.get('sp')
    spine_high = state.get('spine_high')
    spine_low = state.get('spine_low')
    tho = state.get('tho')
    tot = state.get('tot')
    trunk_up = state.get('trunk_up')
    turn = state.get('turn')
    unit = state.get('unit')
    up = state.get('up')
    v = state.get('v')
    v_ = state.get('v_')
    val = state.get('val')
    vel = state.get('vel')
    w_ = state.get('w_')
    wrist_l = state.get('wrist_l')
    wrist_r = state.get('wrist_r')
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
    out = locals().copy()
    out.pop("state", None)
    return out
