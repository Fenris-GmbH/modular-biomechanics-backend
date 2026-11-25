"""
force_screen_modules.py
=======================
Modular front end for the forces-screen response of the force-prediction engine
(force_vectors_v13_forces_screen_contactgate_lefthanded_handling.py).

Every top-level key of the forces-screen response is now its own function. Each one returns
{key: response} and, if output_dir is given, writes <key>.json (optionally <name_prefix><key>.json):

    force_vectors, weight_shift, forces, time, force_vectors_3d, values_for_COP_trajectory,
    COM_MODELS, power, energy, impulse, small_force_vectors, COM_COP_traj_back_view,
    critical_events, frame_offset

    comp = ForcesScreen(pose3d=..., body_mass=..., frame_rate=...,
                        swing_type=..., lefty=..., critical_events=...)
    forces(computation=comp); impulse(computation=comp)
    COM_MODELS(computation=comp, pose2d=...)
    compute_all(computation=comp, pose2d=...)

All core inputs are explicit. Only the legacy output wrapper argument name
json3d=None remains for positional compatibility; passing JSON is unsupported.
Legacy engine one-shot JSON adapters are retained separately.

Each stage below is generated verbatim from the v13 builder; outputs are verified identical to
ForceVectors._build_forces_screen().
"""

import json
import logging
import math
from typing import Optional
import os

import numpy as np
from biomechanics.utils.output_keys import lowercase_output_keys

try:
    from biomechanics.open_movements.forces.force_vectors_v13_forces_screen_contactgate_lefthanded_handling import (
        ForceVectors,
    )
except ImportError as _exc:  # keep the engine module name in one obvious place
    raise ImportError(
        "force_screen_modules needs the engine file "
        "'force_vectors_v13_forces_screen_contactgate_lefthanded_handling.py' next to it "
        "(or edit the import at the top of this file)."
    ) from _exc


# ==========================================================================
# STAGES (generated verbatim from the v13 builder; each stage lists its own inputs)
# ==========================================================================


def _stage_base(fv, S):
    """needs stages: -; reads: ['ctx_raw']"""
    ctx_raw = S["ctx_raw"]
    ctx = ctx_raw
    fr = float(fv.frame_rate)
    dt = 1.0 / fr
    n = int(ctx["n_frames"])
    # Left-handed player: express the analysis in the force-plate frame (180 deg about
    # vertical). Analytical metrics stay in that frame; drawing geometry (3D arrows,
    # 2D video overlays) is converted back to the tracking/video frame at the end.
    rot = bool(fv.lefty_plate_rotation and ctx.get("handedness") == "left")
    if rot:
        logging.info(
            "Left-handed player: rotating the analysis 180 deg about the vertical "
            "axis into the force-plate frame (drawing geometry stays in the tracking frame)."
        )
        ctx = fv._rotate_ctx_180(ctx)
    ev = ctx.get("events") or fv._resolve_events(
        ctx.get("positions"), ctx["markers"], n
    )
    P1, P7, P10 = ev["P1"], ev["P7"], ev["P10"]
    time = np.arange(1, P10 - P1 + 1) / fr  # measured script: range(1, nFrames)/fr
    time_index = 0  # predicted forces need no plate time-sync
    lay = fv._PLATE_LAYOUTS[fv.plate_layout]
    sfx_l, sfx_r = fv._out_side_suffix()
    return {
        "P1": P1,
        "P10": P10,
        "P7": P7,
        "ctx": ctx,
        "dt": dt,
        "fr": fr,
        "lay": lay,
        "n": n,
        "rot": rot,
        "sfx_l": sfx_l,
        "sfx_r": sfx_r,
        "time": time,
    }


def _stage_forces_arrays(fv, S):
    """needs stages: ['base']; reads: ['ctx']"""
    ctx = S["ctx"]
    # ---- reported (contact-gated) forces, output-side labelled
    f = lambda k: np.asarray(ctx[k], dtype=float)
    Fx_l, Fy_l, Fz_l = f("Fx_l"), f("Fy_l"), f("Fz_l")
    Fx_r, Fy_r, Fz_r = f("Fx_r"), f("Fy_r"), f("Fz_r")
    Fx_t, Fy_t, Fz_t = Fx_l + Fx_r, Fy_l + Fy_r, Fz_l + Fz_r
    GRF_l = np.sqrt(Fx_l**2 + Fy_l**2 + Fz_l**2)
    GRF_r = np.sqrt(Fx_r**2 + Fy_r**2 + Fz_r**2)
    GRF_t = GRF_l + GRF_r
    return {
        "Fx_l": Fx_l,
        "Fx_r": Fx_r,
        "Fx_t": Fx_t,
        "Fy_l": Fy_l,
        "Fy_r": Fy_r,
        "Fy_t": Fy_t,
        "Fz_l": Fz_l,
        "Fz_r": Fz_r,
        "Fz_t": Fz_t,
        "GRF_l": GRF_l,
        "GRF_r": GRF_r,
        "GRF_t": GRF_t,
    }


def _stage_geometry(fv, S):
    """needs stages: ['base']; reads: ['ctx', 'n', 'sfx_l', 'sfx_r']"""
    ctx = S["ctx"]
    n = S["n"]
    sfx_l = S["sfx_l"]
    sfx_r = S["sfx_r"]
    # ---- geometry in the tracking frame
    mk, seg = ctx["markers"], ctx["segment_COMs_filtered"]
    depth_shift = np.zeros(n)
    if fv.anchor_foot_depth:
        anchor = fv._b_frame(mk[f"foot_{sfx_l}"])
        depth_shift = anchor[:, 0] - anchor[0, 0]

    def pos_b(a):
        out = fv._b_frame(a).copy()
        out[:, 0] -= depth_shift
        return out

    midfoot_l_3d, midfoot_r_3d = pos_b(mk[f"midfoot_{sfx_l}"]), pos_b(
        mk[f"midfoot_{sfx_r}"]
    )
    foot_l_com, foot_r_com = pos_b(seg[f"foot_{sfx_l}"]), pos_b(seg[f"foot_{sfx_r}"])
    COM = pos_b(ctx["body_COM"])
    return {
        "COM": COM,
        "foot_l_com": foot_l_com,
        "foot_r_com": foot_r_com,
        "midfoot_l_3d": midfoot_l_3d,
        "midfoot_r_3d": midfoot_r_3d,
        "mk": mk,
    }


def _stage_cop(fv, S):
    """needs stages: ['base', 'forces_arrays', 'geometry']; reads: ['Fx_l', 'Fx_r', 'Fy_l', 'Fy_r', 'Fz_l', 'Fz_r', 'GRF_l', 'GRF_t', 'P1', 'foot_l_com', 'foot_r_com', 'midfoot_l_3d', 'midfoot_r_3d', 'n']"""
    Fx_l = S["Fx_l"]
    Fx_r = S["Fx_r"]
    Fy_l = S["Fy_l"]
    Fy_r = S["Fy_r"]
    Fz_l = S["Fz_l"]
    Fz_r = S["Fz_r"]
    GRF_l = S["GRF_l"]
    GRF_t = S["GRF_t"]
    P1 = S["P1"]
    foot_l_com = S["foot_l_com"]
    foot_r_com = S["foot_r_com"]
    midfoot_l_3d = S["midfoot_l_3d"]
    midfoot_r_3d = S["midfoot_r_3d"]
    n = S["n"]
    # ---- COP (marker frame -> plate frame)
    cop_l_m = fv._fs_cop_marker_frame(
        foot_l_com, Fx_l, Fy_l, Fz_l, float(np.min(foot_l_com[:, 2]))
    )
    cop_r_m = fv._fs_cop_marker_frame(
        foot_r_com, Fx_r, Fy_r, Fz_r, float(np.min(foot_r_com[:, 2]))
    )
    offset_x, offset_y, offset_z = fv._fs_plate_origin(midfoot_l_3d, midfoot_r_3d, P1)

    def to_plate(c):
        out = np.zeros((n, 3))
        out[:, 0] = c[:, 1] - offset_y
        out[:, 1] = c[:, 0] - offset_x
        return out

    COP_l, COP_r = to_plate(cop_l_m), to_plate(cop_r_m)
    factor_l = fv._safe_div(GRF_l, GRF_t, fill=0.5)
    COP = COP_r - (COP_r - COP_l) * factor_l[:, None]
    stance = np.sqrt(
        (COP_r[:, 0] - COP_l[:, 0]) ** 2 + (COP_r[:, 1] - COP_l[:, 1]) ** 2
    )
    return {
        "COP": COP,
        "COP_l": COP_l,
        "COP_r": COP_r,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "offset_z": offset_z,
        "stance": stance,
    }


def _stage_vertical_torque(fv, S):
    """needs stages: ['base', 'forces_arrays']; reads: ['Fx_l', 'Fx_r', 'Fy_l', 'Fy_r', 'Fz_l', 'Fz_r', 'ctx']"""
    Fx_l = S["Fx_l"]
    Fx_r = S["Fx_r"]
    Fy_l = S["Fy_l"]
    Fy_r = S["Fy_r"]
    Fz_l = S["Fz_l"]
    Fz_r = S["Fz_r"]
    ctx = S["ctx"]
    # ---- vertical torque (free moment)
    Tz_t, Tz_l, Tz_r = fv._fs_vertical_torque(
        ctx,
        np.column_stack([Fx_l, Fy_l, Fz_l]),
        np.column_stack([Fx_r, Fy_r, Fz_r]),
        Fz_l,
        Fz_r,
    )
    return {"Tz_l": Tz_l, "Tz_r": Tz_r, "Tz_t": Tz_t}


def _stage_body_torques(fv, S):
    """needs stages: ['base', 'forces_arrays', 'geometry', 'cop']; reads: ['COM', 'COP', 'Fx_t', 'Fy_t', 'Fz_t', 'offset_x', 'offset_y', 'offset_z']"""
    COM = S["COM"]
    COP = S["COP"]
    Fx_t = S["Fx_t"]
    Fy_t = S["Fy_t"]
    Fz_t = S["Fz_t"]
    offset_x = S["offset_x"]
    offset_y = S["offset_y"]
    offset_z = S["offset_z"]
    # ---- body torques about the COM + moment arms (GRF line of action vs COM)
    P0 = np.column_stack(
        [COP[:, 1] + offset_x, COP[:, 0] + offset_y, COP[:, 2] + offset_z]
    )
    d = np.column_stack([Fy_t, Fx_t, Fz_t])
    dd = np.sum(d * d, axis=1)
    ok = dd > 1e-12
    th = np.where(ok, np.sum(d * (COM - P0), axis=1) / np.where(ok, dd, 1.0), 0.0)
    proj = np.where(ok[:, None], P0 + d * th[:, None], COM)  # no force -> no moment
    x_torque = (proj[:, 1] - COM[:, 1]) * Fz_t - (
        proj[:, 2] - COM[:, 2]
    ) * Fx_t  # frontal
    y_torque = (proj[:, 0] - COM[:, 0]) * Fz_t - (
        proj[:, 2] - COM[:, 2]
    ) * Fy_t  # sagittal
    z_torque = (proj[:, 0] - COM[:, 0]) * Fx_t - (
        proj[:, 1] - COM[:, 1]
    ) * Fy_t  # horizontal

    x_arm = fv._dist_point_line_2d(
        COM[:, [1, 2]], np.column_stack([Fx_t, Fz_t]), P0[:, [1, 2]]
    )
    y_arm = fv._dist_point_line_2d(
        COM[:, [0, 2]], np.column_stack([Fy_t, Fz_t]), P0[:, [0, 2]]
    )
    z_arm = fv._dist_point_line_2d(
        COM[:, [0, 1]], np.column_stack([Fy_t, Fx_t]), P0[:, [0, 1]]
    )
    x_arm = np.where(x_torque < 0, -x_arm, x_arm) * 100.0  # signed, cm
    y_arm = np.where(y_torque < 0, -y_arm, y_arm) * 100.0
    z_arm = np.where(z_torque < 0, -z_arm, z_arm) * 100.0
    return {
        "x_arm": x_arm,
        "x_torque": x_torque,
        "y_arm": y_arm,
        "y_torque": y_torque,
        "z_arm": z_arm,
        "z_torque": z_torque,
    }


def _stage_com_vs_stance(fv, S):
    """needs stages: ['base', 'geometry', 'cop']; reads: ['COM', 'COP_l', 'COP_r', 'offset_x', 'offset_y', 'stance']"""
    COM = S["COM"]
    COP_l = S["COP_l"]
    COP_r = S["COP_r"]
    offset_x = S["offset_x"]
    offset_y = S["offset_y"]
    stance = S["stance"]
    # ---- COM position of the COM relative to the two COPs (% of half-stance)
    u = np.column_stack([COP_r[:, 1] - COP_l[:, 1], COP_r[:, 0] - COP_l[:, 0]])
    v = np.column_stack([COP_r[:, 1] + offset_x, COP_r[:, 0] + offset_y])
    uu = np.sum(u * u, axis=1)
    okc = uu > 1e-12
    thc = np.sum(u * (COM[:, [0, 1]] - v), axis=1) / np.where(okc, uu, 1.0)
    hproj = v + u * thc[:, None]
    l_pt = np.column_stack([COP_l[:, 1] + offset_x, COP_l[:, 0] + offset_y])
    COM_vs_stance = np.where(
        okc,
        100.0
        * (
            1.0
            - np.linalg.norm(l_pt - hproj, axis=1) / np.where(okc, stance / 2.0, 1.0)
        ),
        0.0,
    )
    return {"COM_vs_stance": COM_vs_stance}


def _stage_kinematics(fv, S):
    """needs stages: ['base', 'geometry', 'cop']; reads: ['COM', 'COP', 'P1', 'fr']"""
    COM = S["COM"]
    COP = S["COP"]
    P1 = S["P1"]
    fr = S["fr"]
    # ---- COP displacement / speed, COM displacement / speed
    COP_AP_disp = (COP[:, 1] - COP[P1, 1]) * 100.0
    COP_ML_disp = (COP[:, 0] - COP[P1, 0]) * 100.0
    cop_speed = (
        np.sqrt(
            np.gradient(COP[:, 0], 1 / fr) ** 2 + np.gradient(COP[:, 1], 1 / fr) ** 2
        )
        * 100.0
    )
    COP_vel = fv._bfilt(cop_speed, 5)

    x_c, y_c = fv._bfilt(COM[:, 0], 3), fv._bfilt(COM[:, 1], 3)
    z_c = fv._bfilt(COM[:, 2], 5)
    COM_disp_x, COM_disp_y, COM_disp_z = (
        (x_c - x_c[0]) * 100.0,
        (y_c - y_c[0]) * 100.0,
        (z_c - z_c[0]) * 100.0,
    )

    def back_diff(x, first):
        v_ = np.diff(x) * fr
        return np.insert(v_, 0, v_[0] if first == "copy" else 0.0)

    vx, vy, vz = back_diff(x_c, 0), back_diff(y_c, 0), back_diff(z_c, "copy")
    COM_vel = np.sqrt(vx**2 + vy**2 + vz**2) * 100.0
    return {
        "COM_disp_x": COM_disp_x,
        "COM_disp_y": COM_disp_y,
        "COM_disp_z": COM_disp_z,
        "COM_vel": COM_vel,
        "COP_AP_disp": COP_AP_disp,
        "COP_ML_disp": COP_ML_disp,
        "COP_vel": COP_vel,
        "back_diff": back_diff,
        "x_c": x_c,
        "y_c": y_c,
        "z_c": z_c,
    }


def _stage_forces(fv, S):
    """needs stages: ['base', 'forces_arrays', 'vertical_torque']; reads: ['Fx_l', 'Fx_r', 'Fx_t', 'Fy_l', 'Fy_r', 'Fy_t', 'Fz_l', 'Fz_r', 'Fz_t', 'P1', 'P10', 'P7', 'Tz_l', 'Tz_r', 'Tz_t', 'time']"""
    Fx_l = S["Fx_l"]
    Fx_r = S["Fx_r"]
    Fx_t = S["Fx_t"]
    Fy_l = S["Fy_l"]
    Fy_r = S["Fy_r"]
    Fy_t = S["Fy_t"]
    Fz_l = S["Fz_l"]
    Fz_r = S["Fz_r"]
    Fz_t = S["Fz_t"]
    P1 = S["P1"]
    P10 = S["P10"]
    P7 = S["P7"]
    Tz_l = S["Tz_l"]
    Tz_r = S["Tz_r"]
    Tz_t = S["Tz_t"]
    time = S["time"]
    # ---- forces dict (+ max / timing / time-to-impact), measured-script layout
    Fz_p1 = Fz_t[P1] if Fz_t[P1] > 0 else fv.body_mass * 9.81
    chan = {
        "Fz": (Fz_t, Fz_l, Fz_r),
        "Fx": (Fx_t, Fx_l, Fx_r),
        "Fy": (Fy_t, Fy_l, Fy_r),
        "Tz": (Tz_t, Tz_l, Tz_r),
    }
    forces, peaks = {}, {}
    for c, arrs in chan.items():
        for s, a in zip("tlr", arrs):
            forces[f"{c}_{s}"] = np.around(a, 2).tolist()
            peaks[f"{c}_{s}"] = fv._fs_peak(a, P1, P10, P7, time)
    wt = {}
    for c in ("Fz", "Fx", "Fy"):
        for s, a in zip("tlr", chan[c]):
            wt[f"{c}_{s}_wt"] = np.around(a / Fz_p1 * 100.0, 2).tolist()
    forces.update(wt)
    forces.update({f"{k}_max": v[0] for k, v in peaks.items()})
    forces.update({f"{k}_time_to_impact": v[2] for k, v in peaks.items()})
    forces.update({f"{k}_max_timing": v[1] for k, v in peaks.items()})
    return {"chan": chan, "forces": forces}


def _stage_torque_peaks(fv, S):
    """needs stages: ['base', 'body_torques']; reads: ['P1', 'P10', 'P7', 'time', 'x_arm', 'x_torque', 'y_arm', 'y_torque', 'z_arm', 'z_torque']"""
    P1 = S["P1"]
    P10 = S["P10"]
    P7 = S["P7"]
    time = S["time"]
    x_arm = S["x_arm"]
    x_torque = S["x_torque"]
    y_arm = S["y_arm"]
    y_torque = S["y_torque"]
    z_arm = S["z_arm"]
    z_torque = S["z_torque"]
    # ---- torque / moment-arm peaks
    tq = {"x_Torque": x_torque, "y_Torque": y_torque, "z_Torque": z_torque}
    ma = {"x_Moment_Arm": x_arm, "y_Moment_Arm": y_arm, "z_Moment_Arm": z_arm}
    tq_pk = {k: fv._fs_peak(v, P1, P10, P7, time) for k, v in tq.items()}
    ma_pk = {k: fv._fs_peak(v, P1, P10, P7, time) for k, v in ma.items()}
    return {"ma": ma, "ma_pk": ma_pk, "tq_pk": tq_pk}


def _stage_arrows(fv, S):
    """needs stages: ['base', 'forces_arrays', 'geometry', 'cop']; reads: ['COP', 'Fx_l', 'Fx_r', 'Fx_t', 'Fy_l', 'Fy_r', 'Fy_t', 'Fz_l', 'Fz_r', 'Fz_t', 'midfoot_l_3d', 'midfoot_r_3d', 'offset_x', 'offset_y', 'offset_z', 'rot']"""
    COP = S["COP"]
    Fx_l = S["Fx_l"]
    Fx_r = S["Fx_r"]
    Fx_t = S["Fx_t"]
    Fy_l = S["Fy_l"]
    Fy_r = S["Fy_r"]
    Fy_t = S["Fy_t"]
    Fz_l = S["Fz_l"]
    Fz_r = S["Fz_r"]
    Fz_t = S["Fz_t"]
    midfoot_l_3d = S["midfoot_l_3d"]
    midfoot_r_3d = S["midfoot_r_3d"]
    offset_x = S["offset_x"]
    offset_y = S["offset_y"]
    offset_z = S["offset_z"]
    rot = S["rot"]
    # ---- 3D arrows (force_vectors_3d)
    sc = fv.FORCE_VECTOR_3D_SCALE
    force_vectors_3d = {
        "x_start_COP_l": midfoot_l_3d[:, 0].tolist(),
        "x_end_COP_l": (midfoot_l_3d[:, 0] + Fy_l / sc).tolist(),
        "y_start_COP_l": midfoot_l_3d[:, 1].tolist(),
        "y_end_COP_l": (midfoot_l_3d[:, 1] + Fx_l / sc).tolist(),
        "z_start_COP_l": midfoot_l_3d[:, 2].tolist(),
        "z_end_COP_l": (midfoot_l_3d[:, 2] + Fz_l / sc).tolist(),
        "x_start_COP_r": midfoot_r_3d[:, 0].tolist(),
        "x_end_COP_r": (midfoot_r_3d[:, 0] + Fy_r / sc).tolist(),
        "y_start_COP_r": midfoot_r_3d[:, 1].tolist(),
        "y_end_COP_r": (midfoot_r_3d[:, 1] + Fx_r / sc).tolist(),
        "z_start_COP_r": midfoot_r_3d[:, 2].tolist(),
        "z_end_COP_r": (midfoot_r_3d[:, 2] + Fz_r / sc).tolist(),
        "x_start_COP_combined": (COP[:, 1] + offset_x).tolist(),
        "x_end_COP_combined": (COP[:, 1] + offset_x + Fy_t / sc).tolist(),
        "y_start_COP_combined": (COP[:, 0] + offset_y).tolist(),
        "y_end_COP_combined": (COP[:, 0] + offset_y + Fx_t / sc).tolist(),
        "z_start_COP_combined": (COP[:, 2] + offset_z).tolist(),
        "z_end_COP_combined": (COP[:, 2] + offset_z + Fz_t / sc).tolist(),
    }

    if rot:
        neg = lambda seq: (-np.asarray(seq, dtype=float)).tolist()
        for k in list(force_vectors_3d):
            if k.startswith(("x_", "y_")):
                force_vectors_3d[k] = neg(force_vectors_3d[k])
    return {"force_vectors_3d": force_vectors_3d}


def _stage_values_cop(fv, S):
    """needs stages: ['base', 'forces_arrays', 'geometry']; reads: ['GRF_l', 'GRF_r', 'GRF_t', 'midfoot_l_3d', 'midfoot_r_3d']"""
    GRF_l = S["GRF_l"]
    GRF_r = S["GRF_r"]
    GRF_t = S["GRF_t"]
    midfoot_l_3d = S["midfoot_l_3d"]
    midfoot_r_3d = S["midfoot_r_3d"]
    # ---- weight shift / stance
    ws_r = 100.0 * fv._safe_div(GRF_r, GRF_t, fill=0.5)
    ws_l = 100.0 * fv._safe_div(GRF_l, GRF_t, fill=0.5)
    stance_mf = np.linalg.norm(midfoot_r_3d - midfoot_l_3d, axis=1) * 100.0
    values_for_COP_trajectory = {
        "weight_shift_right": ws_r.tolist(),
        "weight_shift_left": ws_l.tolist(),
        "stance_midfeet": stance_mf.tolist(),
        "stance_midfeet_inch": (stance_mf * fv._M_TO_IN).tolist(),
    }
    return {"values_for_COP_trajectory": values_for_COP_trajectory}


def _stage_impulse(fv, S):
    """needs stages: ['base', 'forces']; reads: ['P1', 'P10', 'P7', 'chan', 'dt', 'time']"""
    P1 = S["P1"]
    P10 = S["P10"]
    P7 = S["P7"]
    chan = S["chan"]
    dt = S["dt"]
    time = S["time"]
    # ---- impulse
    I = {}
    for ax, (t_, l_, r_) in {"z": chan["Fz"], "y": chan["Fy"], "x": chan["Fx"]}.items():
        I[f"I{ax}_r"], I[f"I{ax}_l"], I[f"I{ax}_t"] = (
            np.cumsum(r_) * dt,
            np.cumsum(l_) * dt,
            np.cumsum(t_) * dt,
        )
    impulse = {k: np.around(v, 2).tolist() for k, v in I.items()}
    for ax in ("x", "y", "z"):
        for s in ("t", "r", "l"):
            pk, tm, tti = fv._fs_peak(
                I[f"I{ax}_{s}"], P1, P10, P7, time, index_timing=True
            )
            (
                impulse[f"I{ax}_{s}_max"],
                impulse[f"I{ax}_{s}_max_timing"],
                impulse[f"I{ax}_{s}_time_to_impact"],
            ) = (pk, int(tm), tti)
    # same key order as the measured script
    impulse = {
        k: impulse[k]
        for k in (
            ["Iz_r", "Iz_l", "Iz_t", "Iy_r", "Iy_l", "Iy_t", "Ix_r", "Ix_l", "Ix_t"]
            + [
                f"I{a}_{s}_{q}"
                for s in ("t", "r", "l")
                for a in ("x", "y", "z")
                for q in ("max", "max_timing", "time_to_impact")
            ]
        )
    }
    return {"impulse": impulse}


def _stage_power_energy(fv, S):
    """needs stages: ['base', 'forces_arrays', 'kinematics']; reads: ['Fx_l', 'Fx_r', 'Fx_t', 'Fy_l', 'Fy_r', 'Fy_t', 'Fz_l', 'Fz_r', 'Fz_t', 'P1', 'P10', 'P7', 'back_diff', 'dt', 'time', 'x_c', 'y_c', 'z_c']"""
    Fx_l = S["Fx_l"]
    Fx_r = S["Fx_r"]
    Fx_t = S["Fx_t"]
    Fy_l = S["Fy_l"]
    Fy_r = S["Fy_r"]
    Fy_t = S["Fy_t"]
    Fz_l = S["Fz_l"]
    Fz_r = S["Fz_r"]
    Fz_t = S["Fz_t"]
    P1 = S["P1"]
    P10 = S["P10"]
    P7 = S["P7"]
    back_diff = S["back_diff"]
    dt = S["dt"]
    time = S["time"]
    x_c = S["x_c"]
    y_c = S["y_c"]
    z_c = S["z_c"]
    # ---- power / energy (COM velocity filtered like the measured script)
    pv0 = back_diff(x_c, 0)
    pv1 = back_diff(y_c, 0)
    pv2 = back_diff(z_c, 0)
    if fv.POWER_AXIS_PAIRING_LIKE_MEASURED:
        vel = {"x": pv0, "y": pv1, "z": pv2}  # Fx*v_AP, Fy*v_lat (as measured script)
    else:
        vel = {"x": pv1, "y": pv0, "z": pv2}  # Fx*v_lat, Fy*v_AP
    Pw, En = {}, {}
    for s, (fx, fy, fz) in {
        "l": (Fx_l, Fy_l, Fz_l),
        "r": (Fx_r, Fy_r, Fz_r),
        "t": (Fx_t, Fy_t, Fz_t),
    }.items():
        Pw[f"Px_{s}"], Pw[f"Py_{s}"], Pw[f"Pz_{s}"] = (
            fx * vel["x"],
            fy * vel["y"],
            fz * vel["z"],
        )
        Pw[f"P_{s}"] = np.sqrt(
            Pw[f"Px_{s}"] ** 2 + Pw[f"Py_{s}"] ** 2 + Pw[f"Pz_{s}"] ** 2
        )
        for a in ("x", "y", "z"):
            En[f"E{a}_{s}"] = np.cumsum(Pw[f"P{a}_{s}"]) * dt
        En[f"E_{s}"] = np.sqrt(
            En[f"Ex_{s}"] ** 2 + En[f"Ey_{s}"] ** 2 + En[f"Ez_{s}"] ** 2
        )

    def peak_block(src, names):
        out = {}
        for nm in names:
            out[nm] = fv._fs_peak(
                src[nm], P1, P10, P7, time, use_abs=True, index_timing=True
            )
        return out

    p_names = [
        "P_t",
        "Px_t",
        "Py_t",
        "Pz_t",
        "Px_r",
        "Py_r",
        "Pz_r",
        "Px_l",
        "Py_l",
        "Pz_l",
    ]
    e_names = [
        "E_t",
        "Ez_t",
        "Ey_t",
        "Ex_t",
        "Ez_r",
        "Ey_r",
        "Ex_r",
        "Ez_l",
        "Ey_l",
        "Ex_l",
    ]
    p_pk, e_pk = peak_block(Pw, p_names), peak_block(En, e_names)
    power = {
        k: np.around(Pw[k], 2).tolist()
        for k in [
            "Px_l",
            "Py_l",
            "Pz_l",
            "Px_r",
            "Py_r",
            "Pz_r",
            "Px_t",
            "Py_t",
            "Pz_t",
            "P_r",
            "P_l",
            "P_t",
        ]
    }
    power.update({f"{k}_max": v[0] for k, v in p_pk.items()})
    power.update({f"{k}_time_to_impact": v[2] for k, v in p_pk.items()})
    power.update({f"{k}_max_timing": v[1] for k, v in p_pk.items()})
    energy = {
        k: np.around(En[k], 2).tolist()
        for k in [
            "Ex_l",
            "Ey_l",
            "Ez_l",
            "Ex_r",
            "Ey_r",
            "Ez_r",
            "Ex_t",
            "Ey_t",
            "Ez_t",
            "E_r",
            "E_l",
            "E_t",
        ]
    }
    energy.update({f"{k}_max": v[0] for k, v in e_pk.items()})
    energy.update({f"{k}_time_to_impact": v[2] for k, v in e_pk.items()})
    energy.update({f"{k}_max_timing": v[1] for k, v in e_pk.items()})
    return {"energy": energy, "power": power}


def _stage_widget(fv, S):
    """needs stages: ['base', 'geometry', 'cop']; reads: ['COM', 'COP', 'COP_l', 'COP_r', 'P1', 'lay', 'midfoot_l_3d', 'midfoot_r_3d', 'offset_x', 'offset_y']"""
    COM = S["COM"]
    COP = S["COP"]
    COP_l = S["COP_l"]
    COP_r = S["COP_r"]
    P1 = S["P1"]
    lay = S["lay"]
    midfoot_l_3d = S["midfoot_l_3d"]
    midfoot_r_3d = S["midfoot_r_3d"]
    offset_x = S["offset_x"]
    offset_y = S["offset_y"]
    # ---- frontend widget pixels (COP / COM trajectories, front view)
    fh, fw = 180.0, 375.0
    dist_pix = math.hypot(fh, fw)
    dist_m = math.hypot(lay["plate_h"], lay["plate_w"])
    k_all = dist_pix / dist_m
    k_h, k_w = fh / lay["plate_h"], fw / lay["plate_w"]
    xo = lay["x_off"]
    foot_len_pix = int(0.2 * k_h)
    foot_wid_pix = int(0.2 / 2.8 * k_w)

    def foot_box(cx, cy):
        return (
            [int(cx - foot_wid_pix / 2), int(cx + foot_wid_pix / 2)],
            [int(cy - foot_len_pix / 2), int(cy + foot_len_pix / 2)],
        )

    COP_l_px, COP_l_py = (COP_l[:, 0] + xo) * k_w, (COP_l[:, 1] + 0.25) * k_h
    COP_r_px, COP_r_py = (COP_r[:, 0] + xo) * k_w, (COP_r[:, 1] + 0.25) * k_h
    COP_px, COP_py = (COP[:, 0] + xo) * k_w, (COP[:, 1] + 0.25) * k_h
    COM_py = (COM[:, 0] - offset_x + 0.25) * k_all
    COM_px = (COM[:, 1] - offset_y + xo) * k_all
    rows_l, cols_l = foot_box(COP_l_px[P1], COP_l_py[P1])
    rows_r, cols_r = foot_box(COP_r_px[P1], COP_r_py[P1])

    vs = lay["vec_scale"]
    N = lambda a: a / vs
    start_X, start_Y = COP[:, 1] + offset_x, COP[:, 0] + offset_y
    start_Z = 0.5 * (midfoot_l_3d[:, 2] + midfoot_r_3d[:, 2])
    return {
        "COM_px": COM_px,
        "COM_py": COM_py,
        "COP_l_px": COP_l_px,
        "COP_l_py": COP_l_py,
        "COP_px": COP_px,
        "COP_py": COP_py,
        "COP_r_px": COP_r_px,
        "COP_r_py": COP_r_py,
        "N": N,
        "cols_l": cols_l,
        "cols_r": cols_r,
        "fh": fh,
        "foot_box": foot_box,
        "fw": fw,
        "k_all": k_all,
        "k_h": k_h,
        "k_w": k_w,
        "rows_l": rows_l,
        "rows_r": rows_r,
        "start_X": start_X,
        "start_Y": start_Y,
        "start_Z": start_Z,
        "xo": xo,
    }


def _stage_pose2d(fv, S):
    """needs stages: ['base', 'forces_arrays', 'geometry', 'widget']; reads: ['Fx_l', 'Fx_r', 'Fx_t', 'Fy_l', 'Fy_r', 'Fy_t', 'Fz_l', 'Fz_r', 'Fz_t', 'GRF_r', 'GRF_t', 'N', 'P1', 'ctx', 'mk', 'n', 'pose2d', 'rot', 'sfx_l', 'sfx_r']"""
    Fx_l = S["Fx_l"]
    Fx_r = S["Fx_r"]
    Fx_t = S["Fx_t"]
    Fy_l = S["Fy_l"]
    Fy_r = S["Fy_r"]
    Fy_t = S["Fy_t"]
    Fz_l = S["Fz_l"]
    Fz_r = S["Fz_r"]
    Fz_t = S["Fz_t"]
    GRF_r = S["GRF_r"]
    GRF_t = S["GRF_t"]
    N = S["N"]
    P1 = S["P1"]
    ctx = S["ctx"]
    mk = S["mk"]
    n = S["n"]
    pose2d = S["pose2d"]
    rot = S["rot"]
    sfx_l = S["sfx_l"]
    sfx_r = S["sfx_r"]
    # ---- 2D-pixel-dependent keys (need pose2d)
    weight_shift = None
    COM_MODELS = None
    pix_keys = {}
    if pose2d is not None:
        p2 = fv._extract_pose2d_array(pose2d, n)
        names = ctx["marker_names"]
        idx2 = {nm: i for i, nm in enumerate(names)}
        mf2_l = p2[:, idx2[f"midfoot_{sfx_l}"]]
        mf2_r = p2[:, idx2[f"midfoot_{sfx_r}"]]
        sh_m = float(np.linalg.norm(mk["shoulder_r"][P1] - mk["shoulder_l"][P1]))
        sh_pix = float(
            math.hypot(
                p2[P1, idx2["shoulder_r"], 1] - p2[P1, idx2["shoulder_l"], 1],
                p2[P1, idx2["shoulder_r"], 0] - p2[P1, idx2["shoulder_l"], 0],
            )
        )
        m2pix = sh_pix / sh_m if sh_m > 0 else 0.0
        tr = lambda a: np.trunc(a).astype(int)
        Fz_t_pix, Fz_l_pix, Fz_r_pix = (
            tr(N(a) * m2pix).astype(float) for a in (Fz_t, Fz_l, Fz_r)
        )
        sg = (
            -1.0 if rot else 1.0
        )  # video overlays use the tracking-frame horizontal forces
        Fx_t_pix, Fx_l_pix, Fx_r_pix = (
            -tr(N(sg * a) * m2pix).astype(float) for a in (Fx_t, Fx_l, Fx_r)
        )
        Fy_t_pix, Fy_l_pix, Fy_r_pix = (
            -tr(N(sg * a) * m2pix).astype(float) for a in (Fy_t, Fy_l, Fy_r)
        )

        dist_mf = np.abs(mf2_r[:, 0] - mf2_l[:, 0])
        wsr = fv._safe_div(GRF_r, GRF_t, fill=0.5)
        factor_mf = dist_mf * wsr
        cx = tr(mf2_l[:, 0] - factor_mf)
        weight_shift = {
            "COP_for_Frontend_X": cx.tolist(),
            "COP_for_Frontend_Y": tr(
                mf2_r[:, 1]
            ).tolist(),  # measured script: midfoot_r here ...
            "factor_for_midfoot_point_frontend": factor_mf.tolist(),
        }
        cy = tr(mf2_l[:, 1])  # ... but midfoot_l inside force_vectors
        COM_models = fv._fs_com_2d(p2, names)
        COM_2d = COM_models["COM_2d"]
        COP_2d = np.column_stack([cx, cy]).astype(float)
        endpoint = COP_2d - np.column_stack([Fx_t_pix, Fz_t_pix])
        nvec = COP_2d - endpoint
        nn = np.linalg.norm(nvec, axis=1)
        okp = nn > 1e-12
        unit = nvec / np.where(okp, nn, 1.0)[:, None]
        proj2 = endpoint + unit * np.sum((COM_2d - endpoint) * unit, axis=1)[:, None]
        for i in range(n):  # zero-force frames: reuse previous
            if not okp[i]:
                proj2[i] = proj2[i - 1] if i > 0 else COM_2d[i]
        pix_keys = {
            "Fz_t_pix": Fz_t_pix.tolist(),
            "Fz_l_pix": Fz_l_pix.tolist(),
            "Fz_r_pix": Fz_r_pix.tolist(),
            "Fx_t_pix": Fx_t_pix.tolist(),
            "Fx_l_pix": Fx_l_pix.tolist(),
            "Fx_r_pix": Fx_r_pix.tolist(),
            "Fy_t_pix": Fy_t_pix.tolist(),
            "Fy_l_pix": Fy_l_pix.tolist(),
            "Fy_r_pix": Fy_r_pix.tolist(),
        }
        pix_keys2 = {
            "COP_for_Frontend_X": cx.tolist(),
            "COP_for_Frontend_Y": cy.tolist(),
            "factor_for_midfoot_point_frontend": factor_mf.tolist(),
            "proj_GRF_COM_2d": proj2.tolist(),
            "COM_2d": COM_2d.tolist(),
        }
        COM_MODELS = {k: v.tolist() for k, v in COM_models.items()}
    else:
        logging.warning(
            "generate_forces_screen: no pose2d supplied -> omitting 2D-pixel keys: top-level "
            "weight_shift, COM_MODELS; force_vectors.{F[xyz]_[tlr]_pix, COP_for_Frontend_X/Y, "
            "factor_for_midfoot_point_frontend, proj_GRF_COM_2d, COM_2d}."
        )
        pix_keys2 = {}
    return {
        "COM_MODELS": COM_MODELS,
        "pix_keys": pix_keys,
        "pix_keys2": pix_keys2,
        "weight_shift": weight_shift,
    }


def _stage_fv_core(fv, S):
    """needs stages: ['base', 'forces_arrays', 'geometry', 'cop', 'body_torques', 'com_vs_stance', 'kinematics', 'torque_peaks', 'widget', 'pose2d']; reads: ['COM_disp_x', 'COM_disp_y', 'COM_disp_z', 'COM_px', 'COM_py', 'COM_vel', 'COM_vs_stance', 'COP', 'COP_AP_disp', 'COP_ML_disp', 'COP_l', 'COP_l_px', 'COP_l_py', 'COP_px', 'COP_py', 'COP_r', 'COP_r_px', 'COP_r_py', 'COP_vel', 'Fx_l', 'Fx_r', 'Fx_t', 'Fy_l', 'Fy_r', 'Fy_t', 'Fz_l', 'Fz_r', 'Fz_t', 'N', 'cols_l', 'cols_r', 'ma', 'ma_pk', 'midfoot_l_3d', 'midfoot_r_3d', 'pix_keys', 'pix_keys2', 'rot', 'rows_l', 'rows_r', 'start_X', 'start_Y', 'start_Z', 'tq_pk', 'x_arm', 'x_torque', 'y_arm', 'y_torque', 'z_arm', 'z_torque']"""
    COM_disp_x = S["COM_disp_x"]
    COM_disp_y = S["COM_disp_y"]
    COM_disp_z = S["COM_disp_z"]
    COM_px = S["COM_px"]
    COM_py = S["COM_py"]
    COM_vel = S["COM_vel"]
    COM_vs_stance = S["COM_vs_stance"]
    COP = S["COP"]
    COP_AP_disp = S["COP_AP_disp"]
    COP_ML_disp = S["COP_ML_disp"]
    COP_l = S["COP_l"]
    COP_l_px = S["COP_l_px"]
    COP_l_py = S["COP_l_py"]
    COP_px = S["COP_px"]
    COP_py = S["COP_py"]
    COP_r = S["COP_r"]
    COP_r_px = S["COP_r_px"]
    COP_r_py = S["COP_r_py"]
    COP_vel = S["COP_vel"]
    Fx_l = S["Fx_l"]
    Fx_r = S["Fx_r"]
    Fx_t = S["Fx_t"]
    Fy_l = S["Fy_l"]
    Fy_r = S["Fy_r"]
    Fy_t = S["Fy_t"]
    Fz_l = S["Fz_l"]
    Fz_r = S["Fz_r"]
    Fz_t = S["Fz_t"]
    N = S["N"]
    cols_l = S["cols_l"]
    cols_r = S["cols_r"]
    ma = S["ma"]
    ma_pk = S["ma_pk"]
    midfoot_l_3d = S["midfoot_l_3d"]
    midfoot_r_3d = S["midfoot_r_3d"]
    pix_keys = S["pix_keys"]
    pix_keys2 = S["pix_keys2"]
    rot = S["rot"]
    rows_l = S["rows_l"]
    rows_r = S["rows_r"]
    start_X = S["start_X"]
    start_Y = S["start_Y"]
    start_Z = S["start_Z"]
    tq_pk = S["tq_pk"]
    x_arm = S["x_arm"]
    x_torque = S["x_torque"]
    y_arm = S["y_arm"]
    y_torque = S["y_torque"]
    z_arm = S["z_arm"]
    z_torque = S["z_torque"]
    force_vectors = {
        "pixel_rows_foot_right_fixed_size": rows_r,
        "pixel_colm_foot_right_fixed_size": cols_r,
        "pixel_rows_foot_left_fixed_size": rows_l,
        "pixel_colm_foot_left_fixed_size": cols_l,
        "COP_l_pix_x_axes": COP_l_px.tolist(),
        "COP_l_pix_y_axes": COP_l_py.tolist(),
        "COP_r_pix_x_axes": COP_r_px.tolist(),
        "COP_r_pix_y_axes": COP_r_py.tolist(),
        "COP_pix_x_axes": COP_px.tolist(),
        "COP_pix_y_axes": COP_py.tolist(),
        "COM_pix_x_axes": COM_px.tolist(),
        "COM_pix_y_axes": COM_py.tolist(),
        "start_point_X": start_X.tolist(),
        "start_point_Y": start_Y.tolist(),
        "start_point_Z": start_Z.tolist(),
        "end_point_X": (start_X + N(Fy_t)).tolist(),
        "end_point_Y": (start_Y + N(Fx_t)).tolist(),
        "end_point_Z": (start_Z + N(Fz_t)).tolist(),
        "start_point_X_l": midfoot_l_3d[:, 0].tolist(),
        "start_point_Y_l": midfoot_l_3d[:, 1].tolist(),
        "start_point_Z_l": midfoot_l_3d[:, 2].tolist(),
        "end_point_X_l": (midfoot_l_3d[:, 0] + N(Fy_l)).tolist(),
        "end_point_Y_l": (midfoot_l_3d[:, 1] + N(Fx_l)).tolist(),
        "end_point_Z_l": (midfoot_l_3d[:, 2] + N(Fz_l)).tolist(),
        "start_point_X_r": midfoot_r_3d[:, 0].tolist(),
        "start_point_Y_r": midfoot_r_3d[:, 1].tolist(),
        "start_point_Z_r": midfoot_r_3d[:, 2].tolist(),
        "end_point_X_r": (midfoot_r_3d[:, 0] + N(Fy_r)).tolist(),
        "end_point_Y_r": (midfoot_r_3d[:, 1] + N(Fx_r)).tolist(),
        "end_point_Z_r": (midfoot_r_3d[:, 2] + N(Fz_r)).tolist(),
    }

    force_vectors.update(pix_keys)
    force_vectors.update(
        {
            "COP_l": COP_l.tolist(),
            "COP_r": COP_r.tolist(),
            "COP": COP.tolist(),
        }
    )
    force_vectors.update(pix_keys2)
    r2 = lambda a: np.around(a, 2)
    force_vectors.update(
        {
            "x_Torque": r2(x_torque).tolist(),
            "y_Torque": r2(y_torque).tolist(),
            "z_Torque": r2(z_torque).tolist(),
            "x_Moment_Arm": r2(x_arm).tolist(),
            "y_Moment_Arm": r2(y_arm).tolist(),
            "z_Moment_Arm": r2(z_arm).tolist(),
            "COM_vs_Stance": [
                [float(v)] for v in COM_vs_stance
            ],  # measured script emits 1-element sublists
            "FilteredCOPSpeed": r2(COP_vel).tolist(),
            "COP_disp_x": r2(COP_ML_disp).tolist(),
            "COP_disp_y": r2(COP_AP_disp).tolist(),
            "COM_disp_x": r2(COM_disp_x).tolist(),
            "COM_disp_y": r2(COM_disp_y).tolist(),
            "COM_disp_z": r2(COM_disp_z).tolist(),
            "COM_vel": r2(COM_vel).tolist(),
        }
    )
    for ax in "xyz":
        force_vectors[f"{ax}_Torque_max"] = tq_pk[f"{ax}_Torque"][0]
    for ax in "xyz":
        force_vectors[f"{ax}_Moment_Arm_max"] = ma_pk[f"{ax}_Moment_Arm"][0]
    for ax in "xyz":
        force_vectors[f"{ax}_Torque_max_timing"] = tq_pk[f"{ax}_Torque"][1]
    for ax in "xyz":
        force_vectors[f"{ax}_Moment_Arm_max_timing"] = ma_pk[f"{ax}_Moment_Arm"][1]
    for ax in "xyz":
        force_vectors[f"{ax}_Torque_time_to_impact"] = tq_pk[f"{ax}_Torque"][2]
    for ax in "xyz":
        force_vectors[f"{ax}_Moment_Arm_time_to_impact"] = ma_pk[f"{ax}_Moment_Arm"][2]
    inch = fv._M_TO_IN
    force_vectors.update(
        {
            "COP_disp_x_inch": r2(COP_ML_disp * inch).tolist(),
            "COP_disp_y_inch": r2(COP_AP_disp * inch).tolist(),
            "COM_disp_x_inch": r2(COM_disp_x * inch).tolist(),
            "COM_disp_y_inch": r2(COM_disp_y * inch).tolist(),
            "COM_disp_z_inch": r2(COM_disp_z * inch).tolist(),
        }
    )
    for ax in "xyz":
        force_vectors[f"{ax}_Moment_Arm_max_inch"] = ma_pk[f"{ax}_Moment_Arm"][0] * inch
    for ax in "xyz":
        force_vectors[f"{ax}_Moment_Arm_inch"] = r2(
            ma[f"{ax}_Moment_Arm"] * inch
        ).tolist()
    force_vectors.update(
        {
            "COM_vel_ips": r2(COM_vel * inch).tolist(),
            "COP_vel_ips": r2(COP_vel * inch).tolist(),
        }
    )

    if rot:
        neg = lambda seq: (-np.asarray(seq, dtype=float)).tolist()
        for suf in ("", "_l", "_r"):
            for pre in ("start_point_", "end_point_"):
                for ax in ("X", "Y"):
                    key = f"{pre}{ax}{suf}"
                    force_vectors[key] = neg(force_vectors[key])
    return {"force_vectors": force_vectors}


def _stage_small(fv, S):
    """needs stages: ['base', 'forces_arrays', 'cop', 'widget']; reads: ['COP', 'COP_l', 'COP_r', 'Fx_l', 'Fx_r', 'Fx_t', 'Fz_l', 'Fz_r', 'Fz_t', 'lay', 'n', 'xo']"""
    COP = S["COP"]
    COP_l = S["COP_l"]
    COP_r = S["COP_r"]
    Fx_l = S["Fx_l"]
    Fx_r = S["Fx_r"]
    Fx_t = S["Fx_t"]
    Fz_l = S["Fz_l"]
    Fz_r = S["Fz_r"]
    Fz_t = S["Fz_t"]
    lay = S["lay"]
    n = S["n"]
    xo = S["xo"]
    # ---- small (frontend side-panel) force vectors; independent of pose2d
    sh_, sw_ = 75.0, 223.0
    kh_s = sh_ / lay["plate_h"]
    kw_s = sw_ / lay["plate_w"]
    base_y = int(sh_ - 30)
    sx = lambda c: (c[:, 0] + xo) * kw_s + 14.5
    s_lead, s_trail, s_comb = sx(COP_l), sx(COP_r), sx(COP)
    scale_s = max(float(np.max(Fz_t)), 1e-9) / base_y
    pxs = lambda a: np.trunc(a / scale_s)

    def end_pts(s_x, Fx_, Fz_):
        fxp = -pxs(Fx_)
        fzp = pxs(Fz_)
        return (
            np.trunc(s_x - fxp).astype(int).tolist(),
            np.trunc(base_y - fzp).astype(int).tolist(),
        )

    ex_l, ey_l = end_pts(s_lead, Fx_l, Fz_l)
    ex_r, ey_r = end_pts(s_trail, Fx_r, Fz_r)
    ex_t, ey_t = end_pts(s_comb, Fx_t, Fz_t)
    ys = [base_y] * n
    small_force_vectors = {
        "force_vectors_lead": {
            "starting_x": np.trunc(s_lead).astype(int).tolist(),
            "starting_y": ys,
            "ending_x": ex_l,
            "ending_y": ey_l,
        },
        "force_vectors_trail": {
            "starting_x": np.trunc(s_trail).astype(int).tolist(),
            "starting_y": ys,
            "ending_x": ex_r,
            "ending_y": ey_r,
        },
        "force_vectors_combined": {
            "starting_x": np.trunc(s_comb).astype(int).tolist(),
            "starting_y": ys,
            "ending_x": ex_t,
            "ending_y": ey_t,
        },
    }
    return {"small_force_vectors": small_force_vectors}


def _stage_backview(fv, S):
    """needs stages: ['base', 'geometry', 'cop', 'widget']; reads: ['COM', 'COP', 'COP_l', 'COP_r', 'P1', 'fh', 'foot_box', 'fw', 'k_all', 'k_h', 'k_w', 'offset_x', 'offset_y', 'xo']"""
    COM = S["COM"]
    COP = S["COP"]
    COP_l = S["COP_l"]
    COP_r = S["COP_r"]
    P1 = S["P1"]
    fh = S["fh"]
    foot_box = S["foot_box"]
    fw = S["fw"]
    k_all = S["k_all"]
    k_h = S["k_h"]
    k_w = S["k_w"]
    offset_x = S["offset_x"]
    offset_y = S["offset_y"]
    xo = S["xo"]
    # ---- back-view COM / COP trajectories
    bl_py = fh - (COP_l[:, 1] + 0.25) * k_h
    br_py = fh - (COP_r[:, 1] + 0.25) * k_h
    bc_py = fh - (COP[:, 1] + 0.25) * k_h
    bl_px = fw - (COP_l[:, 0] + xo) * k_w
    br_px = fw - (COP_r[:, 0] + xo) * k_w
    bc_px = fw - (COP[:, 0] + xo) * k_w
    bcom_py = fh - ((COM[:, 0] - offset_x + 0.25) * k_all)
    if fv.plate_layout == "amti":
        bcom_px = (
            COM[:, 1] - offset_y + xo
        ) * k_all  # (not mirrored in the AMTI branch)
    else:
        bcom_px = fw - ((COM[:, 1] - offset_y + xo) * k_all)
    brows_l, bcols_l = foot_box(bl_px[P1], bl_py[P1])
    brows_r, bcols_r = foot_box(br_px[P1], br_py[P1])
    COM_COP_traj_back_view = {
        "pixel_rows_foot_right_fixed_size": brows_r,
        "pixel_colm_foot_right_fixed_size": bcols_r,
        "pixel_rows_foot_left_fixed_size": brows_l,
        "pixel_colm_foot_left_fixed_size": bcols_l,
        "COP_l_pix_x_axes": bl_px.tolist(),
        "COP_l_pix_y_axes": bl_py.tolist(),
        "COP_r_pix_x_axes": br_px.tolist(),
        "COP_r_pix_y_axes": br_py.tolist(),
        "COP_pix_x_axes": bc_px.tolist(),
        "COP_pix_y_axes": bc_py.tolist(),
        "COM_pix_x_axes": bcom_px.tolist(),
        "COM_pix_y_axes": bcom_py.tolist(),
    }
    return {"COM_COP_traj_back_view": COM_COP_traj_back_view}


_STAGES = {
    "base": (_stage_base, []),
    "forces_arrays": (_stage_forces_arrays, ["base"]),
    "geometry": (_stage_geometry, ["base"]),
    "cop": (_stage_cop, ["base", "forces_arrays", "geometry"]),
    "vertical_torque": (_stage_vertical_torque, ["base", "forces_arrays"]),
    "body_torques": (_stage_body_torques, ["base", "forces_arrays", "geometry", "cop"]),
    "com_vs_stance": (_stage_com_vs_stance, ["base", "geometry", "cop"]),
    "kinematics": (_stage_kinematics, ["base", "geometry", "cop"]),
    "forces": (_stage_forces, ["base", "forces_arrays", "vertical_torque"]),
    "torque_peaks": (_stage_torque_peaks, ["base", "body_torques"]),
    "arrows": (_stage_arrows, ["base", "forces_arrays", "geometry", "cop"]),
    "values_cop": (_stage_values_cop, ["base", "forces_arrays", "geometry"]),
    "impulse": (_stage_impulse, ["base", "forces"]),
    "power_energy": (_stage_power_energy, ["base", "forces_arrays", "kinematics"]),
    "widget": (_stage_widget, ["base", "geometry", "cop"]),
    "pose2d": (_stage_pose2d, ["base", "forces_arrays", "geometry", "widget"]),
    "fv_core": (
        _stage_fv_core,
        [
            "base",
            "forces_arrays",
            "geometry",
            "cop",
            "body_torques",
            "com_vs_stance",
            "kinematics",
            "torque_peaks",
            "widget",
            "pose2d",
        ],
    ),
    "small": (_stage_small, ["base", "forces_arrays", "cop", "widget"]),
    "backview": (_stage_backview, ["base", "geometry", "cop", "widget"]),
}

_OUT_STAGE = {
    "time": "base",
    "forces": "forces",
    "force_vectors_3d": "arrows",
    "values_for_COP_trajectory": "values_cop",
    "impulse": "impulse",
    "power": "power_energy",
    "energy": "power_energy",
    "force_vectors": "fv_core",
    "weight_shift": "pose2d",
    "COM_MODELS": "pose2d",
    "small_force_vectors": "small",
    "COM_COP_traj_back_view": "backview",
}


OUTPUT_KEYS = (
    "force_vectors",
    "weight_shift",
    "forces",
    "time",
    "force_vectors_3d",
    "values_for_COP_trajectory",
    "COM_MODELS",
    "power",
    "energy",
    "impulse",
    "small_force_vectors",
    "COM_COP_traj_back_view",
    "critical_events",
    "frame_offset",
)
CORE_CONFIG = {
    "configuration.video.frame_rate": "Hz - motion differentiation, filter cut-offs, time axis",
    "configuration.user.weight": "kg - segment masses (body mass)",
    "configuration.swing.swingType": "'hitting' | 'pitching' | 'full swing' - selects the movement preset (filters, contact handling, club on/off)",
    "configuration.swing.clubType": "golf only ('driver' | iron types) - club mass",
    "configuration.swing.handedness": "'leftHanded' -> analysis rotated into the force-plate frame; otherwise unchanged",
    "configuration.swing.critical_events": "P1 (start / weight reference), P7 (impact / release), P10 (end); missing ones fall back (first frame / estimated peak hand speed / last frame)",
}
CORE_DATA = [
    "data.pose  (29 markers x frames x 3)",
    "data.club  (golf only, include_club=True: 3D start/mid/end per frame)",
]
_SPEC = {
    "force_vectors": {
        "needs_core": True,
        "config": [
            "configuration.video.frame_rate",
            "configuration.swing.critical_events (P1, P7, P10)",
            "configuration.swing.handedness",
        ],
        "data": ["pose2d (optional: 2D pixel keys)"],
        "note": "Biggest block: COP, body torques x/y/z, moment arms, COM/COP displacement and speed, 3D start/end points, widget pixels. Engine options: plate_layout, anchor_foot_depth.",
    },
    "weight_shift": {
        "needs_core": True,
        "config": [],
        "data": ["pose2d (required)"],
        "note": "2D pixel positions of the feet scaled by the load split. Needs pose2d. (Today it shares a stage with the pixel force arrows, so it also runs the force computation; strictly it only needs pose2d + the Fz split.)",
    },
    "forces": {
        "needs_core": True,
        "config": [
            "configuration.video.frame_rate",
            "configuration.swing.critical_events (P1, P7, P10)",
            "configuration.swing.handedness",
            "configuration.user.weight (fallback weight reference)",
        ],
        "data": [],
        "note": "Fz/Fx/Fy/Tz total/left/right, % of P1 weight, max, timing, time-to-impact.",
    },
    "time": {
        "needs_core": True,
        "config": [
            "configuration.video.frame_rate",
            "configuration.swing.critical_events (P1, P10)",
        ],
        "data": ["data.pose (frame count / valid frames, for event fallbacks)"],
        "note": "Time axis from P1 to P10, seconds (measured-script convention: starts at 1/frame_rate).",
    },
    "force_vectors_3d": {
        "needs_core": True,
        "config": [
            "configuration.swing.critical_events (P1, P7, P10)",
            "configuration.swing.handedness",
        ],
        "data": [],
        "note": "3D arrows per foot and combined, drawn in the tracking frame (lefty rotation converted back).",
    },
    "values_for_COP_trajectory": {
        "needs_core": True,
        "config": [],
        "data": [],
        "note": "Weight shift % per foot and midfoot stance width (cm, inch).",
    },
    "COM_MODELS": {
        "needs_core": True,
        "config": [],
        "data": ["pose2d (required)"],
        "note": "2D COM models (full body, without arms, lower, upper). Technically only needs pose2d; shares a stage with the pixel force arrows today.",
    },
    "power": {
        "needs_core": True,
        "config": [
            "configuration.video.frame_rate",
            "configuration.swing.critical_events (P1, P7, P10)",
            "configuration.swing.handedness",
        ],
        "data": [],
        "note": "Force x COM velocity per axis/foot, with max/timing/time-to-impact.",
    },
    "energy": {
        "needs_core": True,
        "config": [
            "configuration.video.frame_rate",
            "configuration.swing.critical_events (P1, P7, P10)",
            "configuration.swing.handedness",
        ],
        "data": [],
        "note": "Cumulative integral of power, with max/timing/time-to-impact.",
    },
    "impulse": {
        "needs_core": True,
        "config": [
            "configuration.video.frame_rate",
            "configuration.swing.critical_events (P1, P7, P10)",
            "configuration.swing.handedness",
        ],
        "data": [],
        "note": "Cumulative integral of force, with max/timing/time-to-impact.",
    },
    "small_force_vectors": {
        "needs_core": True,
        "config": [
            "configuration.swing.critical_events (P1, P7, P10)",
            "configuration.swing.handedness",
        ],
        "data": [],
        "note": "Side-panel force arrows (pixels). Engine option: plate_layout.",
    },
    "COM_COP_traj_back_view": {
        "needs_core": True,
        "config": [
            "configuration.swing.critical_events (P1, P7, P10)",
            "configuration.swing.handedness",
        ],
        "data": [],
        "note": "Back-view COM/COP trajectory pixels. Engine option: plate_layout.",
    },
    "critical_events": {
        "needs_core": False,
        "config": ["configuration.swing.critical_events"],
        "data": [],
        "note": "Echo of the input events (shifted by frame_offset = 0). Does NOT run the force computation.",
    },
    "frame_offset": {
        "needs_core": False,
        "config": [],
        "data": [],
        "note": "Always 0 for computed forces (no plate time-sync). Does NOT run the force computation.",
    },
}

# ==========================================================================
# PUBLIC FUNCTIONS - one per output key
# ==========================================================================


def force_vectors(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"force_vectors": ...} (and writes "force_vectors.json" if output_dir is given).

    Biggest block: COP, body torques x/y/z, moment arms, COM/COP displacement and speed, 3D start/end points, widget pixels. Engine options: plate_layout, anchor_foot_depth.

    Needs
      core:   all CORE items (see CORE_CONFIG / describe_requirements()) - the force computation feeds this output
      direct configuration:
        - configuration.video.frame_rate
        - configuration.swing.critical_events (P1, P7, P10)
        - configuration.swing.handedness
      direct data:
        - pose2d (optional: 2D pixel keys)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "force_vectors", json3d, pose2d, computation, output_dir, name_prefix, overrides
    )


def weight_shift(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"weight_shift": ...} (and writes "weight_shift.json" if output_dir is given).

    2D pixel positions of the feet scaled by the load split. Needs pose2d. (Today it shares a stage with the pixel force arrows, so it also runs the force computation; strictly it only needs pose2d + the Fz split.)

    Needs
      core:   all CORE items (see CORE_CONFIG / describe_requirements()) - the force computation feeds this output
      direct configuration:
        (none beyond CORE)
      direct data:
        - pose2d (required)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "weight_shift", json3d, pose2d, computation, output_dir, name_prefix, overrides
    )


def forces(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"forces": ...} (and writes "forces.json" if output_dir is given).

    Fz/Fx/Fy/Tz total/left/right, % of P1 weight, max, timing, time-to-impact.

    Needs
      core:   all CORE items (see CORE_CONFIG / describe_requirements()) - the force computation feeds this output
      direct configuration:
        - configuration.video.frame_rate
        - configuration.swing.critical_events (P1, P7, P10)
        - configuration.swing.handedness
        - configuration.user.weight (fallback weight reference)
      direct data:
        (none beyond CORE)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "forces", json3d, pose2d, computation, output_dir, name_prefix, overrides
    )


def time(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"time": ...} (and writes "time.json" if output_dir is given).

    Time axis from P1 to P10, seconds (measured-script convention: starts at 1/frame_rate).

    Needs
      core:   all CORE items (see CORE_CONFIG / describe_requirements()) - the force computation feeds this output
      direct configuration:
        - configuration.video.frame_rate
        - configuration.swing.critical_events (P1, P10)
      direct data:
        - data.pose (frame count / valid frames, for event fallbacks)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "time", json3d, pose2d, computation, output_dir, name_prefix, overrides
    )


def force_vectors_3d(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"force_vectors_3d": ...} (and writes "force_vectors_3d.json" if output_dir is given).

    3D arrows per foot and combined, drawn in the tracking frame (lefty rotation converted back).

    Needs
      core:   all CORE items (see CORE_CONFIG / describe_requirements()) - the force computation feeds this output
      direct configuration:
        - configuration.swing.critical_events (P1, P7, P10)
        - configuration.swing.handedness
      direct data:
        (none beyond CORE)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "force_vectors_3d",
        json3d,
        pose2d,
        computation,
        output_dir,
        name_prefix,
        overrides,
    )


def values_for_COP_trajectory(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"values_for_COP_trajectory": ...} (and writes "values_for_COP_trajectory.json" if output_dir is given).

    Weight shift % per foot and midfoot stance width (cm, inch).

    Needs
      core:   all CORE items (see CORE_CONFIG / describe_requirements()) - the force computation feeds this output
      direct configuration:
        (none beyond CORE)
      direct data:
        (none beyond CORE)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "values_for_COP_trajectory",
        json3d,
        pose2d,
        computation,
        output_dir,
        name_prefix,
        overrides,
    )


def COM_MODELS(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"COM_MODELS": ...} (and writes "COM_MODELS.json" if output_dir is given).

    2D COM models (full body, without arms, lower, upper). Technically only needs pose2d; shares a stage with the pixel force arrows today.

    Needs
      core:   all CORE items (see CORE_CONFIG / describe_requirements()) - the force computation feeds this output
      direct configuration:
        (none beyond CORE)
      direct data:
        - pose2d (required)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "COM_MODELS", json3d, pose2d, computation, output_dir, name_prefix, overrides
    )


def power(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"power": ...} (and writes "power.json" if output_dir is given).

    Force x COM velocity per axis/foot, with max/timing/time-to-impact.

    Needs
      core:   all CORE items (see CORE_CONFIG / describe_requirements()) - the force computation feeds this output
      direct configuration:
        - configuration.video.frame_rate
        - configuration.swing.critical_events (P1, P7, P10)
        - configuration.swing.handedness
      direct data:
        (none beyond CORE)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "power", json3d, pose2d, computation, output_dir, name_prefix, overrides
    )


def energy(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"energy": ...} (and writes "energy.json" if output_dir is given).

    Cumulative integral of power, with max/timing/time-to-impact.

    Needs
      core:   all CORE items (see CORE_CONFIG / describe_requirements()) - the force computation feeds this output
      direct configuration:
        - configuration.video.frame_rate
        - configuration.swing.critical_events (P1, P7, P10)
        - configuration.swing.handedness
      direct data:
        (none beyond CORE)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "energy", json3d, pose2d, computation, output_dir, name_prefix, overrides
    )


def impulse(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"impulse": ...} (and writes "impulse.json" if output_dir is given).

    Cumulative integral of force, with max/timing/time-to-impact.

    Needs
      core:   all CORE items (see CORE_CONFIG / describe_requirements()) - the force computation feeds this output
      direct configuration:
        - configuration.video.frame_rate
        - configuration.swing.critical_events (P1, P7, P10)
        - configuration.swing.handedness
      direct data:
        (none beyond CORE)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "impulse", json3d, pose2d, computation, output_dir, name_prefix, overrides
    )


def small_force_vectors(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"small_force_vectors": ...} (and writes "small_force_vectors.json" if output_dir is given).

    Side-panel force arrows (pixels). Engine option: plate_layout.

    Needs
      core:   all CORE items (see CORE_CONFIG / describe_requirements()) - the force computation feeds this output
      direct configuration:
        - configuration.swing.critical_events (P1, P7, P10)
        - configuration.swing.handedness
      direct data:
        (none beyond CORE)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "small_force_vectors",
        json3d,
        pose2d,
        computation,
        output_dir,
        name_prefix,
        overrides,
    )


def COM_COP_traj_back_view(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"COM_COP_traj_back_view": ...} (and writes "COM_COP_traj_back_view.json" if output_dir is given).

    Back-view COM/COP trajectory pixels. Engine option: plate_layout.

    Needs
      core:   all CORE items (see CORE_CONFIG / describe_requirements()) - the force computation feeds this output
      direct configuration:
        - configuration.swing.critical_events (P1, P7, P10)
        - configuration.swing.handedness
      direct data:
        (none beyond CORE)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "COM_COP_traj_back_view",
        json3d,
        pose2d,
        computation,
        output_dir,
        name_prefix,
        overrides,
    )


def critical_events(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"critical_events": ...} (and writes "critical_events.json" if output_dir is given).

    Echo of the input events (shifted by frame_offset = 0). Does NOT run the force computation.

    Needs
      core:   none - does not run the force computation
      direct configuration:
        - configuration.swing.critical_events
      direct data:
        (none beyond CORE)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "critical_events",
        json3d,
        pose2d,
        computation,
        output_dir,
        name_prefix,
        overrides,
    )


def frame_offset(
    json3d=None,
    pose2d=None,
    *,
    computation=None,
    output_dir=None,
    name_prefix="",
    **overrides,
):
    """Returns {"frame_offset": ...} (and writes "frame_offset.json" if output_dir is given).

    Always 0 for computed forces (no plate time-sync). Does NOT run the force computation.

    Needs
      core:   none - does not run the force computation
      direct configuration:
        (none beyond CORE)
      direct data:
        (none beyond CORE)
    overrides are passed to the engine (e.g. body_mass=..., frame_rate=..., kinematic_contact_gate=True).
    """
    return _produce(
        "frame_offset", json3d, pose2d, computation, output_dir, name_prefix, overrides
    )


# ==========================================================================
# Shared lazy computation
# ==========================================================================
class ForcesScreen:
    """One trial. The heavy force computation (inverse dynamics + left/right split) runs at most
    once, and each stage runs only if an output needs it. Create it once and pass it to several
    output functions via computation=... to avoid recomputing."""

    def __init__(
        self,
        pose3d: np.ndarray,
        body_mass: float,
        frame_rate: float,
        swing_type: str,
        lefty: bool,
        *,
        club_3d: Optional[dict] = None,
        club_type: str = "driver",
        critical_events: Optional[dict] = None,
        **model_overrides,
    ):
        self.pose3d = pose3d
        self.club_3d = club_3d
        self.club_type = club_type
        self.critical_events = critical_events
        self.lefty = lefty
        self.pose2d = None  # supplied to individual output functions
        self.fv = ForceVectors.from_swing_type(
            swing_type,
            body_mass=body_mass,
            frame_rate=frame_rate,
            **model_overrides,
        )
        self._ctx, self._S, self._done = None, None, set()

    @property
    def ctx(self):
        if self._ctx is None:
            self._ctx = self.fv._run_pipeline(
                self.pose3d,
                club_3d=self.club_3d,
                club_type=self.club_type,
                critical_events=self.critical_events,
                lefty=self.lefty,
            )
        return self._ctx

    def _run(self, stage):
        if self._S is None:
            self._S = {"ctx_raw": self.ctx, "pose2d": self.pose2d}
        if stage in self._done:
            return
        func, deps = _STAGES[stage]
        for d in deps:
            self._run(d)
        self._S.update(func(self.fv, self._S))
        self._done.add(stage)

    def get(self, key, *, pose2d=None):
        # Changing 2D data resets ONLY presentation-stage caches; reuse numerical _ctx.
        if pose2d is not None and pose2d is not self.pose2d:
            self.pose2d = pose2d
            self._S, self._done = None, set()
        if key not in OUTPUT_KEYS:
            raise KeyError(f"unknown output {key!r}; choose from {OUTPUT_KEYS}")
        if key == "frame_offset":  # predicted forces need no plate time-sync
            return 0
        if key == "critical_events":  # echo of the input events; no force computation
            pos = self.critical_events
            return lowercase_output_keys({
                k: (v + 0 if isinstance(v, (int, float)) else v)
                for k, v in (pos if isinstance(pos, dict) else {}).items()
            })
        if key in ("weight_shift", "COM_MODELS") and self.pose2d is None:
            logging.warning(
                "%s needs pose2d (2D pixel markers); not produced without it.", key
            )
            return None
        self._run(_OUT_STAGE[key])
        val = self._S[key]
        if key == "time":
            val = val.tolist()
        return None if val is None else lowercase_output_keys(self.fv._json_safe(val))


def _produce(key, json3d, pose2d, computation, output_dir, name_prefix, overrides):
    if computation is None:
        if json3d is not None:
            raise TypeError("JSON ingestion moved to Context; initialize ForcesScreen with explicit inputs")
        computation = ForcesScreen(**overrides)
    val = computation.get(key, pose2d=pose2d)
    if val is None:
        return {}
    result = lowercase_output_keys({key: val})
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{name_prefix}{key}.json"), "w") as fh:
            json.dump(result, fh, allow_nan=False)
    return result


def compute_all(
    *, computation=None, pose2d=None, keys=None, output_dir=None,
    name_prefix="", **inputs,
):
    """All (or the chosen) outputs from ONE computation; returns the merged response (identical to
    the v13 forces-screen response) and writes <key>.json per key if output_dir is given.
    """
    comp = computation if computation is not None else ForcesScreen(**inputs)
    merged = {}
    for k in keys or OUTPUT_KEYS:
        merged.update(_produce(k, None, pose2d, comp, output_dir, name_prefix, {}))
    return merged


def describe_requirements(key=None):
    """Configuration fields / data each output function needs (plus CORE for force-derived ones)."""
    if key is None:
        return {
            "CORE": {"configuration": dict(CORE_CONFIG), "data": list(CORE_DATA)},
            **_SPEC,
        }
    return {
        "CORE (force-derived outputs)": (
            {"configuration": dict(CORE_CONFIG), "data": list(CORE_DATA)}
            if _SPEC[key]["needs_core"]
            else None
        ),
        key: _SPEC[key],
    }


def resolved_configuration(
    *, swing_type, body_mass, frame_rate, lefty, club_type="driver",
    critical_events=None, **model_overrides,
):
    """Show the resolved numerical model configuration without parsing JSON."""
    fv = ForceVectors.from_swing_type(
        swing_type, body_mass=body_mass, frame_rate=frame_rate, **model_overrides
    )
    return {
        "swingType": swing_type,
        "handedness": "left" if lefty else "right",
        "clubType": club_type,
        "body_mass_kg": fv.body_mass,
        "frame_rate_hz": fv.frame_rate,
        "continuous_contact": fv.continuous_contact,
        "include_club": fv.include_club,
        "critical_events": critical_events,
    }


# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 3:
#         print(
#             "usage: python force_screen_modules.py <pose_3d.json> <out_dir> [pose_2d.json]"
#         )
#         sys.exit(0)
#     j3 = json.load(open(sys.argv[1]))
#     j2 = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
#     compute_all(j3, j2, output_dir=sys.argv[2])
