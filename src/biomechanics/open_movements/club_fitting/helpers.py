import numpy as np
from scipy import signal
from scipy.signal import savgol_filter


def _lp(x, fc, frame_rate, order=4):
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


def _deriv(x, order, frame_rate, dt):
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


def _stats(arr, unit, dt):
    a = np.asarray(arr, dtype=float)
    if a.size == 0 or not np.isfinite(a).any():
        return {"unit": unit}
    imax, imin = int(np.nanargmax(a)), int(np.nanargmin(a))
    return {
        "unit": unit,
        "min": float(a[imin]),
        "max": float(a[imax]),
        "mean": float(np.nanmean(a)),
        "range": float(a[imax] - a[imin]),
        "max_frame": imax,
        "max_time_s": imax * dt,
        "min_frame": imin,
        "min_time_s": imin * dt,
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


WIDGET_EQUIVALENTS = {
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


def filter_hand(hand, frame_rate):
    return np.column_stack(
        [
            _lp(hand[:, 0], 3, frame_rate),
            _lp(hand[:, 1], 3, frame_rate),
            _lp(hand[:, 2], 5, frame_rate),
        ]
    )


def hand_force(filtered_hand, dt, W):
    # A constant translation does not affect the time derivatives.
    velocity = np.gradient(filtered_hand - filtered_hand[0], dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    return W * (0.61 / 100) * acceleration


def hand_force_axes(filtered_hand, elbow, force):
    tangent = filtered_hand[1:] - filtered_hand[:-1]
    lengths = np.linalg.norm(tangent, axis=1)
    tangent[lengths > 0] /= lengths[lengths > 0, None]
    tangent = np.vstack([np.zeros((1, 3)), tangent])
    axial = filtered_hand - elbow
    lengths = np.linalg.norm(axial, axis=1)
    axial[lengths > 0] /= lengths[lengths > 0, None]
    tangential_force = np.sum(force * tangent, axis=1)
    axial_force = np.sum(force * axial, axis=1)
    return tangential_force, axial_force, tangent, axial
