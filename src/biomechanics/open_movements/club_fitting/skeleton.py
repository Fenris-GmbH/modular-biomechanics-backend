from .math_helpers import *

def calculate(state):
    """Compute skeleton stage with cached dependencies."""
    ankle_l = state.get('ankle_l')
    ankle_r = state.get('ankle_r')
    club_head = state.get('club_head')
    club_main = state.get('club_main')
    club_mid = state.get('club_mid')
    elbow_l = state.get('elbow_l')
    elbow_r = state.get('elbow_r')
    foot_l = state.get('foot_l')
    foot_r = state.get('foot_r')
    hand_l = state.get('hand_l')
    hand_r = state.get('hand_r')
    hip_l = state.get('hip_l')
    hip_r = state.get('hip_r')
    knee_l = state.get('knee_l')
    knee_r = state.get('knee_r')
    neck = state.get('neck')
    pelvis = state.get('pelvis')
    shoulder_l = state.get('shoulder_l')
    shoulder_r = state.get('shoulder_r')
    skull = state.get('skull')
    spine_high = state.get('spine_high')
    spine_low = state.get('spine_low')
    swing_type = state.get('swing_type')
    wrist_l = state.get('wrist_l')
    wrist_r = state.get('wrist_r')
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
    out = locals().copy()
    out.pop("state", None)
    return out
