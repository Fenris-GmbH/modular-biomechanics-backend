from typing import Any

from .math_helpers import *

def calculate(state: dict[str, Any]) -> dict[str, Any]:
    """Compute generic_widgets stage with cached dependencies."""
    _WIDGET_EQUIVALENTS = state.get('_WIDGET_EQUIVALENTS')
    _cfg_swing = state.get('_cfg_swing')
    _deriv = state.get('_deriv')
    _k = state.get('_k')
    _limb_widgets = state.get('_limb_widgets')
    _name = state.get('_name')
    _peak_speed_frame = state.get('_peak_speed_frame')
    _sp = state.get('_sp')
    _w = state.get('_w')
    _wo = state.get('_wo')
    _wr_by = state.get('_wr_by')
    dt = state.get('dt')
    ex_limb = state.get('ex_limb')
    ex_other = state.get('ex_other')
    g_n = state.get('g_n')
    g_other = state.get('g_other')
    g_ref = state.get('g_ref')
    g_ref_o = state.get('g_ref_o')
    g_sel = state.get('g_sel')
    g_side = state.get('g_side')
    generic_outputs = state.get('generic_outputs')
    lefty = state.get('lefty')
    movement_type = state.get('movement_type')
    s_ = state.get('s_')
    sfx = state.get('sfx')
    wr_a = state.get('wr_a')
    wrist_l = state.get('wrist_l')
    wrist_r = state.get('wrist_r')
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
    out = locals().copy()
    out.pop("state", None)
    return out
