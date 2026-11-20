"""Standalone output: M2C_force_vectors_hand_r_3d."""
from typing import Any
from .core import lowercase_output_keys, resolve

def M2C_force_vectors_hand_r_3d(ctx: Any = None, *, computation: Any = None) -> dict[str, Any]:
    """Return the M2C_force_vectors_hand_r_3d output dictionary for the supplied context or computation."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'M2C_force_vectors_hand_r_3d': comp.get('M2C_force_vectors_hand_r_3d')})
