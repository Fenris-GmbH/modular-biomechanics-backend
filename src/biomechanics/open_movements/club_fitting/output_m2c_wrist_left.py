"""Individually callable generic output: M2C_wrist_left."""
from .core import lowercase_output_keys, resolve

def M2C_wrist_left(ctx=None, *, computation=None):
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'M2C_wrist_left': comp.get('M2C_wrist_left')})
