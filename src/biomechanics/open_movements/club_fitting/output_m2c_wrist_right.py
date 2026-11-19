"""Individually callable generic output: M2C_wrist_right."""
from .core import resolve

def M2C_wrist_right(ctx=None, *, computation=None):
    comp = resolve(computation=computation, ctx=ctx)
    return {'M2C_wrist_right': comp.get('M2C_wrist_right')}
