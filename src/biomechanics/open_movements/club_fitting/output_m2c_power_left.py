"""Individually callable generic output: M2C_power_left."""
from .core import resolve

def M2C_power_left(ctx=None, *, computation=None):
    comp = resolve(computation=computation, ctx=ctx)
    return {'M2C_power_left': comp.get('M2C_power_left')}
