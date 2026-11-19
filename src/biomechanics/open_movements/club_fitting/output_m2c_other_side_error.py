"""Individually callable generic output: M2C_other_side_error."""
from .core import resolve

def M2C_other_side_error(ctx=None, *, computation=None):
    comp = resolve(computation=computation, ctx=ctx)
    return {'M2C_other_side_error': comp.get('M2C_other_side_error')}
