"""Individually callable generic output: M2C_other_side_error."""
from .core import lowercase_output_keys, resolve

def M2C_other_side_error(ctx=None, *, computation=None):
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'M2C_other_side_error': comp.get('M2C_other_side_error')})
