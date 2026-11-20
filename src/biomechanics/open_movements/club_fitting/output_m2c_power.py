"""Standalone output: M2C_power."""
from .core import lowercase_output_keys, resolve

def M2C_power(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'M2C_power': comp.get('M2C_power')})
