"""Standalone output: M2C_power."""
from .core import resolve

def M2C_power(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return {'M2C_power': comp.get('M2C_power')}
