"""Standalone output: movement_type."""
from .core import resolve

def movement_type(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return {'movement_type': comp.get('movement_type')}
