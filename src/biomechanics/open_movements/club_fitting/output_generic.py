"""Standalone output: generic."""
from .core import resolve

def generic(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return {'generic': comp.get('generic')}
