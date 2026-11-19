"""Standalone output: M2C_insights."""
from .core import resolve

def M2C_insights(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return {'M2C_insights': comp.get('M2C_insights')}
