"""Standalone output: inPlaneMetrics."""
from .core import resolve

def inPlaneMetrics(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return {'inPlaneMetrics': comp.get('inPlaneMetrics')}
