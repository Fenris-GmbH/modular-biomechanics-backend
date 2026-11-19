"""Standalone output: outOfPlaneMetrics."""
from .core import resolve

def outOfPlaneMetrics(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return {'outOfPlaneMetrics': comp.get('outOfPlaneMetrics')}
