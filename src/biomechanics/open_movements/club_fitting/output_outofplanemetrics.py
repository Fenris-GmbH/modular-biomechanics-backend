"""Standalone output: outOfPlaneMetrics."""
from .core import lowercase_output_keys, resolve

def outOfPlaneMetrics(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'outOfPlaneMetrics': comp.get('outOfPlaneMetrics')})
