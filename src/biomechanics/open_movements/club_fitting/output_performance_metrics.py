"""Standalone output: Performance_metrics."""
from .core import resolve

def Performance_metrics(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return {'Performance_metrics': comp.get('Performance_metrics')}
