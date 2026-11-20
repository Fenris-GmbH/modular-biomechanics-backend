"""Standalone output: pitcher_metric_equivalents."""
from .core import lowercase_output_keys, resolve

def pitcher_metric_equivalents(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'pitcher_metric_equivalents': comp.get('pitcher_metric_equivalents')})
