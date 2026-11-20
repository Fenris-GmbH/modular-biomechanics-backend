"""Individually callable generic output: generic_metric_equivalents."""
from .core import lowercase_output_keys, resolve

def generic_metric_equivalents(ctx=None, *, computation=None):
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'generic_metric_equivalents': comp.get('generic_metric_equivalents')})
