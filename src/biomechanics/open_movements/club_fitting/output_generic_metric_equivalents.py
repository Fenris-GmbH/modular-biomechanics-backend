"""Individually callable generic output: generic_metric_equivalents."""
from .core import resolve

def generic_metric_equivalents(ctx=None, *, computation=None):
    comp = resolve(computation=computation, ctx=ctx)
    return {'generic_metric_equivalents': comp.get('generic_metric_equivalents')}
