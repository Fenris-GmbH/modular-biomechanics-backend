"""Individually callable generic output: generic_widgets_error."""
from .core import resolve

def generic_widgets_error(ctx=None, *, computation=None):
    comp = resolve(computation=computation, ctx=ctx)
    return {'generic_widgets_error': comp.get('generic_widgets_error')}
