"""Standalone output: generic."""
from .core import lowercase_output_keys, resolve

def generic(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'generic': comp.get('generic')})
