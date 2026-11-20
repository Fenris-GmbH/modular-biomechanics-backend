"""Standalone output: M2C_wrist."""
from .core import lowercase_output_keys, resolve

def M2C_wrist(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'M2C_wrist': comp.get('M2C_wrist')})
