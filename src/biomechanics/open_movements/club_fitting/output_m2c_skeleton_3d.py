"""Standalone output: M2C_skeleton_3d."""
from .core import lowercase_output_keys, resolve

def M2C_skeleton_3d(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'M2C_skeleton_3d': comp.get('M2C_skeleton_3d')})
