"""Standalone output: M2C_inplane_vectors_3d."""
from .core import resolve

def M2C_inplane_vectors_3d(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return {'M2C_inplane_vectors_3d': comp.get('M2C_inplane_vectors_3d')}
