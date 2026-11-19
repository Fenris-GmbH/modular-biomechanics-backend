"""Standalone output: M2C_club_mechanics."""
from .core import resolve

def M2C_club_mechanics(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return {'M2C_club_mechanics': comp.get('M2C_club_mechanics')}
