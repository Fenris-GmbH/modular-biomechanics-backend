"""Standalone output: M2C_club_mechanics_highlights."""
from .core import resolve

def M2C_club_mechanics_highlights(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return {'M2C_club_mechanics_highlights': comp.get('M2C_club_mechanics_highlights')}
