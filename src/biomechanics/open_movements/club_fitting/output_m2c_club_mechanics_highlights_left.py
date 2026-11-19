"""Individually callable generic output: M2C_club_mechanics_highlights_left."""
from .core import resolve

def M2C_club_mechanics_highlights_left(ctx=None, *, computation=None):
    comp = resolve(computation=computation, ctx=ctx)
    return {'M2C_club_mechanics_highlights_left': comp.get('M2C_club_mechanics_highlights_left')}
