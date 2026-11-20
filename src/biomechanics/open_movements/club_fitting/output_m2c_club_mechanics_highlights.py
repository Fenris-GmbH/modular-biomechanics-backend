"""Standalone output: M2C_club_mechanics_highlights."""
from .core import lowercase_output_keys, resolve

def M2C_club_mechanics_highlights(ctx=None, *, computation=None):
    """Return the original one-key response, calculating only needed stages."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'M2C_club_mechanics_highlights': comp.get('M2C_club_mechanics_highlights')})
