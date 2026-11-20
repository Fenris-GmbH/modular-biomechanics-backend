"""Standalone output: M2C_club_mechanics_highlights."""
from typing import Any
from .core import lowercase_output_keys, resolve

def M2C_club_mechanics_highlights(ctx: Any = None, *, computation: Any = None) -> dict[str, Any]:
    """Return the M2C_club_mechanics_highlights output dictionary for the supplied context or computation."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'M2C_club_mechanics_highlights': comp.get('M2C_club_mechanics_highlights')})
