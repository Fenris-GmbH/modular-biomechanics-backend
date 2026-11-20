"""Individually callable generic output: M2C_other_side_error."""
from typing import Any
from .core import lowercase_output_keys, resolve

def M2C_other_side_error(ctx: Any = None, *, computation: Any = None) -> dict[str, Any]:
    """Return the M2C_other_side_error output dictionary for the supplied context or computation."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'M2C_other_side_error': comp.get('M2C_other_side_error')})
