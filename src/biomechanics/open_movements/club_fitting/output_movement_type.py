"""Standalone output: movement_type."""
from typing import Any
from .core import lowercase_output_keys, resolve

def movement_type(ctx: Any = None, *, computation: Any = None) -> dict[str, Any]:
    """Return the movement_type output dictionary for the supplied context or computation."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'movement_type': comp.get('movement_type')})
