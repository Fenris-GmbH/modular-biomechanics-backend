"""Standalone output: generic."""
from typing import Any
from .core import lowercase_output_keys, resolve

def generic(ctx: Any = None, *, computation: Any = None) -> dict[str, Any]:
    """Return the generic output dictionary for the supplied context or computation."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'generic': comp.get('generic')})
