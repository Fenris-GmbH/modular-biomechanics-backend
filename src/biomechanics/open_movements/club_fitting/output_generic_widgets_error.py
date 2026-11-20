"""Individually callable generic output: generic_widgets_error."""
from typing import Any
from .core import lowercase_output_keys, resolve

def generic_widgets_error(ctx: Any = None, *, computation: Any = None) -> dict[str, Any]:
    """Return the generic_widgets_error output dictionary for the supplied context or computation."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'generic_widgets_error': comp.get('generic_widgets_error')})
