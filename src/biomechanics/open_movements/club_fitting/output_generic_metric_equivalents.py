"""Individually callable generic output: generic_metric_equivalents."""
from typing import Any
from .core import lowercase_output_keys, resolve

def generic_metric_equivalents(ctx: Any = None, *, computation: Any = None) -> dict[str, Any]:
    """Return the generic_metric_equivalents output dictionary for the supplied context or computation."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'generic_metric_equivalents': comp.get('generic_metric_equivalents')})
