"""Standalone output: inPlaneMetrics."""
from typing import Any
from .core import lowercase_output_keys, resolve

def inPlaneMetrics(ctx: Any = None, *, computation: Any = None) -> dict[str, Any]:
    """Return the inPlaneMetrics output dictionary for the supplied context or computation."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'inPlaneMetrics': comp.get('inPlaneMetrics')})
