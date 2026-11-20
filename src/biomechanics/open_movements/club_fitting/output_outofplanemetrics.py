"""Standalone output: outOfPlaneMetrics."""
from typing import Any
from .core import lowercase_output_keys, resolve

def outOfPlaneMetrics(ctx: Any = None, *, computation: Any = None) -> dict[str, Any]:
    """Return the outOfPlaneMetrics output dictionary for the supplied context or computation."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'outOfPlaneMetrics': comp.get('outOfPlaneMetrics')})
