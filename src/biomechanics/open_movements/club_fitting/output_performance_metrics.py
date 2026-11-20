"""Standalone output: Performance_metrics."""
from typing import Any
from .core import lowercase_output_keys, resolve

def Performance_metrics(ctx: Any = None, *, computation: Any = None) -> dict[str, Any]:
    """Return the Performance_metrics output dictionary for the supplied context or computation."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'Performance_metrics': comp.get('Performance_metrics')})
