"""Standalone output: pitcher_metric_equivalents."""
from typing import Any
from .core import lowercase_output_keys, resolve

def pitcher_metric_equivalents(ctx: Any = None, *, computation: Any = None) -> dict[str, Any]:
    """Return the pitcher_metric_equivalents output dictionary for the supplied context or computation."""
    comp = resolve(computation=computation, ctx=ctx)
    return lowercase_output_keys({'pitcher_metric_equivalents': comp.get('pitcher_metric_equivalents')})
