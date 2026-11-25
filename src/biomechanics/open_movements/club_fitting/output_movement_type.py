"""Standalone output: movement_type."""
from typing import Any
from .core import lowercase_output_keys, resolve

def movement_type(*, computation: Any = None, pose3d=None, movement_type=None, lefty=None, body_mass=None, frame_rate=None, down_the_line=None, handedness_specified=True, club_3d=None, bat_type=None, critical_events=None) -> dict[str, Any]:
    """Return the movement_type output dictionary for the supplied explicit inputs or computation."""
    comp = resolve(computation=computation, pose3d=pose3d, movement_type=movement_type, lefty=lefty, body_mass=body_mass, frame_rate=frame_rate, down_the_line=down_the_line, handedness_specified=handedness_specified)
    return lowercase_output_keys({'movement_type': comp.get('movement_type', club_3d=club_3d, bat_type=bat_type, critical_events=critical_events)})
