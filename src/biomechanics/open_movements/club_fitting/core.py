"""Shared state and dependency scheduler for standalone club-fitting outputs."""

from copy import deepcopy
from . import (
    common,
    hitting,
    skeleton,
    rotational,
    pitcher_metrics,
    helpers,
    generic_widgets,
    generic_metrics,
)

KEYS_HITTING = (
    "movement_type",
    "M2C_insights",
    "Performance_metrics",
    "M2C_power",
    "M2C_club_mechanics",
    "M2C_club_mechanics_highlights",
    "M2C_force_vectors_bat_3d",
    "M2C_wrist",
    "inPlaneMetrics",
    "outOfPlaneMetrics",
    "M2C_inplane_vectors",
    "M2C_inplane_vectors_3d",
    "M2C_kinetics_3d",
    "M2C_skeleton_3d",
    "generic",
)
KEYS_PITCHING = (
    "movement_type",
    "M2C_insights",
    "Performance_metrics",
    "M2C_power",
    "M2C_club_mechanics",
    "M2C_club_mechanics_highlights",
    "M2C_wrist",
    "M2C_force_vectors_hand_r_3d",
    "M2C_skeleton_3d",
    "pitcher_metric_equivalents",
    "generic",
)
GENERIC_WIDGET_KEYS = (
    "M2C_power",
    "M2C_club_mechanics",
    "M2C_club_mechanics_highlights",
    "M2C_wrist",
)


def lowercase_output_keys(value):
    """Lowercase string keys recursively in a returned output value."""
    if isinstance(value, dict):
        lowered = {}
        for key, item in value.items():
            lowered_key = key.lower() if isinstance(key, str) else key
            if lowered_key in lowered:
                raise ValueError(
                    f"Output keys collide after lower-casing: {lowered_key!r}"
                )
            lowered[lowered_key] = lowercase_output_keys(item)
        return lowered
    if isinstance(value, list):
        return [lowercase_output_keys(item) for item in value]
    if isinstance(value, tuple):
        return tuple(lowercase_output_keys(item) for item in value)
    return value

MODULES = {
    "common": common,
    "hitting": hitting,
    "skeleton": skeleton,
    "rotational": rotational,
    "pitcher_metrics": pitcher_metrics,
    "helpers": helpers,
    "generic_widgets": generic_widgets,
    "generic_metrics": generic_metrics,
}


class ClubFittingComputation:
    """Per-movement staged calculations. Do not reuse after changing inputs."""

    def __init__(self, ctx):
        raw = ctx if isinstance(ctx, dict) else getattr(ctx, "raw_json_3d", None)
        if raw is None:
            raise TypeError(
                "Context needs raw_json_3d; transformed ctx.markers alone is insufficient"
            )
        # Legacy expects the wrapper, while prepare(ctx) usually receives the inner JSON.
        if "data_3d_bio" not in raw and "configuration" in raw and "data" in raw:
            raw = {"data_3d_bio": raw}
        self._state = {"json3d": deepcopy(raw)}
        self._done = set()
        self._stage("common")

    @property
    def movement_type(self):
        return self._state["movement_type"]

    @property
    def available_keys(self):
        if self.movement_type == "hitting":
            return KEYS_HITTING
        if self.movement_type == "pitching":
            return KEYS_PITCHING
        self._stage("generic_widgets")
        extra = tuple(
            k for k in self._state["generic_outputs"] if k not in GENERIC_WIDGET_KEYS
        )
        return (
            "movement_type",
            *GENERIC_WIDGET_KEYS,
            *extra,
            "generic",
            "M2C_skeleton_3d",
        )

    def _stage(self, name):
        if name in self._done:
            return
        dependencies = {
            "hitting": (),
            "skeleton": (
                ("hitting",) if self._state.get("movement_type") == "hitting" else ()
            ),
            "rotational": (),
            "pitcher_metrics": ("rotational",),
            "helpers": ("rotational",),
            "generic_widgets": ("helpers",),
            "generic_metrics": ("helpers",),
        }.get(name, ())
        for dep in dependencies:
            self._stage(dep)
        self._state.update(MODULES[name].calculate(self._state))
        self._done.add(name)

    def get(self, key):
        if key == "movement_type":
            return self.movement_type
        if key not in self.available_keys:
            raise KeyError(
                f"{key!r} not produced for {self.movement_type!r}; available: {self.available_keys}"
            )
        if key == "M2C_skeleton_3d":
            self._stage("skeleton")
            return lowercase_output_keys(self._state[key])
        if key == "generic":
            self._stage("generic_metrics")
            return lowercase_output_keys(self._state[key])
        if self.movement_type == "hitting":
            self._stage("hitting")
            return lowercase_output_keys(
                self._state["M2C_force_vectors_shaft_3d"]
                if key == "M2C_force_vectors_bat_3d"
                else self._state[key]
            )
        if self.movement_type == "pitching":
            if key in ("Performance_metrics", "M2C_force_vectors_hand_r_3d"):
                self._stage("pitcher_metrics")
                return lowercase_output_keys(self._state[
                    (
                        "Pitcher_performance_metrics"
                        if key == "Performance_metrics"
                        else "M2C_force_vectors_hand_3d"
                    )
                ])
            self._stage("generic_widgets")
            return lowercase_output_keys(self._state["pitcher_outputs"][key])
        self._stage("generic_widgets")
        return lowercase_output_keys(self._state["generic_outputs"][key])

    def compute_all(self):
        return lowercase_output_keys(
            {key: self.get(key) for key in self.available_keys}
        )


def resolve(computation=None, ctx=None):
    if computation is None:
        if ctx is None:
            raise ValueError("Pass ctx or computation=ClubFittingComputation(ctx)")
        computation = ClubFittingComputation(ctx)
    return computation
