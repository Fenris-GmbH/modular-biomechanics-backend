"""Demand-driven club-fitting calculations and per-recording cache."""

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real
from threading import RLock
from typing import Any, Optional

import numpy as np

from biomechanics.utils.output_keys import lowercase_output_keys

from .graph import GENERIC_INFERRED_GRAPH, GENERIC_WIDGET_KEYS, GRAPHS
from .common import raw_pose as prepare_pose
from .helpers import _clean

EVENTS = frozenset(f"p{index}" for index in range(1, 9))
TRACKS = {"club_start_raw": "start", "club_mid_raw": "mid", "club_end_raw": "end"}
OPTIONS = EVENTS | TRACKS.keys() | {"bat_type"}


class ClubFittingComputation:
    """One recording, with optional inputs bound per dependency on demand."""

    def __init__(
        self,
        *,
        pose3d,
        movement_type: str,
        lefty: bool,
        body_mass,
        frame_rate: float,
        down_the_line: bool,
        handedness_specified: bool = True,
    ) -> None:
        missing = [
            name
            for name, value in (
                ("pose3d", pose3d),
                ("movement_type", movement_type),
                ("lefty", lefty),
                ("frame_rate", frame_rate),
                ("down_the_line", down_the_line),
            )
            if value is None
        ]
        if missing:
            raise ValueError("Missing club-fitting initialization fields: " + ", ".join(missing))
        if not isinstance(frame_rate, Real) or not np.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("frame_rate must be a positive finite number")
        for name, value in (("lefty", lefty), ("down_the_line", down_the_line)):
            if value not in (True, False):
                raise ValueError(f"{name} must be a boolean")

        kind = str(movement_type).lower()
        if kind not in ("hitting", "pitching"):
            kind = "generic"
        self._fixed = {
            "pose3d": deepcopy(pose3d),
            "movement_type": kind,
            "lefty": bool(lefty),
            "weight": body_mass,
            "frame_rate": frame_rate,
            "down_the_line": bool(down_the_line),
            "handedness_specified": True
            if handedness_specified is None
            else bool(handedness_specified),
            "zero": 0,
        }
        self._graph = (
            GENERIC_INFERRED_GRAPH
            if kind == "generic" and not self._fixed["handedness_specified"]
            else GRAPHS[kind]
        )
        self._options = {}
        self._input_sources = {}
        self._input_defaults = {}
        self._validated_options = set()
        self._revisions = {}
        self._cache = {}
        self._lock = RLock()

    @property
    def movement_type(self) -> str:
        return self._fixed["movement_type"]

    @property
    def available_keys(self) -> tuple[str, ...]:
        """Return applicable outputs without calculating their numerical payloads."""
        if self.movement_type != "generic":
            return tuple(self._graph.metrics)
        with self._lock:
            side = self._run(("primary_side",), {})["primary_side"]
        other_suffix = "_right" if side == "l" else "_left"
        return (
            "movement_type",
            *GENERIC_WIDGET_KEYS,
            *(key + other_suffix for key in GENERIC_WIDGET_KEYS),
            "generic_metric_equivalents",
            "M2C_skeleton_3d",
            "generic",
        )

    def _generic_root(self, key, side):
        if key in GENERIC_WIDGET_KEYS:
            return key + "_" + side
        other = "right" if side == "l" else "left"
        suffix = "_" + other
        if key.endswith(suffix) and key[: -len(suffix)] in GENERIC_WIDGET_KEYS:
            return key[: -len(suffix)] + "_" + other[0]
        return self._graph.metrics[key]

    @staticmethod
    def _track(source, point):
        if not isinstance(source, Mapping) or not source:
            raise ValueError("club_3d must be a nonempty mapping of frames to points")
        try:
            rows = np.array([frame[point] for frame in source.values()], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"club_3d must provide {point!r} coordinates for each frame") from exc
        if rows.shape != (len(source), 3):
            raise ValueError(f"club_3d {point!r} coordinates must have shape (frames, 3)")
        return rows

    def _set_input_defaults(self, **defaults):
        """Attach recording defaults at the facade boundary before first use."""
        self._input_defaults = defaults

    def _prepare_options(self, key, required, supplied):
        options = self._options.copy()
        revisions = self._revisions.copy()
        sources = self._input_sources.copy()
        validated = self._validated_options.copy()
        for argument, leaves in (
            ("critical_events", EVENTS),
            ("club_3d", TRACKS.keys()),
            ("bat_type", {"bat_type"}),
        ):
            if not required & leaves:
                continue
            incoming = supplied.get(argument)
            if incoming is None and argument not in sources:
                incoming = self._input_defaults.get(argument)
            if incoming is not None:
                # Bind the source as a unit so tracks from two recordings
                # cannot be mixed when different metrics use different points.
                sources[argument] = deepcopy(incoming)
                validated.difference_update(leaves)
        events = sources.get("critical_events")
        track = sources.get("club_3d")
        bat = sources.get("bat_type")
        raw_pose = None

        for name in sorted((required & OPTIONS) - validated):
            if name in EVENTS and events is not None:
                if not isinstance(events, Mapping):
                    raise ValueError(f"{key} requires critical_events as a mapping")
                if name not in events:
                    raise ValueError(f"{key} requires critical_events[{name!r}]")
                value = events[name]
                if raw_pose is None:
                    # Shape validation only. Cache the pose with numerical nodes
                    # after the entire request succeeds.
                    cached_pose = self._cache.get("raw_pose")
                    raw_pose = (
                        cached_pose[1][0] if cached_pose else prepare_pose(self._fixed["pose3d"])
                    )
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Integral)
                    or not 0 <= value < raw_pose.shape[1]
                ):
                    raise ValueError(
                        f"{key}: critical_events[{name!r}] must be an integer frame within pose3d"
                    )
                value = int(value)
            elif name in TRACKS and track is not None:
                try:
                    value = self._track(track, TRACKS[name])
                except ValueError as exc:
                    raise ValueError(f"{key}: {exc}") from exc
            elif name == "bat_type" and bat is not None:
                if not isinstance(bat, str) or not bat.strip():
                    raise ValueError(f"{key} requires bat_type as a nonempty string")
                value = bat
            else:
                label = (
                    f"critical_events[{name!r}]"
                    if name in EVENTS
                    else "club_3d"
                    if name in TRACKS
                    else name
                )
                raise ValueError(f"{key} requires {label}")

            old = options.get(name)
            unchanged = (
                np.array_equal(old, value, equal_nan=True)
                if isinstance(value, np.ndarray)
                else old == value
            )
            if not unchanged:
                options[name] = value
                revisions[name] = revisions.get(name, 0) + 1
            validated.add(name)

        return options, revisions, sources, validated

    def _run(self, roots, supplied, *, label=None):
        required = set().union(*(self._graph.required(root) for root in roots))
        options, revisions, sources, validated = self._prepare_options(
            label or roots[0], required, supplied
        )
        staged = {}
        resolved = {}

        def evaluate(value):
            if value in self._fixed:
                return self._fixed[value], ()
            if value in options:
                return options[value], ((value, revisions[value]),)
            if value in resolved:
                return resolved[value]

            node, _ = self._graph.producers[value]
            arguments = [evaluate(dep) for dep in node.dependencies]
            dependencies = tuple(sorted({item for _, stamp in arguments for item in stamp}))
            existing = self._cache.get(node.name)
            if existing is not None and existing[0] == dependencies:
                results = existing[1]
            else:
                try:
                    result = node.calculate(*(argument for argument, _ in arguments))
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"{label or roots[0]} ({node.name}): {exc}") from exc
                results = (result,) if len(node.outputs) == 1 else result
                staged[node.name] = dependencies, results
            for output, result in zip(node.outputs, results):
                resolved[output] = result, dependencies
            return resolved[value]

        results = {root: evaluate(root)[0] for root in roots}
        self._cache.update(staged)
        self._options = options
        self._revisions = revisions
        self._input_sources = sources
        self._validated_options = validated
        return results

    def get(self, key: str, *, club_3d=None, bat_type=None, critical_events=None) -> Any:
        """Calculate one output. Omitted arguments reuse successfully supplied values."""
        supplied = {"club_3d": club_3d, "bat_type": bat_type, "critical_events": critical_events}
        with self._lock:
            if self.movement_type != "generic":
                root = self._graph.metrics[key]
                result = self._run((root,), supplied, label=key)[root]
            else:
                result = self._get_generic(key)
            return lowercase_output_keys(result)

    def _get_generic(self, key):
        if key in ("M2C_other_side_error", "generic_widgets_error"):
            side = self._run(("primary_side",), {})["primary_side"]
            if key == "M2C_other_side_error":
                side = "r" if side == "l" else "l"
            try:
                self._run(
                    tuple(widget + "_" + side for widget in GENERIC_WIDGET_KEYS), {}, label=key
                )
            except (TypeError, ValueError, IndexError, FloatingPointError) as exc:
                cause = exc.__cause__ or exc
                return f"{type(cause).__name__}: {cause}"
            raise KeyError(f"{key!r} is only produced if the corresponding widgets fail")

        if key in self._graph.metrics and key != "generic_metric_equivalents":
            return self._run((self._graph.metrics[key],), {}, label=key)[self._graph.metrics[key]]
        side = self._run(("primary_side",), {})["primary_side"]
        root = self._generic_root(key, side)
        return _clean(self._run((root,), {}, label=key)[root])

    def compute_all(self, *, club_3d=None, bat_type=None, critical_events=None) -> dict[str, Any]:
        """Assemble the applicable public outputs using the same graph."""
        supplied = {"club_3d": club_3d, "bat_type": bat_type, "critical_events": critical_events}
        with self._lock:
            if self.movement_type == "generic":
                return lowercase_output_keys(self._all_generic())

            roots = self._graph.metrics
            values = self._run(
                tuple(roots.values()), supplied, label=f"{self.movement_type} club fitting"
            )
            return lowercase_output_keys({key: values[root] for key, root in roots.items()})

    def _all_generic(self):
        result = {"movement_type": self.movement_type}
        try:
            side = self._run(("primary_side",), {})["primary_side"]
            roots = {key: key + "_" + side for key in GENERIC_WIDGET_KEYS}
            primary = self._run(tuple(roots.values()), {})
            result.update({key: primary[root] for key, root in roots.items()})
        except (TypeError, ValueError, IndexError, FloatingPointError) as exc:
            cause = exc.__cause__ or exc
            result["generic_widgets_error"] = f"{type(cause).__name__}: {cause}"
        else:
            other = "r" if side == "l" else "l"
            suffix = "_right" if other == "r" else "_left"
            try:
                roots = {key + suffix: key + "_" + other for key in GENERIC_WIDGET_KEYS}
                secondary = self._run(tuple(roots.values()), {})
                result.update({key: secondary[root] for key, root in roots.items()})
            except (TypeError, ValueError, IndexError, FloatingPointError) as exc:
                cause = exc.__cause__ or exc
                result["M2C_other_side_error"] = f"{type(cause).__name__}: {cause}"
            result["generic_metric_equivalents"] = self._get_generic("generic_metric_equivalents")
        for key in ("M2C_skeleton_3d", "generic"):
            result[key] = self._get_generic(key)
        return _clean(result)


def resolve(
    computation: Optional[ClubFittingComputation] = None,
    *,
    pose3d=None,
    movement_type=None,
    lefty=None,
    body_mass=None,
    frame_rate=None,
    down_the_line=None,
    handedness_specified=True,
) -> ClubFittingComputation:
    """Use a cached computation or construct one from explicit movement fields."""
    if computation is not None:
        return computation
    required = {
        "pose3d": pose3d,
        "movement_type": movement_type,
        "lefty": lefty,
        "frame_rate": frame_rate,
        "down_the_line": down_the_line,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Pass computation=ClubFittingComputation(...) or provide "
            "explicit initialization fields: " + ", ".join(missing)
        )
    return ClubFittingComputation(
        pose3d=pose3d,
        movement_type=movement_type,
        lefty=lefty,
        body_mass=body_mass,
        frame_rate=frame_rate,
        down_the_line=down_the_line,
        handedness_specified=handedness_specified,
    )
