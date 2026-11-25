"""Declare club-fitting calculations and the values each consumes."""

from dataclasses import dataclass
from functools import partial
from inspect import signature
from typing import Callable

import numpy as np

from . import (
    common,
    generic_metrics,
    generic_widgets as widgets,
    helpers,
    hitting,
    pitcher_metrics,
    skeleton,
)
from .rotational import rotational_velocity_about_spine


@dataclass(frozen=True)
class Node:
    name: str
    calculate: Callable
    dependencies: tuple[str, ...]
    outputs: tuple[str, ...]


class Graph:
    def __init__(self):
        self.producers = {}
        self.metrics = {}
        self.node_names = set()

    def add(self, function, outputs, *, inputs=None, name=None):
        if isinstance(outputs, str):
            outputs = tuple(outputs.split())
        if inputs is None:
            inputs = tuple(signature(function).parameters)
        node = Node(name or function.__name__, function, tuple(inputs), outputs)
        if node.name in self.node_names:
            raise ValueError(f"Duplicate node name {node.name}")
        for index, output in enumerate(outputs):
            if output in self.producers:
                raise ValueError(f"Duplicate producer for {output}")
            self.producers[output] = node, index
        self.node_names.add(node.name)

    def output(self, key, value=None):
        self.metrics[key] = key if value is None else value

    def required(self, value):
        """Inspect leaf dependencies without executing any calculations."""
        if value not in self.producers:
            return {value}
        node, _ = self.producers[value]
        return set().union(*(self.required(dep) for dep in node.dependencies))


def _marker(name, oriented_pose, lefty, movement_type):
    names = common.LEFT_MARKERS if lefty else common.RIGHT_MARKERS
    marker = oriented_pose[names.index(name)]
    if movement_type == "generic" and not np.isfinite(marker).all():
        return common.interpolate_marker(marker)
    return marker


def _missing_joints(oriented_pose, lefty, movement_type):
    if movement_type != "generic":
        return []
    names = common.LEFT_MARKERS if lefty else common.RIGHT_MARKERS
    return [
        name
        for name, marker in zip(names, oriented_pose)
        if not name.startswith("pose_club_") and not np.isfinite(marker).all()
    ]


def _trajectory(track, raw_pose):
    frames = len(raw_pose[0])
    if len(track) < frames:
        raise ValueError(f"club_3d has {len(track)} frames, expected at least {frames}")
    return track[:frames]


def _skeleton(keys, *markers):
    return {name: marker.tolist() for name, marker in zip(keys, markers)}


def _common(graph, kind):
    graph.add(common.raw_pose, "raw_pose")
    graph.add(common.orient_pose, "oriented_pose")
    graph.add(lambda frame_rate: 1 / frame_rate, "dt", name="frame_interval")
    graph.add(common.body_mass, "W mass_assumed")
    graph.add(lambda raw_pose: raw_pose.shape[1], "nframes", name="frame_count")
    for name in common.RIGHT_MARKERS:
        graph.add(partial(_marker, name), name, name="marker." + name)
    graph.add(_missing_joints, "generic_missing_joints")
    keys = skeleton.BODY_SKELETON
    if kind == "hitting":
        keys += ("club_main", "club_mid", "club_head")
    graph.add(partial(_skeleton, keys), "M2C_skeleton_3d", inputs=keys, name="skeleton")
    graph.add(_body_com, "body_com body_segments", inputs=BODY_COM_MARKERS, name="body_com")
    graph.add(widgets.filter_body_com, "filtered_com")
    graph.add(generic_metrics.diagnostics, "generic")
    graph.output("movement_type")
    if kind != "generic":
        graph.add(common.torso_turn, "x_factor_raw")
        graph.add(common.turn_insight, "x_factor x_factor_stretch")
        graph.add(common.hand_arc, "arc_width arc_width_inch")


def _hitting_graph(graph):
    for field, point in (("club_head", "end"), ("club_main", "start"), ("club_mid", "mid")):
        graph.add(_trajectory, field, inputs=("club_" + point + "_raw", "raw_pose"), name=field)
    graph.add(hitting.raw_midhand, "MH_point", inputs=("lefty", "raw_pose"))
    graph.add(
        hitting.shaft_angle,
        "lead_forearm_shaft_angle_deg_2d",
        inputs=("club_mid", "lefty", "raw_pose"),
    )
    graph.add(hitting.ground_power, "vertical_power")
    graph.add(hitting.lead_hand_power, "hand_power")
    graph.add(hitting.equipment_mass, "shaft_mass club_head_mass club_mass")
    graph.add(hitting.equipment_com, "club_COM")
    graph.add(hitting.head_motion, "club_speed F_club_head")
    graph.add(hitting.head_power, "club_head_power")
    graph.add(hitting.power_index, "power_index2")
    graph.add(hitting.com_motion, "club_COM_speed_X club_COM_speed_Y club_COM_speed_Z")
    graph.add(
        hitting.com_velocity, "club_COM_speed_X_filt club_COM_speed_Y_filt club_COM_speed_Z_filt"
    )
    graph.add(hitting.com_acceleration, "club_COM_acc_X club_COM_acc_Y club_COM_acc_Z")
    graph.add(hitting.shaft_power, "P_shaft")
    graph.add(hitting.downswing_start, "index_min_club_COM_speed")
    graph.add(hitting.midhand_force, "F_MH_vec")
    graph.add(hitting.shaft_moment, "h_dot_C_vec")
    graph.add(hitting.midhand_moments, "MH_force_moment_vec M_MH_vec")
    graph.add(hitting.net_moment, "T_net_vec")
    graph.add(hitting.plane_axis, "B_hat")
    graph.add(hitting.tangent_axis, "T_hat")
    graph.add(hitting.normal_axis, "N_hat")
    graph.add(hitting.tangential_force, "F_tan_shaft")
    graph.add(hitting.normal_force, "F_MH_normal F_rad_shaft")
    graph.add(hitting.grip_force, "Grip")
    graph.add(hitting.couple_projection, "C_in_plane")
    graph.add(hitting.force_moment_projection, "M_in_plane")
    graph.add(hitting.net_moment_projection, "T_in_plane")
    graph.add(
        hitting.out_of_plane_moments,
        "F_MH_swing_axis M_MH_tangential M_MH_normal MHforce_moment_tangential MHforce_moment_normal",
    )
    graph.add(hitting.out_of_plane_net_moment, "T_net_tangential T_net_normal")
    graph.add(hitting.shaft_force_vectors, "F_tan_vec F_rad_vec")
    graph.add(
        hitting.projected_com,
        "proj_club_COM club_COM_2d",
        inputs=(
            "F_rad_vec",
            "F_tan_vec",
            "club_end_raw",
            "club_head_mass",
            "club_main",
            "club_mass",
            "index_min_club_COM_speed",
            "shaft_mass",
        ),
    )
    graph.add(hitting.highlights, "M2C_club_mechanics_highlights")
    graph.add(hitting.insights, "M2C_insights")
    graph.add(hitting.power, "M2C_power")
    graph.add(hitting.mechanics, "M2C_club_mechanics")
    graph.add(
        hitting.bat_force_vectors,
        "M2C_force_vectors_shaft_3d",
        inputs=("F_rad_vec", "F_tan_vec", "lefty", "raw_pose"),
    )
    graph.add(hitting.wrist, "M2C_wrist")
    graph.add(hitting.in_plane_metrics, "inPlaneMetrics")
    graph.add(hitting.out_of_plane_metrics, "outOfPlaneMetrics")
    graph.add(hitting.kinetics, "M2C_kinetics_3d")
    graph.add(hitting.in_plane_vectors, "M2C_inplane_vectors")
    graph.add(hitting.in_plane_vectors_3d, "M2C_inplane_vectors_3d")
    graph.add(hitting.performance, "Performance_metrics")
    for key in (
        "M2C_insights",
        "Performance_metrics",
        "M2C_power",
        "M2C_club_mechanics",
        "M2C_club_mechanics_highlights",
        "M2C_wrist",
        "inPlaneMetrics",
        "outOfPlaneMetrics",
        "M2C_inplane_vectors",
        "M2C_inplane_vectors_3d",
        "M2C_kinetics_3d",
    ):
        graph.output(key)
    graph.output("M2C_force_vectors_bat_3d", "M2C_force_vectors_shaft_3d")


GENERIC_WIDGET_KEYS = (
    "M2C_power",
    "M2C_club_mechanics",
    "M2C_club_mechanics_highlights",
    "M2C_wrist",
)

BODY_COM_MARKERS = (
    "pelvis",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
    "foot_l",
    "foot_r",
    "skull",
    "skullbase",
    "neck",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "hand_l",
    "hand_r",
)


def _body_com(*markers):
    return helpers._body_com(dict(zip(BODY_COM_MARKERS, markers)))


def _wrist_speed(wrist, frame_rate, dt):
    return np.linalg.norm(helpers._deriv(wrist, 1, frame_rate, dt), axis=1)


def _faster_wrist(left, right):
    peaks = {"l": np.max(left), "r": np.max(right)}
    return max(peaks, key=peaks.get)


def _throwing_marker(lefty, left, right):
    return left if lefty else right


def _limb(graph, suffix, *, pitching):
    if pitching:
        shoulder, elbow, wrist, hand = (
            "throwing_" + name for name in ("shoulder", "elbow", "wrist", "hand")
        )
        start, stop, end, reference = "p4", "release_stop", "p8", "p7"
        highlights_start = "p1"
    else:
        shoulder, elbow, wrist, hand = (
            name + suffix for name in ("shoulder", "elbow", "wrist", "hand")
        )
        start = highlights_start = "zero"
        stop, end, reference = "nframes", "last_frame", "reference" + suffix

    def limb(function, output, inputs):
        graph.add(
            function,
            " ".join(value + suffix for value in output.split()),
            inputs=inputs.split(),
            name=function.__name__ + suffix,
        )

    limb(helpers.filter_hand, "filtered_hand", f"{hand} frame_rate")
    limb(helpers.hand_force, "hand_force", f"filtered_hand{suffix} dt W")
    limb(widgets.hand_power, "hand_power", f"{hand} hand_force{suffix} frame_rate")
    limb(
        widgets.arm_power,
        "arm_power distal_power",
        f"{shoulder} {elbow} {wrist} {hand} frame_rate dt W",
    )
    limb(
        helpers.hand_force_axes,
        "tangential_force axial_force tangent_axis axial_axis",
        f"filtered_hand{suffix} {elbow} hand_force{suffix}",
    )
    limb(
        widgets.elbow_kinetics,
        "elbow_force elbow_torque",
        f"{elbow} {wrist} {hand} frame_rate dt W",
    )
    graph.add(
        partial(np.linalg.norm, axis=1),
        "hand_magnitude" + suffix,
        inputs=("hand_force" + suffix,),
        name="hand_magnitude" + suffix,
    )
    limb(widgets.elbow_flexion, "elbow_flexion", f"{shoulder} {elbow} {wrist}")
    limb(widgets.wrist, "M2C_wrist", f"{elbow} {wrist} {hand} elbow_flexion{suffix} frame_rate dt")
    limb(
        widgets.power,
        "M2C_power",
        f"vertical_power hand_power{suffix} arm_power{suffix} distal_power{suffix} {start} {stop} W",
    )
    mechanics_inputs = " ".join(
        key + suffix
        for key in (
            "tangential_force",
            "axial_force",
            "elbow_force",
            "hand_magnitude",
            "elbow_torque",
        )
    )
    limb(widgets.mechanics, "M2C_club_mechanics", mechanics_inputs)
    limb(
        widgets.highlights,
        "M2C_club_mechanics_highlights",
        f"{mechanics_inputs} {highlights_start} {end} {reference} frame_rate",
    )


def _body_widgets(graph, kind):
    pitching = kind == "pitching"
    graph.add(
        widgets.vertical_power,
        "vertical_power",
        inputs=("filtered_com", "dt", "W"),
    )
    if pitching:
        for part in ("shoulder", "elbow", "wrist", "hand"):
            graph.add(
                _throwing_marker,
                "throwing_" + part,
                inputs=("lefty", part + "_l", part + "_r"),
                name="throwing_" + part,
            )
        graph.add(lambda p7: p7 + 1, "release_stop", name="release_stop")
        _limb(graph, "", pitching=True)
        for key in GENERIC_WIDGET_KEYS:
            graph.output(key)
        _pitcher_graph(graph)
        return

    graph.add(lambda nframes: nframes - 1, "last_frame", name="last_frame")
    for side in ("l", "r"):
        suffix = "_" + side
        graph.add(
            _wrist_speed,
            "wrist_speed" + suffix,
            inputs=("wrist" + suffix, "frame_rate", "dt"),
            name="wrist_speed" + suffix,
        )
        graph.add(
            lambda speed: int(np.argmax(speed)),
            "reference" + suffix,
            inputs=("wrist_speed" + suffix,),
            name="reference" + suffix,
        )
        _limb(graph, suffix, pitching=False)
    graph.add(
        widgets.generic_equivalents,
        "generic_metric_equivalents",
        inputs=("primary_side", "reference_l", "reference_r", "handedness_specified", "dt"),
    )
    graph.output("generic_metric_equivalents")


def _pitcher_graph(graph):
    rotations = {
        "pelvis_rot_speed": ("hip_l", "hip_r"),
        "UT_rot_speed": ("shoulder_l", "shoulder_r"),
        "arm_rot_speed": ("shoulder_l", "elbow_l"),
        "hand_rot_speed": ("elbow_l", "wrist_l"),
        "trail_arm_rot_speed": ("shoulder_r", "elbow_r"),
        "trail_hand_rot_speed": ("elbow_r", "wrist_r"),
    }
    for value, endpoints in rotations.items():
        graph.add(
            partial(rotational_velocity_about_spine, filter_win=41),
            value,
            inputs=(*endpoints, "spine_high", "spine_low", "dt"),
            name=value,
        )
    graph.add(
        _throwing_marker,
        "throwing_arm_rot_speed",
        inputs=("lefty", "arm_rot_speed", "trail_arm_rot_speed"),
        name="throwing_arm_rotation",
    )
    graph.add(
        _throwing_marker,
        "throwing_hand_rot_speed",
        inputs=("lefty", "hand_rot_speed", "trail_hand_rot_speed"),
        name="throwing_hand_rotation",
    )
    graph.add(pitcher_metrics.performance, "Performance_metrics")
    graph.add(
        pitcher_metrics.throwing_hand_vectors,
        "M2C_force_vectors_hand_r_3d",
        inputs=(
            "throwing_hand",
            "hand_force",
            "tangential_force",
            "axial_force",
            "tangent_axis",
            "axial_axis",
            "frame_rate",
        ),
    )
    graph.add(widgets.pitcher_insights, "M2C_insights")
    graph.add(
        lambda: dict(
            helpers.WIDGET_EQUIVALENTS, time_to_impact="for pitchers = ms relative to release (p7)"
        ),
        "pitcher_metric_equivalents",
        name="pitcher_metric_equivalents",
    )
    for key in (
        "Performance_metrics",
        "M2C_force_vectors_hand_r_3d",
        "M2C_insights",
        "pitcher_metric_equivalents",
    ):
        graph.output(key)


def build_graph(kind, handedness_specified=True):
    graph = Graph()
    _common(graph, kind)
    if kind == "hitting":
        _hitting_graph(graph)
    else:
        _body_widgets(graph, kind)
    if kind == "generic":
        if handedness_specified:
            graph.add(lambda lefty: "l" if lefty else "r", "primary_side", name="primary_side")
        else:
            graph.add(
                _faster_wrist,
                "primary_side",
                inputs=("wrist_speed_l", "wrist_speed_r"),
                name="primary_side",
            )
    graph.output("M2C_skeleton_3d")
    graph.output("generic")
    return graph


GRAPHS = {kind: build_graph(kind) for kind in ("hitting", "pitching", "generic")}

GENERIC_INFERRED_GRAPH = build_graph("generic", handedness_specified=False)
