"""Golden tests for modular force outputs against monolithic force-screen output."""

import json
import math
import numbers
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "tests" / "fixtures" / "golden_inputs_generic"
REF_OUTPUTS_DIR = PROJECT_ROOT / "ref_outputs"
EXCLUDE_KEYS = ["error_messages","generic","m2c_skeleton_3d","generic_metric_equivalents"]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from biomechanics.main import movement  # noqa: E402
from biomechanics.utils.test_helpers import CLUB_FITTING_METHODS,data_3d_bio_parser  # noqa: E402


def _compare_values(golden, actual, path="$", summary=None):
    """Collect all key and value differences without dumping long force arrays."""
    if summary is None:
        summary = {
            "missing": [],
            "extra": [],
            "values": [],
            "count": 0,
            "counts_by_output": {},
        }

    def record_value_difference(message):
        summary["count"] += 1
        output_name = path.removeprefix("$.").split(".", 1)[0].split("[", 1)[0]
        counts = summary["counts_by_output"]
        counts[output_name] = counts.get(output_name, 0) + 1
        if len(summary["values"]) < 20:
            summary["values"].append(message)

    if isinstance(golden, dict) and isinstance(actual, dict):
        golden_keys = set(golden)
        actual_keys = set(actual)
        summary["missing"].extend(
            f"{path}.{key}" for key in sorted(golden_keys - actual_keys)
        )
        summary["extra"].extend(
            f"{path}.{key}" for key in sorted(actual_keys - golden_keys)
        )
        for key in sorted(golden_keys & actual_keys):
            _compare_values(golden[key], actual[key], f"{path}.{key}", summary)
        return summary

    if isinstance(golden, list) and isinstance(actual, list):
        if len(golden) != len(actual):
            record_value_difference(
                f"{path}: list lengths differ ({len(golden)} != {len(actual)})"
            )
        for index, (expected_item, actual_item) in enumerate(zip(golden, actual)):
            _compare_values(expected_item, actual_item, f"{path}[{index}]", summary)
        return summary

    numeric = (
        isinstance(golden, numbers.Real)
        and not isinstance(golden, bool)
        and isinstance(actual, numbers.Real)
        and not isinstance(actual, bool)
    )
    if numeric:
        matches = (math.isnan(golden) and math.isnan(actual)) or math.isclose(
            golden, actual, rel_tol=1e-9, abs_tol=1e-12
        )
    else:
        matches = type(golden) is type(actual) and golden == actual

    if not matches:
        record_value_difference(f"{path}: expected {golden!r}, got {actual!r}")
    return summary


INPUT_FILES = sorted(
    file for file in INPUT_DIR.glob("*.json")
    if not file.stem.endswith("_2d")
)

FORCE_METHODS = (
    "calculate_force_vectors",
    "calculate_weight_shift",
    "calculate_forces",
    "calculate_force_time",
    "calculate_force_vectors_3d",
    "calculate_values_for_COP_trajectory",
    "calculate_COM_MODELS",
    "calculate_power",
    "calculate_energy",
    "calculate_impulse",
    "calculate_small_force_vectors",
    "calculate_COM_COP_traj_back_view",
    "calculate_critical_events",
    "calculate_frame_offset",
)
FORCE_METHOD_INPUTS = {
    "calculate_force_vectors": ("pose2d",),
    "calculate_weight_shift": ("pose2d",),
    "calculate_COM_MODELS": ("pose2d",),
}


@pytest.mark.parametrize("input_path", INPUT_FILES, ids=lambda path: path.stem)
def test_modular_force_output_matches_monolithic_golden(input_path):
    swing_id = input_path.stem
    golden_path = REF_OUTPUTS_DIR / swing_id / f"{swing_id}_forcescreen.json"
    assert golden_path.is_file(), f"Missing monolithic golden output: {golden_path}"

    input_json = _load_json(input_path)
    golden_output = _lowercase_keys(_load_json(golden_path))
    pose2d_path = input_path.with_name(f"{swing_id}_2d.json")
    pose2d = _load_json(pose2d_path) if pose2d_path.is_file() else None

    recording = input_json["data_3d_bio"]
    swing = recording["configuration"]["swing"]

    pose3d,frame_rate,weight,orientation,handedness,movement_name,input_critical_events,club_3d_markers = data_3d_bio_parser(input_json['data_3d_bio'])

    fixture_inputs = {
    "club_3d": club_3d_markers,
    "club_type": swing.get("clubType"),
    "critical_events": input_critical_events,
    "pose2d": pose2d,
    }
    session = movement(pose3d=pose3d,movement_type=movement_name,frame_rate =frame_rate,weight=weight,orientation=orientation,handedness=handedness)
    include_club = (
        movement_name == "full swing"
        and fixture_inputs["club_3d"] is not None
    )
    session.initialize_forces(
        club_3d=fixture_inputs["club_3d"],
        include_club=include_club,
        club_type=fixture_inputs["club_type"],
        critical_events=fixture_inputs["critical_events"],
    )
    modular_output = {}
    for method_name in FORCE_METHODS:
        method = getattr(session, method_name)
        method_kwargs = {
            name: fixture_inputs[name]
            for name in FORCE_METHOD_INPUTS.get(method_name, ())
            if fixture_inputs[name] is not None
        }
        result = method(**method_kwargs)
        if result:
            modular_output.update(result)

    differences = _compare_values(golden_output, modular_output)

    _assert_matches_golden(differences, "force screen")


def _load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _lowercase_keys(value):
    """Return a copy with every string key lower-cased recursively."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered_key = key.lower() if isinstance(key, str) else key
            if lowered_key in result:
                raise ValueError(
                    f"Reference keys collide after lower-casing: {lowered_key!r}"
                )
            result[lowered_key] = _lowercase_keys(item)
        return result
    if isinstance(value, list):
        return [_lowercase_keys(item) for item in value]
    return value


def _assert_matches_golden(differences, label):
    assert not differences[
        "missing"
    ], f"{label} output is missing monolithic keys:\n" + "\n".join(
        differences["missing"][:20]
    )
    assert not differences[
        "extra"
    ], f"{label} output has keys absent from the monolithic golden:\n" + "\n".join(
        differences["extra"][:20]
    )
    assert differences["count"] == 0, (
        f"{differences['count']} {label} values differ from the monolithic golden. "
        f"Counts by output: {differences['counts_by_output']}. "
        "First differences:\n" + "\n".join(differences["values"])
    )


@pytest.mark.parametrize("input_path", INPUT_FILES, ids=lambda path: path.stem)
def test_modular_basic_screen_methods_match_monolithic_golden(input_path):
    swing_id = input_path.stem
    golden_path = REF_OUTPUTS_DIR / swing_id / f"{swing_id}_basicscreen.json"
    assert golden_path.is_file(), f"Missing basic screen golden output: {golden_path}"

    input_json = _load_json(input_path)
    pose3d,frame_rate,weight,orientation,handedness,movement_name,input_critical_events,club_3d_markers = data_3d_bio_parser(input_json['data_3d_bio'])

    golden_output = _lowercase_keys(_load_json(golden_path))
    session = movement(pose3d=pose3d,movement_type=movement_name,frame_rate =frame_rate,weight=weight,orientation=orientation,handedness=handedness)

    modular_output = {
        **session.calculate_kinematic_sequence(),
        **session.calculate_COM(),
        **session.calculate_segment_angles(),
        **session.calculate_segment_displacements(),
        "time": session.ctx.time.tolist(),
    }
    # error_messages is monolithic diagnostic metadata, not a returned modular
    # method result. Compare every method result and the modular context time.
    golden_methods = {key: golden_output[key] for key in golden_output}
    for key in EXCLUDE_KEYS:
        try:
            golden_methods.pop(key)
        except Exception as e:
            pass
    differences = _compare_values(golden_methods, modular_output)
    _assert_matches_golden(differences, "basic screen")

HITTING_METHOD_INPUTS = {
    "m2c_insights": ("club_3d", "critical_events"),
    "m2c_wrist": ("club_3d",),
    "m2c_skeleton_3d": ("club_3d",),
    "performance_metrics": ("club_3d", "bat_type", "critical_events"),
    "m2c_power": ("club_3d", "bat_type", "critical_events"),
    "m2c_club_mechanics": ("club_3d", "bat_type", "critical_events"),
    "m2c_club_mechanics_highlights": ("club_3d", "bat_type", "critical_events"),
    "m2c_force_vectors_bat_3d": ("club_3d", "bat_type", "critical_events"),
    "inplanemetrics": ("club_3d", "bat_type", "critical_events"),
    "outofplanemetrics": ("club_3d", "bat_type", "critical_events"),
    "m2c_inplane_vectors": ("club_3d", "bat_type", "critical_events"),
    "m2c_inplane_vectors_3d": ("club_3d", "bat_type", "critical_events"),
    "m2c_kinetics_3d": ("club_3d", "bat_type", "critical_events"),
}

PITCHING_METHOD_INPUTS = {
    "m2c_insights": ("critical_events",),
    "performance_metrics": ("critical_events",),
    "m2c_power": ("critical_events",),
    "m2c_club_mechanics_highlights": ("critical_events",),
}


@pytest.mark.parametrize("input_path", INPUT_FILES, ids=lambda path: path.stem)
def test_modular_applicable_club_methods_match_monolithic_golden(input_path):
    swing_id = input_path.stem
    golden_path = REF_OUTPUTS_DIR / swing_id / f"{swing_id}_clubFitting.json"
    assert golden_path.is_file(), f"Missing club fitting golden output: {golden_path}"

    input_json = _load_json(input_path)
    recording = input_json["data_3d_bio"]
    swing = recording["configuration"]["swing"]

    bat_type = swing.get("BatType")


    golden_output = _lowercase_keys(_load_json(golden_path))
    for key in EXCLUDE_KEYS:
        golden_output.pop(key, None)

    pose3d,frame_rate,weight,orientation,handedness,movement_name,input_critical_events,club_3d_markers = data_3d_bio_parser(input_json['data_3d_bio'])

    fixture_inputs = {
        "club_3d": club_3d_markers,
        "bat_type": bat_type,
        "critical_events": input_critical_events,
    }
    fixture_paths = {
        "club_3d": "data.club",
        "bat_type": "configuration.swing.BatType",
        "critical_events": "configuration.swing.critical_events",
    }

    session = movement(pose3d=pose3d,movement_type=movement_name,frame_rate =frame_rate,weight=weight,orientation=orientation,handedness=handedness)
    session.initialize_club_fitting()

    modular_output = session.calculate_movement_type()
    movement_type = modular_output["movement_type"]
    if movement_type not in ["full swing","hitting"]:
        club_3d=None

    if movement_type == "hitting":
        method_inputs = HITTING_METHOD_INPUTS
    elif movement_type == "pitching":
        method_inputs = PITCHING_METHOD_INPUTS
    else:
        assert movement_type == "generic", f"Unexpected movement type: {movement_type}"
        method_inputs = {}

    available_keys = {
        key.lower() for key in session._get_club_fitting().available_keys
    }

    for output_key, method_name in CLUB_FITTING_METHODS.items():
        output_key = output_key.lower()
        if output_key == "movement_type" or output_key not in available_keys:
            continue

        kwargs = {}
        for name in method_inputs.get(output_key, ()):
            value = fixture_inputs[name]
            assert value is not None, (
                f"{swing_id}: {method_name} requires {fixture_paths[name]}"
            )
            kwargs[name] = value

        result = getattr(session, method_name)(**kwargs)
        if result:
            modular_output.update(result)

    differences = _compare_values(golden_output, modular_output)
    _assert_matches_golden(differences, "club fitting")
