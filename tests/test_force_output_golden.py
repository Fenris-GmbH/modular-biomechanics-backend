"""Golden tests for modular force outputs against monolithic force-screen output."""

import json
import math
import numbers
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "tests" / "fixtures" / "force_inputs"
REF_OUTPUTS_DIR = PROJECT_ROOT / "ref_outputs"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from biomechanics.main import movement  # noqa: E402
from biomechanics.utils.test_helpers import CLUB_FITTING_METHODS  # noqa: E402


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


INPUT_FILES = sorted(INPUT_DIR.glob("*.json"))
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


@pytest.mark.parametrize("input_path", INPUT_FILES, ids=lambda path: path.stem)
def test_modular_force_output_matches_monolithic_golden(input_path):
    swing_id = input_path.stem
    golden_path = (
        REF_OUTPUTS_DIR / swing_id / f"{swing_id}_forcescreen.json"
    )
    assert golden_path.is_file(), f"Missing monolithic golden output: {golden_path}"

    input_json = _load_json(input_path)
    golden_output = _load_json(golden_path)

    session = movement(input_json["data_3d_bio"])
    modular_output = {}
    for method_name in FORCE_METHODS:
        result = getattr(session, method_name)()
        if result:
            modular_output.update(result)
    differences = _compare_values(golden_output, modular_output)

    _assert_matches_golden(differences, "force screen")


def _load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _assert_matches_golden(differences, label):
    assert not differences["missing"], (
        f"{label} output is missing monolithic keys:\n"
        + "\n".join(differences["missing"][:20])
    )
    assert not differences["extra"], (
        f"{label} output has keys absent from the monolithic golden:\n"
        + "\n".join(differences["extra"][:20])
    )
    assert differences["count"] == 0, (
        f"{differences['count']} {label} values differ from the monolithic golden. "
        f"Counts by output: {differences['counts_by_output']}. "
        "First differences:\n"
        + "\n".join(differences["values"])
    )


@pytest.mark.parametrize("input_path", INPUT_FILES, ids=lambda path: path.stem)
def test_modular_basic_screen_methods_match_monolithic_golden(input_path):
    swing_id = input_path.stem
    golden_path = (
        REF_OUTPUTS_DIR / swing_id / f"{swing_id}_basicscreen.json"
    )
    assert golden_path.is_file(), f"Missing basic screen golden output: {golden_path}"

    input_json = _load_json(input_path)
    golden_output = _load_json(golden_path)
    session = movement(input_json["data_3d_bio"])

    modular_output = {
        "kinematic_sequence": session.calculate_kinematic_sequence(),
        "com": session.calculate_COM(),
        "segment_angles": session.calculate_segment_angles(),
        "segment_displacements": session.calculate_segment_displacements(),
        "time": session.ctx.time.tolist(),
    }
    # error_messages is monolithic diagnostic metadata, not a returned modular
    # method result. Compare every method result and the modular context time.
    golden_methods = {key: golden_output[key] for key in modular_output}
    differences = _compare_values(golden_methods, modular_output)
    _assert_matches_golden(differences, "basic screen")


@pytest.mark.parametrize("input_path", INPUT_FILES, ids=lambda path: path.stem)
def test_modular_applicable_club_methods_match_monolithic_golden(input_path):
    swing_id = input_path.stem
    golden_path = (
        REF_OUTPUTS_DIR / swing_id / f"{swing_id}_clubFitting.json"
    )
    assert golden_path.is_file(), f"Missing club fitting golden output: {golden_path}"

    input_json = _load_json(input_path)
    golden_output = _load_json(golden_path)
    session = movement(input_json["data_3d_bio"])

    modular_output = session.calculate_movement_type()
    available_keys = set(session._get_club_fitting().available_keys)
    for output_key, method_name in CLUB_FITTING_METHODS.items():
        if output_key == "movement_type" or output_key not in available_keys:
            continue
        result = getattr(session, method_name)()
        if result:
            modular_output.update(result)

    differences = _compare_values(golden_output, modular_output)
    _assert_matches_golden(differences, "club fitting")
