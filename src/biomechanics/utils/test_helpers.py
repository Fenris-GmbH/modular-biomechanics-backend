import math
import numpy as np
import logging
from typing import Any

logger = logging.getLogger(__name__)


def replace_nans(obj: Any, sentinel: str = "__NaN__") -> Any:
    """Recursively replace floating-point NaNs with a comparison sentinel."""
    if isinstance(obj, float) and math.isnan(obj):
        return sentinel
    elif isinstance(obj, list):
        return [replace_nans(x, sentinel) for x in obj]
    elif isinstance(obj, dict):
        return {k: replace_nans(v, sentinel) for k, v in obj.items()}
    return obj


def compare_dict_subset(
    dict_a: dict[str, Any],
    dict_b: dict[str, Any],
    path: str = "",
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> dict[str, bool]:
    """Compare keys and values in ``dict_a`` against a reference dictionary."""
    results = {}

    dict_a = replace_nans(dict_a)
    dict_b = replace_nans(dict_b)

    for key in dict_a:
        current_path = f"{path}.{key}" if path else key

        if key not in dict_b:
            results[current_path] = False
            print(f"❌ Key missing in dict_b: {current_path}")
        else:
            val_a = dict_a[key]
            val_b = dict_b[key]

            if isinstance(val_a, dict) and isinstance(val_b, dict):
                results.update(
                    compare_dict_subset(val_a, val_b, current_path, rel_tol, abs_tol)
                )
            elif isinstance(val_a, list) and isinstance(val_b, list):
                if len(val_a) != len(val_b):
                    results[current_path] = False
                    print(
                        f"❌ List length mismatch at {current_path}: {len(val_a)} vs {len(val_b)}"
                    )
                else:
                    all_close = True
                    for i, (a, b) in enumerate(zip(val_a, val_b)):
                        if isinstance(a, float) and isinstance(b, float):
                            if not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
                                print(
                                    f"❌ Float mismatch at {current_path}[{i}]: {a} vs {b}"
                                )
                                all_close = False
                        elif a != b:
                            print(
                                f"❌ Value mismatch at {current_path}[{i}]: {a} vs {b}"
                            )
                            all_close = False
                    results[current_path] = all_close
            elif isinstance(val_a, float) and isinstance(val_b, float):
                close = math.isclose(val_a, val_b, rel_tol=rel_tol, abs_tol=abs_tol)
                if not close:
                    print(f"❌ Float mismatch at {current_path}: {val_a} vs {val_b}")
                results[current_path] = close
            else:
                equal = val_a == val_b
                if not equal:
                    print(f"❌ Value mismatch at {current_path}: {val_a} vs {val_b}")
                results[current_path] = equal

    return results


def _compare(local: dict[str, Any], ref: dict[str, Any]) -> bool:
    """Recursively compare dictionaries and lists using exact value equality."""
    if local.keys() != ref.keys():
        print(f"Keys mismatch: {local.keys()} vs {ref.keys()}")
        return False

    for key in local.keys():
        if isinstance(local[key], dict) and isinstance(ref[key], dict):
            if not _compare(local[key], ref[key]):
                print(f"Mismatch in nested dict for key: {key}")
                return False
        elif isinstance(local[key], list) and isinstance(ref[key], list):
            if len(local[key]) != len(ref[key]):
                print(f"List length mismatch for key: {key}")
                return False
            for i, (l_item, r_item) in enumerate(zip(local[key], ref[key])):
                if isinstance(l_item, dict) and isinstance(r_item, dict):
                    if not _compare(l_item, r_item):
                        print(
                            f"Mismatch in nested dict in list for key: {key} at index {i}"
                        )
                        return False
                elif l_item != r_item:
                    print(
                        f"Value mismatch for key: {key} at index {i}: {l_item} vs {r_item}"
                    )
                    return False
        else:
            if local[key] != ref[key]:
                print(f"Value mismatch for key: {key}: {local[key]} vs {ref[key]}")
                return False

    return True


def all_passed(
    values: dict[str, Any],
    reference: dict[str, Any],
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> None:
    """Assert that every value in ``values`` matches the reference subset."""
    results = compare_dict_subset(values, reference, rel_tol=rel_tol, abs_tol=abs_tol)
    print(results)
    assert all(value == True for value in list(results.values()))


# All individual callable output functions
CLUB_FITTING_METHODS = {
    "movement_type": "calculate_movement_type",
    "M2C_insights": "calculate_M2C_insights",
    "Performance_metrics": "calculate_Performance_metrics",
    "M2C_power": "calculate_M2C_power",
    "M2C_club_mechanics": "calculate_M2C_club_mechanics",
    "M2C_club_mechanics_highlights": "calculate_M2C_club_mechanics_highlights",
    "M2C_wrist": "calculate_M2C_wrist",
    "M2C_force_vectors_hand_r_3d": "calculate_M2C_force_vectors_hand_r_3d",
    "M2C_force_vectors_bat_3d": "calculate_M2C_force_vectors_bat_3d",
    # "M2C_skeleton_3d": "calculate_M2C_skeleton_3d",
    "pitcher_metric_equivalents": "calculate_pitcher_metric_equivalents",
    "inPlaneMetrics": "calculate_inPlaneMetrics",
    "outOfPlaneMetrics": "calculate_outOfPlaneMetrics",
    "M2C_inplane_vectors": "calculate_M2C_inplane_vectors",
    "M2C_inplane_vectors_3d": "calculate_M2C_inplane_vectors_3d",
    "M2C_kinetics_3d": "calculate_M2C_kinetics_3d",
    # "generic": "calculate_generic",
    # "generic_metric_equivalents": "calculate_generic_metric_equivalents",
    "M2C_other_side_error": "calculate_M2C_other_side_error",
    "generic_widgets_error": "calculate_generic_widgets_error",
    "M2C_power_left": "calculate_M2C_power_left",
    "M2C_power_right": "calculate_M2C_power_right",
    "M2C_club_mechanics_left": "calculate_M2C_club_mechanics_left",
    "M2C_club_mechanics_right": "calculate_M2C_club_mechanics_right",
    "M2C_club_mechanics_highlights_left": "calculate_M2C_club_mechanics_highlights_left",
    "M2C_club_mechanics_highlights_right": "calculate_M2C_club_mechanics_highlights_right",
    "M2C_wrist_left": "calculate_M2C_wrist_left",
    "M2C_wrist_right": "calculate_M2C_wrist_right",
}

COMMON_KEYS = {
    "movement_type",
    "M2C_power",
    "M2C_club_mechanics",
    "M2C_club_mechanics_highlights",
    "M2C_wrist",
    # "M2C_skeleton_3d",
    # "generic",
}

HITTING_KEYS = COMMON_KEYS | {
    "M2C_insights",
    "Performance_metrics",
    "M2C_force_vectors_bat_3d",
    "inPlaneMetrics",
    "outOfPlaneMetrics",
    "M2C_inplane_vectors",
    "M2C_inplane_vectors_3d",
    "M2C_kinetics_3d",
}

PITCHING_KEYS = COMMON_KEYS | {
    "M2C_insights",
    "Performance_metrics",
    "M2C_force_vectors_hand_r_3d",
    "pitcher_metric_equivalents",
}

GENERIC_KEYS = COMMON_KEYS | {
    # "generic_metric_equivalents",
    "M2C_other_side_error",
    "generic_widgets_error",
    "M2C_power_left",
    "M2C_power_right",
    "M2C_club_mechanics_left",
    "M2C_club_mechanics_right",
    "M2C_club_mechanics_highlights_left",
    "M2C_club_mechanics_highlights_right",
    "M2C_wrist_left",
    "M2C_wrist_right",
}

KEYS_BY_MOVEMENT = {
    "hitting": HITTING_KEYS,
    "pitching": PITCHING_KEYS,
    "generic": GENERIC_KEYS,
}


def calculate_club_metrics(
    session: Any,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Run the registered club outputs for a session and collect failures/skips."""
    comp = session._get_club_fitting()

    swing_type = comp.movement_type
    available_keys = set(comp.available_keys)

    if swing_type not in KEYS_BY_MOVEMENT:
        raise ValueError(f"Unsupported movement type: {swing_type}")

    requested_keys = KEYS_BY_MOVEMENT[swing_type]

    response = {}
    errors = {}
    skipped = []

    for key, method_name in CLUB_FITTING_METHODS.items():

        if key not in requested_keys or key not in available_keys:
            skipped.append(key)
            continue

        try:
            method = getattr(session, method_name)
            result = method()
            response.update(result)

        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to calculate %s", key)

    return response, errors, skipped


def numpy_int_parser(obj: Any) -> int:
    """Convert a NumPy int64 to a built-in int for JSON serialization."""
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError
