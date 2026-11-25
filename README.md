# Biomechanics Modular Backend

Python package for processing tracked movement data into basic movement, force,
and club-fitting outputs. The modular API exposes calculations individually so
callers can request only the outputs they need. `scripts/generic_runner.py`
demonstrates a sequence of method calls for one configured trial.

## Requirements and installation

The project currently targets **Python 3.9** (`pyproject.toml` specifies
`>=3.9,<3.10`) and depends on NumPy and SciPy.

From the repository root, install the package into the active Python environment:

```powershell
python -m pip install -e .
```

Pytest is used for the golden suite but is not a runtime dependency:

```powershell
python -m pip install pytest
```

Use the same Python environment for installation, running the package, and
running pytest. Running with a Python version outside the declared range may
work in some environments, but is not the supported configuration.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/biomechanics/main.py` | `movement` session and individual basic, force, and club-fitting methods |
| `src/biomechanics/open_movements/forces/` | Force prediction engine and staged force-screen outputs |
| `src/biomechanics/open_movements/club_fitting/` | Movement-specific club-fitting calculations and output methods |
| `scripts/generic_runner.py` | Example runner for one configured local input trial |
| `tests/test_golden.py` | Per-input golden comparisons for basic, force, and applicable club outputs |
| `tests/fixtures/` | Local test input copies for the golden suite |
| `inputs/` | Local input trials |
| `outputs/` | JSON results written by the runner |
| `ref_outputs/` | Local reference outputs |
| `monolithic_scripts/` | Legacy scripts used as monolithic reference implementations |

The `.gitignore` excludes local input, output, reference, and test-fixture JSONs,
as well as the top-level `monolithic_scripts/` directory. Keep private or large
trial data local. The Python test files and Markdown documentation remain
trackable.

## Input format

The example runner reads one JSON file at:

```text
inputs/<swing_id>/<swing_id>.json
```

It reads the full input document and creates the session from its
`data_3d_bio` object. This payload is expected to contain `configuration` and
`data`, including the 3D pose. Configuration commonly supplies frame rate,
body weight, handedness, orientation, swing type, and critical event frames.

The force engine can infer movement presets and metadata from supported input
layouts. The modular `movement` session explicitly forwards body weight and
frame rate to force calculations. This matters when calling lower-level engine
APIs directly: pass the outer input object or explicitly provide the metadata
when supplying only the inner `data_3d_bio` object.

## Running the modular methods

`scripts/generic_runner.py` reads the `swing_id` configured in its `__main__`
block, runs that trial, and writes nonempty method results to
`outputs/<swing_id>/<metric>.json`. Change `swing_id` in the script to select a
different trial. Run it from the repository root:

```powershell
python scripts/generic_runner.py
```

Importing the module does not start processing. Within a `movement` session,
force methods share a cached force computation; club-fitting methods share a
cached club-fitting computation.

## Mode configuration reference

`configs/mode_configs.py` is a reference for constructing sessions, setting up
shared computations, and calling methods with their required inputs. It is not
currently consumed by `scripts/generic_runner.py`.

The `movement` constructor takes these parsed values as named arguments:

```python
session = movement(
    pose3d=pose3d,
    movement_type=movement_type,
    frame_rate=frame_rate,
    weight=weight,
    orientation=orientation,
    handedness=handedness,
)
```

The mode reference uses `baseball` with a fixed movement type of `hitting` or
`pitching`, and `golf` with `full swing`. Its generic mode uses `movement` and
passes through the input movement type. For example, `jump` is a generic
club-fitting movement:

```python
session = movement(
    pose3d=pose3d,
    movement_type="jump",
    frame_rate=frame_rate,
    weight=weight,
    orientation=orientation,
    handedness=handedness,
)
```

Force setup is shared by the force methods. Golf requires `club_3d` when
`include_club=True`; the mode reference also passes `club_type`, which has a
method default. `critical_events` can also be passed.
Hitting, pitching, and generic force setup use `include_club=False`. Only
`calculate_force_vectors`, `calculate_weight_shift`, and
`calculate_COM_MODELS` accept `pose2d`. The force engine has no dedicated jump
preset, so unknown force movement types use its pitching preset. The generic
mode refers to the club-fitting graph, not a generic force model.

Club-fitting methods take inputs per calculation. Hitting metrics that use club
tracking receive `club_3d`; metrics that depend on the bat model also receive
`bat_type`; event-based metrics receive `critical_events`. Pitching event-based
metrics receive `critical_events`. Golf and other generic movements use the
generic club-fitting graph. `calculate_all_club_fitting()` returns its combined
outputs, including side-specific widgets, skeleton data, diagnostics, and
metric equivalents.

## Basic screen methods

The example runner calls these basic methods for its configured trial:

| Output | Method |
| --- | --- |
| Kinematic sequence | `calculate_kinematic_sequence()` |
| Center of mass | `calculate_COM()` |
| Segment angles | `calculate_segment_angles()` |
| Segment displacements | `calculate_segment_displacements()` |

`calculate_COM()` is called before segment angles because the angle calculation
can reuse the computed center of mass. The session context also contains the
basic time vector.

## Force methods

The example runner calls these force methods for its configured trial. The
shared force pipeline is cached, and each method returns its own output
dictionary.

| Output | Method |
| --- | --- |
| Force vectors | `calculate_force_vectors()` |
| Weight shift | `calculate_weight_shift()` |
| Forces | `calculate_forces()` |
| Force time | `calculate_force_time()` |
| 3D force vectors | `calculate_force_vectors_3d()` |
| COP trajectory values | `calculate_values_for_COP_trajectory()` |
| COM models | `calculate_COM_MODELS()` |
| Power | `calculate_power()` |
| Energy | `calculate_energy()` |
| Impulse | `calculate_impulse()` |
| Small force vectors | `calculate_small_force_vectors()` |
| Back-view COM/COP trajectory | `calculate_COM_COP_traj_back_view()` |
| Critical events | `calculate_critical_events()` |
| Frame offset | `calculate_frame_offset()` |

Some force-screen fields need 2D pose data. These methods accept an optional
`pose2d` argument:

- `calculate_force_vectors(pose2d=...)`
- `calculate_weight_shift(pose2d=...)`
- `calculate_COM_MODELS(pose2d=...)`

The session caches its force computation. A later call can add or replace
`pose2d`; the force pipeline reuses its numerical context and refreshes the
2D-dependent output stages. Without 2D pose data, `weight_shift` and
`COM_MODELS` are omitted, along with the 2D pixel fields inside `force_vectors`.

## Club-fitting methods by movement type

Club-fitting classifies the movement type. It recognizes the lower-cased values
`hitting` and `pitching`; other values are classified as `generic`. The
available methods depend on this classification.

### Available outputs

| Method output | Hitting | Pitching | Generic / other |
| --- |:---:|:---:|:---:|
| `movement_type` | Yes | Yes | Yes |
| `M2C_power` | Yes | Yes | Yes |
| `M2C_club_mechanics` | Yes | Yes | Yes |
| `M2C_club_mechanics_highlights` | Yes | Yes | Yes |
| `M2C_wrist` | Yes | Yes | Yes |
| `M2C_insights` | Yes | Yes | — |
| `Performance_metrics` | Yes | Yes | — |
| `M2C_force_vectors_bat_3d` | Yes | — | — |
| `inPlaneMetrics` | Yes | — | — |
| `outOfPlaneMetrics` | Yes | — | — |
| `M2C_inplane_vectors` | Yes | — | — |
| `M2C_inplane_vectors_3d` | Yes | — | — |
| `M2C_kinetics_3d` | Yes | — | — |
| `M2C_force_vectors_hand_r_3d` | — | Yes | — |
| `pitcher_metric_equivalents` | — | Yes | — |
| `M2C_other_side_error` | — | — | Yes |
| `generic_widgets_error` | — | — | Yes |
| `M2C_power_left`, `M2C_power_right` | — | — | Yes |
| `M2C_club_mechanics_left`, `M2C_club_mechanics_right` | — | — | Yes |
| `M2C_club_mechanics_highlights_left`, `M2C_club_mechanics_highlights_right` | — | — | Yes |
| `M2C_wrist_left`, `M2C_wrist_right` | — | — | Yes |

The `M2C_force_vectors_hand_r_3d` output name is retained for compatibility; the
pitching calculation handles the pitcher’s handedness. The force engine has a
separate preset map: it recognizes `hitting`, `pitching`, and `full swing` for
force calculations. A `full swing` input is still classified as `generic` by
club-fitting, so it receives the generic club-fitting methods shown above.

## Golden tests

The suite compares modular results directly against the matching files in
`ref_outputs/<swing_id>/` for every JSON input copied into
`tests/fixtures/golden_inputs/`. It calls individual force
methods, the four basic screen methods, and all club methods available for each
movement type. The comparison checks missing and extra nested keys and all
returned scalar/list values using `rel_tol=1e-9` and `abs_tol=1e-12`.

Run the suite from the repository root:

```powershell
python -m pytest tests/test_golden.py -q
```

Fixture and reference JSONs are intentionally ignored by Git. A fresh checkout
therefore needs the local fixture inputs and `ref_outputs/<swing_id>/` files
restored before this suite can run. The basic screen reference includes
an `error_messages` diagnostic block that is not returned by a modular method,
so the test compares every basic method result and the modular time vector
against the corresponding reference fields.

## Legacy implementation

`monolithic_scripts/` contains the older force scripts and is ignored by Git as
requested. The modular force and club packages under `src/biomechanics/` are the
application code. Keep the legacy scripts locally when regenerating or
investigating monolithic comparison outputs.
