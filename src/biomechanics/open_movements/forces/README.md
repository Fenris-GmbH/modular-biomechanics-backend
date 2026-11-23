# force_screen_modules — forces-screen outputs from computed (predicted) forces

Modular Python API that turns a 3D pose JSON (and optionally a 2D pose JSON) into the same
"forces screen" response that the measured-force-plate script produces — **without force plates**.
Every top-level key (all lower case) of that response is its own function, so a backend can request only what it needs
(`energy`, `forces`, `force_vectors_3d`, …) and each result is returned as `{key: response}` and/or
written to `<key>.json`.

| | |
|---|---|
| Module | `force_screen_modules.py` (this API) |
| Engine (unchanged, required) | `force_vectors_v13_forces_screen_contactgate_lefthanded_handling.py` — must sit in the same folder / on `PYTHONPATH` |
| Tested with | Python 3.12, numpy 2.5, scipy 1.18 |
| Dependencies | `numpy`, `scipy` (no other third-party packages) |
| Inputs | One 3D JSON object containing configuration and pose data (optionally under `data_3d_bio`), plus an optional separate 2D pose JSON |
| Output keys | **all lower case** (main and sub keys), e.g. `com_models` → `com_2d` |
| Verified | every function returns exactly what the engine's original builder (`ForceVectors._build_forces_screen`) returns, on real left- and right-handed golf trials |

---

## 1. Quick start

```python
import json
import force_screen_modules as fs

json3d = json.load(open("trial_3d.json"))      # configuration + data.pose (and data.club for golf)
pose2d = json.load(open("trial_2d.json"))      # optional; markers under data or data.pose

# One output only (returns {"energy": {...}}; also writes out/energy.json)
res = fs.energy(json3d, pose2d=pose2d, output_dir="out")

# Several outputs, heavy computation done once
comp    = fs.ForcesScreen(json3d, pose2d=pose2d)
forces  = fs.forces(computation=comp)
impulse = fs.impulse(computation=comp, output_dir="out")

# Everything; one <key>.json per key
all_out = fs.compute_all(json3d, pose2d=pose2d, output_dir="out", name_prefix="swing42_")
```

The engine reads configuration and pose data from `json3d`; `pose2d` is an optional separate argument. There is no active command-line interface in `force_screen_modules.py`.

## 2. Public functions

All functions share one signature:

```python
fn(json3d=None, pose2d=None, *, computation=None, output_dir=None,
   name_prefix="", **overrides) -> dict
```

| Argument | Meaning |
|---|---|
| `json3d` | Required input object containing the 3D pose at `data.pose` or `data_3d_bio.data.pose`; it also carries configuration and, for golf, club data. |
| `pose2d` | Optional 2D pose object, with markers under `data` or `data.pose` (see §3.2). |
| `computation` | A `ForcesScreen` object to reuse (skips recomputing the forces). |
| `output_dir` | If set, writes `<name_prefix><key>.json` there (`{key: response}` inside the file). |
| `**overrides` | Passed to the engine (see §7), e.g. `body_mass=`, `frame_rate=`, `kinematic_contact_gate=True`. |

Returns `{key: response}`; returns `{}` (and writes nothing) when an output cannot be produced
(`weight_shift` / `com_models` without `pose2d`).

| Function | Output key content |
|---|---|
| `force_vectors` | COP (plate-frame and pixels), body torques x/y/z, moment arms, COM/COP displacement and speed, 3D start/end points, widget pixels (+ 2D pixel arrows if `pose2d`) |
| `weight_shift` | 2D pixel positions of the feet scaled by the load split — **needs `pose2d`** |
| `forces` | fz/fx/fy/tz total/left/right, % of P1 weight, max, timing, time-to-impact |
| `time` | Time axis P1→P10 (s) |
| `force_vectors_3d` | 3D arrows per foot and combined (tracking frame) |
| `values_for_cop_trajectory` | Weight shift % per foot, midfoot stance width |
| `com_models` | 2D COM models (full / without arms / lower / upper) — **needs `pose2d`** |
| `power`, `energy`, `impulse` | Force×COM-velocity, its integral, force integral, each with max/timing/time-to-impact |
| `small_force_vectors` | Side-panel force arrows (pixels) |
| `com_cop_traj_back_view` | Back-view COM/COP trajectory (pixels) |
| `critical_events` | Echo of the input events (does **not** run the force computation) |
| `frame_offset` | Always `0` for computed forces (does **not** run the force computation) |

Helpers: `compute_all(...)`, `ForcesScreen(json3d, pose2d=..., **overrides)`,
`describe_requirements(key=None)` (what each function needs, as data),
`resolved_configuration(json3d, **overrides)` (what the engine will *actually* use for this file).

`OUTPUT_KEYS` lists all keys. A function runs only the internal stages its output needs
(e.g. `energy` runs 6 of 19 stages); `ForcesScreen` caches stages so nothing is computed twice.

---

## 3. Input contract

### 3.1 3D inputs

| File | Format |
|---|---|
| 3D pose | `json3d["data"]["pose"]` or `json3d["data_3d_bio"]["data"]["pose"]`; a numeric array with markers on axis 0, frames on axis 1, and 3 coordinates on axis 2. At least 29 markers are required. |
| Club data (golf) | In the matching `data` block, `club` is keyed by frame index; each frame must contain 3D `start`, `mid`, and `end` coordinates. Club frame count must match the pose. `shaft` is unused. |

Marker order and sides: see the engine; sides are **anatomical** and fixed
(index 1–5 right leg, 6–10 left leg, 14–17 right arm, 18–21 left arm).

### 3.2 2D input (`pose2d`) — optional

`{"data": [frame][marker][x, y]}` pixels (or `{"data": {"pose": [...]}}`). At least **29 markers** (31 is fine — the
first 29 are used); the frame count **must equal** the 3D frame count. Without `pose2d`, only the keys
that depend on 2D pixels are omitted: top-level `weight_shift`, `com_models`, and inside `force_vectors`:
`f{x,y,z}_{t,l,r}_pix`, `cop_for_frontend_x/y`, `factor_for_midfoot_point_frontend`,
`proj_grf_com_2d`, `com_2d`. A warning is logged; nothing is faked.

## 4. Configuration fields used

### 4.1 Field reference

| Config path | Type / unit | Used for | If missing |
|---|---|---|---|
| `configuration.video.frame_rate` | number, Hz | Differentiation, filter cut-offs, time axis, impulse/power/energy | Engine default **240 Hz** unless overridden (see 4.2) |
| `configuration.user.weight` | number, **kg** (no unit conversion) | Segment masses → every force | Preset default (golf/hitting 70, pitching 75) unless overridden (see 4.2) |
| `configuration.swing.swingType` | string | Selects the movement preset: `"hitting"`, `"pitching"`, `"full swing"` (golf). Exact, case-sensitive match | **Any other value / missing → pitching preset** (INFO log) |
| `configuration.swing.clubType` | string | Golf club mass: `"driver"` → driver masses, **any other value → iron** | `"driver"` |
| `configuration.swing.handedness` | string | `"leftHanded"` (case-insensitive, starts with "left") → analysis rotated 180° into the plate frame (§6.3). Anything else → untouched | Treated as right-handed |
| `configuration.swing.critical_events` | ints (frame index) | `P1` start / weight reference, `P7` impact (or release), `P10` end of window (`p8` is accepted if no `p10`). Upper or lower case keys | Fallbacks, see §6.4 |

Not used: `video.height/width/shutter/iso`, `processing.*`, `user.height/id`, `launch_data`, `PPMM`,
`boundingBox`, `orientation`.

Which function needs which field is in each function's docstring and in `describe_requirements(key)`.
Every force-derived output needs the whole table above; the direct extras are:

| Function | Direct config needs (besides the table above being available for the force computation) |
|---|---|
| `forces` | `frame_rate`, events P1/P7/P10, `handedness`, `user.weight` (reference if the force at P1 is ≤ 0) |
| `impulse`, `power`, `energy` | `frame_rate`, events P1/P7/P10, `handedness` |
| `force_vectors` | `frame_rate`, events, `handedness`; `pose2d` optional |
| `force_vectors_3d`, `small_force_vectors`, `com_cop_traj_back_view` | events (P1), `handedness` |
| `time` | `frame_rate`, events P1/P10 |
| `weight_shift`, `com_models` | `pose2d` (required) |
| `values_for_cop_trajectory` | nothing beyond the pose |
| `critical_events` | `critical_events` only (no force computation) |
| `frame_offset` | nothing |

### 4.2 Mass and frame rate

The engine builds movement-specific presets from the swing type and extracts mass/frame rate from
the configuration metadata when available. Explicit `body_mass=` and `frame_rate=` overrides take
precedence. The constructor defaults are used when metadata is unavailable; inspect
`fs.resolved_configuration(json3d)` to see the effective values.

---

## 5. Output contract

**Every returned key — main and sub keys — is lower case** (e.g. `com_models.com_2d`, `forces.fz_t`, `critical_events.p1`); values are unchanged. Output filenames use the internal output key spelling, so keys such as `COM_MODELS` may produce an uppercase filename.

All values are plain JSON types. Non-finite numbers are returned as `null`. Per-frame arrays have
length **n = number of frames in the 3D pose** unless stated. Numbers follow the measured-force
script's rounding (e.g. 2 decimals for forces).

| Key | Shape | Units / notes |
|---|---|---|
| `forces` | 57 fields: per-frame lists `fz/fx/fy/tz _t/_l/_r` and `fz/fx/fy _t/_l/_r _wt`; scalars `*_max`, `*_max_timing`, `*_time_to_impact` | N; `Tz` N·m; `_wt` % of total vertical force at P1; timing / time-to-impact in **ms** (window P1→P10) |
| `time` | list, length **P10−P1** | seconds; starts at 1/frame_rate (measured-script convention) |
| `force_vectors` | 92 fields: per-frame lists, 2-element lists (`pixel_rows/colm_foot_*`), scalars (`*_torque_max`, `*_moment_arm_max*`, timings) | Torques N·m; moment arms and COM/COP displacement **cm** (+ `_inch`); `com_vel` cm/s (`_ips` in/s); `com_vs_stance` is `[[value], …]` (1-element lists, as in the measured script); pixels in a 375×180 widget |
| `force_vectors_3d` | 18 per-frame lists `{x,y,z}_{start,end}_cop_{l,r,combined}` | Tracking-frame coordinates (m); arrow end = start + force/1500 |
| `values_for_cop_trajectory` | 4 per-frame lists | `weight_shift_*` %; `stance_midfeet` cm (+ `_inch`) |
| `weight_shift` | 3 per-frame lists | pixels (needs `pose2d`) |
| `com_models` | 4 per-frame lists of `[x, y]` | pixels (needs `pose2d`) |
| `power` / `energy` | 42 fields each: per-frame lists + scalars `*_max`, `*_max_timing`, `*_time_to_impact` | W / J. **`*_max_timing` here is a frame index relative to P1, not ms** (measured-script convention); `*_time_to_impact` ms |
| `impulse` | 36 fields | N·s; `*_max_timing` is an integer frame index; `*_time_to_impact` ms |
| `small_force_vectors` | 3 objects (`force_vectors_lead/_trail/_combined`), each with `starting_x/y`, `ending_x/y` int lists | pixels, 223×75 panel |
| `com_cop_traj_back_view` | 12 fields | pixels, 375×180 widget |
| `critical_events` | the input event dict (e.g. `p1…p10`, extra keys such as `p7_by` passed through) | frame indices; **only the events that were in the input** — resolved fallbacks are not added |
| `frame_offset` | int `0` | |

---

## 6. Conventions and behaviour

### 6.1 Left / right
`_l` = anatomical **left** foot, `_r` = anatomical **right** foot for every player (so for a
left-handed golfer `_l` is the lead foot). This follows the source marker order; the engine's
`swap_lr=True` default implements it. All geometry (arrows, COP) is taken from the marker set of the
same foot as the force.

### 6.2 Coordinate frames and plate-frame COP
Force convention: `Fx` lateral, `Fy` anterior-posterior, `Fz` vertical (positive up). Geometry is in
the tracking frame (axis0 depth/AP, axis1 lateral, axis2 vertical). There are no physical plates:
the "plate frame" COP (`force_vectors.cop_l/_r/cop`) is **synthetic** — registered to the midpoint of
the two midfoot markers at P1 (vertical: left-midfoot height at P1). COP per foot is the point where
the predicted force's line of action meets the ground, taken through the foot-segment COM.

### 6.3 Left-handed players
Measured left-handed data showed the horizontal forces reversed relative to the plates while the 3D
tracking frame is the same as for right-handers. For `leftHanded` files the analysis is therefore
rotated 180° about the vertical axis into the plate frame: `fx`, `fy`, the x/y torques and the
displacements along those axes are sign-flipped; `fz`, `tz`, the z torque and ratios are unchanged.
Drawing geometry (`force_vectors_3d`, `start/end_point_*`, 2D pixel arrows) stays in the tracking /
video frame so it still overlays the skeleton. Right-handed files never take this path. Disable with
`lefty_plate_rotation=False` (e.g. a lefty session whose plates were not turned). *The rule was
derived from one left- and one right-handed trial; see §9.*

### 6.4 Missing events (graceful fallbacks, all logged as warnings)
| Event | Fallback |
|---|---|
| `P1` missing / invalid | first frame with fully valid markers |
| `P10` (or `p8`) missing / invalid / not after P1 | last frame with fully valid markers |
| `P7` missing / outside (P1, P10) | **estimated** as the frame of peak hand speed inside the window; if that fails, the last frame of the window |

An empty window (P10 ≤ P1 after fallbacks) raises `ValueError`.

### 6.5 Movement presets (`swingType`)
| `swingType` | Preset | Key settings |
|---|---|---|
| `"hitting"` | `for_baseball_swing` | contact gating **on** (back foot may leave the ground) |
| `"pitching"` | `for_pitching` | contact gating on, longer debounce |
| `"full swing"` | `for_golf` | continuous contact (both feet grounded), club modelled (`include_club=True`) |
| other / missing | `for_pitching` | INFO log |

---

## 7. Engine options (`**overrides`)

Passed to the engine constructor; explicit values always win over file values and preset defaults.

| Option | Effect |
|---|---|
| `body_mass` (kg), `frame_rate` (Hz) | See §4.2 |
| `kinematic_contact_gate` (default `False`) | Re-applies the marker-height contact confidence after the force low-pass so an airborne foot carries no force (use for hitting/pitching if force appears while a foot is in the air). Heel-raised loaded feet can be shrunk if `contact_height_threshold + contact_confidence_band` is too small |
| `contact_threshold`, `contact_hysteresis`, `min_contact_frames`, `contact_height_threshold`, `contact_confidence_band` | Contact detection (non-continuous presets) |
| `include_club` | `False` runs body-only (the golf preset requires club data otherwise) |
| `plate_layout` (`"amti"` / `"kistler"`) | Frontend widget mapping only |
| `anchor_foot_depth` (default `False`) | Reproduces the measured script's depth anchoring for geometry (not forces) |
| `lefty_plate_rotation` (default `True`) | See §6.3 |
| `fc_pos`, `fc_dyn`, `fc_grf`, `filter_order` | Filter settings |

---

## 8. Errors and failure behaviour

The module **fails hard** (no partial or degraded outputs) so results stay consistent. Catch these
in the backend and surface the message:

| Condition | Exception | Message gist |
|---|---|---|
| Golf (`include_club=True`) but no club block | `KeyError` | "include_club=True … but no club data found" |
| Club frames ≠ pose frames | `ValueError` | reports the club and pose frame counts |
| 3D pose has fewer than 29 markers or unexpected array shape | `ValueError` | reports the shape / marker count |
| Missing `json3d` when no computation is supplied | `ValueError` | pass the input JSON or a `ForcesScreen` computation |
| Club points not 3D (e.g. 2D pixel coordinates) | `ValueError` | names the frame/part and the offending value |
| `pose2d` frame count ≠ 3D frame count | `ValueError` | shape vs frames |
| `pose2d` < 29 markers / wrong shape / no `data` | `ValueError` / `KeyError` | |
| Empty analysis window after event fallbacks | `ValueError` | "Analysis window is empty" |
| Unknown output key | `KeyError` | |
| `json3d` and `computation` both missing | `ValueError` | |

Missing `pose2d` is **not** an error (see §3.2). Missing events are **not** errors (see §6.4).
When writing files, nothing is written if the computation raised.

---

## 9. Operations

- **Performance:** the force computation dominates: ≈ **12 s for a 336-frame trial** (≈ 35 ms/frame,
  roughly linear in frames, single core). Deriving all 14 outputs from it takes ≈ 0.2 s. Peak extra
  memory is a few MB. Always reuse a `ForcesScreen` when you need several outputs.
- **Concurrency:** create **one `ForcesScreen` / call per request**; do not share instances across
  threads (the engine keeps per-run state). Separate processes are fine.
- **Determinism:** same input and options give the same output.
- **Logging:** standard `logging`. The engine calls `logging.basicConfig(level=INFO)` at import —
  configure your own handlers after importing if you need different behaviour. Fallbacks and omitted
  keys are WARNINGs; log them.
- **Files:** `<name_prefix><key>.json`, UTF-8, strict JSON (`allow_nan=False`), content is
  `{key: response}`.
- **Idempotent / stateless:** no files or network are touched except the requested `output_dir`.

---

## 10. Validation status and known limitations

Validated: structure and values of every function equal the engine's original builder on real
left- and right-handed trials; the legacy response is unchanged for the right-handed trial
(compared with the previous engine version) and identical to the original v11 on synthetic trials;
the measured-script formulas were reproduced numerically (all 324 keys identical when the new
builder was fed the measured script's own forces, COP and COM on a synthetic trial). **Not** validated: absolute force accuracy at scale — it depends on tracking quality
(e.g. a trial with 140 of 336 interpolated frames agreed noticeably worse with the plates than one
with 79).

Known limitations to communicate downstream:
1. **`tz_*` (vertical torque) is a closure estimate**, not a measurement (the part of the vertical
   torque the predicted shear couple does not explain). Treat as qualitative.
2. **COP is model-based**, not measured; plate-frame COP is synthetic (§6.2).
3. **Left-handed rule** is based on one left- and one right-handed trial (§6.3). Check the sign of
   `fx/fy` on a few more lefty sessions with plates before relying on it.
4. **`weight_shift` / `com_models`** run the force computation although they only need `pose2d`
   (they share a stage with the pixel force arrows); a later split will remove this.
5. **Flat-layout mass/frame-rate** must be passed explicitly (§4.2).
6. **Club data must be 3D** for golf; files whose `club` block is 2D pixel data fail by design.
7. Unrecognised `swingType` silently uses the pitching preset (logged at INFO) — validate upstream.
8. `critical_events` only echoes input events; estimated/fallback events are not added.

---

## 11. Integration checklist

- [ ] Engine file and `force_screen_modules.py` deployed together; `numpy`, `scipy` installed.
- [ ] The `json3d` configuration carries `user.weight` (kg) and `video.frame_rate` (Hz), or pass `body_mass=` and `frame_rate=` explicitly.
- [ ] `swingType` is one of `hitting` / `pitching` / `full swing`.
- [ ] Golf `json3d` inputs include a 3D `data.club` block with one entry per pose frame.
- [ ] `pose2d` (≥ 29 markers, same frame count) supplied when the 2D pixel keys are needed.
- [ ] One `ForcesScreen` per request; reuse it for multiple outputs.
- [ ] `ValueError` / `KeyError` from §8 handled and shown to the user; WARNING logs collected.
- [ ] Acceptance test: `fs.compute_all(...)` for a known trial equals the stored reference response
      (`json.dumps(..., sort_keys=True)` equality).

## 12. Input layout

The module passes `json3d` to the engine, which reads configuration and the 3D pose from that
object. The pose is expected at `data.pose` or `data_3d_bio.data.pose`, with markers on axis 0,
frames on axis 1, and XYZ coordinates on axis 2. Golf club coordinates are read from the same
data block. The optional 2D pose is passed separately as `pose2d`.
