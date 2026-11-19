# force_screen_modules — forces-screen outputs from computed (predicted) forces

Modular Python API that turns a 3D pose JSON (and optionally a 2D pose JSON) into the same
"forces screen" response that the measured-force-plate script produces — **without force plates**.
Every top-level key of that response is its own function, so a backend can request only what it needs
(`energy`, `forces`, `force_vectors_3d`, …) and each result is returned as `{key: response}` and/or
written to `<key>.json`.

| | |
|---|---|
| Module | `force_screen_modules.py` (this API) |
| Engine (unchanged, required) | `force_vectors_v13_forces_screen_contactgate_lefthanded_handling.py` — must sit in the same folder / on `PYTHONPATH` |
| Tested with | Python 3.12, numpy 2.5, scipy 1.18 |
| Dependencies | `numpy`, `scipy` (no other third-party packages) |
| Inputs | unchanged from the engine: 3D pose JSON (+ optional 2D pose JSON) |
| Verified | every function returns exactly what the engine's original builder (`ForceVectors._build_forces_screen`) returns, on real left- and right-handed golf trials |

---

## 1. Quick start

```python
import json
import force_screen_modules as fs

json3d = json.load(open("pose_3d.json"))
pose2d = json.load(open("pose_2d.json"))          # optional

# One output only (returns {"energy": {...}}; also writes out/energy.json)
res = fs.energy(json3d, pose2d, output_dir="out", body_mass=97.5, frame_rate=240)

# Several outputs, heavy computation done once
comp = fs.ForcesScreen(json3d, pose2d, body_mass=97.5, frame_rate=240)
forces  = fs.forces(computation=comp)
impulse = fs.impulse(computation=comp, output_dir="out")

# Everything (identical to the engine's forces-screen response); one <key>.json per key
all_out = fs.compute_all(json3d, pose2d, output_dir="out", name_prefix="swing42_",
                         body_mass=97.5, frame_rate=240)
```

> **Pass `body_mass` and `frame_rate` explicitly** until the engine's input reader is extended
> (see §4.2). Explicit values always win over anything read from the file and are safe for every layout.

Command line (writes all outputs):
`python force_screen_modules.py pose_3d.json out_dir [pose_2d.json]`

---

## 2. Public functions

All functions share one signature:

```python
fn(json3d=None, pose2d=None, *, computation=None, output_dir=None, name_prefix="", **overrides) -> dict
```

| Argument | Meaning |
|---|---|
| `json3d` | Parsed 3D pose JSON (dict). Required unless `computation` is given. |
| `pose2d` | Parsed 2D pose JSON (dict), optional (see §3). |
| `computation` | A `ForcesScreen` object to reuse (skips recomputing the forces). |
| `output_dir` | If set, writes `<name_prefix><key>.json` there (`{key: response}` inside the file). |
| `**overrides` | Passed to the engine (see §7), e.g. `body_mass=`, `frame_rate=`, `kinematic_contact_gate=True`. |

Returns `{key: response}`; returns `{}` (and writes nothing) when an output cannot be produced
(`weight_shift` / `COM_MODELS` without `pose2d`).

| Function | Output key content |
|---|---|
| `force_vectors` | COP (plate-frame and pixels), body torques x/y/z, moment arms, COM/COP displacement and speed, 3D start/end points, widget pixels (+ 2D pixel arrows if `pose2d`) |
| `weight_shift` | 2D pixel positions of the feet scaled by the load split — **needs `pose2d`** |
| `forces` | Fz/Fx/Fy/Tz total/left/right, % of P1 weight, max, timing, time-to-impact |
| `time` | Time axis P1→P10 (s) |
| `force_vectors_3d` | 3D arrows per foot and combined (tracking frame) |
| `values_for_COP_trajectory` | Weight shift % per foot, midfoot stance width |
| `COM_MODELS` | 2D COM models (full / without arms / lower / upper) — **needs `pose2d`** |
| `power`, `energy`, `impulse` | Force×COM-velocity, its integral, force integral, each with max/timing/time-to-impact |
| `small_force_vectors` | Side-panel force arrows (pixels) |
| `COM_COP_traj_back_view` | Back-view COM/COP trajectory (pixels) |
| `critical_events` | Echo of the input events (does **not** run the force computation) |
| `frame_offset` | Always `0` for computed forces (does **not** run the force computation) |

Helpers: `compute_all(...)`, `ForcesScreen(json3d, pose2d, **overrides)`,
`describe_requirements(key=None)` (what each function needs, as data),
`resolved_configuration(json3d, **overrides)` (what the engine will *actually* use for this file).

`OUTPUT_KEYS` lists all keys. A function runs only the internal stages its output needs
(e.g. `energy` runs 6 of 19 stages); `ForcesScreen` caches stages so nothing is computed twice.

---

## 3. Input contract

### 3.1 3D pose JSON (`json3d`) — unchanged from the engine

Two layouts are accepted. Required content in both:

| Content | Flat layout | Nested layout |
|---|---|---|
| Pose | `data.pose` — 29 markers × frames × 3 (metres) | `data_3d_bio.data.pose` |
| Club (golf only, 3D) | `data.club` — dict keyed by frame index (`"0"…`), each with 3-component `start`, `mid`, `end` | `data_3d_bio.data.club` |
| Configuration | `configuration.{video,swing,user,processing}` | `data_3d_bio.configuration.{…}` |

Marker order (index → name in the source file): see the engine; sides are **anatomical** and fixed
(index 1–5 right leg, 6–10 left leg, 14–17 right arm, 18–21 left arm).

### 3.2 2D pose JSON (`pose2d`) — optional

`pose2d["data"]` = frames × markers × (x, y) pixels (marker-first is also accepted).
At least **29 markers** are required (files with more, e.g. 31, are fine — the first 29 are used);
the frame count **must equal** the 3D frame count. Without `pose2d`, only the keys that depend on
2D pixels are omitted: top-level `weight_shift`, `COM_MODELS`, and inside `force_vectors`:
`F{x,y,z}_{t,l,r}_pix`, `COP_for_Frontend_X/Y`, `factor_for_midfoot_point_frontend`,
`proj_GRF_COM_2d`, `COM_2d`. A warning is logged; nothing is faked.

---

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
| `force_vectors_3d`, `small_force_vectors`, `COM_COP_traj_back_view` | events (P1), `handedness` |
| `time` | `frame_rate`, events P1/P10 |
| `weight_shift`, `COM_MODELS` | `pose2d` (required) |
| `values_for_COP_trajectory` | nothing beyond the pose |
| `critical_events` | `critical_events` only (no force computation) |
| `frame_offset` | nothing |

### 4.2 Known input-reader limitation (important for integration)

The engine reads **mass and frame rate from the file only in the nested layout**
(`data_3d_bio.configuration…`). For the **flat layout** it silently uses defaults
(70 kg / 240 Hz) — wrong for most files and it changes every force (acceleration scales with
frame_rate²). Until the reader is extended, the backend must pass them:

```python
cfg  = json3d["configuration"]                       # flat layout (nested: json3d["data_3d_bio"]["configuration"])
comp = fs.ForcesScreen(json3d, pose2d,
                       body_mass=cfg["user"]["weight"],
                       frame_rate=cfg["video"]["frame_rate"])
```

`fs.resolved_configuration(json3d)` returns what the engine would use *without* overrides — log it
to catch this. When `config.json` becomes a separate input, read `user.weight` and `video.frame_rate`
there and pass them as above.

---

## 5. Output contract

All values are plain JSON types. Non-finite numbers are returned as `null`. Per-frame arrays have
length **n = number of frames in the 3D pose** unless stated. Numbers follow the measured-force
script's rounding (e.g. 2 decimals for forces).

| Key | Shape | Units / notes |
|---|---|---|
| `forces` | 57 fields: per-frame lists `Fz/Fx/Fy/Tz _t/_l/_r` and `Fz/Fx/Fy _t/_l/_r _wt`; scalars `*_max`, `*_max_timing`, `*_time_to_impact` | N; `Tz` N·m; `_wt` % of total vertical force at P1; timing / time-to-impact in **ms** (window P1→P10) |
| `time` | list, length **P10−P1** | seconds; starts at 1/frame_rate (measured-script convention) |
| `force_vectors` | 92 fields: per-frame lists, 2-element lists (`pixel_rows/colm_foot_*`), scalars (`*_Torque_max`, `*_Moment_Arm_max*`, timings) | Torques N·m; moment arms and COM/COP displacement **cm** (+ `_inch`); `COM_vel` cm/s (`_ips` in/s); `COM_vs_Stance` is `[[value], …]` (1-element lists, as in the measured script); pixels in a 375×180 widget |
| `force_vectors_3d` | 18 per-frame lists `{x,y,z}_{start,end}_COP_{l,r,combined}` | Tracking-frame coordinates (m); arrow end = start + force/1500 |
| `values_for_COP_trajectory` | 4 per-frame lists | `weight_shift_*` %; `stance_midfeet` cm (+ `_inch`) |
| `weight_shift` | 3 per-frame lists | pixels (needs `pose2d`) |
| `COM_MODELS` | 4 per-frame lists of `[x, y]` | pixels (needs `pose2d`) |
| `power` / `energy` | 42 fields each: per-frame lists + scalars `*_max`, `*_max_timing`, `*_time_to_impact` | W / J. **`*_max_timing` here is a frame index relative to P1, not ms** (measured-script convention); `*_time_to_impact` ms |
| `impulse` | 36 fields | N·s; `*_max_timing` is an integer frame index; `*_time_to_impact` ms |
| `small_force_vectors` | 3 objects (`force_vectors_lead/_trail/_combined`), each with `starting_x/y`, `ending_x/y` int lists | pixels, 223×75 panel |
| `COM_COP_traj_back_view` | 12 fields | pixels, 375×180 widget |
| `critical_events` | the input event dict (e.g. `P1…P10`, extra keys such as `p7_by` passed through) | frame indices; **only the events that were in the input** — resolved fallbacks are not added |
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
the "plate frame" COP (`force_vectors.COP_l/_r/COP`) is **synthetic** — registered to the midpoint of
the two midfoot markers at P1 (vertical: left-midfoot height at P1). COP per foot is the point where
the predicted force's line of action meets the ground, taken through the foot-segment COM.

### 6.3 Left-handed players
Measured left-handed data showed the horizontal forces reversed relative to the plates while the 3D
tracking frame is the same as for right-handers. For `leftHanded` files the analysis is therefore
rotated 180° about the vertical axis into the plate frame: `Fx`, `Fy`, the x/y torques and the
displacements along those axes are sign-flipped; `Fz`, `Tz`, the z torque and ratios are unchanged.
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
| Club frames ≠ pose frames | `ValueError` | "Club data has N frames but pose data has M" |
| Club points not 3D (e.g. 2D pixel coordinates) | `ValueError` | names the frame/part and component count |
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
1. **`Tz_*` (vertical torque) is a closure estimate**, not a measurement (the part of the vertical
   torque the predicted shear couple does not explain). Treat as qualitative.
2. **COP is model-based**, not measured; plate-frame COP is synthetic (§6.2).
3. **Left-handed rule** is based on one left- and one right-handed trial (§6.3). Check the sign of
   `Fx/Fy` on a few more lefty sessions with plates before relying on it.
4. **`weight_shift` / `COM_MODELS`** run the force computation although they only need `pose2d`
   (they share a stage with the pixel force arrows); a later split will remove this.
5. **Flat-layout mass/frame-rate** must be passed explicitly (§4.2).
6. **Club data must be 3D** for golf; files whose `club` block is 2D pixel data fail by design.
7. Unrecognised `swingType` silently uses the pitching preset (logged at INFO) — validate upstream.
8. `critical_events` only echoes input events; estimated/fallback events are not added.

---

## 11. Integration checklist

- [ ] Engine file and `force_screen_modules.py` deployed together; `numpy`, `scipy` installed.
- [ ] `body_mass` (kg) and `frame_rate` (Hz) passed explicitly from the configuration.
- [ ] `swingType` is one of `hitting` / `pitching` / `full swing`.
- [ ] Golf files carry a 3D `data.club` block with one entry per pose frame.
- [ ] `pose2d` (≥ 29 markers, same frame count) supplied when the 2D pixel keys are needed.
- [ ] One `ForcesScreen` per request; reuse it for multiple outputs.
- [ ] `ValueError` / `KeyError` from §8 handled and shown to the user; WARNING logs collected.
- [ ] Acceptance test: `fs.compute_all(...)` for a known trial equals the stored reference response
      (`json.dumps(..., sort_keys=True)` equality).

## 12. Preparing for split inputs (`config.json`, `pose_3d.json`, `pose_2d.json`)

`describe_requirements()` already groups needs by input: configuration paths (§4), `data.pose` /
`data.club` (pose_3d), pose2d (pose_2d). To split, build the same `json3d` dict from the parts
(`{"configuration": cfg, "data": {"pose": …, "club": …}}`) and pass `body_mass` / `frame_rate`
explicitly; no module change is needed.
