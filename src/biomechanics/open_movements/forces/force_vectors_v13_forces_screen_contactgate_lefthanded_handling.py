import json
import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt
from scipy.optimize import minimize
from scipy.ndimage import median_filter
import os
import math
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def numpy_int_parser(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError


class ForceVectors:
    # -------------------- Rotational-inertia configuration --------------------
    # Which two raw markers define each segment's long axis (proximal, distal),
    # matching the same pairs already used for compute_COM in generate_force_vectors.
    # Used to estimate each segment's own angular velocity/acceleration (see
    # _compute_segment_rotational_moment). Keys use the *segment* name (e.g.
    # "foot_l"); "trunk"/"head" have no l/r suffix since they are split between
    # sides downstream, not duplicated per side.
    # -------------------- Phase-position labels --------------------
    # Display names for the p1-p8 event-frame indices found under
    # json3d["configuration"]["swing"]["critical_events"] (or the nested
    # data_3d_bio location -- see extract_positions). Two movement-specific
    # sets are provided since the same p1-p8 keys mean different things for a
    # pitching delivery vs. a hitting swing. Also used by _extract_swing_type
    # / from_swing_type to pick sensible defaults for whichever movement the
    # input JSON declares itself as.
    PITCHING_PHASE_LABELS = {
        "p1": "Set", "p2": "Lift", "p3": "Initiation", "p4": "Contact",
        "p5": "Max Separation", "p6": "Layback", "p7": "Release", "p8": "Finish",
    }
    HITTING_PHASE_LABELS = {
        "p1": "Stance", "p2": "Load", "p3": "First Move", "p4": "Foot Contact",
        "p5": "Max Separation", "p6": "Swing Initiation", "p7": "Impact", "p8": "Finish",
    }

    SEGMENT_AXIS_MARKERS = {
        "foot_l": ("ankle_l", "foot_l"), "shank_l": ("knee_l", "ankle_l"), "thigh_l": ("hip_l", "knee_l"),
        "foot_r": ("ankle_r", "foot_r"), "shank_r": ("knee_r", "ankle_r"), "thigh_r": ("hip_r", "knee_r"),
        "trunk": ("pelvis", "neck"), "head": ("skullbase", "skull"),
        "upperarm_l": ("shoulder_l", "elbow_l"), "forearm_l": ("elbow_l", "wrist_l"), "hand_l": ("wrist_l", "hand_l"),
        "upperarm_r": ("shoulder_r", "elbow_r"), "forearm_r": ("elbow_r", "wrist_r"), "hand_r": ("wrist_r", "hand_r"),
        "club": ("club_grip", "club_headpt"),
    }

    # Approximate radius-of-gyration / segment-length ratios about each segment's
    # TRANSVERSE (perpendicular-to-long-axis) principal axis through its own CG,
    # from published anthropometric tables (Winter, "Biomechanics and Motor Control
    # of Human Movement", 4th ed., Table 4.1). These are population-average
    # constants, not measured from your subject -- treat as approximate, and
    # override per-key if you have better data (e.g. from DXA or segment-specific
    # regression equations) for your population. Keyed by the segment's base name
    # (no l/r suffix, no trunk/head distinction beyond their own entries).
    SEGMENT_RADIUS_OF_GYRATION_RATIO = {
        "foot": 0.475, "shank": 0.302, "thigh": 0.323,
        "trunk": 0.50, "head": 0.495,
        "upperarm": 0.322, "forearm": 0.303, "hand": 0.297,
        # Rough thin-rod approximation (1/sqrt(12) ~= 0.289). A real club's mass is
        # concentrated toward the head, not uniform, so its true rotational inertia
        # about its own COM is smaller than a uniform rod's -- override this with a
        # measured/spec value (clubmaker swingweight data or a MOI meter reading) if
        # you have one; this default is a placeholder, not an anthropometric constant.
        "club": 0.289,
    }

    def __init__(
        self,
        body_mass: float = 95,
        frame_rate: int = 240,
        fc_pos: float = 18.0,
        fc_dyn: float = 10.0,
        fc_grf: float = 7.0,
        contact_threshold: float = 10,
        contact_hysteresis: float = 5.0,
        min_contact_frames: int = 5,
        contact_height_threshold: float = 0.00, #0.05
        contact_confidence_band: float = 0.1, #0.1
        continuous_contact: bool = True,
        filter_order: int = 6,
        swap_xy: bool = True,
        legacy_lateral_weight_direction: bool = True,
        lateral_weight_source: str = "body_com", #change to "pelvis" or "body_com" if desired
        include_rotational_inertia: bool = True,
        despike_markers: bool = True,
        despike_window: int = 10,
        despike_n_sigma: float = 3.0,
        mu_friction: float = 1.0,
        swap_lr: bool = True,
        include_club: bool = False,
        club_shaft_mass: float = None,
        club_head_mass: float = None,
        plate_layout: str = "amti",
        anchor_foot_depth: bool = False,
        lefty_plate_rotation: bool = True,
        kinematic_contact_gate: bool = False,

    ) -> None:
        """
        Parameters
        ----------
        body_mass : subject mass in kg
        frame_rate : capture frame rate in Hz
        fc_pos : Hz, low-pass cutoff applied to segment COM positions before they are
            differentiated. This is the ONLY position-domain smoothing in the whole
            pipeline (see module docstring) -- velocity/acceleration are obtained by
            plain differentiation of this filtered position, not re-filtered. Slower
            movements (walking) can use a lower value (~5-8Hz); fast athletic
            movements (golf, baseball) generally need this higher (~15-20Hz) or real
            acceleration peaks get smoothed away, which paradoxically makes the
            *remaining* signal look noisier relative to the flattened peaks. Default
            (5 Hz, order 4) matches the reference single-pass "Script A" tuning.
            Also used to filter the raw axis markers feeding the rotational-inertia
            estimate, for the same reason (single band-limiting pass before any
            differentiation).
        fc_dyn : Hz, low-pass cutoff applied to the summed force/moment signal right
            after inverse dynamics, before it's split into left/right/COP. This is a
            cleanup pass for summation artifacts across segments -- keep it <= fc_pos.
        fc_grf : Hz, final low-pass cutoff applied to the reported GRF traces. This is
            your "how smooth do I want the plot to look" knob -- keep it <= fc_dyn.
        contact_threshold : N, Fz value above which a foot is considered "in contact"
            (rising edge of the hysteresis band).
        contact_hysteresis : N, how far below contact_threshold Fz must fall before the
            foot is considered "out of contact" (falling edge = contact_threshold -
            contact_hysteresis). This prevents chatter when Fz hovers near the threshold.
            Used as a final reporting-level cleanup pass on the already-optimized GRF
            (see contact_height_threshold below for the primary contact signal used
            for movements with a real flight phase).
        min_contact_frames : minimum length (in frames) a contact/no-contact state must
            persist to be kept; shorter runs are merged into the surrounding state
            (debounce). Shared by both the Fz-threshold hysteresis above and the
            kinematic height-based contact detection below.
        contact_height_threshold : distance (in the same length units as your marker
            data -- meters if that's what your tracking system uses) above a foot's
            own self-calibrated ground level, below which that foot is considered
            FULLY "in contact" (confidence = 1.0 -- see contact_confidence_band).
            Ground level for each foot is estimated independently as the lowest
            height that foot's COM reaches anywhere in the trial (handles a sloped
            surface like a pitching mound fine, since each foot gets its own
            reference rather than assuming both feet touch at the same absolute
            height). This drives the PRIMARY contact signal for
            continuous_contact=False movements (pitching, gait, jumps): it's
            computed purely from marker kinematics, independent of any force
            computation, and is passed into optimize_bilateral_grf as a continuous
            per-frame confidence (not a boolean -- see contact_confidence_band) so a
            foot that's kinematically airborne gets its GRF pulled toward zero by
            the optimizer's own bounds. This matters because the optimizer's
            moment-equilibrium split has NO contact awareness on its own -- without
            a contact signal telling it a foot is off the ground, it will still
            assign some nonzero force to that foot on every single frame just to
            balance the moment equations, and that residual can easily sit above
            contact_threshold, making the downstream Fz-hysteresis pass (which only
            sees the optimizer's already-computed output) wrongly conclude the foot
            is "in contact" when kinematically it isn't. Default 0.05 is a
            reasonable starting point for meter-scale data; tighten it if a foot is
            being marked "in contact" too early during landing, loosen it if a
            planted foot's small vertical wobble is incorrectly flagging brief false
            liftoffs.
        contact_confidence_band : additional height (same units as
            contact_height_threshold) over which contact confidence ramps linearly
            from 1.0 (at contact_height_threshold) down to 0.0 (at
            contact_height_threshold + contact_confidence_band), instead of
            switching instantly at a single height. This is what makes contact
            gating produce a smooth, physically plausible loading/unloading ramp
            during foot strike/toe-off rather than an instantaneous jump: the
            optimizer's per-foot force bounds are scaled by this confidence every
            frame, so they shrink toward zero gradually as a foot lifts off (and
            expand back to normal gradually as it plants) rather than snapping
            between "fully free" and "hard-pinned to zero" at one exact frame.
            Default 0.10 means confidence reaches 0 once a foot is 15cm (0.05 +
            0.10) above its own ground reference for meter-scale data -- widen this
            if transitions still look too abrupt, narrow it if the ramp feels too
            sluggish relative to how fast the real foot strike/toe-off happens.
        continuous_contact : if True, skip contact gating entirely and treat both feet
            as in contact for the whole trial. Appropriate ONLY for movements with no
            flight phase, e.g. a golf swing or a baseball swing from a fixed stance --
            use the `for_golf`/`for_baseball_swing` presets for those rather than
            setting this directly. Defaults to False (contact gating ON) because that
            is the safe choice for the general case: gait, jumps, box lifts/carries,
            pitching, or any movement where a foot actually leaves the ground. Getting
            this wrong the "safe" direction (continuous_contact=False on a trial that's
            actually continuous) just costs you an unnecessary hysteresis pass on
            already-nonzero-Fz data; getting it wrong the other way (continuous_contact
            =True on a trial with a real flight/swing phase) silently reports nonzero
            GRF during frames where the foot was never in contact with anything.
        filter_order : Butterworth filter order used everywhere sosfiltfilt is called.
            Higher = steeper rolloff at the same cutoff frequency (more aggressive
            rejection of content just above fc_pos/fc_dyn/fc_grf), at the cost of a
            slightly longer effective filter transient at the start/end of the trial.
            Default 4 matches the reference single-pass "Script A" tuning; 6-8 gives
            noticeably harder filtering per pass if you need it, but remember each
            stage (fc_pos, fc_dyn, fc_grf) is a *separate* pass, so raising order
            compounds across stages faster than raising order in a single-pass filter.
        swap_xy : if True (default), swap the X/Y axes read from the tracking data before
            any processing, so the pipeline's internal/output convention matches standard
            force-plate labeling (Fx = lateral/ML, Fy = anterior-posterior/AP, Fz = vertical)
            instead of the tracking system's own convention (X = depth/AP). Set False if
            your tracking data already uses the force-plate convention natively.
        legacy_lateral_weight_direction : if False (default, recommended), the
            medio-lateral HAT (trunk/head) weight-transfer split assigns MORE weight to
            whichever foot the lateral_weight_source signal is closer to -- the
            anatomically correct direction (e.g. that signal shifting toward the left
            foot -> more Fz under the left foot). If True, restores the old, inverted
            convention (closer to a foot -> more weight assigned to the OTHER foot).
            Only set True to reproduce/compare against prior runs that used the old
            convention -- new work should leave this False.
        lateral_weight_source : which signal drives the left/right HAT weight-transfer
            split (weight_ratio). Options:
              "hip_midpoint" (default) -- midpoint of the hip_l/hip_r joint-center
                  markers, lateral position. This is the standard general-purpose
                  proxy: in the absence of an external unmodeled load, horizontal COM
                  projection tracks hip position closely across gait, squats, lunges,
                  jumps, cutting, and most gym/sport movements. More anatomically
                  precise than a single "pelvis" marker (true hip-joint-center
                  midpoint vs. one sacral/pelvis marker whose placement can vary by
                  tracking system).
              "pelvis"       -- single pelvis marker lateral position. Kept for
                  back-compat / comparison; hip_midpoint is generally preferable when
                  both hip markers are reliably tracked.
              "body_com"     -- whole-body mass-weighted COM (all segments) lateral
                  position. Can work for bilateral movements with no external load,
                  but for any task with an unmodeled external load (carrying,
                  lifting), it can be actively misleading: reaching to one side
                  without stepping often causes the trunk/head to counter-lean AWAY
                  from the reach side to balance a load that isn't in the model,
                  which can flip weight_ratio in the wrong direction relative to
                  which foot is actually bearing more weight.
            None of these model an external hand-held load explicitly -- for
            carry/lift/place tasks with a real payload, treat whichever option you
            pick as an approximation and validate it against known left/right
            transitions in your own data before trusting it in production.
        mu_friction : effective ground-contact friction coefficient (shear/normal
            ratio) used as the soft friction-cone guard in optimize_bilateral_grf's
            per-foot Fx/Fy split, i.e. the ceiling on |Fx|/Fz (or |Fy|/Fz) a foot is
            allowed to exert before the friction_penalty term kicks in. Default 1.0
            reflects cleated/spiked athletic footwear (golf, baseball) digging into
            turf, where effective traction is routinely well above classic bare-shoe
            Coulomb estimates and can exceed 1.0. The previous hardcoded value here
            was 0.5 (plain-shoe-on-hard-floor territory), which actively suppressed
            legitimate high-shear/low-normal-load frames -- e.g. a foot briefly
            unloaded in Fz during a fast weight transfer while still pushing hard
            laterally -- by fighting the moment-equilibrium term for those frames
            instead of just guarding against genuinely unphysical splits. Lower this
            back toward 0.5-0.8 for barefoot/flat-shoe movements (gait, lifting) if
            you see the optimizer now allowing unrealistically high per-foot shear.
        swap_lr : if True, swap the left/right foot GRF and moment outputs (GRF_left
            <-> GRF_right, M_left <-> M_right, and everything downstream: contact
            gating, COP_l/COP_r, and the Fx_l/Fy_l/Fz_l vs Fx_r/Fy_r/Fz_r channels in
            the returned/plotted data). Fx_t/Fy_t/Fz_t totals are unaffected since
            they're a symmetric sum. Default False. Set True if your per-side traces
            are shaped correctly but consistently attached to the wrong foot -- e.g.
            predicted Fz_l tracking the SHAPE of measured Fz_r (and vice versa) frame
            for frame, including matching peaks/dips at the same times, just under
            the other foot's label. That pattern points to a left/right label
            mismatch somewhere between marker capture and your ground-truth
            comparison (a common cause: force plates or markers labeled by
            camera/screen-view side rather than the subject's own anatomical left/
            right, which are mirror images when the subject faces the camera) rather
            than an error in the force computation itself -- this flag corrects for
            that at the output stage without needing to trace the mislabeling back
            to its source.
        include_club : if True (default False here), model the club as a mass-bearing
            segment using the club's own "start" (grip), "mid" (shaft), and "end"
            (head) coordinates from the input JSON's dedicated club data block -- NOT
            the "club_main"/"club_head" pose markers, which are mislabeled (they're
            actually clavicle_r/clavicle_l) and are never used for the club. Its
            momentum change (real mass, and very high deceleration right around
            impact) is included in the total-body force/moment sum this way. This
            targets errors that concentrate specifically in the impact window
            (P4-P9-ish) and don't respond to filter tuning, since they're a
            genuinely missing physics term, not noise: a fast-swinging club's
            momentum is a real external contribution to the true ground reaction
            force that a body-only model can't see. Only relevant for movements
            where a club is actually tracked (golf) -- leave False for pitching/
            hitting/gait/etc, where there's no club data block to read.

            The club is modeled as two point masses -- shaft (at the "mid"
            coordinate) and head (at the "end"/"club_headpt" coordinate) -- rather
            than a single point, since head mass is concentrated enough at the tip
            that a single-point approximation biases both the COM location and the
            club's own lever arm in the moment balance. club_COM is the
            mass-weighted average of those two points every frame.
        club_shaft_mass, club_head_mass : masses in kg for the two-point club model
            above. Default None on both means: read
            configuration.swing.clubType from the input JSON each time
            generate_force_vectors runs, and use real-world estimates -- driver:
            shaft 0.059 kg, head 0.192 kg; anything else (irons etc.): shaft 0.120
            kg, head 0.256 kg. Set either or both explicitly to override this
            per-swing lookup with a fixed value.
        plate_layout : "amti" (default) or "kistler". Only affects the frontend-widget
            mapping in generate_forces_screen (widget size, x-offset, 2D arrow scale),
            i.e. which of the measured-force script's two layouts is mimicked. There are
            no physical plates behind computed forces, so this is purely a presentation
            choice; keep whichever layout your frontend widget is built for.
        anchor_foot_depth : generate_forces_screen only. If True, reproduces the
            measured-force script's "depth correction": the depth (AP) travel of the
            output-left foot is subtracted from every marker before COM / moment-arm /
            arrow geometry is built. That correction targets monocular depth noise for a
            planted-feet golf swing; for pitching (stride foot really travels) leave it
            False. Never touches the force computation itself.
        lefty_plate_rotation : generate_forces_screen only. When the input JSON says the
            player is left-handed, the analysis is rotated 180 degrees about the vertical
            axis into the force-plate frame before the response is built (horizontal force /
            torque / displacement metrics are then reported the way the plates report them;
            see _build_forces_screen). Checked on a measured left-handed vs right-handed
            pair: relative to the plates, a lefty's horizontal forces are reversed while the
            3D tracking frame itself is not. Right-handed files never take this path.
            Set False to switch the rotation off (e.g. a lefty session whose plates were not
            turned). The legacy response is unaffected either way.
        kinematic_contact_gate : bool, default False (no change to existing outputs). Only
            used when continuous_contact=False. Re-applies the marker-height contact
            confidence (contact_height_threshold / contact_confidence_band) to each foot's
            force AFTER the fc_grf low-pass, i.e. at the very end, scaling a foot's force by
            its confidence (exactly 0 once the foot is contact_height_threshold +
            contact_confidence_band above its own lowest point). Without it the height signal
            only limits the optimizer's allowed force; the low-pass then smears force from the
            last grounded frames into the flight phase and the final Fz-threshold gate cannot
            tell that smear from real contact (airborne foot still shows force, drops to 0,
            comes back). Whatever the gate removes from one foot is added to the other foot so
            the total ground force stays equal to the whole-body inverse-dynamics total. Caveat:
            it is a height rule -- a loaded foot whose heel/foot COM rises more than
            threshold + band (e.g. a rear foot pivoting on its toes) would be shrunk too, so
            tune the two height values for the movement.
        include_rotational_inertia : if True (default), add each segment's own
            rotational "free moment" (I_perp * alpha_perp, from that segment's
            proximal->distal marker axis swinging over time) into total_moment,
            on top of the existing translational r x F term. See module docstring
            and _compute_segment_rotational_moment for what this does and does not
            capture (notably: no axial/twist-spin contribution, since that isn't
            observable from a 2-marker-per-segment axis). Set False to reproduce the
            old point-mass-only behavior.
        despike_markers : if True (default), run a Hampel-filter-style despike +
            NaN/Inf gap-fill pass on every raw marker's trajectory BEFORE compute_COM
            and before the single fc_pos Butterworth pass (see _despike_marker and
            the module docstring's single-pass filtering design). This is NOT a
            second low-pass filter: it only replaces individual samples that deviate
            sharply from their own local median (a targeted fix for single-frame
            tracking glitches/teleports/occlusion gaps), and otherwise leaves real
            fast athletic motion untouched, so it doesn't fight fc_pos's job of
            frequency-domain band-limiting. Set False to reproduce the old
            behavior (raw markers passed straight to compute_COM).
        despike_window : frames, size of the centered window used to compute the
            local median/MAD for the Hampel despike. Smaller = more locally
            sensitive (catches shorter glitches, more chance of nudging real fast
            motion); larger = more conservative. Default 7 is intentionally narrow.
        despike_n_sigma : robust-standard-deviation multiple (1.4826 * local MAD)
            a sample must exceed its local median by to be flagged as an outlier
            and replaced. Default 4.0 is intentionally conservative -- it should
            only catch genuine single-frame glitches, not normal high-velocity
            athletic motion. Lower this if glitches are getting through; raise it
            if real fast-motion samples are being incorrectly flattened.
        """
        self.body_mass = body_mass
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate
        self.fc_pos = fc_pos
        self.fc_dyn = fc_dyn
        self.fc_grf = fc_grf
        self.contact_threshold = contact_threshold
        self.contact_hysteresis = max(contact_hysteresis, 0.0)
        self.min_contact_frames = max(int(min_contact_frames), 1)
        self.contact_height_threshold = contact_height_threshold
        self.contact_confidence_band = contact_confidence_band
        self.continuous_contact = continuous_contact
        self.filter_order = filter_order
        self.swap_xy = swap_xy
        self.legacy_lateral_weight_direction = legacy_lateral_weight_direction
        if lateral_weight_source not in ("hip_midpoint", "pelvis", "body_com"):
            raise ValueError(
                f"lateral_weight_source must be one of 'hip_midpoint', 'pelvis', 'body_com'; "
                f"got {lateral_weight_source!r}"
            )
        self.lateral_weight_source = lateral_weight_source
        self.include_rotational_inertia = include_rotational_inertia
        self.despike_markers = despike_markers
        self.despike_window = despike_window
        self.despike_n_sigma = despike_n_sigma
        self.mu_friction = mu_friction
        self.swap_lr = swap_lr
        self.include_club = include_club
        self.club_shaft_mass = club_shaft_mass
        self.club_head_mass = club_head_mass
        if plate_layout not in ("amti", "kistler"):
            raise ValueError(f"plate_layout must be 'amti' or 'kistler'; got {plate_layout!r}")
        self.plate_layout = plate_layout
        self.anchor_foot_depth = anchor_foot_depth
        self.lefty_plate_rotation = bool(lefty_plate_rotation)
        self.kinematic_contact_gate = bool(kinematic_contact_gate)

    # -------------------- Movement presets --------------------
    @classmethod
    def for_golf(cls, body_mass: float = 70, frame_rate: int = 240, **overrides):
        """Golf (full swing): both feet grounded throughout (continuous contact, no
        contact gating), same filter set as the baseball presets, club modeled as a
        two-point mass (include_club=True; needs the club data block in the input)."""
        defaults = dict(
            fc_pos=18.0, fc_dyn=12.0, fc_grf=3.5, continuous_contact=True,
            include_club=True,
        )
        defaults.update(overrides)
        return cls(body_mass=body_mass, frame_rate=frame_rate, **defaults)

    @classmethod
    def for_baseball_swing(cls, body_mass: float = 70, frame_rate: int = 240, **overrides):
        """Baseball/softball hitting: very fast rotation. Contact gating is ON
        (continuous_contact=False) on purpose -- the back foot can come off the ground
        during the swing, so a foot's GRF is shrunk to zero while it is airborne."""
        defaults = dict(
            fc_pos=18.0, fc_dyn=12.0, fc_grf=3.5, continuous_contact=False,
        )
        defaults.update(overrides)
        return cls(body_mass=body_mass, frame_rate=frame_rate, **defaults)

    @classmethod
    def for_pitching(cls, body_mass: float = 75, frame_rate: int = 240, **overrides):
        """Baseball pitching: stride foot briefly leaves the ground, so contact gating
        stays on, but with a longer debounce to avoid chatter during the fast plant."""
        defaults = dict(
            fc_pos=18.0, fc_dyn=12.0, fc_grf=3.5, continuous_contact=False,
            contact_threshold=50.0, contact_hysteresis=20.0, min_contact_frames=8,
        )
        defaults.update(overrides)
        return cls(body_mass=body_mass, frame_rate=frame_rate, **defaults)

    # -------------------- Movement-type auto-detection --------------------
    # Maps a swingType value found in the input JSON (configuration.swing.
    # swingType) to the preset (classmethod name) used to build a ForceVectors
    # instance for it. Extend this dict as more movement types get their own
    # preset -- e.g. add "golf": "for_golf" once golf inputs carry a matching
    # swingType value.
    _SWING_TYPE_PRESETS = {
        "hitting": "for_baseball_swing",
        "pitching": "for_pitching",
        "full swing": "for_golf",
    }

    @staticmethod
    def _extract_swing_type(json3d: dict):
        """Read the movement type (e.g. "pitching", "hitting") out of the input
        JSON's configuration.swing.swingType field, checking both the nested
        data_3d_bio location and the flat/older location -- mirrors the
        dual-format handling used elsewhere in this file (e.g.
        extract_positions, _extract_pose_array). Returns None (not an error)
        if neither location has it.
        """
        try:
            return json3d["data_3d_bio"]["configuration"]["swing"]["swingType"]
        except (KeyError, TypeError):
            pass
        try:
            return json3d["configuration"]["swing"]["swingType"]
        except (KeyError, TypeError):
            pass
        return None

    @staticmethod
    def _extract_handedness(json3d: dict):
        """"left" / "right" / None, from either input layout (nested data_3d_bio first,
        then the flat configuration). Looks at configuration.handedness and at any
        sub-block of the configuration (swing, ...) carrying a handedness value."""
        configs = []
        try:
            configs.append(json3d["data_3d_bio"]["configuration"])
        except (KeyError, TypeError):
            pass
        if isinstance(json3d.get("configuration"), dict):
            configs.append(json3d["configuration"])
        for cfg in configs:
            blocks = [cfg] + [v for v in cfg.values() if isinstance(v, dict)]
            for blk in blocks:
                val = blk.get("handedness")
                if isinstance(val, str) and val.strip():
                    v = val.strip().lower()
                    if v.startswith("left"):
                        return "left"
                    if v.startswith("right"):
                        return "right"
        return None

    @staticmethod
    def _rot180(arr):
        """Rotate 180 degrees about the vertical axis (internal frame: columns 0 and 1
        are the horizontal ones). Exact (sign change only); vertical untouched."""
        out = np.array(arr, dtype=float, copy=True)
        out[..., 0] *= -1.0
        out[..., 1] *= -1.0
        return out

    def _rotate_ctx_180(self, ctx):
        """Copy of ctx with every position, force and moment rotated 180 deg about the
        vertical axis. Fx/Fy change sign; (Mx, My) change sign, Mz does not (a rotation
        about z leaves the vertical pseudovector component unchanged)."""
        r = dict(ctx)
        r["markers"] = {k: self._rot180(v) for k, v in ctx["markers"].items()}
        r["segment_COMs_filtered"] = {k: self._rot180(v) for k, v in ctx["segment_COMs_filtered"].items()}
        r["body_COM"] = self._rot180(ctx["body_COM"])
        inv = dict(ctx["inv_dyn"])
        inv["total_moment"] = self._rot180(ctx["inv_dyn"]["total_moment"])
        inv["total_force"] = self._rot180(ctx["inv_dyn"]["total_force"])
        r["inv_dyn"] = inv
        for s_ in ("l", "r", "t"):
            r[f"Fx_{s_}"] = -np.asarray(ctx[f"Fx_{s_}"], dtype=float)
            r[f"Fy_{s_}"] = -np.asarray(ctx[f"Fy_{s_}"], dtype=float)
        return r

    @staticmethod
    def _extract_club_type(json3d: dict) -> str:
        """Read configuration.swing.clubType from either input format -- the nested
        data_3d_bio location first (any movement sub-block under its configuration that
        carries a clubType, "swing" preferred), then the flat/older location. Returns a
        lower-cased, stripped string; "driver" when nothing is found (the previous
        default). Only "driver" selects the driver masses; every other value is treated
        as an iron (see the include_club block in _run_pipeline)."""
        found = None
        try:
            config = json3d["data_3d_bio"]["configuration"]
            if isinstance(config.get("swing"), dict) and config["swing"].get("clubType"):
                found = config["swing"]["clubType"]
            else:
                for val in config.values():
                    if isinstance(val, dict) and val.get("clubType"):
                        found = val["clubType"]
                        break
        except (KeyError, TypeError, AttributeError):
            pass
        if found is None:
            try:
                found = json3d["configuration"]["swing"].get("clubType")
            except (KeyError, TypeError, AttributeError):
                found = None
        if not isinstance(found, str) or not found.strip():
            return "driver"
        return found.strip().lower()

    @classmethod
    def from_swing_type(cls, json3d: dict, **overrides) -> "ForceVectors":
        """Build the ForceVectors preset matching this input JSON's movement
        type, read from configuration.swing.swingType (see
        _extract_swing_type).

        Currently recognized swingType values -> preset:
            "hitting"    -> for_baseball_swing (contact gating on: the back foot
                            can leave the ground during the swing)
            "pitching"   -> for_pitching (stride foot briefly leaves the ground,
                            contact gating on, longer debounce)
            "full swing" -> for_golf (both feet grounded throughout, no contact
                            gating; club modeled via include_club=True)
        Anything else -- an unrecognized swingType, or none found at all --
        falls back to for_pitching's settings. That's a deliberate choice, not
        an accident: it's the most conservative non-continuous-contact preset, and is the safer default for a movement that might involve a foot
        actually leaving the ground (e.g. a jump) until that movement gets its
        own dedicated preset -- silently treating an unrecognized movement as
        continuous-contact (for_baseball_swing/for_golf) would risk reporting
        nonzero GRF during frames where a foot was never on the ground. Extend
        _SWING_TYPE_PRESETS when adding support for more movement types.

        Also pulls body_mass/frame_rate from extract_meta(json3d) when
        available, so the preset uses the athlete's actual recorded
        weight/frame rate rather than the preset's generic default. Explicit
        **overrides (or an explicit body_mass/frame_rate passed here) always
        win over both the extracted metadata and the preset's own default.
        """
        swing_type = cls._extract_swing_type(json3d)
        preset_name = cls._SWING_TYPE_PRESETS.get(swing_type, "for_pitching")
        if swing_type not in cls._SWING_TYPE_PRESETS:
            logging.info(
                "swingType=%r not recognized (expected one of %s); "
                "defaulting to the '%s' preset.",
                swing_type, sorted(cls._SWING_TYPE_PRESETS), preset_name,
            )
        preset_fn = getattr(cls, preset_name)

        meta = cls.extract_meta(json3d)
        kwargs = {}
        if "body_mass" in meta:
            kwargs["body_mass"] = meta["body_mass"]
        if "frame_rate" in meta:
            kwargs["frame_rate"] = meta["frame_rate"]
        kwargs.update(overrides)  # explicit overrides always win

        return preset_fn(**kwargs)

    @classmethod
    def process(cls, json3d: dict, **overrides) -> dict:
        """One-shot entry point: detect the movement type from the input JSON
        (see from_swing_type), build the matching ForceVectors preset, and run
        generate_force_vectors. Returns the same result dict
        generate_force_vectors always returns (data.forces / data.
        force_vectors / data.body_COM / data.time) regardless of which preset
        got auto-selected -- callers don't need to know or care which movement
        type was detected to consume the output.

        **overrides are passed straight through to the preset constructor
        (e.g. swap_lr=True, include_club=True, body_mass=88) -- use this to
        apply the same fixed overrides regardless of which preset gets
        auto-selected for a given input file.
        """
        fv = cls.from_swing_type(json3d, **overrides)
        return fv.generate_force_vectors(json3d)

    # -------------------- Main processing --------------------
    def generate_force_vectors(self, json3d):
        """Legacy response: {"data": {force_vectors_3d, force_vectors(COP*), forces,
        body_COM, time}}. Unchanged from v11. For the measured-force-equivalent
        response (Tz, moment arms, torques, COP trajectory, power, ...) use
        generate_forces_screen()."""
        ctx = self._run_pipeline(json3d)
        return self._format_legacy_response(ctx)

    def generate_forces_screen(self, json3d, pose2d=None):
        """Response with the same top-level keys / sub-keys as the measured-force
        script's generate_force_vectors(), built from the COMPUTED forces. See
        _build_forces_screen for the definition of every channel and the list of
        keys that need pose2d."""
        ctx = self._run_pipeline(json3d)
        return self._build_forces_screen(ctx, pose2d=pose2d)

    @classmethod
    def process_forces_screen(cls, json3d: dict, pose2d: dict = None, **overrides) -> dict:
        """One-shot: auto-detect movement preset (see from_swing_type) and return the
        measured-equivalent response."""
        fv = cls.from_swing_type(json3d, **overrides)
        return fv.generate_forces_screen(json3d, pose2d=pose2d)

    def _format_legacy_response(self, ctx):
        return {
            "data": {
                "force_vectors_3d": ctx["force_vectors_3d"],
                "force_vectors": {
                    "COP_l": ctx["COP_l"].tolist(),
                    "COP_r": ctx["COP_r"].tolist(),
                    "COP": ctx["COP_total"].tolist(),
                },
                "forces": ctx["forces"],
                "body_COM": np.around(ctx["body_COM"], 4).tolist(),
                "time": ctx["time"].tolist(),
            }
        }

    def _run_pipeline(self, json3d):
        """Shared computation behind both responses. Returns a context dict with
        everything either formatter needs (nothing here changes v11 numerics)."""
        self._diag = {}  # fresh diagnostics for this call
        M_data = self._extract_pose_array(json3d)

        if self.swap_xy:
            # Tracking convention: axis 0 = depth (AP), axis 1 = lateral (ML), axis 2 = vertical.
            # Force-plate convention: Fx = lateral (ML), Fy = anterior-posterior (AP), Fz = vertical.
            # Swap X/Y here, once, so every downstream force/moment/COP output already lines
            # up with the force-plate labeling -- no remapping needed when comparing traces.
            M_data = M_data.copy()
            M_data[..., [0, 1]] = M_data[..., [1, 0]]

        marker_names = [
            "pelvis", "hip_l", "knee_l", "ankle_l", "foot_l", "midfoot_l",
            "hip_r", "knee_r", "ankle_r", "foot_r", "midfoot_r",
            "nose", "spine_low", "spine_high", "shoulder_l", "elbow_l", "wrist_l", "hand_l",
            "shoulder_r", "elbow_r", "wrist_r", "hand_r",
            "neck", "skullbase", "skull", "clavicle_r", "clavicle_l", "ear_l", "ear_r"
        ]
        if M_data.shape[0] < len(marker_names):
            raise ValueError(f"Not enough markers in data: expected {len(marker_names)}, got {M_data.shape[0]}")
        markers = {name: np.squeeze(M_data[i]) for i, name in enumerate(marker_names)}

        if self.include_club:
            club_start, club_end, club_mid = self._extract_club_array(json3d, n_frames_expected=M_data.shape[1])
            markers["club_grip"] = club_start
            markers["club_headpt"] = club_end
            markers["club_mid"] = club_mid

        if self.despike_markers:
            markers = {name: self._despike_marker(pos, name=name) for name, pos in markers.items()}

        # --- Compute segment COMs ---
        def compute_COM(p1, p2, factor):
            return p1 - (p1 - p2) * factor

        segment_COMs = {
            "foot_l": compute_COM(markers["ankle_l"], markers["foot_l"], 0.4415),
            "shank_l": compute_COM(markers["knee_l"], markers["ankle_l"], 0.4459),
            "thigh_l": compute_COM(markers["hip_l"], markers["knee_l"], 0.4095),
            "foot_r": compute_COM(markers["ankle_r"], markers["foot_r"], 0.4415),
            "shank_r": compute_COM(markers["knee_r"], markers["ankle_r"], 0.4459),
            "thigh_r": compute_COM(markers["hip_r"], markers["knee_r"], 0.4095),
            "pelvis": markers["pelvis"],
            # Raw hip-joint-center markers (not a computed segment COM, zero mass like
            # "pelvis" above) -- kept as reference points only, for the hip-midpoint
            # lateral weight-transfer driver signal (see lateral_weight_source).
            "hip_l": markers["hip_l"],
            "hip_r": markers["hip_r"],
            "trunk": compute_COM(markers["neck"], markers["pelvis"], 0.4486),
            "head": compute_COM(markers["skull"], markers["skullbase"], 0.5976),
            "upperarm_l": compute_COM(markers["shoulder_l"], markers["elbow_l"], 0.5772),
            "forearm_l": compute_COM(markers["elbow_l"], markers["wrist_l"], 0.4574),
            "hand_l": compute_COM(markers["wrist_l"], markers["hand_l"], 0.79),
            "upperarm_r": compute_COM(markers["shoulder_r"], markers["elbow_r"], 0.5772),
            "forearm_r": compute_COM(markers["elbow_r"], markers["wrist_r"], 0.4574),
            "hand_r": compute_COM(markers["wrist_r"], markers["hand_r"], 0.79),
        }
        # if self.include_club:
        #     segment_COMs["club"] = compute_COM(
        #         markers["club_grip"], markers["club_headpt"], self.club_com_factor
        #     )

        if self.include_club:
            if self.club_shaft_mass is None or self.club_head_mass is None:
                # Default masses based on clubType in the input JSON (driver vs. iron)
                club_type = self._extract_club_type(json3d)
                if club_type == "driver":
                    self.club_shaft_mass = 0.059  # kg
                    self.club_head_mass = 0.192  # kg
                else:
                    self.club_shaft_mass = 0.120  # kg
                    self.club_head_mass = 0.256  # kg
            self.club_mass = self.club_shaft_mass + self.club_head_mass
            segment_COMs["club"] = (
                markers["club_mid"] * self.club_shaft_mass + markers["club_headpt"] * self.club_head_mass
            ) / self.club_mass
            

        # --- Segment masses (scaled by body mass) ---
        segment_masses = {
            "foot_l": 0.0137 * self.body_mass, "shank_l": 0.0433 * self.body_mass, "thigh_l": 0.1416 * self.body_mass,
            "foot_r": 0.0137 * self.body_mass, "shank_r": 0.0433 * self.body_mass, "thigh_r": 0.1416 * self.body_mass,
            "pelvis": 0.0,
            "trunk": 0.4336 * self.body_mass, "head": 0.0694 * self.body_mass,
            "upperarm_l": 0.0271 * self.body_mass, "forearm_l": 0.0162 * self.body_mass,
            "hand_l": 0.0061 * self.body_mass,
            "upperarm_r": 0.0271 * self.body_mass, "forearm_r": 0.0162 * self.body_mass,
            "hand_r": 0.0061 * self.body_mass,
        }
        if self.include_club:
            segment_masses["club"] = self.club_mass

        total_mass = sum(segment_masses.values())
        expected_mass = self.body_mass + (self.club_mass if self.include_club else 0.0)
        if abs(total_mass - expected_mass) > 0.5:
            logging.warning(f"Mass budget mismatch: {total_mass:.2f} vs {expected_mass}")

        # --- Smooth COM positions ONCE with an explicit low-pass filter. This is the
        # only position smoothing in the whole pipeline; compute_force below assumes
        # its COM input is already filtered at fc_pos and does NOT re-filter it, and
        # does NOT re-filter the velocity/acceleration derived from it either (see
        # module docstring -- this matches Script A's single-pass approach). ---
        segment_COMs_filtered = {k: self._filter_lowpass(v, self.fc_pos) for k, v in segment_COMs.items()}

        # --- Filter the raw axis markers used for rotational-inertia estimation, at
        # the same fc_pos and in the same single-pass fashion as segment_COMs_filtered
        # above, so position is band-limited exactly once before any differentiation
        # (translational OR rotational). See SEGMENT_AXIS_MARKERS / module docstring. ---
        axis_markers_filtered = None
        if self.include_rotational_inertia:
            axis_marker_names = set()
            for seg_key, (prox_name, dist_name) in self.SEGMENT_AXIS_MARKERS.items():
                if seg_key == "club" and not self.include_club:
                    # club_grip/club_headpt are only extracted into `markers` when
                    # include_club=True (see the include_club block above) -- for
                    # movements with no tracked club (pitching, hitting, gait, etc.)
                    # they simply don't exist, so skip this entry entirely rather
                    # than asking _filter_lowpass to filter a marker that was never
                    # populated.
                    continue
                axis_marker_names.add(prox_name)
                axis_marker_names.add(dist_name)
            axis_markers_filtered = {
                name: self._filter_lowpass(markers[name], self.fc_pos) for name in axis_marker_names
            }

        # --- Compute total body COM ---
        weighted_COMs = np.zeros_like(next(iter(segment_COMs_filtered.values())))
        total_mass_used = 0.0

        for seg_name, com in segment_COMs_filtered.items():
            mass = segment_masses.get(seg_name, 0.0)
            weighted_COMs += com * mass
            total_mass_used += mass

        if total_mass_used == 0:
            raise ValueError("Total segment mass is zero, cannot compute body COM.")

        body_COM = weighted_COMs / total_mass_used

        # --- Inverse dynamics ---
        # body_COM is passed in so the left/right weight-transfer split (weight_ratio)
        # can be driven by the true whole-body mass-weighted COM rather than pelvis
        # position alone -- pelvis position misses trunk lean / arm reach / carried-load
        # shifts that move the real COM without much pelvis translation (e.g. reaching
        # to place something overhead to one side while the feet stay planted).
        inv_dyn = self.estimate_inverse_dynamics_grf(
            segment_COMs_filtered, segment_masses, body_COM=body_COM,
            axis_markers_filtered=axis_markers_filtered,
        )
        GRF_left, GRF_right = np.array(inv_dyn["GRF_left"]), np.array(inv_dyn["GRF_right"])

        M_left, M_right = np.array(inv_dyn["M_left"]), np.array(inv_dyn["M_right"])

        if self.swap_lr:
            # See swap_lr docstring: corrects a suspected left/right label
            # mismatch (predicted per-side traces match the ground-truth SHAPE
            # of the opposite foot, frame for frame) at the output stage.
            GRF_left, GRF_right = GRF_right, GRF_left
            M_left, M_right = M_right, M_left

        # --- Final reporting-level smoothing of GRFs ---
        GRF_left = self._filter_lowpass(GRF_left, self.fc_grf)
        GRF_right = self._filter_lowpass(GRF_right, self.fc_grf)


        # --- Optional: re-apply the marker-height contact confidence after the low-pass ---
        if self.kinematic_contact_gate and not self.continuous_contact:
            conf_a = self._contact_confidence(segment_COMs_filtered["foot_l"])
            conf_b = self._contact_confidence(segment_COMs_filtered["foot_r"])
            # swap_lr relabelled the forces but not the markers: with swap_lr the output-left
            # force belongs to the 'r' marker group (same pairing the optimizer used).
            g_left, g_right = (conf_b, conf_a) if self.swap_lr else (conf_a, conf_b)
            GRF_left, GRF_right = self._apply_kinematic_gate(GRF_left, GRF_right, g_left, g_right)

        # --- Apply contact gating (or pass through for continuous_contact sports) ---
        Fx_l, Fy_l, Fz_l = self._apply_contact(GRF_left)
        Fx_r, Fy_r, Fz_r = self._apply_contact(GRF_right)

        # --- Recompute totals ---
        Fx_t, Fy_t, Fz_t = Fx_l + Fx_r, Fy_l + Fy_r, Fz_l + Fz_r

        # --- COP ---

        # passed back in here as ref_point to recover an absolute-frame COP.
        COP_l, contact_l = self._compute_cop_dynamic(GRF_left, M_left, segment_COMs_filtered["foot_l"])
        COP_r, contact_r = self._compute_cop_dynamic(GRF_right, M_right, segment_COMs_filtered["foot_r"])

        Fz_total = Fz_l + Fz_r
        COP_total = np.full_like(COP_l, np.nan)
        mask = Fz_total > 0
        COP_total[mask] = (
                                  np.nan_to_num(COP_l[mask]) * Fz_l[mask, None]
                                  + np.nan_to_num(COP_r[mask]) * Fz_r[mask, None]
                          ) / Fz_total[mask, None]

        # (Earlier ad hoc per-channel sign-flip experiments removed -- see the
        # swap_lr constructor option instead, which addresses the suspected
        # left/right mislabeling at its source rather than flipping signs.)
        positions = self.extract_positions(json3d)
        events = self._resolve_events(positions, markers, len(Fz_t))
        p1 = events["P1"]

        forces = self._save_forces(Fx_t, Fx_l, Fx_r, Fy_t, Fy_l, Fy_r, Fz_t, Fz_l, Fz_r, p1)
        time = np.arange(len(Fz_t)) / self.frame_rate

        
        force_vectors_3d = self._compute_force_vectors_3d(
            ankle_l=axis_markers_filtered["ankle_l"],
            ankle_r=axis_markers_filtered["ankle_r"],

            Fx_l=Fx_l,
            Fy_l=Fy_l,
            Fz_l=Fz_l,

            Fx_r=Fx_r,
            Fy_r=Fy_r,
            Fz_r=Fz_r,

            Fx_t=Fx_t,
            Fy_t=Fy_t,
            Fz_t=Fz_t,
        )



        return {
            "marker_names": marker_names,
            "n_frames": int(len(Fz_t)),
            "markers": markers,
            "segment_COMs_filtered": segment_COMs_filtered,
            "body_COM": body_COM,
            "axis_markers_filtered": axis_markers_filtered,
            "inv_dyn": inv_dyn,
            "positions": positions,
            "events": events,
            "handedness": self._extract_handedness(json3d),
            "Fx_l": Fx_l, "Fy_l": Fy_l, "Fz_l": Fz_l,
            "Fx_r": Fx_r, "Fy_r": Fy_r, "Fz_r": Fz_r,
            "Fx_t": Fx_t, "Fy_t": Fy_t, "Fz_t": Fz_t,
            "COP_l": COP_l, "COP_r": COP_r, "COP_total": COP_total,
            "forces": forces,
            "force_vectors_3d": force_vectors_3d,
            "time": time,
        }

    # -------------------- Marker cleanup (pre-filter) --------------------
    def _despike_marker(self, pos: np.ndarray, name: str = None) -> np.ndarray:
        """Hampel-filter-style despike + NaN/Inf gap-fill for one raw marker's
        (n_frames, 3) trajectory, run BEFORE the single fc_pos Butterworth pass
        (see module docstring / __init__ docstring for despike_markers). This is
        a targeted outlier-replacement step, not a second smoothing pass -- it
        only touches individual samples that deviate sharply from their own
        local median, so real fast athletic motion (which tracks its own local
        trend even when moving quickly) passes through untouched. Unlike a
        low-pass filter, it doesn't attenuate genuine high-velocity content, so
        it doesn't cascade with or fight fc_pos's job of frequency-domain
        band-limiting -- it just removes glitches fc_pos shouldn't have to
        absorb in the first place (e.g. a single-frame tracking teleport).

        Two passes, per axis:
          1. NaN/Inf gap-fill: linearly interpolate any non-finite samples from
             their nearest finite neighbors. Necessary because a single missing
             sample would otherwise corrupt sosfiltfilt's ENTIRE output for that
             marker, not just the missing frame(s).
          2. Hampel despike: compare each sample to the median of a small
             centered window (self.despike_window frames) computed via a fast
             vectorized median filter; flag it as an outlier only if it deviates
             from that local median by more than self.despike_n_sigma robust
             standard deviations (1.4826 * local median absolute deviation).
             Flagged samples are replaced by the local median.

        name : optional marker name, used only for logging -- if any sample on
            any axis actually gets corrected, this logs which marker/axis/how
            many frames, so a trial with clean input data (nothing corrected)
            can be told apart from one where despiking silently fixed something.
            If nothing is corrected anywhere, nothing is logged -- this is meant
            to be quiet on clean data, not noisy by default.
        """
        pos = np.asarray(pos, dtype=float).copy()
        n = pos.shape[0]
        if n < 3:
            return pos

        window = min(self.despike_window, n if n % 2 == 1 else n - 1)
        window = max(window, 3)
        axis_labels = ("x", "y", "z")

        for axis in range(pos.shape[1]):
            col = pos[:, axis]

            # --- 1. NaN/Inf gap-fill ---
            finite = np.isfinite(col)
            n_nan = int((~finite).sum())
            if n_nan:
                if finite.sum() < 2:
                    logging.warning(
                        "[marker-clean] %s axis %s has fewer than 2 finite samples; "
                        "cannot interpolate, leaving non-finite values in place.",
                        name or "<marker>", axis_labels[axis] if axis < 3 else axis,
                    )
                else:
                    idx = np.arange(n)
                    col = col.copy()
                    col[~finite] = np.interp(idx[~finite], idx[finite], col[finite])
                    logging.info(
                        "[marker-clean] %s axis %s: interpolated %d non-finite frame(s).",
                        name or "<marker>", axis_labels[axis] if axis < 3 else axis, n_nan,
                    )

            # --- 2. Hampel despike (vectorized via scipy.ndimage.median_filter) ---
            med = median_filter(col, size=window, mode="nearest")
            abs_dev = np.abs(col - med)
            mad = median_filter(abs_dev, size=window, mode="nearest")
            robust_std = 1.4826 * mad
            outliers = (robust_std > 0) & (abs_dev > self.despike_n_sigma * robust_std)
            if outliers.any():
                worst = int(np.argmax(abs_dev * outliers))
                logging.info(
                    "[marker-clean] %s axis %s: despiked %d frame(s) (worst: %.4f m "
                    "at frame %d, replaced with local median).",
                    name or "<marker>", axis_labels[axis] if axis < 3 else axis,
                    int(outliers.sum()), float(abs_dev[worst]), worst,
                )
                col = np.where(outliers, med, col)

            pos[:, axis] = col

        return pos

    # -------------------- Filtering helpers --------------------
    def _pad_linear(self, arr: np.ndarray, pad: int) -> np.ndarray:
        """Extend arr by `pad` samples at each end via linear extrapolation of the
        local trend, using a least-squares fit over a small window at each edge
        (not just the first/last two samples, which would be noise-sensitive).

        This is deliberately NOT flat/constant padding: a golf recording's start
        (address) is genuinely near-static, but its end (finish/follow-through) is
        usually still in fast motion when the recording cuts off. Flat padding at
        a moving boundary fabricates a "motion suddenly stops" discontinuity that,
        once differentiated twice for acceleration, produces a large spurious
        deceleration spike right at that edge. Fitting and extending the local
        slope at each edge independently gets this right in both cases: it
        collapses to ~flat padding where the true slope is ~0 (address), and
        continues the real trend where it isn't (finish).
        """
        n = arr.shape[0]
        fit_n = max(3, min(10, n // 3))
        #fit_n = min(max(5, pad // 6), min(20, n // 2))
        t_fit = np.arange(fit_n)

        # Start edge: fit the first fit_n samples, extrapolate backward.
        start_coefs = np.polyfit(t_fit, arr[:fit_n], 1)  # shape (2, *trailing_dims)
        t_start_pad = np.arange(-pad, 0)
        start_pad = np.polyval(start_coefs, t_start_pad[:, None] if arr.ndim > 1 else t_start_pad)

        # End edge: fit the last fit_n samples, extrapolate forward.
        t_fit_end = np.arange(n - fit_n, n)
        end_coefs = np.polyfit(t_fit_end, arr[-fit_n:], 1)
        t_end_pad = np.arange(n, n + pad)
        end_pad = np.polyval(end_coefs, t_end_pad[:, None] if arr.ndim > 1 else t_end_pad)

        return np.concatenate([start_pad, arr, end_pad], axis=0)

    def _filter_lowpass(self, arr: np.ndarray, fc: float, order: int = None) -> np.ndarray:
        """Zero-phase Butterworth low-pass filter along axis 0.

        Manually pads with linear-extrapolation extension (see _pad_linear) before
        filtering and crops back to the original length after, rather than relying
        solely on scipy's internal filtfilt padding. Rationale: this pipeline calls
        _filter_lowpass three separate times in cascade (fc_pos on position, fc_dyn
        on total force/moment, fc_grf on final GRF), each with its own default
        boundary padding/settling. Scipy's default 'odd' reflection padding can
        leave a residual decaying transient right at the true edges of the data --
        visible as predicted traces starting noticeably off from measured for the
        first several dozen frames before settling onto the true trace, even at a
        genuinely near-static address. A generous buffer -- several periods of the
        cutoff frequency -- gives filtfilt's own internal transient room to settle
        *before* reaching the real data.
        """
        if fc is None or fc <= 0:
            return arr.copy()
        if order is None:
            order = self.filter_order
        nyq = self.frame_rate / 2.0
        fc_eff = min(fc, nyq * 0.98)
        sos = butter(order, fc_eff / nyq, btype="low", output="sos")

        n = arr.shape[0]
        # Buffer sized to several periods of the cutoff frequency, so filtfilt's
        # own boundary transient (which settles over roughly a few periods of
        # the lowest frequency it's passing) has room to die out before reaching
        # the real data. Capped so we never pad past a small multiple of the
        # actual sequence length.
        pad = int(np.ceil(4 * self.frame_rate / fc_eff))
        pad = max(pad, 10)
        pad = min(pad, 3 * n)

        try:
            if pad > 0 and n > 1:
                padded = self._pad_linear(arr, pad)
                filtered = sosfiltfilt(sos, padded, axis=0)
                return filtered[pad:pad + n]
            return sosfiltfilt(sos, arr, axis=0)
        except ValueError:
            # Sequence too short for filtfilt's default edge padding; better to skip
            # filtering than to raise on short trials.
            logging.warning(
                "Sequence of length %d too short to low-pass filter at %.1f Hz; using raw values.",
                arr.shape[0], fc,
            )
            return arr.copy()

    def _differentiate(self, arr: np.ndarray) -> np.ndarray:
        """Plain 2nd-order-accurate central difference, dt apart, along axis 0.
        No smoothing happens here -- the only position-domain smoothing in the whole
        pipeline is the single _filter_lowpass(fc_pos) call applied once upstream,
        per the module docstring (Script-A-style single-pass filtering)."""
        return np.gradient(arr, self.dt, axis=0, edge_order=2)

    #####################################################################
    # 3D FORCE VECTOR GENERATION FOR FRONTEND
    #####################################################################

    def generate_force_vectors_3d(
        self,
        forces,
        cop_data
    ):
        """
        Generate frontend-ready 3D force vectors.

        Inputs
        ------
        forces:
            dict containing:
                Fx_l, Fy_l, Fz_l
                Fx_r, Fy_r, Fz_r
                Fx_t, Fy_t, Fz_t

        cop_data:
            dict containing:
                x_start_COP_l
                y_start_COP_l
                z_start_COP_l
                x_end_COP_l
                ...

        Returns
        -------
        dict:
            force_vectors_3d
        """

        def build_vectors(
            side,
            Fx,
            Fy,
            Fz
        ):

            vectors = []

            xs = cop_data[f"x_start_COP_{side}"]
            ys = cop_data[f"y_start_COP_{side}"]
            zs = cop_data[f"z_start_COP_{side}"]

            xe = cop_data[f"x_end_COP_{side}"]
            ye = cop_data[f"y_end_COP_{side}"]
            ze = cop_data[f"z_end_COP_{side}"]


            n = len(Fx)


            for i in range(n):

                vectors.append(
                    {
                        "start":
                        [
                            float(xs[i]),
                            float(ys[i]),
                            float(zs[i])
                        ],

                        "end":
                        [
                            float(xe[i]),
                            float(ye[i]),
                            float(ze[i])
                        ],

                        "force":
                        [
                            float(Fx[i]),
                            float(Fy[i]),
                            float(Fz[i])
                        ]
                    }
                )

            return vectors



        result = {

            "left":
            build_vectors(
                "l",
                forces["Fx_l"],
                forces["Fy_l"],
                forces["Fz_l"]
            ),


            "right":
            build_vectors(
                "r",
                forces["Fx_r"],
                forces["Fy_r"],
                forces["Fz_r"]
            ),


            "combined":
            build_vectors(
                "combined",
                forces["Fx_t"],
                forces["Fy_t"],
                forces["Fz_t"]
            )

        }


        return result

    # -------------------- Rotational inertia --------------------
    def _compute_segment_rotational_moment(self, p_prox: np.ndarray, p_dist: np.ndarray,
                                             mass: float, radius_of_gyration_ratio: float) -> np.ndarray:
        """Approximate the free moment (couple) produced by a segment's own angular
        acceleration: M_rot = I_perp * alpha_perp.

        p_prox, p_dist : (n_frames, 3) already fc_pos-filtered proximal/distal marker
            positions defining the segment's long axis (see SEGMENT_AXIS_MARKERS).
        mass : segment mass in kg (use half the segment mass for trunk/head, to match
            how their translational force is already split 50/50 between L/R sides
            downstream -- see call site).
        radius_of_gyration_ratio : SEGMENT_RADIUS_OF_GYRATION_RATIO entry for this
            segment (dimensionless, fraction of segment length).

        What this captures: rotation PERPENDICULAR to the segment's long axis
        (flexion/extension, ab/adduction-type swinging), estimated from how the
        proximal->distal unit vector u(t) changes direction over time:
            du/dt = omega x u   (only the omega component perpendicular to u affects u)
            => omega_perp = u x du/dt   (since u . u = 1)
        differentiated again for alpha_perp = d(omega_perp)/dt.

        What this does NOT capture: axial spin about the segment's own long axis
        (e.g. forearm pronation/supination, femoral internal/external rotation, a
        club/bat twisting in the hands) -- that is not observable from a 2-marker
        axis and would require a third, off-axis marker per segment. This method
        assumes the segment is axisymmetric about its long axis (equal transverse
        principal moments of inertia), which is standard for limb-like segments and
        also makes the gyroscopic omega x (I*omega) term vanish analytically for a
        purely-perpendicular omega -- so no axial-spin term is silently dropped from
        an otherwise-complete equation; the simplification is exact under that
        assumption, not an approximation layered on top of a fuller one.
        """
        axis_vec = p_dist - p_prox
        norms = np.linalg.norm(axis_vec, axis=1)
        length = float(np.median(norms))
        if length < 1e-6:
            return np.zeros_like(axis_vec)

        safe_norms = np.where(norms < 1e-9, 1e-9, norms)
        u = axis_vec / safe_norms[:, None]

        du_dt = self._differentiate(u)
        omega_perp = np.cross(u, du_dt)  # rad/s, perpendicular component only
        alpha_perp = self._differentiate(omega_perp)  # rad/s^2

        I_perp = mass * (radius_of_gyration_ratio * length) ** 2  # kg*m^2
        return I_perp * alpha_perp

    # -------------------- Contact detection --------------------
    def _estimate_kinematic_contact(self, foot_pos: np.ndarray) -> np.ndarray:
        """Rough foot-ground contact estimate from marker height ALONE, computed
        entirely independently of any force/moment computation -- see
        contact_height_threshold docstring in __init__ for the full rationale.

        Ground level for this foot is self-calibrated as the lowest height
        (vertical/column-2 position) it reaches anywhere in the trial -- this
        assumes the foot does touch the ground at some point during the
        recording, true for essentially any grounded athletic movement, and
        handles each foot having a different absolute ground height fine (e.g.
        a pitching mound's slope, where the stride foot lands lower than the
        back/pivot foot started), since each foot gets its own independently
        calibrated reference rather than assuming a single shared ground plane.

        A frame is "in contact" if this foot's height above its own ground
        reference is below self.contact_height_threshold, then debounced with
        the same self.min_contact_frames used by the Fz-hysteresis pass, to
        avoid single-frame marker-noise flicker right at the contact boundary.

        This is deliberately independent of Fz/GRF entirely -- the whole point
        is to give optimize_bilateral_grf a contact signal that ISN'T derived
        from a force computation that had no contact awareness in the first
        place (see contact_height_threshold docstring).
        """
        z = foot_pos[:, 2]
        ground_z = np.min(z)
        height_above_ground = z - ground_z
        contact = height_above_ground < self.contact_height_threshold
        return self._debounce(contact, self.min_contact_frames)

    def _contact_confidence(self, foot_pos: np.ndarray) -> np.ndarray:
        """Continuous per-frame contact confidence in [0, 1], from marker height
        ALONE (same self-calibrated per-foot ground reference as
        _estimate_kinematic_contact), ramped linearly rather than switched at a
        single threshold -- see contact_confidence_band docstring in __init__.

        confidence = 1.0  for height <= contact_height_threshold (fully grounded)
        confidence = 0.0  for height >= contact_height_threshold + contact_confidence_band
        linear ramp in between.

        This (not the boolean _estimate_kinematic_contact) is what
        optimize_bilateral_grf actually uses: passing a boolean mask there means
        that foot's allowed force bounds snap instantly between "fully free" and
        "pinned to zero" at the exact frame the mask flips, which produces a real
        step discontinuity in the reported GRF right at that frame once you
        differentiate/observe it on a plot -- a physically implausible sudden
        jump rather than the gradual loading/unloading ramp a real foot strike or
        toe-off actually has. Feeding in this continuous confidence instead lets
        the bounds shrink/expand smoothly over the ramp band, converging to
        exactly the same fully-free and fully-pinned endpoints as the boolean
        version at the extremes, but without the discontinuity in between.
        """
        z = foot_pos[:, 2]
        ground_z = np.min(z)
        height_above_ground = z - ground_z
        band = max(self.contact_confidence_band, 1e-9)
        confidence = 1.0 - (height_above_ground - self.contact_height_threshold) / band
        return np.clip(confidence, 0.0, 1.0)

    def _hysteresis_contact(self, Fz: np.ndarray) -> np.ndarray:
        """Schmitt-trigger contact detection: separate on/off thresholds plus a
        debounce pass, so brief oscillations of Fz around the threshold don't cause the
        contact mask (and therefore the reported force) to chatter."""
        on_thr = self.contact_threshold
        off_thr = max(self.contact_threshold - self.contact_hysteresis, 0.0)
        n = len(Fz)
        contact = np.zeros(n, dtype=bool)
        state = Fz[0] > on_thr
        for i in range(n):
            if state:
                if Fz[i] < off_thr:
                    state = False
            else:
                if Fz[i] > on_thr:
                    state = True
            contact[i] = state
        return self._debounce(contact, self.min_contact_frames)

    @staticmethod
    def _debounce(mask: np.ndarray, min_len: int) -> np.ndarray:
        """Merge any True/False run shorter than min_len frames into the state that
        preceded it, removing single-frame (or few-frame) flicker."""
        if min_len <= 1:
            return mask
        out = mask.copy()
        n = len(mask)
        i = 0
        while i < n:
            j = i
            while j < n and mask[j] == mask[i]:
                j += 1
            run_len = j - i
            if run_len < min_len and i > 0:
                out[i:j] = out[i - 1]
            i = j
        return out

    @staticmethod
    def _apply_kinematic_gate(GRF_l, GRF_r, g_l, g_r):
        """Scale each foot's force vector by its contact confidence g in [0, 1] and give
        what was removed from one foot to the other (as long as that other foot is itself
        in contact), so GRF_l + GRF_r keeps the whole-body total. If neither foot is in
        contact there is nothing to transfer to and the forces are simply scaled."""
        g_l = np.clip(np.asarray(g_l, dtype=float), 0.0, 1.0)[:, None]
        g_r = np.clip(np.asarray(g_r, dtype=float), 0.0, 1.0)[:, None]
        removed_l = GRF_l * (1.0 - g_l)
        removed_r = GRF_r * (1.0 - g_r)
        out_l = GRF_l * g_l + removed_r * (g_l > 0)
        out_r = GRF_r * g_r + removed_l * (g_r > 0)
        return out_l, out_r

    def _apply_contact(self, GRF: np.ndarray):
        Fx, Fy, Fz = GRF[:, 0].copy(), GRF[:, 1].copy(), GRF[:, 2].copy()
        # Fy = Fy * -1  # To match AMTI force plate convention #Changed
        # Match Script A's convention of never reporting a negative vertical GRF.
        Fz[Fz < 0] = 0.0
        if self.continuous_contact:
            # No flight phase for this movement type (e.g. golf swing, batting stance):
            # skip gating so we don't introduce artificial dropouts.
            return Fx, Fy, Fz
        contact = self._hysteresis_contact(Fz)
        Fx[~contact] = 0.0
        Fy[~contact] = 0.0
        Fz[~contact] = 0.0
        return Fx, Fy, Fz

    def _compute_cop_dynamic(self, F: np.ndarray, M: np.ndarray, ref_point: np.ndarray):
        """ref_point is the point M was taken about (here, the foot COM used as the
        moment-arm origin in estimate_inverse_dynamics_grf's `COM - foot_COM` terms).
        -My/Fz and Mx/Fz only give the COP's OFFSET from ref_point, not an absolute
        marker-space position -- ref_point must be added back in, or COP_l/COP_r end
        up expressed relative to two different, independently-moving origins (each
        foot's own COM), which also silently breaks the Fz-weighted COP_l/COP_r blend
        used to compute COP_total downstream, since that blend assumes a shared frame."""
        COP = np.full_like(F, np.nan)
        Fz = F[:, 2]
        if self.continuous_contact:
            contact = np.ones(len(Fz), dtype=bool)
        else:
            contact = self._hysteresis_contact(Fz)
        COP[contact, 0] = -M[contact, 1] / Fz[contact] + ref_point[contact, 0]
        COP[contact, 1] = M[contact, 0] / Fz[contact] + ref_point[contact, 1]
        COP[:, 2] = 0
        return COP, contact

    #positions = self.extract_positions(json3d)
    #p1 = positions["p1"]

    def _save_forces(self, Fx_t, Fx_l, Fx_r, Fy_t, Fy_l, Fy_r, Fz_t, Fz_l, Fz_r, p1):
        # Force components expressed as % of P1 total vertical force.
        p1_fz = Fz_t[p1]

        return {
            "Fx_t": np.around(Fx_t, 2).tolist(), "Fx_l": np.around(Fx_l, 2).tolist(), "Fx_r": np.around(Fx_r, 2).tolist(),
            "Fy_t": np.around(Fy_t, 2).tolist(), "Fy_l": np.around(Fy_l, 2).tolist(), "Fy_r": np.around(Fy_r, 2).tolist(),
            "Fz_t": np.around(Fz_t, 2).tolist(), "Fz_l": np.around(Fz_l, 2).tolist(), "Fz_r": np.around(Fz_r, 2).tolist(),

            "Fx_t_wt": np.around(100.0 * Fx_t / p1_fz, 2).tolist(),
            "Fx_l_wt": np.around(100.0 * Fx_l / p1_fz, 2).tolist(),
            "Fx_r_wt": np.around(100.0 * Fx_r / p1_fz, 2).tolist(),
            "Fy_t_wt": np.around(100.0 * Fy_t / p1_fz, 2).tolist(),
            "Fy_l_wt": np.around(100.0 * Fy_l / p1_fz, 2).tolist(),
            "Fy_r_wt": np.around(100.0 * Fy_r / p1_fz, 2).tolist(),
            "Fz_t_wt": np.around(100.0 * Fz_t / p1_fz, 2).tolist(),
            "Fz_l_wt": np.around(100.0 * Fz_l / p1_fz, 2).tolist(),
            "Fz_r_wt": np.around(100.0 * Fz_r / p1_fz, 2).tolist(),
        }

    def _compute_force_vectors_3d(
            self,
            ankle_l,
            ankle_r,
            Fx_l,
            Fy_l,
            Fz_l,
            Fx_r,
            Fy_r,
            Fz_r,
            Fx_t,
            Fy_t,
            Fz_t,
            scale=0.001
    ):
        """
        Generate 3D force arrows for visualization.

        All arrows originate from mid-ankles COM point.

        Coordinate convention:
            start = application point
            end   = start + scaled force vector
        """

        # Ensure arrays
        ankle_l = np.asarray(ankle_l)
        ankle_r = np.asarray(ankle_r)

        # Common origin
        mid_feet = (ankle_l + ankle_r) / 2.0


        def make_arrow(start_point, force):

            force = np.asarray(force)

            start = start_point
            end = start + force * scale

            return start, end


        # Left GRF
        start_l, end_l = make_arrow(
            ankle_l,
            np.column_stack([Fx_l, Fy_l, Fz_l])
        )


        start_r, end_r = make_arrow(
            ankle_r,
            np.column_stack([Fx_r, Fy_r, Fz_r])
        )


        start_t, end_t = make_arrow(
            mid_feet,
            np.column_stack([Fx_t, Fy_t, Fz_t])
        )


        return {

            "x_start_COP_l": start_l[:, 0].tolist(),
            "x_end_COP_l": end_l[:, 0].tolist(),

            "y_start_COP_l": start_l[:, 1].tolist(),
            "y_end_COP_l": end_l[:, 1].tolist(),

            "z_start_COP_l": start_l[:, 2].tolist(),
            "z_end_COP_l": end_l[:, 2].tolist(),


            "x_start_COP_r": start_r[:, 0].tolist(),
            "x_end_COP_r": end_r[:, 0].tolist(),

            "y_start_COP_r": start_r[:, 1].tolist(),
            "y_end_COP_r": end_r[:, 1].tolist(),

            "z_start_COP_r": start_r[:, 2].tolist(),
            "z_end_COP_r": end_r[:, 2].tolist(),


            "x_start_COP_combined": start_t[:, 0].tolist(),
            "x_end_COP_combined": end_t[:, 0].tolist(),

            "y_start_COP_combined": start_t[:, 1].tolist(),
            "y_end_COP_combined": end_t[:, 1].tolist(),

            "z_start_COP_combined": start_t[:, 2].tolist(),
            "z_end_COP_combined": end_t[:, 2].tolist(),
        }

    def _extract_pose_array(self, json3d: dict) -> np.ndarray:
        """Locate the marker pose array, handling both input JSON formats seen
        so far:
          - Old/flat format:    json3d["data"]["pose"]
          - New nested format:  json3d["data_3d_bio"]["data"]["pose"]
        Returns the squeezed array (same np.squeeze behavior as before)."""
        if "data_3d_bio" in json3d:
            try:
                pose = json3d["data_3d_bio"]["data"]["pose"]
            except (KeyError, TypeError):
                raise KeyError(
                    "Input JSON has 'data_3d_bio' but is missing "
                    "['data_3d_bio']['data']['pose']."
                )
        elif "data" in json3d and "pose" in json3d.get("data", {}):
            pose = json3d["data"]["pose"]
        else:
            raise KeyError(
                "Could not locate marker pose data. Expected either "
                "['data']['pose'] (old format) or "
                "['data_3d_bio']['data']['pose'] (new format)."
            )
        return np.squeeze(np.asarray(pose))

    def _extract_club_array(self, json3d: dict, n_frames_expected: int) -> tuple:
        """Extract per-frame club grip ("start"), shaft ("mid"), and head ("end")
        coordinates from the input JSON's dedicated club data block -- NOT the
        pose array's "club_main"/"club_head" markers, which are mislabeled
        (they're actually clavicle_r/clavicle_l) and don't represent the club
        at all.

        Expected structure (frame index as string key):
            json3d["data"]["club"]["0"]["start"/"mid"/"end"/"shaft"] -> [x, y, z]
        (or the nested json3d["data_3d_bio"]["data"]["club"] location, mirroring
        _extract_pose_array's old/new format handling.) "start"/"mid"/"end" are
        used for the two-point-mass club model (shaft mass at "mid", head mass at
        "end"); "shaft" is unused.

        Returns (club_start, club_end, club_mid), each shape (n_frames, 3),
        frame-ordered to match the pose array (sorted by integer frame index).
        """
        if "data_3d_bio" in json3d:
            club = json3d["data_3d_bio"].get("data", {}).get("club")
        else:
            club = json3d.get("data", {}).get("club")
        if not club:
            raise KeyError(
                "include_club=True (the golf preset, swingType 'full swing') but no club "
                "data found. Expected ['data']['club'] (or ['data_3d_bio']['data']['club']), "
                "a dict keyed by frame index with 'start'/'mid'/'end' coordinates per frame. "
                "Outputs are not produced without it, so results stay consistent; pass "
                "include_club=False explicitly only if a body-only run is really wanted."
            )

        frame_keys = sorted(club.keys(), key=lambda k: int(k))
        # Every club point must be a 3D [x, y, z] coordinate. Some files carry the club in
        # 2D image (pixel) coordinates instead; fail with a clear message, not an IndexError.
        for k in frame_keys:
            for part in ("start", "mid", "end"):
                pt = club[k].get(part) if isinstance(club[k], dict) else None
                if not (isinstance(pt, (list, tuple)) and len(pt) == 3):
                    got = len(pt) if isinstance(pt, (list, tuple)) else type(pt).__name__
                    raise ValueError(
                        f"Club data frame {k!r} '{part}' has {got} components (value: {pt!r}); "
                        "3D [x, y, z] coordinates are required. A 2-component value usually "
                        "means the club block holds 2D image (pixel) coordinates, not 3D data. "
                        "No outputs are produced for this file."
                    )
        club_start = np.array([club[k]["start"] for k in frame_keys], dtype=float)
        club_mid = np.array([club[k]["mid"] for k in frame_keys], dtype=float)
        club_end = np.array([club[k]["end"] for k in frame_keys], dtype=float)

        if len(frame_keys) != n_frames_expected:
            raise ValueError(
                f"Club data has {len(frame_keys)} frames but pose data has "
                f"{n_frames_expected} frames -- these must align 1:1 per frame. "
                "Outputs are not produced for mismatched inputs."
            )

        if self.swap_xy:
            # Match the same axis convention correction applied to M_data.
            club_start = club_start[:, [1, 0, 2]]
            club_mid = club_mid[:, [1, 0, 2]]
            club_end = club_end[:, [1, 0, 2]]

        return club_start, club_end, club_mid

    @staticmethod
    def extract_meta(json3d: dict) -> dict:
        """Pull body_mass (kg), frame_rate (Hz), handedness, and movement type
        out of the new nested format's configuration block, if present.
        Returns {} (not an error) for the old flat format, which didn't carry
        this metadata -- callers should fall back to their own defaults in
        that case. Use this to build a ForceVectors preset with the athlete's
        REAL weight/frame_rate instead of a fixed assumed value, e.g.:
            meta = ForceVectors.extract_meta(json3d)
            fv = ForceVectors.for_pitching(
                body_mass=meta.get("body_mass", 92),
                frame_rate=meta.get("frame_rate", 240),
            )
        """
        meta = {}
        if "data_3d_bio" not in json3d:
            return meta
        config = json3d["data_3d_bio"].get("configuration", {})
        user = config.get("user", {})
        video = config.get("video", {})
        if "weight" in user:
            meta["body_mass"] = user["weight"]
        if "frame_rate" in video:
            meta["frame_rate"] = video["frame_rate"]
        # handedness/movement live under a movement-specific sub-dict (e.g.
        # "swing" for hitting); key name for pitching is unconfirmed, so search.
        for val in config.values():
            if isinstance(val, dict) and "handedness" in val:
                meta["handedness"] = val["handedness"]
                if "swingType" in val:
                    meta["movement"] = val["swingType"]
                break
        return meta

    # -------------------- Phase-position debug plotting --------------------
    @staticmethod
    def extract_positions(json3d: dict):
        """Pull the p1-p8 phase-event frame indices out of
        either the old flat format (json3d['configuration']['positions']) or
        the new nested format (json3d['data_3d_bio']['configuration'][*]
        ['critical_events'], under a movement-specific key like "swing").
        Returns None (and logs a warning, not an error) if neither is found,
        so callers can skip the debug-plot annotations gracefully."""
        try:
            return json3d["configuration"]["swing"]["critical_events"]
        except (KeyError, TypeError):
            pass

        config = json3d.get("data_3d_bio", {}).get("configuration", {})
        if isinstance(config, dict):
            for val in config.values():
                if isinstance(val, dict) and "critical_events" in val:
                    return val["critical_events"]

        logging.warning(
            "No phase-position data found (checked ['configuration']"
            "['swing']['critical_events'] and ['data_3d_bio']['configuration'][*]"
            "['critical_events'])."
        )
        return None

    # -------------------- Optimized bilateral GRF split --------------------
    def optimize_bilateral_grf(
        self,
        total_force,
        total_moment,
        foot_l,
        foot_r,
        body_COM,
        cop_target=None,
        contact_conf_l=None,
        contact_conf_r=None,
    ):
        """
        Optimize left/right GRF partition.

        Replaces heuristic weight_ratio splitting.

        Constraints:
            FL + FR = total_force

        Uses:
            - total moment equilibrium
            - COP location
            - temporal smoothness
            - contact_conf_l/contact_conf_r (optional, continuous [0,1] per-frame
              contact confidence from _contact_confidence): smoothly shrinks each
              foot's allowed force-bound box toward zero (or, for the other foot,
              toward the full total_force) as that foot's confidence drops, so a
              foot lifting off gets its GRF ramped down over the confidence
              transition band instead of snapped to zero at one frame. Pass None
              (default) for movements where both feet are assumed grounded
              throughout (continuous_contact=True) -- see generate_force_vectors.

        Returns:
            GRF_left
            GRF_right
        """

        total_force = np.asarray(total_force, dtype=float)
        total_moment = np.asarray(total_moment, dtype=float)

        n = total_force.shape[0]

        GRF_left = np.zeros_like(total_force)
        GRF_right = np.zeros_like(total_force)

        previous_left = total_force[0].copy() * 0.5

        # ==========================================
        # TRACK WORST MOMENT ERROR DURING PITCH
        # ==========================================
        pitch_start = 45
        pitch_end = 536

        worst_pitch_frame = None
        worst_pitch_error = -np.inf
        worst_pitch_data = {}

        for i in range(n):

            previous_left = np.asarray(previous_left).reshape(3,)

            F = total_force[i]
            # ----------------------------------------------------------
            # NEW: no physical ground reaction if total vertical force
            # is zero or negative.
            # ----------------------------------------------------------
            if F[2] <= 0:
                GRF_left[i] = F * 0.5
                GRF_right[i] = F * 0.5
                previous_left = GRF_left[i].copy()
                continue

            # (Contact gating now happens via smooth bound-shrinking below, driven
            # by contact_conf_l/contact_conf_r, instead of a hard override branch
            # here -- see optimize_bilateral_grf docstring / _contact_confidence.)

            # initial guess
            x0 = previous_left.copy()

            def objective(x):

                nonlocal worst_pitch_error
                nonlocal worst_pitch_frame
                nonlocal worst_pitch_data

                FL = x
                FR = F - FL
                


                # -----------------------------------
                # Fx should follow vertical load share
                # -----------------------------------

                fz_sum = FL[2] + FR[2] + 1e-8

                left_fz_ratio = FL[2] / fz_sum

                expected_left_fx = F[0] * left_fz_ratio

                fx_balance_penalty = (
                    FL[0] - expected_left_fx
                )**2

                # ----------------------------------
                # Shear force should require vertical load
                # ----------------------------------

                mu = self.mu_friction

                shear_left = max(0, abs(FL[0]) - mu * FL[2])
                shear_right = max(0, abs(FR[0]) - mu * FR[2])

                friction_penalty = (
                    shear_left**2 +
                    shear_right**2
                )

                # ---------------------------------
                # Bilateral shear symmetry regularizer
                # Small tie-breaker only, for numerically ill-conditioned
                # frames (e.g. feet laterally aligned with COM so moment
                # equilibrium alone doesn't fully pin down the Fx split).
                # This must NOT overwrite fx_balance_penalty above -- the
                # load-share penalty is the physically-motivated term and
                # was previously being clobbered by this block, which
                # forced FL[0] ~= FR[0] on every frame regardless of the
                # true (usually asymmetric) bilateral weight-shift split.
                # ---------------------------------

                fx_symmetry_reg = (
                    FL[0] - FR[0]
                )**2
                # ---------------------------------
                # Geometry and predicted moment
                # ---------------------------------
                rL = foot_l[i] - body_COM[i]
                rR = foot_r[i] - body_COM[i]

                M_left = np.cross(rL, FL)
                M_right = np.cross(rR, FR)

                Mx_from_Fx = (
                    np.cross(rL, [FL[0], 0, 0]) +
                    np.cross(rR, [FR[0], 0, 0])
                )

                Mx_from_Fy = (
                    np.cross(rL, [0, FL[1], 0]) +
                    np.cross(rR, [0, FR[1], 0])
                )

                Mx_from_Fz = (
                    np.cross(rL, [0, 0, FL[2]]) +
                    np.cross(rR, [0, 0, FR[2]])
                )

                moment_pred = M_left + M_right

                # ==========================================
                # TRACK WORST MOMENT ERROR DURING PITCH
                # ==========================================
                if pitch_start <= i <= pitch_end:

                    current_error = moment_pred - total_moment[i]
                    error_norm = np.linalg.norm(current_error)

                    if error_norm > worst_pitch_error:
                        worst_pitch_error = error_norm
                        worst_pitch_frame = i

                        worst_pitch_data = {
                            "moment_pred": moment_pred.copy(),
                            "total_moment": total_moment[i].copy(),
                            "moment_error": current_error.copy(),
                            "rL": rL.copy(),
                            "rR": rR.copy(),
                            "M_left": M_left.copy(),
                            "M_right": M_right.copy(),
                        }

                moment_error = np.sum(
                    (moment_pred - total_moment[i]) ** 2
                )

                # -------------------------------
                # 3. Positive vertical forces
                # -------------------------------
                # 1. Temporal smoothness #
                smooth = 0.02 * np.sum((FL - previous_left) ** 2)


                return (
                    0.1*smooth
                    + 100.0*moment_error
                    + 50.0*friction_penalty
                    + 1.0*fx_balance_penalty
                    + 0.01*fx_symmetry_reg
                )

            constraints = [
                {
                    "type": "ineq",
                    "fun": lambda x: F[2] - x[2]
                }
            ]

            fz_total = max(F[2], 0)

            base_lower = np.array([-0.8 * fz_total, -0.8 * fz_total, 0.0])
            base_upper = np.array([0.8 * fz_total, 0.8 * fz_total, fz_total])

            # Smoothly shrink the feasible box toward FL=0 as left-foot contact
            # confidence drops, and toward FL=F (i.e. FR=0) as right-foot contact
            # confidence drops -- see _contact_confidence / optimize_bilateral_grf
            # docstring. At confidence=1 this is a no-op (identical to the old
            # fixed friction bounds); at confidence=0 it collapses lower=upper to
            # a single point, equivalent to the old hard override, but everything
            # in between ramps continuously instead of jumping.
            lower, upper = base_lower, base_upper
            if contact_conf_l is not None:
                squeeze_l = 1.0 - contact_conf_l[i]
                lower = (1 - squeeze_l) * lower + squeeze_l * 0.0
                upper = (1 - squeeze_l) * upper + squeeze_l * 0.0
            if contact_conf_r is not None:
                squeeze_r = 1.0 - contact_conf_r[i]
                lower = (1 - squeeze_r) * lower + squeeze_r * F
                upper = (1 - squeeze_r) * upper + squeeze_r * F

            bounds = list(zip(lower, upper))
            x0 = np.clip(x0, lower, upper)

            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                options={
                    "maxiter": 50,
                    "ftol": 1e-6,
                },
            )

            if result.success:
                FL = result.x
            else:
                FL = x0

            FR = F - FL

            GRF_left[i] = FL
            GRF_right[i] = FR

            err = np.linalg.norm(GRF_left[i] + GRF_right[i] - F)


        

            previous_left = FL.copy()

      

        return GRF_left, GRF_right

    # -------------------- Inverse dynamics --------------------
    def estimate_inverse_dynamics_grf(self, segment_COMs, segment_masses, body_COM, g=9.81,
                                        axis_markers_filtered=None):
        """segment_COMs must already be position-filtered at fc_pos (see
        _filter_lowpass / generate_force_vectors). compute_force below does NOT
        re-filter position, velocity, or acceleration -- it goes straight from the
        (already-filtered) COM to two plain differentiations, so position is
        band-limited at fc_pos exactly once across the whole pipeline, matching the
        module docstring's Script-A-style single-pass filtering. If you call this
        method directly with raw/unfiltered COM data, filter it at fc_pos yourself
        first -- otherwise position noise will pass straight into the
        velocity/acceleration differentiation stage unfiltered.

        body_COM : (n_frames, 3) whole-body mass-weighted center of mass (already
            fc_pos-filtered), used to drive the left/right weight-transfer split --
            see the weight_ratio comment below for why this is body_COM and not
            pelvis position.

        axis_markers_filtered : optional dict of {marker_name: (n_frames, 3) already
            fc_pos-filtered marker position}, covering every marker referenced in
            SEGMENT_AXIS_MARKERS. Required for the rotational-inertia moment term to
            be added (see _compute_segment_rotational_moment); if None (or if
            self.include_rotational_inertia is False), rotational moment is skipped
            and total_moment includes only the translational r x F term, matching the
            pre-rotational-inertia behavior. generate_force_vectors always builds and
            passes this automatically when include_rotational_inertia=True -- you
            only need to supply it yourself if calling this method directly.
        """
        n_frames = len(next(iter(segment_COMs.values())))

        GRF_left = np.zeros((n_frames,3))
        GRF_right = np.zeros((n_frames,3))

        M_left = np.zeros((n_frames,3))
        M_right = np.zeros((n_frames,3))

        total_force = np.zeros((n_frames,3))
        total_moment = np.zeros((n_frames,3))

        body_force_total = np.zeros((n_frames,3))
        body_moment_total = np.zeros((n_frames,3))

        # How much of the lateral (Fy) HAT load should follow the pelvis-based split.
        # 0.0 = equal split, 1.0 = fully dynamic like Fz.
        ALPHA_LATERAL_SHARE = 0.50

        use_rotational = self.include_rotational_inertia and axis_markers_filtered is not None
        if self.include_rotational_inertia and axis_markers_filtered is None:
            logging.warning(
                "include_rotational_inertia=True but axis_markers_filtered was not "
                "provided to estimate_inverse_dynamics_grf; skipping rotational moment "
                "term for this call (translational r x F only)."
            )

        def compute_force(COM, mass, add_g=True):
            COM = np.atleast_2d(COM)
            # Script-A-style: COM is assumed already filtered at fc_pos by the caller
            # (see this method's docstring). Differentiate straight through with two
            # plain np.gradient calls -- no re-filtering of velocity or acceleration.
            # Position is therefore band-limited exactly once for the whole pipeline,
            # instead of being cascaded through 2-3 extra Butterworth passes.
            vel = self._differentiate(COM)
            acc = self._differentiate(vel)
            if add_g:
                acc = acc.copy()
                acc[:, -1] += g
            mass = np.asarray(mass)
            if mass.ndim == 1:
                mass = mass[:, None]
            return mass * acc

        def add_rotational_moment(seg_key, mass_for_inertia):
            """Look up SEGMENT_AXIS_MARKERS/SEGMENT_RADIUS_OF_GYRATION_RATIO for
            seg_key and return its rotational moment, or zeros if unavailable."""
            if not use_rotational or seg_key not in self.SEGMENT_AXIS_MARKERS:
                return None
            prox_name, dist_name = self.SEGMENT_AXIS_MARKERS[seg_key]
            base_name = seg_key.rsplit("_", 1)[0] if seg_key.endswith(("_l", "_r")) else seg_key
            k = self.SEGMENT_RADIUS_OF_GYRATION_RATIO.get(base_name)
            if k is None:
                return None
            p_prox = axis_markers_filtered.get(prox_name)
            p_dist = axis_markers_filtered.get(dist_name)
            if p_prox is None or p_dist is None:
                return None
            return self._compute_segment_rotational_moment(p_prox, p_dist, mass_for_inertia, k)

        for side in ["l", "r"]:
            seg_order = [f"foot_{side}", f"shank_{side}", f"thigh_{side}", f"upperarm_{side}", f"forearm_{side}",
                         f"hand_{side}"]
            HAT_segs = ["trunk", "head"]
            if self.include_club:
                # Club is held with both hands -- shared contribution like HAT, not
                # attributed to one leg. Its own force contribution (mass * accel)
                # is included in the loop below; the L/R split of that force is left
                # to the same moment-equilibrium optimizer that resolves everything
                # else, rather than assumed to be an even split here.
                HAT_segs = HAT_segs + ["club"]
            nFrames = segment_COMs[seg_order[0]].shape[0]
                # reset side-specific accumulators only
            total_force = np.zeros((nFrames, 3))
            total_moment = np.zeros((nFrames, 3))
            #foot_COM = segment_COMs[f"foot_{side}"]

            # --- Limb segments ---
            for seg in seg_order:
                COM, mass = segment_COMs[seg], segment_masses[seg]
                F = compute_force(COM, mass, add_g=True)
                total_force += F

                total_moment += np.cross(COM - body_COM, F)

                # Rotational (free-moment) contribution from this segment's own
                # angular acceleration, on top of the translational r x F above.
                # Free moments/couples add directly regardless of reference point.
                M_rot = add_rotational_moment(seg, mass)
                if M_rot is not None:
                    total_moment += M_rot

            # --- Lateral (medio-lateral) driver signal & feet lateral positions for
            # HAT weight-transfer splitting. Which signal drives this is controlled by
            # self.lateral_weight_source -- see __init__ docstring. "hip_midpoint" is
            # the general-purpose default: hip-joint-center midpoint tracks horizontal
            # COM projection closely across gait/gym/sport movements with no external
            # unmodeled load. After the swap_xy normalization, column 0 = lateral/ML,
            # column 1 = AP, column 2 = vertical -- this MUST read column 0. ---
            if self.lateral_weight_source == "hip_midpoint":
                driver_ml = 0.5 * (segment_COMs["hip_l"][:, 0] + segment_COMs["hip_r"][:, 0])
            elif self.lateral_weight_source == "pelvis":
                driver_ml = segment_COMs["pelvis"][:, 0]
            else:  # "body_com"
                driver_ml = body_COM[:, 0]

            foot_l_ml = segment_COMs["foot_l"][:, 0]
            foot_r_ml = segment_COMs["foot_r"][:, 0]

            if self.legacy_lateral_weight_direction:
                # Old (inverted) convention: driver signal proximity to a foot assigns
                # MORE weight to the OTHER foot. Kept only for back-compat comparisons
                # against prior force-plate runs that used this convention.
                weight_ratio = (driver_ml - foot_l_ml) / (foot_r_ml - foot_l_ml + 1e-8)
            else:
                # Anatomically correct convention (default): driver signal proximity
                # to a foot assigns MORE weight to that SAME foot. E.g. driver signal
                # shifting toward the left foot -> more Fz (and more of the dynamic
                # lateral share) under the left foot.
                weight_ratio = (foot_r_ml - driver_ml) / (foot_r_ml - foot_l_ml + 1e-8)

            weight_ratio = np.clip(weight_ratio, 0, 1)  # fraction assigned to LEFT

            # --- HAT segments: AP (col 1) unchanged; Fz (col 2) dynamic; lateral (col 0) blended ---
            for seg in HAT_segs:
                COM = segment_COMs[seg]
                seg_mass = segment_masses[seg]
                # Full HAT contribution calculated once.
                # Bilateral optimizer will handle left/right allocation.
                base_mass = 0.5 * seg_mass
                F = compute_force(COM, base_mass, add_g=True)
                # Equal HAT vertical allocation.


                # Equal HAT lateral allocation.
                


                total_force += F
                total_moment += np.cross(COM - body_COM, F)

                # Rotational contribution for trunk/head: use base_mass (half the
                # segment's real mass), the same "each side carries half" accounting
                # already used for the F baseline above, so that summing the L and R
                # totals recovers the full segment's rotational moment exactly once
                # (rotation isn't naturally tied to the L/R weight-transfer split the
                # way vertical/lateral force is, so it isn't run through weight_ratio).
                M_rot = add_rotational_moment(seg, base_mass)
                if M_rot is not None:
                    total_moment += M_rot

                       # --- Final force/moment domain smoothing (mops up residual differentiation noise) ---
            # --- Final force/moment domain smoothing (mops up residual differentiation noise) ---
            total_force_f = self._filter_lowpass(total_force, self.fc_dyn)
            total_moment_f = self._filter_lowpass(total_moment, self.fc_dyn)
            body_force_total += total_force_f
            body_moment_total += total_moment_f

            # Preserve current left/right storage during loop
            if side == "l":
                GRF_left = total_force_f.copy()
                M_left = total_moment_f.copy()
            else:
                GRF_right = total_force_f.copy()
                M_right = total_moment_f.copy()

 

        # ---------------------------------------------------------
        # Optimization-based bilateral GRF partition
        # ---------------------------------------------------------

        total_force = body_force_total
        total_moment = body_moment_total

        # Kinematic (marker-height-only) contact CONFIDENCE, computed independently
        # of any force/moment values -- see contact_height_threshold/
        # contact_confidence_band docstrings. Passed into the optimizer so it can
        # smoothly shrink an airborne-going foot's GRF bounds toward zero over the
        # confidence ramp band, rather than snapping to a hard override at a single
        # frame; without this, the moment-equilibrium split has no way to know a
        # foot is off the ground and will keep assigning it nonzero force every
        # frame just to balance the equations, which the downstream Fz-threshold
        # hysteresis (in _apply_contact) then can't reliably catch, since it's
        # thresholding a result that was never actually small to begin with. Only
        # computed for movements with a real flight phase (continuous_contact=
        # False) -- for continuous-stance movements (golf, baseball swing) both
        # feet are assumed grounded throughout by design.
        if self.continuous_contact:
            contact_conf_l = None
            contact_conf_r = None
        else:
            contact_conf_l = self._contact_confidence(segment_COMs["foot_l"])
            contact_conf_r = self._contact_confidence(segment_COMs["foot_r"])

        GRF_left, GRF_right = self.optimize_bilateral_grf(
            total_force,
            total_moment,
            segment_COMs["foot_l"],
            segment_COMs["foot_r"],
            body_COM,
            contact_conf_l=contact_conf_l,
            contact_conf_r=contact_conf_r,
        )

        force_error = np.max(
            np.linalg.norm(
                GRF_left + GRF_right - total_force,
                axis=1,
            )
        )

        moment_pred = (
            np.cross(
                segment_COMs["foot_l"] - body_COM,
                GRF_left
            )
            +
            np.cross(
                segment_COMs["foot_r"] - body_COM,
                GRF_right
            )
        )

        moment_error = moment_pred - total_moment


        # optimizer temporarily disabled for baseline check

        return {
            "GRF_left": GRF_left,
            "GRF_right": GRF_right,
            "M_left": M_left,
            "M_right": M_right,
            # Extra (v12): whole-body force / moment-about-COM that the ground must
            # supply (both sides summed, fc_dyn-filtered, before the L/R optimizer).
            "total_force": total_force,
            "total_moment": total_moment,
        }

    # ======================================================================
    # MEASURED-EQUIVALENT ("forces screen") RESPONSE  (v12)
    # ======================================================================
    # Everything below turns the COMPUTED per-foot GRF into the same response the
    # measured-force script builds from AMTI/Kistler plates (same top-level keys,
    # same sub-keys, same units, same window/timing conventions). It never changes
    # the force computation above.
    #
    # Conventions
    #   * "_l" / "_r" are the OUTPUT sides, i.e. after swap_lr. Every piece of
    #     geometry (arrow origin, COP, midfoot) is taken from the marker set that
    #     belongs to that output side (see _out_side_suffix), so forces and
    #     positions always refer to the same foot.
    #   * Position geometry is expressed in the "tracking frame" of the measured
    #     script: axis0 = depth/AP (acted on by Fy), axis1 = lateral (acted on by
    #     Fx), axis2 = vertical. Internally this pipeline's columns are
    #     (Fx=lateral, Fy=AP, Fz) so the tracking frame is internal[..., [1, 0, 2]].
    #   * "plate frame" COP (what force_vectors.COP_l/_r/COP hold in the measured
    #     response) has no physical plates here. It is defined as
    #         plate = marker_frame - offset,   offset = midpoint of the two midfoot
    #     markers at P1 (horizontal) / left midfoot height at P1 (vertical),
    #     so that  COP[:,1]+offset_x / COP[:,0]+offset_y  map straight back to marker
    #     coordinates exactly as the measured script's formulas expect.
    #   * Events: p1 -> P1, p7 -> P7 (impact/release), p8 -> P10 (analysis end).
    #     (A p10 key, if the JSON has one, wins over p8.) Missing events fall back
    #     gracefully -- see _resolve_events.
    _PLATE_LAYOUTS = {
        "amti":    {"plate_h": 0.5, "plate_w": 1.0, "x_off": 0.75, "vec_scale": 1600.0},
        "kistler": {"plate_h": 0.5, "plate_w": 1.2, "x_off": 0.90, "vec_scale": 1500.0},
    }
    FORCE_VECTOR_3D_SCALE = 1500.0   # measured script hard-codes 1/1500 for force_vectors_3d
    # The measured script multiplies Fx by COM velocity along tracking-axis0 (AP) and Fy by
    # velocity along axis1 (lateral), although Fx acts laterally / Fy acts AP. True keeps
    # that pairing so predicted and measured power traces are directly comparable; False
    # pairs matching axes (Fx*v_lateral, Fy*v_AP).
    POWER_AXIS_PAIRING_LIKE_MEASURED = True
    _M_TO_IN = 0.3937007874
    # (mass fraction) used by the 2D-stick-figure COM models, same as the measured script
    _SEG_MASS_FRAC = {
        "head": 0.0694, "trunk": 0.4336, "arm": 0.0271, "forearm": 0.0162, "hand": 0.0061,
        "thigh": 0.1416, "shank": 0.0433, "foot": 0.0137,
    }

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _b_frame(arr):
        """internal (Fx-lateral, Fy-AP, up) -> tracking frame (AP, lateral, up)."""
        return np.asarray(arr, dtype=float)[..., [1, 0, 2]]

    @staticmethod
    def _safe_div(a, b, fill=0.0, eps=1e-9):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ok = np.abs(b) > eps
        return np.where(ok, a / np.where(ok, b, 1.0), fill)

    def _out_side_suffix(self):
        """Marker-name suffix ('l'/'r') that belongs to each OUTPUT side. With
        swap_lr the optimizer's 'left' force is relabelled 'right' (and vice versa),
        so geometry for output-left must come from the 'r' markers."""
        return ("r", "l") if self.swap_lr else ("l", "r")

    def _bfilt(self, x, fc, order=4):
        """Butterworth + filtfilt exactly as the measured script does it (default
        padding, order 4). Falls back to the raw signal if the trial is too short."""
        x = np.asarray(x, dtype=float)
        b, a = butter(order, fc / (self.frame_rate / 2.0))
        try:
            return filtfilt(b, a, x)
        except ValueError:
            logging.warning("Trial too short to filter at %.1f Hz; using raw values.", fc)
            return x.copy()

    def _resolve_events(self, positions, markers, n_frames):
        """Resolve the three events the analysis needs, tolerating missing phase data.

        P1  (start of analysis / weight reference): p1, else the first frame with fully
            valid markers.
        P10 (end of analysis window):               p10 or p8, else the last frame with
            fully valid markers.
        P7  (impact for hitting / golf, release for pitching): p7, else ESTIMATED as the
            frame of peak hand speed inside (P1, P10) -- the usual kinematic marker of
            impact/release. If that cannot be computed, the last frame of the window.
        An event that is absent, non-numeric, outside the trial, or inconsistent
        (P10 <= P1, or P7 outside (P1, P10)) is treated as missing. Nothing is raised
        for missing events; every fallback is logged.
        Returns dict(P1, P7, P10, source={"P1": "input"|"fallback"|..., ...})."""
        positions = positions if isinstance(positions, dict) else {}

        def get(*keys):
            for k in keys:
                v = positions.get(k)
                try:
                    if v is not None and math.isfinite(float(v)):
                        v = int(round(float(v)))
                        if 0 <= v < n_frames:
                            return v
                except (TypeError, ValueError):
                    pass
            return None

        finite = np.ones(n_frames, dtype=bool)
        for arr in markers.values():
            finite &= np.isfinite(np.asarray(arr)).all(axis=1)
        valid = np.flatnonzero(finite)
        first = int(valid[0]) if valid.size else 0
        last = int(valid[-1]) if valid.size else n_frames - 1
        src = {}

        P1 = get("p1", "P1")
        src["P1"] = "input"
        if P1 is None:
            P1, src["P1"] = first, "fallback: first valid frame"
            logging.warning("Phase event p1 missing/invalid -> using first valid frame (%d).", P1)

        P10 = get("p10", "P10", "p8", "P8")
        src["P10"] = "input"
        if P10 is not None and P10 <= P1:
            logging.warning("Phase event p8/p10 (%d) is not after P1 (%d) -> using last frame.", P10, P1)
            P10 = None
        if P10 is None:
            P10, src["P10"] = last, "fallback: last valid frame"
            logging.warning("Phase event p8/p10 missing/invalid -> using last valid frame (%d).", P10)
        if P10 <= P1:
            raise ValueError(f"Analysis window is empty (P1={P1}, P10={P10}); trial too short or markers invalid.")

        P7 = get("p7", "P7")
        src["P7"] = "input"
        if P7 is not None and not (P1 < P7 < P10):
            logging.warning("Phase event p7 (%d) lies outside (P1=%d, P10=%d) -> estimating.", P7, P1, P10)
            P7 = None
        if P7 is None:
            P7 = self._estimate_impact_frame(markers, P1, P10)
            if P7 is None:
                P7, src["P7"] = P10 - 1, "fallback: end of window"
                logging.warning("Phase event p7 missing and impact could not be estimated -> using "
                                "end of window (%d); *_time_to_impact is relative to that.", P7)
            else:
                src["P7"] = "estimated: peak hand speed"
                logging.warning("Phase event p7 (impact/release) missing -> estimated at frame %d "
                                "(peak hand speed).", P7)
        return {"P1": P1, "P7": P7, "P10": P10, "source": src}

    def _estimate_impact_frame(self, markers, P1, P10):
        """Frame of peak hand speed (faster of the two hands) strictly inside (P1, P10)."""
        try:
            speeds = []
            for nm in ("hand_l", "hand_r"):
                if nm in markers:
                    x = self._filter_lowpass(np.asarray(markers[nm], dtype=float), self.fc_dyn)
                    speeds.append(np.linalg.norm(np.gradient(x, 1.0 / self.frame_rate, axis=0), axis=1))
            if not speeds:
                return None
            sp = np.max(np.vstack(speeds), axis=0)
            lo, hi = P1 + 1, P10 - 1
            if hi <= lo or not np.isfinite(sp[lo:hi + 1]).all():
                return None
            return int(lo + np.argmax(sp[lo:hi + 1]))
        except Exception as exc:
            logging.warning("Impact estimation failed (%s).", exc)
            return None

    @staticmethod
    def _fs_peak(x, P1, P10, P7, time, use_abs=False, index_timing=False):
        """max over [P1, P10) (same window as the measured script) -> (peak,
        timing, time_to_impact). timing/time_to_impact are ms unless index_timing
        (the measured script reports a raw frame index for impulse/power/energy)."""
        win = np.asarray(x[P1:P10], dtype=float)
        if use_abs:
            win = np.abs(win)
        idx = int(np.argmax(win))
        t_imp = time[min(max(P7 - P1, 0), len(time) - 1)]
        tti = float((t_imp - time[idx]) * 1000.0)
        timing = float(idx) if index_timing else float(time[idx] * 1000.0)
        return float(win[idx]), timing, tti

    @staticmethod
    def _json_safe(obj):
        """Replace non-finite floats (invalid in strict JSON) with None, recursively."""
        if isinstance(obj, dict):
            return {k: ForceVectors._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ForceVectors._json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj) if math.isfinite(float(obj)) else None
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    @staticmethod
    def _dist_point_line_2d(p, a, b):
        """Perpendicular distance from point p to the line through b with direction a
        (all (n,2)). Zero-length direction (no force) -> 0."""
        aa = np.sum(a * a, axis=1)
        ok = aa > 1e-12
        th = np.sum(a * (p - b), axis=1) / np.where(ok, aa, 1.0)
        h = b + a * th[:, None]
        return np.where(ok, np.linalg.norm(p - h, axis=1), 0.0)

    # ------------------------------------------------------------ COP / Tz
    @staticmethod
    def _fs_plate_origin(midfoot_l_3d, midfoot_r_3d, P1):
        """(offset_x, offset_y, offset_z): marker-frame position of the synthetic
        plate-frame origin. Horizontal = midpoint of the two midfoot markers at P1
        (AP, lateral); vertical = left midfoot height at P1 (the ground reference the
        measured script uses). Override this if you want a fixed plate layout."""
        return (
            float(0.5 * (midfoot_l_3d[P1, 0] + midfoot_r_3d[P1, 0])),
            float(0.5 * (midfoot_l_3d[P1, 1] + midfoot_r_3d[P1, 1])),
            float(midfoot_l_3d[P1, 2]),
        )

    def _fs_cop_marker_frame(self, P_app, Fx, Fy, Fz, ground_z):
        """Per-foot COP (tracking frame, horizontal) from the optimizer's own model:
        each foot's GRF is applied at that foot's segment COM (see
        optimize_bilateral_grf: r = foot_COM - body_COM). The COP is where that line
        of action meets the ground:
            COP_xy = P_xy - (P_z - ground_z) * F_xy / F_z
        so shear tilts the COP away from the foot COM, consistently with the moment
        balance the force split was solved with. Shear/Fz is capped at mu_friction (the
        same cone the optimizer guards). Unloaded foot (Fz<=1 N) -> the foot COM
        projection, which keeps the trace continuous instead of NaN."""
        mu = max(self.mu_friction, 1e-6)
        loaded = Fz > 1.0
        Fz_safe = np.where(loaded, Fz, 1.0)
        slope_ap = np.clip(Fy / Fz_safe, -mu, mu) * loaded
        slope_lat = np.clip(Fx / Fz_safe, -mu, mu) * loaded
        h = np.maximum(P_app[:, 2] - ground_z, 0.0)
        cop = np.zeros((len(Fz), 3))
        cop[:, 0] = P_app[:, 0] - h * slope_ap      # AP  (acted on by Fy)
        cop[:, 1] = P_app[:, 1] - h * slope_lat     # lateral (acted on by Fx)
        return cop

    def _fs_vertical_torque(self, ctx, Fl, Fr, Fz_l, Fz_r):
        """Vertical free moment (Tz_t / Tz_l / Tz_r), N*m.

        This is a CLOSURE estimate -- the one channel here that cannot be measured
        from kinematics:  the whole-body inverse dynamics says the ground must supply
        a vertical torque about the COM of total_moment_z; the predicted horizontal
        shear forces produce a couple (r x F)_z about the COM; whatever the couple
        does not cover is attributed to the feet's free moment:
            Tz_t = LPF_fc_grf(total_moment_z) - sum_i [(foot_i - COM) x F_i]_z
        (both terms at fc_grf bandwidth so filtering differences are not read as
        torque). Split between feet in proportion to Fz. Sign follows the same
        tracking-frame cross-product convention as z_Torque. See the module notes for
        why this is qualitative only."""
        sfx_l, sfx_r = self._out_side_suffix()
        com = ctx["body_COM"]
        seg = ctx["segment_COMs_filtered"]
        r_l = seg[f"foot_{sfx_l}"] - com
        r_r = seg[f"foot_{sfx_r}"] - com
        couple_z = np.cross(r_l, Fl)[:, 2] + np.cross(r_r, Fr)[:, 2]
        total_mz = self._filter_lowpass(ctx["inv_dyn"]["total_moment"], self.fc_grf)[:, 2]
        # internal frame (lat, AP, up) is a mirror of the tracking frame (AP, lat, up):
        # swapping two axes flips the sign of the z-component of a cross product.
        Tz_t = -(total_mz - couple_z)
        Fz_tot = Fz_l + Fz_r
        Tz_t = np.where(Fz_tot > 1.0, Tz_t, 0.0)
        share_l = self._safe_div(Fz_l, Fz_tot, fill=0.5)
        return Tz_t, Tz_t * share_l, Tz_t * (1.0 - share_l)

    # ------------------------------------------------------------- 2D (pose2d)
    @staticmethod
    def _extract_pose2d_array(pose2d, n_frames):
        """Returns (n_frames, 29, 2) pixel markers. Accepts the measured script's
        format pose2d["data"] = [frame][marker][x, y] (also marker-first, and a
        pose2d["data"]["pose"] nesting)."""
        data = pose2d.get("data") if isinstance(pose2d, dict) else None
        if isinstance(data, dict) and "pose" in data:
            data = data["pose"]
        if data is None:
            raise KeyError("pose2d must carry its markers under ['data'] (or ['data']['pose']).")
        arr = np.squeeze(np.asarray(data, dtype=float))
        if arr.ndim != 3:
            raise ValueError(f"pose2d markers have unexpected shape {arr.shape}")
        # frame axis = the one matching the 3D frame count (frame-first or marker-first)
        if arr.shape[0] != n_frames and arr.shape[1] == n_frames:
            arr = arr.transpose(1, 0, 2)
        if arr.shape[0] != n_frames:
            raise ValueError(f"pose2d has shape {arr.shape} but the 3D pose has {n_frames} frames.")
        if arr.shape[1] < 29:
            raise ValueError(f"pose2d needs at least 29 markers; got shape {arr.shape}")
        if arr.shape[1] > 29:
            # the 2D skeleton may carry extra trailing markers (e.g. 31); only indices 0-28
            # (the 29-marker layout shared with the 3D pose) are used.
            logging.info("pose2d has %d markers; using the first 29.", arr.shape[1])
            arr = arr[:, :29, :]
        return arr[:, :, :2]

    def _fs_com_2d(self, p2, names):
        """2D (pixel) COM models of the measured script: full body, without arms,
        lower body, upper body."""
        idx = {nm: i for i, nm in enumerate(names)}
        def com(a, b, f):
            pa, pb = p2[:, idx[a]], p2[:, idx[b]]
            return pa - (pa - pb) * f
        m = self._SEG_MASS_FRAC
        seg = {
            "head": (com("skull", "skullbase", 0.5976), m["head"]),
            "trunk": (com("neck", "pelvis", 0.4486), m["trunk"]),
            "arm_r": (com("shoulder_r", "elbow_r", 0.5772), m["arm"]),
            "forearm_r": (com("elbow_r", "wrist_r", 0.4574), m["forearm"]),
            "hand_r": (com("wrist_r", "hand_r", 0.79), m["hand"]),
            "arm_l": (com("shoulder_l", "elbow_l", 0.5772), m["arm"]),
            "forearm_l": (com("elbow_l", "wrist_l", 0.4574), m["forearm"]),
            "hand_l": (com("wrist_l", "hand_l", 0.79), m["hand"]),
            "thigh_r": (com("hip_r", "knee_r", 0.4095), m["thigh"]),
            "shank_r": (com("knee_r", "ankle_r", 0.4459), m["shank"]),
            "foot_r": (com("ankle_r", "foot_r", 0.4415), m["foot"]),
            "thigh_l": (com("hip_l", "knee_l", 0.4095), m["thigh"]),
            "shank_l": (com("knee_l", "ankle_l", 0.4459), m["shank"]),
            "foot_l": (com("ankle_l", "foot_l", 0.4415), m["foot"]),
        }
        def wsum(keys):
            return sum(seg[k][0] * seg[k][1] for k in keys)
        arm_m = m["arm"] + m["forearm"] + m["hand"]
        leg_m = m["thigh"] + m["shank"] + m["foot"]
        W_no_arms = 1 - 2 * arm_m
        W_lower = 1 - 2 * arm_m - m["trunk"] - m["head"]
        W_upper = 1 - 2 * leg_m
        legs = ["thigh_r", "shank_r", "foot_r", "thigh_l", "shank_l", "foot_l"]
        arms = ["arm_r", "forearm_r", "hand_r", "arm_l", "forearm_l", "hand_l"]
        return {
            "COM_2d": wsum(list(seg)),
            "COM_2d_woArms": wsum(["head", "trunk"] + legs) / W_no_arms,
            "COM_2d_lowerBody": wsum(legs) / W_lower,
            "COM_2d_upperBody": wsum(["head", "trunk"] + arms) / W_upper,
        }

    # ------------------------------------------------------------------ main
    def _build_forces_screen(self, ctx, pose2d=None):
        """Build the measured-script-equivalent response from computed forces.

        Returns the same top-level keys as the measured script: force_vectors,
        weight_shift, forces, time, force_vectors_3d, values_for_COP_trajectory,
        COM_MODELS, power, energy, impulse, small_force_vectors,
        COM_COP_traj_back_view, critical_events, frame_offset.

        Keys that fundamentally need 2D pixel data (the measured script reads them from
        its 2D stick figure) are only produced when pose2d is supplied:
        top-level weight_shift and COM_MODELS, and inside force_vectors:
        Fx/Fy/Fz_{t,l,r}_pix, COP_for_Frontend_X/Y, factor_for_midfoot_point_frontend,
        proj_GRF_COM_2d, COM_2d. Without pose2d they are omitted (and logged), not
        faked.
        """
        fr = float(self.frame_rate)
        dt = 1.0 / fr
        n = int(ctx["n_frames"])
        # Left-handed player: express the analysis in the force-plate frame (180 deg about
        # vertical). Analytical metrics stay in that frame; drawing geometry (3D arrows,
        # 2D video overlays) is converted back to the tracking/video frame at the end.
        rot = bool(self.lefty_plate_rotation and ctx.get("handedness") == "left")
        if rot:
            logging.info("Left-handed player: rotating the analysis 180 deg about the vertical "
                         "axis into the force-plate frame (drawing geometry stays in the tracking frame).")
            ctx = self._rotate_ctx_180(ctx)
        ev = ctx.get("events") or self._resolve_events(ctx.get("positions"), ctx["markers"], n)
        P1, P7, P10 = ev["P1"], ev["P7"], ev["P10"]
        time = np.arange(1, P10 - P1 + 1) / fr          # measured script: range(1, nFrames)/fr
        time_index = 0                                   # predicted forces need no plate time-sync
        lay = self._PLATE_LAYOUTS[self.plate_layout]
        sfx_l, sfx_r = self._out_side_suffix()

        # ---- reported (contact-gated) forces, output-side labelled
        f = lambda k: np.asarray(ctx[k], dtype=float)
        Fx_l, Fy_l, Fz_l = f("Fx_l"), f("Fy_l"), f("Fz_l")
        Fx_r, Fy_r, Fz_r = f("Fx_r"), f("Fy_r"), f("Fz_r")
        Fx_t, Fy_t, Fz_t = Fx_l + Fx_r, Fy_l + Fy_r, Fz_l + Fz_r
        GRF_l = np.sqrt(Fx_l**2 + Fy_l**2 + Fz_l**2)
        GRF_r = np.sqrt(Fx_r**2 + Fy_r**2 + Fz_r**2)
        GRF_t = GRF_l + GRF_r

        # ---- geometry in the tracking frame
        mk, seg = ctx["markers"], ctx["segment_COMs_filtered"]
        depth_shift = np.zeros(n)
        if self.anchor_foot_depth:
            anchor = self._b_frame(mk[f"foot_{sfx_l}"])
            depth_shift = anchor[:, 0] - anchor[0, 0]

        def pos_b(a):
            out = self._b_frame(a).copy()
            out[:, 0] -= depth_shift
            return out

        midfoot_l_3d, midfoot_r_3d = pos_b(mk[f"midfoot_{sfx_l}"]), pos_b(mk[f"midfoot_{sfx_r}"])
        foot_l_com, foot_r_com = pos_b(seg[f"foot_{sfx_l}"]), pos_b(seg[f"foot_{sfx_r}"])
        COM = pos_b(ctx["body_COM"])

        # ---- COP (marker frame -> plate frame)
        cop_l_m = self._fs_cop_marker_frame(foot_l_com, Fx_l, Fy_l, Fz_l, float(np.min(foot_l_com[:, 2])))
        cop_r_m = self._fs_cop_marker_frame(foot_r_com, Fx_r, Fy_r, Fz_r, float(np.min(foot_r_com[:, 2])))
        offset_x, offset_y, offset_z = self._fs_plate_origin(midfoot_l_3d, midfoot_r_3d, P1)

        def to_plate(c):
            out = np.zeros((n, 3))
            out[:, 0] = c[:, 1] - offset_y
            out[:, 1] = c[:, 0] - offset_x
            return out

        COP_l, COP_r = to_plate(cop_l_m), to_plate(cop_r_m)
        factor_l = self._safe_div(GRF_l, GRF_t, fill=0.5)
        COP = COP_r - (COP_r - COP_l) * factor_l[:, None]
        stance = np.sqrt((COP_r[:, 0] - COP_l[:, 0]) ** 2 + (COP_r[:, 1] - COP_l[:, 1]) ** 2)

        # ---- vertical torque (free moment)
        Tz_t, Tz_l, Tz_r = self._fs_vertical_torque(
            ctx, np.column_stack([Fx_l, Fy_l, Fz_l]), np.column_stack([Fx_r, Fy_r, Fz_r]), Fz_l, Fz_r
        )

        # ---- body torques about the COM + moment arms (GRF line of action vs COM)
        P0 = np.column_stack([COP[:, 1] + offset_x, COP[:, 0] + offset_y, COP[:, 2] + offset_z])
        d = np.column_stack([Fy_t, Fx_t, Fz_t])
        dd = np.sum(d * d, axis=1)
        ok = dd > 1e-12
        th = np.where(ok, np.sum(d * (COM - P0), axis=1) / np.where(ok, dd, 1.0), 0.0)
        proj = np.where(ok[:, None], P0 + d * th[:, None], COM)    # no force -> no moment
        x_torque = (proj[:, 1] - COM[:, 1]) * Fz_t - (proj[:, 2] - COM[:, 2]) * Fx_t   # frontal
        y_torque = (proj[:, 0] - COM[:, 0]) * Fz_t - (proj[:, 2] - COM[:, 2]) * Fy_t   # sagittal
        z_torque = (proj[:, 0] - COM[:, 0]) * Fx_t - (proj[:, 1] - COM[:, 1]) * Fy_t   # horizontal

        x_arm = self._dist_point_line_2d(COM[:, [1, 2]], np.column_stack([Fx_t, Fz_t]), P0[:, [1, 2]])
        y_arm = self._dist_point_line_2d(COM[:, [0, 2]], np.column_stack([Fy_t, Fz_t]), P0[:, [0, 2]])
        z_arm = self._dist_point_line_2d(COM[:, [0, 1]], np.column_stack([Fy_t, Fx_t]), P0[:, [0, 1]])
        x_arm = np.where(x_torque < 0, -x_arm, x_arm) * 100.0      # signed, cm
        y_arm = np.where(y_torque < 0, -y_arm, y_arm) * 100.0
        z_arm = np.where(z_torque < 0, -z_arm, z_arm) * 100.0

        # ---- COM position of the COM relative to the two COPs (% of half-stance)
        u = np.column_stack([COP_r[:, 1] - COP_l[:, 1], COP_r[:, 0] - COP_l[:, 0]])
        v = np.column_stack([COP_r[:, 1] + offset_x, COP_r[:, 0] + offset_y])
        uu = np.sum(u * u, axis=1)
        okc = uu > 1e-12
        thc = np.sum(u * (COM[:, [0, 1]] - v), axis=1) / np.where(okc, uu, 1.0)
        hproj = v + u * thc[:, None]
        l_pt = np.column_stack([COP_l[:, 1] + offset_x, COP_l[:, 0] + offset_y])
        COM_vs_stance = np.where(
            okc, 100.0 * (1.0 - np.linalg.norm(l_pt - hproj, axis=1) / np.where(okc, stance / 2.0, 1.0)), 0.0
        )

        # ---- COP displacement / speed, COM displacement / speed
        COP_AP_disp = (COP[:, 1] - COP[P1, 1]) * 100.0
        COP_ML_disp = (COP[:, 0] - COP[P1, 0]) * 100.0
        cop_speed = np.sqrt(np.gradient(COP[:, 0], 1 / fr) ** 2 + np.gradient(COP[:, 1], 1 / fr) ** 2) * 100.0
        COP_vel = self._bfilt(cop_speed, 5)

        x_c, y_c = self._bfilt(COM[:, 0], 3), self._bfilt(COM[:, 1], 3)
        z_c = self._bfilt(COM[:, 2], 5)
        COM_disp_x, COM_disp_y, COM_disp_z = (x_c - x_c[0]) * 100.0, (y_c - y_c[0]) * 100.0, (z_c - z_c[0]) * 100.0

        def back_diff(x, first):
            v_ = np.diff(x) * fr
            return np.insert(v_, 0, v_[0] if first == "copy" else 0.0)

        vx, vy, vz = back_diff(x_c, 0), back_diff(y_c, 0), back_diff(z_c, "copy")
        COM_vel = np.sqrt(vx**2 + vy**2 + vz**2) * 100.0

        # ---- forces dict (+ max / timing / time-to-impact), measured-script layout
        Fz_p1 = Fz_t[P1] if Fz_t[P1] > 0 else self.body_mass * 9.81
        chan = {"Fz": (Fz_t, Fz_l, Fz_r), "Fx": (Fx_t, Fx_l, Fx_r), "Fy": (Fy_t, Fy_l, Fy_r), "Tz": (Tz_t, Tz_l, Tz_r)}
        forces, peaks = {}, {}
        for c, arrs in chan.items():
            for s, a in zip("tlr", arrs):
                forces[f"{c}_{s}"] = np.around(a, 2).tolist()
                peaks[f"{c}_{s}"] = self._fs_peak(a, P1, P10, P7, time)
        wt = {}
        for c in ("Fz", "Fx", "Fy"):
            for s, a in zip("tlr", chan[c]):
                wt[f"{c}_{s}_wt"] = np.around(a / Fz_p1 * 100.0, 2).tolist()
        forces.update(wt)
        forces.update({f"{k}_max": v[0] for k, v in peaks.items()})
        forces.update({f"{k}_time_to_impact": v[2] for k, v in peaks.items()})
        forces.update({f"{k}_max_timing": v[1] for k, v in peaks.items()})

        # ---- torque / moment-arm peaks
        tq = {"x_Torque": x_torque, "y_Torque": y_torque, "z_Torque": z_torque}
        ma = {"x_Moment_Arm": x_arm, "y_Moment_Arm": y_arm, "z_Moment_Arm": z_arm}
        tq_pk = {k: self._fs_peak(v, P1, P10, P7, time) for k, v in tq.items()}
        ma_pk = {k: self._fs_peak(v, P1, P10, P7, time) for k, v in ma.items()}

        # ---- 3D arrows (force_vectors_3d)
        sc = self.FORCE_VECTOR_3D_SCALE
        force_vectors_3d = {
            "x_start_COP_l": midfoot_l_3d[:, 0].tolist(), "x_end_COP_l": (midfoot_l_3d[:, 0] + Fy_l / sc).tolist(),
            "y_start_COP_l": midfoot_l_3d[:, 1].tolist(), "y_end_COP_l": (midfoot_l_3d[:, 1] + Fx_l / sc).tolist(),
            "z_start_COP_l": midfoot_l_3d[:, 2].tolist(), "z_end_COP_l": (midfoot_l_3d[:, 2] + Fz_l / sc).tolist(),
            "x_start_COP_r": midfoot_r_3d[:, 0].tolist(), "x_end_COP_r": (midfoot_r_3d[:, 0] + Fy_r / sc).tolist(),
            "y_start_COP_r": midfoot_r_3d[:, 1].tolist(), "y_end_COP_r": (midfoot_r_3d[:, 1] + Fx_r / sc).tolist(),
            "z_start_COP_r": midfoot_r_3d[:, 2].tolist(), "z_end_COP_r": (midfoot_r_3d[:, 2] + Fz_r / sc).tolist(),
            "x_start_COP_combined": (COP[:, 1] + offset_x).tolist(),
            "x_end_COP_combined": (COP[:, 1] + offset_x + Fy_t / sc).tolist(),
            "y_start_COP_combined": (COP[:, 0] + offset_y).tolist(),
            "y_end_COP_combined": (COP[:, 0] + offset_y + Fx_t / sc).tolist(),
            "z_start_COP_combined": (COP[:, 2] + offset_z).tolist(),
            "z_end_COP_combined": (COP[:, 2] + offset_z + Fz_t / sc).tolist(),
        }

        # ---- weight shift / stance
        ws_r = 100.0 * self._safe_div(GRF_r, GRF_t, fill=0.5)
        ws_l = 100.0 * self._safe_div(GRF_l, GRF_t, fill=0.5)
        stance_mf = np.linalg.norm(midfoot_r_3d - midfoot_l_3d, axis=1) * 100.0
        values_for_COP_trajectory = {
            "weight_shift_right": ws_r.tolist(), "weight_shift_left": ws_l.tolist(),
            "stance_midfeet": stance_mf.tolist(), "stance_midfeet_inch": (stance_mf * self._M_TO_IN).tolist(),
        }

        # ---- impulse
        I = {}
        for ax, (t_, l_, r_) in {"z": chan["Fz"], "y": chan["Fy"], "x": chan["Fx"]}.items():
            I[f"I{ax}_r"], I[f"I{ax}_l"], I[f"I{ax}_t"] = np.cumsum(r_) * dt, np.cumsum(l_) * dt, np.cumsum(t_) * dt
        impulse = {k: np.around(v, 2).tolist() for k, v in I.items()}
        for ax in ("x", "y", "z"):
            for s in ("t", "r", "l"):
                pk, tm, tti = self._fs_peak(I[f"I{ax}_{s}"], P1, P10, P7, time, index_timing=True)
                impulse[f"I{ax}_{s}_max"], impulse[f"I{ax}_{s}_max_timing"], impulse[f"I{ax}_{s}_time_to_impact"] = pk, int(tm), tti
        # same key order as the measured script
        impulse = {k: impulse[k] for k in (
            ["Iz_r", "Iz_l", "Iz_t", "Iy_r", "Iy_l", "Iy_t", "Ix_r", "Ix_l", "Ix_t"]
            + [f"I{a}_{s}_{q}" for s in ("t", "r", "l") for a in ("x", "y", "z") for q in ("max", "max_timing", "time_to_impact")]
        )}

        # ---- power / energy (COM velocity filtered like the measured script)
        pv0 = back_diff(x_c, 0)
        pv1 = back_diff(y_c, 0)
        pv2 = back_diff(z_c, 0)
        if self.POWER_AXIS_PAIRING_LIKE_MEASURED:
            vel = {"x": pv0, "y": pv1, "z": pv2}                 # Fx*v_AP, Fy*v_lat (as measured script)
        else:
            vel = {"x": pv1, "y": pv0, "z": pv2}                 # Fx*v_lat, Fy*v_AP
        Pw, En = {}, {}
        for s, (fx, fy, fz) in {"l": (Fx_l, Fy_l, Fz_l), "r": (Fx_r, Fy_r, Fz_r), "t": (Fx_t, Fy_t, Fz_t)}.items():
            Pw[f"Px_{s}"], Pw[f"Py_{s}"], Pw[f"Pz_{s}"] = fx * vel["x"], fy * vel["y"], fz * vel["z"]
            Pw[f"P_{s}"] = np.sqrt(Pw[f"Px_{s}"] ** 2 + Pw[f"Py_{s}"] ** 2 + Pw[f"Pz_{s}"] ** 2)
            for a in ("x", "y", "z"):
                En[f"E{a}_{s}"] = np.cumsum(Pw[f"P{a}_{s}"]) * dt
            En[f"E_{s}"] = np.sqrt(En[f"Ex_{s}"] ** 2 + En[f"Ey_{s}"] ** 2 + En[f"Ez_{s}"] ** 2)

        def peak_block(src, names):
            out = {}
            for nm in names:
                out[nm] = self._fs_peak(src[nm], P1, P10, P7, time, use_abs=True, index_timing=True)
            return out

        p_names = ["P_t", "Px_t", "Py_t", "Pz_t", "Px_r", "Py_r", "Pz_r", "Px_l", "Py_l", "Pz_l"]
        e_names = ["E_t", "Ez_t", "Ey_t", "Ex_t", "Ez_r", "Ey_r", "Ex_r", "Ez_l", "Ey_l", "Ex_l"]
        p_pk, e_pk = peak_block(Pw, p_names), peak_block(En, e_names)
        power = {k: np.around(Pw[k], 2).tolist() for k in
                 ["Px_l", "Py_l", "Pz_l", "Px_r", "Py_r", "Pz_r", "Px_t", "Py_t", "Pz_t", "P_r", "P_l", "P_t"]}
        power.update({f"{k}_max": v[0] for k, v in p_pk.items()})
        power.update({f"{k}_time_to_impact": v[2] for k, v in p_pk.items()})
        power.update({f"{k}_max_timing": v[1] for k, v in p_pk.items()})
        energy = {k: np.around(En[k], 2).tolist() for k in
                  ["Ex_l", "Ey_l", "Ez_l", "Ex_r", "Ey_r", "Ez_r", "Ex_t", "Ey_t", "Ez_t", "E_r", "E_l", "E_t"]}
        energy.update({f"{k}_max": v[0] for k, v in e_pk.items()})
        energy.update({f"{k}_time_to_impact": v[2] for k, v in e_pk.items()})
        energy.update({f"{k}_max_timing": v[1] for k, v in e_pk.items()})

        # ---- frontend widget pixels (COP / COM trajectories, front view)
        fh, fw = 180.0, 375.0
        dist_pix = math.hypot(fh, fw)
        dist_m = math.hypot(lay["plate_h"], lay["plate_w"])
        k_all = dist_pix / dist_m
        k_h, k_w = fh / lay["plate_h"], fw / lay["plate_w"]
        xo = lay["x_off"]
        foot_len_pix = int(0.2 * k_h)
        foot_wid_pix = int(0.2 / 2.8 * k_w)

        def foot_box(cx, cy):
            return ([int(cx - foot_wid_pix / 2), int(cx + foot_wid_pix / 2)],
                    [int(cy - foot_len_pix / 2), int(cy + foot_len_pix / 2)])

        COP_l_px, COP_l_py = (COP_l[:, 0] + xo) * k_w, (COP_l[:, 1] + 0.25) * k_h
        COP_r_px, COP_r_py = (COP_r[:, 0] + xo) * k_w, (COP_r[:, 1] + 0.25) * k_h
        COP_px, COP_py = (COP[:, 0] + xo) * k_w, (COP[:, 1] + 0.25) * k_h
        COM_py = (COM[:, 0] - offset_x + 0.25) * k_all
        COM_px = (COM[:, 1] - offset_y + xo) * k_all
        rows_l, cols_l = foot_box(COP_l_px[P1], COP_l_py[P1])
        rows_r, cols_r = foot_box(COP_r_px[P1], COP_r_py[P1])

        vs = lay["vec_scale"]
        N = lambda a: a / vs
        start_X, start_Y = COP[:, 1] + offset_x, COP[:, 0] + offset_y
        start_Z = 0.5 * (midfoot_l_3d[:, 2] + midfoot_r_3d[:, 2])

        force_vectors = {
            "pixel_rows_foot_right_fixed_size": rows_r, "pixel_colm_foot_right_fixed_size": cols_r,
            "pixel_rows_foot_left_fixed_size": rows_l, "pixel_colm_foot_left_fixed_size": cols_l,
            "COP_l_pix_x_axes": COP_l_px.tolist(), "COP_l_pix_y_axes": COP_l_py.tolist(),
            "COP_r_pix_x_axes": COP_r_px.tolist(), "COP_r_pix_y_axes": COP_r_py.tolist(),
            "COP_pix_x_axes": COP_px.tolist(), "COP_pix_y_axes": COP_py.tolist(),
            "COM_pix_x_axes": COM_px.tolist(), "COM_pix_y_axes": COM_py.tolist(),
            "start_point_X": start_X.tolist(), "start_point_Y": start_Y.tolist(), "start_point_Z": start_Z.tolist(),
            "end_point_X": (start_X + N(Fy_t)).tolist(), "end_point_Y": (start_Y + N(Fx_t)).tolist(),
            "end_point_Z": (start_Z + N(Fz_t)).tolist(),
            "start_point_X_l": midfoot_l_3d[:, 0].tolist(), "start_point_Y_l": midfoot_l_3d[:, 1].tolist(),
            "start_point_Z_l": midfoot_l_3d[:, 2].tolist(),
            "end_point_X_l": (midfoot_l_3d[:, 0] + N(Fy_l)).tolist(),
            "end_point_Y_l": (midfoot_l_3d[:, 1] + N(Fx_l)).tolist(),
            "end_point_Z_l": (midfoot_l_3d[:, 2] + N(Fz_l)).tolist(),
            "start_point_X_r": midfoot_r_3d[:, 0].tolist(), "start_point_Y_r": midfoot_r_3d[:, 1].tolist(),
            "start_point_Z_r": midfoot_r_3d[:, 2].tolist(),
            "end_point_X_r": (midfoot_r_3d[:, 0] + N(Fy_r)).tolist(),
            "end_point_Y_r": (midfoot_r_3d[:, 1] + N(Fx_r)).tolist(),
            "end_point_Z_r": (midfoot_r_3d[:, 2] + N(Fz_r)).tolist(),
        }

        # ---- 2D-pixel-dependent keys (need pose2d)
        weight_shift = None
        COM_MODELS = None
        pix_keys = {}
        if pose2d is not None:
            p2 = self._extract_pose2d_array(pose2d, n)
            names = ctx["marker_names"]
            idx2 = {nm: i for i, nm in enumerate(names)}
            mf2_l = p2[:, idx2[f"midfoot_{sfx_l}"]]
            mf2_r = p2[:, idx2[f"midfoot_{sfx_r}"]]
            sh_m = float(np.linalg.norm(mk["shoulder_r"][P1] - mk["shoulder_l"][P1]))
            sh_pix = float(math.hypot(p2[P1, idx2["shoulder_r"], 1] - p2[P1, idx2["shoulder_l"], 1],
                                      p2[P1, idx2["shoulder_r"], 0] - p2[P1, idx2["shoulder_l"], 0]))
            m2pix = sh_pix / sh_m if sh_m > 0 else 0.0
            tr = lambda a: np.trunc(a).astype(int)
            Fz_t_pix, Fz_l_pix, Fz_r_pix = (tr(N(a) * m2pix).astype(float) for a in (Fz_t, Fz_l, Fz_r))
            sg = -1.0 if rot else 1.0          # video overlays use the tracking-frame horizontal forces
            Fx_t_pix, Fx_l_pix, Fx_r_pix = (-tr(N(sg * a) * m2pix).astype(float) for a in (Fx_t, Fx_l, Fx_r))
            Fy_t_pix, Fy_l_pix, Fy_r_pix = (-tr(N(sg * a) * m2pix).astype(float) for a in (Fy_t, Fy_l, Fy_r))

            dist_mf = np.abs(mf2_r[:, 0] - mf2_l[:, 0])
            wsr = self._safe_div(GRF_r, GRF_t, fill=0.5)
            factor_mf = dist_mf * wsr
            cx = tr(mf2_l[:, 0] - factor_mf)
            weight_shift = {
                "COP_for_Frontend_X": cx.tolist(),
                "COP_for_Frontend_Y": tr(mf2_r[:, 1]).tolist(),      # measured script: midfoot_r here ...
                "factor_for_midfoot_point_frontend": factor_mf.tolist(),
            }
            cy = tr(mf2_l[:, 1])                                      # ... but midfoot_l inside force_vectors
            COM_models = self._fs_com_2d(p2, names)
            COM_2d = COM_models["COM_2d"]
            COP_2d = np.column_stack([cx, cy]).astype(float)
            endpoint = COP_2d - np.column_stack([Fx_t_pix, Fz_t_pix])
            nvec = COP_2d - endpoint
            nn = np.linalg.norm(nvec, axis=1)
            okp = nn > 1e-12
            unit = nvec / np.where(okp, nn, 1.0)[:, None]
            proj2 = endpoint + unit * np.sum((COM_2d - endpoint) * unit, axis=1)[:, None]
            for i in range(n):                                        # zero-force frames: reuse previous
                if not okp[i]:
                    proj2[i] = proj2[i - 1] if i > 0 else COM_2d[i]
            pix_keys = {
                "Fz_t_pix": Fz_t_pix.tolist(), "Fz_l_pix": Fz_l_pix.tolist(), "Fz_r_pix": Fz_r_pix.tolist(),
                "Fx_t_pix": Fx_t_pix.tolist(), "Fx_l_pix": Fx_l_pix.tolist(), "Fx_r_pix": Fx_r_pix.tolist(),
                "Fy_t_pix": Fy_t_pix.tolist(), "Fy_l_pix": Fy_l_pix.tolist(), "Fy_r_pix": Fy_r_pix.tolist(),
            }
            pix_keys2 = {
                "COP_for_Frontend_X": cx.tolist(), "COP_for_Frontend_Y": cy.tolist(),
                "factor_for_midfoot_point_frontend": factor_mf.tolist(),
                "proj_GRF_COM_2d": proj2.tolist(), "COM_2d": COM_2d.tolist(),
            }
            COM_MODELS = {k: v.tolist() for k, v in COM_models.items()}
        else:
            logging.warning(
                "generate_forces_screen: no pose2d supplied -> omitting 2D-pixel keys: top-level "
                "weight_shift, COM_MODELS; force_vectors.{F[xyz]_[tlr]_pix, COP_for_Frontend_X/Y, "
                "factor_for_midfoot_point_frontend, proj_GRF_COM_2d, COM_2d}."
            )
            pix_keys2 = {}

        force_vectors.update(pix_keys)
        force_vectors.update({
            "COP_l": COP_l.tolist(), "COP_r": COP_r.tolist(), "COP": COP.tolist(),
        })
        force_vectors.update(pix_keys2)
        r2 = lambda a: np.around(a, 2)
        force_vectors.update({
            "x_Torque": r2(x_torque).tolist(), "y_Torque": r2(y_torque).tolist(), "z_Torque": r2(z_torque).tolist(),
            "x_Moment_Arm": r2(x_arm).tolist(), "y_Moment_Arm": r2(y_arm).tolist(), "z_Moment_Arm": r2(z_arm).tolist(),
            "COM_vs_Stance": [[float(v)] for v in COM_vs_stance],   # measured script emits 1-element sublists
            "FilteredCOPSpeed": r2(COP_vel).tolist(),
            "COP_disp_x": r2(COP_ML_disp).tolist(), "COP_disp_y": r2(COP_AP_disp).tolist(),
            "COM_disp_x": r2(COM_disp_x).tolist(), "COM_disp_y": r2(COM_disp_y).tolist(),
            "COM_disp_z": r2(COM_disp_z).tolist(),
            "COM_vel": r2(COM_vel).tolist(),
        })
        for ax in "xyz":
            force_vectors[f"{ax}_Torque_max"] = tq_pk[f"{ax}_Torque"][0]
        for ax in "xyz":
            force_vectors[f"{ax}_Moment_Arm_max"] = ma_pk[f"{ax}_Moment_Arm"][0]
        for ax in "xyz":
            force_vectors[f"{ax}_Torque_max_timing"] = tq_pk[f"{ax}_Torque"][1]
        for ax in "xyz":
            force_vectors[f"{ax}_Moment_Arm_max_timing"] = ma_pk[f"{ax}_Moment_Arm"][1]
        for ax in "xyz":
            force_vectors[f"{ax}_Torque_time_to_impact"] = tq_pk[f"{ax}_Torque"][2]
        for ax in "xyz":
            force_vectors[f"{ax}_Moment_Arm_time_to_impact"] = ma_pk[f"{ax}_Moment_Arm"][2]
        inch = self._M_TO_IN
        force_vectors.update({
            "COP_disp_x_inch": r2(COP_ML_disp * inch).tolist(), "COP_disp_y_inch": r2(COP_AP_disp * inch).tolist(),
            "COM_disp_x_inch": r2(COM_disp_x * inch).tolist(), "COM_disp_y_inch": r2(COM_disp_y * inch).tolist(),
            "COM_disp_z_inch": r2(COM_disp_z * inch).tolist(),
        })
        for ax in "xyz":
            force_vectors[f"{ax}_Moment_Arm_max_inch"] = ma_pk[f"{ax}_Moment_Arm"][0] * inch
        for ax in "xyz":
            force_vectors[f"{ax}_Moment_Arm_inch"] = r2(ma[f"{ax}_Moment_Arm"] * inch).tolist()
        force_vectors.update({
            "COM_vel_ips": r2(COM_vel * inch).tolist(),
            "COP_vel_ips": r2(COP_vel * inch).tolist(),
        })

        # ---- small (frontend side-panel) force vectors; independent of pose2d
        sh_, sw_ = 75.0, 223.0
        kh_s = sh_ / lay["plate_h"]
        kw_s = sw_ / lay["plate_w"]
        base_y = int(sh_ - 30)
        sx = lambda c: (c[:, 0] + xo) * kw_s + 14.5
        s_lead, s_trail, s_comb = sx(COP_l), sx(COP_r), sx(COP)
        scale_s = max(float(np.max(Fz_t)), 1e-9) / base_y
        pxs = lambda a: np.trunc(a / scale_s)
        def end_pts(s_x, Fx_, Fz_):
            fxp = -pxs(Fx_)
            fzp = pxs(Fz_)
            return (np.trunc(s_x - fxp).astype(int).tolist(), np.trunc(base_y - fzp).astype(int).tolist())
        ex_l, ey_l = end_pts(s_lead, Fx_l, Fz_l)
        ex_r, ey_r = end_pts(s_trail, Fx_r, Fz_r)
        ex_t, ey_t = end_pts(s_comb, Fx_t, Fz_t)
        ys = [base_y] * n
        small_force_vectors = {
            "force_vectors_lead": {"starting_x": np.trunc(s_lead).astype(int).tolist(), "starting_y": ys,
                                   "ending_x": ex_l, "ending_y": ey_l},
            "force_vectors_trail": {"starting_x": np.trunc(s_trail).astype(int).tolist(), "starting_y": ys,
                                    "ending_x": ex_r, "ending_y": ey_r},
            "force_vectors_combined": {"starting_x": np.trunc(s_comb).astype(int).tolist(), "starting_y": ys,
                                       "ending_x": ex_t, "ending_y": ey_t},
        }

        # ---- back-view COM / COP trajectories
        bl_py = fh - (COP_l[:, 1] + 0.25) * k_h
        br_py = fh - (COP_r[:, 1] + 0.25) * k_h
        bc_py = fh - (COP[:, 1] + 0.25) * k_h
        bl_px = fw - (COP_l[:, 0] + xo) * k_w
        br_px = fw - (COP_r[:, 0] + xo) * k_w
        bc_px = fw - (COP[:, 0] + xo) * k_w
        bcom_py = fh - ((COM[:, 0] - offset_x + 0.25) * k_all)
        if self.plate_layout == "amti":
            bcom_px = (COM[:, 1] - offset_y + xo) * k_all            # (not mirrored in the AMTI branch)
        else:
            bcom_px = fw - ((COM[:, 1] - offset_y + xo) * k_all)
        brows_l, bcols_l = foot_box(bl_px[P1], bl_py[P1])
        brows_r, bcols_r = foot_box(br_px[P1], br_py[P1])
        COM_COP_traj_back_view = {
            "pixel_rows_foot_right_fixed_size": brows_r, "pixel_colm_foot_right_fixed_size": bcols_r,
            "pixel_rows_foot_left_fixed_size": brows_l, "pixel_colm_foot_left_fixed_size": bcols_l,
            "COP_l_pix_x_axes": bl_px.tolist(), "COP_l_pix_y_axes": bl_py.tolist(),
            "COP_r_pix_x_axes": br_px.tolist(), "COP_r_pix_y_axes": br_py.tolist(),
            "COP_pix_x_axes": bc_px.tolist(), "COP_pix_y_axes": bc_py.tolist(),
            "COM_pix_x_axes": bcom_px.tolist(), "COM_pix_y_axes": bcom_py.tolist(),
        }

        if rot:
            neg = lambda seq: (-np.asarray(seq, dtype=float)).tolist()
            for k in list(force_vectors_3d):
                if k.startswith(("x_", "y_")):
                    force_vectors_3d[k] = neg(force_vectors_3d[k])
            for suf in ("", "_l", "_r"):
                for pre in ("start_point_", "end_point_"):
                    for ax in ("X", "Y"):
                        key = f"{pre}{ax}{suf}"
                        force_vectors[key] = neg(force_vectors[key])

        critical_events = {}
        for k, v in (ctx["positions"] if isinstance(ctx.get("positions"), dict) else {}).items():
            critical_events[k] = v + time_index if isinstance(v, (int, float)) else v

        response = {
            "force_vectors": force_vectors,
            "forces": forces,
            "time": time.tolist(),
            "force_vectors_3d": force_vectors_3d,
            "values_for_COP_trajectory": values_for_COP_trajectory,
            "power": power,
            "energy": energy,
            "impulse": impulse,
            "small_force_vectors": small_force_vectors,
            "COM_COP_traj_back_view": COM_COP_traj_back_view,
            "critical_events": critical_events,
            "frame_offset": time_index,
        }
        if weight_shift is not None:
            response["weight_shift"] = weight_shift
        if COM_MODELS is not None:
            response["COM_MODELS"] = COM_MODELS
        return self._json_safe(response)

# ------------------------- Batch Processing -------------------------
if __name__ == "__main__":
    input_jsons = "./input/"
    output_jsons = "./output/"
    os.makedirs(output_jsons, exist_ok=True)
    json_list = [f for f in os.listdir(input_jsons) if f.endswith(".json")]

    for file in json_list:
        name, ext = os.path.splitext(file)
        if name[-3:] in ['_2d', 'ut1', 'seq', 'mal', 'raw', 'end', 'all', 'een', 'ing', '_cv', 'sim', '487', '489'] or name[-4:] == 'desc':
            logging.info(f"Skipped file: {name}")
            continue

        logging.info(f"Processing file: {name}")
        with open(os.path.join(input_jsons, file)) as f:
            forces_data = json.load(f)

        # Optional 2D stick figure next to the 3D file (<name>_2d.json). Only needed for the
        # pixel-based keys of the measured-equivalent response (see _build_forces_screen).
        pose2d = None
        p2d = os.path.join(input_jsons, f"{name}_2d.json")
        if os.path.exists(p2d):
            with open(p2d) as f:
                pose2d = json.load(f)
        else:
            logging.info("No %s_2d.json found; 2D-pixel keys will be omitted.", name)

        try:
            # Auto-detects movement type from configuration.swing.swingType
            # ("hitting" / "pitching" / "full swing") and picks the matching preset -- see
            # ForceVectors.from_swing_type. The pipeline runs ONCE; both responses are
            # formatted from the same computation.
            fv = ForceVectors.from_swing_type(forces_data)
            ctx = fv._run_pipeline(forces_data)

            # Build and serialize BOTH responses before touching disk, so a failure never
            # leaves a partial or empty output file behind.
            #   1) legacy response (unchanged from v11)
            #   2) measured-equivalent response (same keys as the measured-force script)
            legacy_txt = json.dumps(fv._format_legacy_response(ctx))
            screen_txt = json.dumps(fv._build_forces_screen(ctx, pose2d=pose2d), allow_nan=False)

            with open(os.path.join(output_jsons, f"{name}_predicted_forces.json"), "w") as f:
                f.write(legacy_txt)
            with open(os.path.join(output_jsons, f"{name}_forcesScreen.json"), "w") as f:
                f.write(screen_txt)
        except Exception as exc:
            # Hard failure on purpose (no degraded/partial outputs), but say which file
            # and why before the traceback so the cause is easy to trace in batch logs.
            logging.error("FAILED on %s: %s: %s", file, type(exc).__name__, exc)
            raise

    logging.info("All files processed.")
