from .math_helpers import *

def calculate(state):
    """Compute pitcher_metrics stage with cached dependencies."""
    F_axial_hand = state.get('F_axial_hand')
    F_axial_hand_vec = state.get('F_axial_hand_vec')
    F_axial_hand_vec_norm = state.get('F_axial_hand_vec_norm')
    F_axial_hand_vec_vis = state.get('F_axial_hand_vec_vis')
    F_combined_hand_vec = state.get('F_combined_hand_vec')
    F_combined_hand_vec_vis = state.get('F_combined_hand_vec_vis')
    F_hand_vec = state.get('F_hand_vec')
    F_tan_hand = state.get('F_tan_hand')
    F_tan_hand_vec = state.get('F_tan_hand_vec')
    F_tan_hand_vec_norm = state.get('F_tan_hand_vec_norm')
    F_tan_hand_vec_vis = state.get('F_tan_hand_vec_vis')
    Fx_t_est_hand_l = state.get('Fx_t_est_hand_l')
    Fx_t_est_hand_r = state.get('Fx_t_est_hand_r')
    Fy_t_est_hand_l = state.get('Fy_t_est_hand_l')
    Fy_t_est_hand_r = state.get('Fy_t_est_hand_r')
    Fz_t_est_hand_l = state.get('Fz_t_est_hand_l')
    Fz_t_est_hand_r = state.get('Fz_t_est_hand_r')
    N = state.get('N')
    Pitcher_performance_metrics = state.get('Pitcher_performance_metrics')
    UT_rot_speed = state.get('UT_rot_speed')
    W = state.get('W')
    actual_path = state.get('actual_path')
    ankle_l = state.get('ankle_l')
    ankle_r = state.get('ankle_r')
    arm_acceleration = state.get('arm_acceleration')
    arm_peak = state.get('arm_peak')
    arm_rot_speed = state.get('arm_rot_speed')
    c = state.get('c')
    d = state.get('d')
    direct_path = state.get('direct_path')
    dt = state.get('dt')
    efficiency = state.get('efficiency')
    elbow_angle = state.get('elbow_angle')
    elbow_l = state.get('elbow_l')
    elbow_r = state.get('elbow_r')
    elbow_velocity = state.get('elbow_velocity')
    end_x_axial_force_hand_3d = state.get('end_x_axial_force_hand_3d')
    end_x_combined_force_hand_3d = state.get('end_x_combined_force_hand_3d')
    end_x_tan_force_hand_3d = state.get('end_x_tan_force_hand_3d')
    end_y_axial_force_hand_3d = state.get('end_y_axial_force_hand_3d')
    end_y_combined_force_hand_3d = state.get('end_y_combined_force_hand_3d')
    end_y_tan_force_hand_3d = state.get('end_y_tan_force_hand_3d')
    end_z_axial_force_hand_3d = state.get('end_z_axial_force_hand_3d')
    end_z_combined_force_hand_3d = state.get('end_z_combined_force_hand_3d')
    end_z_tan_force_hand_3d = state.get('end_z_tan_force_hand_3d')
    energy_transfer = state.get('energy_transfer')
    extension_vector = state.get('extension_vector')
    fc = state.get('fc')
    forearm = state.get('forearm')
    forearm_peak_frame = state.get('forearm_peak_frame')
    frame_rate = state.get('frame_rate')
    hand_force_arrow_scale = state.get('hand_force_arrow_scale')
    hand_l = state.get('hand_l')
    hand_l_acc_x = state.get('hand_l_acc_x')
    hand_l_acc_y = state.get('hand_l_acc_y')
    hand_l_acc_z = state.get('hand_l_acc_z')
    hand_l_disp_x = state.get('hand_l_disp_x')
    hand_l_disp_y = state.get('hand_l_disp_y')
    hand_l_disp_z = state.get('hand_l_disp_z')
    hand_l_f = state.get('hand_l_f')
    hand_l_m = state.get('hand_l_m')
    hand_l_speed_x = state.get('hand_l_speed_x')
    hand_l_speed_y = state.get('hand_l_speed_y')
    hand_l_speed_z = state.get('hand_l_speed_z')
    hand_path = state.get('hand_path')
    hand_peak = state.get('hand_peak')
    hand_r = state.get('hand_r')
    hand_r_acc_x = state.get('hand_r_acc_x')
    hand_r_acc_y = state.get('hand_r_acc_y')
    hand_r_acc_z = state.get('hand_r_acc_z')
    hand_r_disp_x = state.get('hand_r_disp_x')
    hand_r_disp_y = state.get('hand_r_disp_y')
    hand_r_disp_z = state.get('hand_r_disp_z')
    hand_r_f = state.get('hand_r_f')
    hand_r_m = state.get('hand_r_m')
    hand_r_power_x = state.get('hand_r_power_x')
    hand_r_power_y = state.get('hand_r_power_y')
    hand_r_power_z = state.get('hand_r_power_z')
    hand_r_speed_X = state.get('hand_r_speed_X')
    hand_r_speed_Y = state.get('hand_r_speed_Y')
    hand_r_speed_Z = state.get('hand_r_speed_Z')
    hand_r_speed_x = state.get('hand_r_speed_x')
    hand_r_speed_y = state.get('hand_r_speed_y')
    hand_r_speed_z = state.get('hand_r_speed_z')
    hand_rot_speed = state.get('hand_rot_speed')
    hand_segment = state.get('hand_segment')
    hand_speed = state.get('hand_speed')
    hand_velocity = state.get('hand_velocity')
    hip_l = state.get('hip_l')
    hip_r = state.get('hip_r')
    idx = state.get('idx')
    is_pitcher = state.get('is_pitcher')
    knee_accel_window = state.get('knee_accel_window')
    knee_angle = state.get('knee_angle')
    knee_extension_acceleration = state.get('knee_extension_acceleration')
    knee_extension_velocity = state.get('knee_extension_velocity')
    knee_l = state.get('knee_l')
    knee_r = state.get('knee_r')
    label = state.get('label')
    late_window = state.get('late_window')
    lead_ankle = state.get('lead_ankle')
    lead_hip = state.get('lead_hip')
    lead_knee = state.get('lead_knee')
    lead_leg = state.get('lead_leg')
    lead_leg_bracing = state.get('lead_leg_bracing')
    lefty = state.get('lefty')
    movement_type = state.get('movement_type')
    nonzero = state.get('nonzero')
    norms = state.get('norms')
    p1 = state.get('p1')
    p4 = state.get('p4')
    p5 = state.get('p5')
    p7 = state.get('p7')
    p8 = state.get('p8')
    path_waste = state.get('path_waste')
    peak_acceleration = state.get('peak_acceleration')
    peak_bracing = state.get('peak_bracing')
    peak_hand_frame = state.get('peak_hand_frame')
    peak_idx = state.get('peak_idx')
    pelvis_peak = state.get('pelvis_peak')
    pelvis_rot_speed = state.get('pelvis_rot_speed')
    release_extension = state.get('release_extension')
    release_frame = state.get('release_frame')
    release_position = state.get('release_position')
    release_velocity_variability = state.get('release_velocity_variability')
    release_window = state.get('release_window')
    rotational_velocity_about_spine = state.get('rotational_velocity_about_spine')
    segment_lengths = state.get('segment_lengths')
    shank = state.get('shank')
    shoulder_l = state.get('shoulder_l')
    shoulder_peak_idx = state.get('shoulder_peak_idx')
    shoulder_peak_velocity = state.get('shoulder_peak_velocity')
    shoulder_r = state.get('shoulder_r')
    spine_high = state.get('spine_high')
    spine_low = state.get('spine_low')
    stability = state.get('stability')
    start_frame = state.get('start_frame')
    start_x_hand_3d = state.get('start_x_hand_3d')
    start_y_hand_3d = state.get('start_y_hand_3d')
    start_z_hand_3d = state.get('start_z_hand_3d')
    swing_type = state.get('swing_type')
    thigh = state.get('thigh')
    throwing_arm_rot_speed = state.get('throwing_arm_rot_speed')
    throwing_elbow = state.get('throwing_elbow')
    throwing_forearm_rot_speed = state.get('throwing_forearm_rot_speed')
    throwing_hand_rot_speed = state.get('throwing_hand_rot_speed')
    throwing_shoulder = state.get('throwing_shoulder')
    throwing_wrist = state.get('throwing_wrist')
    transfer = state.get('transfer')
    trunk_peak = state.get('trunk_peak')
    trunk_peak_idx = state.get('trunk_peak_idx')
    trunk_rot_velocity = state.get('trunk_rot_velocity')
    trunk_rotation_angle = state.get('trunk_rotation_angle')
    trunk_rotation_range = state.get('trunk_rotation_range')
    trunk_rotation_window = state.get('trunk_rotation_window')
    upper_arm = state.get('upper_arm')
    window = state.get('window')
    wrist_l = state.get('wrist_l')
    wrist_r = state.get('wrist_r')
    x = state.get('x')
    y = state.get('y')
    z = state.get('z')
    Pitcher_performance_metrics = {}
    swing_type = {"hitting": 0, "pitching": 1, "generic": 2}[movement_type]
    is_pitcher = swing_type == 1
    if is_pitcher:

        if lefty == 1:

            throwing_wrist = wrist_l
            throwing_elbow = elbow_l
            throwing_shoulder = shoulder_l

            lead_leg = ankle_r
            lead_knee = knee_r

        else:

            throwing_wrist = wrist_r
            throwing_elbow = elbow_r
            throwing_shoulder = shoulder_r

            lead_leg = ankle_l
            lead_knee = knee_l
    if is_pitcher:


        hand_velocity = np.gradient(
            throwing_wrist,
            dt,
            axis=0
        )


        hand_speed = np.linalg.norm(
            hand_velocity,
            axis=1
        )


        peak_hand_frame = np.argmax(
            hand_speed[p4:p7]
        ) + p4


        Pitcher_performance_metrics["peak_hand_speed"] = {

            "value_mps":
                float(
                    hand_speed[peak_hand_frame]
                ),

            "frame":
                int(
                    peak_hand_frame
                ),

            "time_to_release_ms":
                float(
                    (peak_hand_frame-p7)
                    /
                    frame_rate
                    *
                    1000
                )
        }
    if is_pitcher:


        release_position = throwing_wrist[p7]


        Pitcher_performance_metrics["release_height"] = {

            "value_m":
                float(
                    release_position[2]
                ),

            "frame":
                int(p7)

        }
    if is_pitcher:


        extension_vector = (
            throwing_wrist[p7]
            -
            lead_leg[p7]
        )


        release_extension = abs(
            extension_vector[0]
        )


        Pitcher_performance_metrics["release_extension"]={

            "value_m":
                float(
                    release_extension
                ),

            "frame":
                int(p7)

        }
    if is_pitcher:


        #################################################################
        # Throwing forearm rotational velocity (handedness corrected)
        #################################################################

        throwing_forearm_rot_speed = rotational_velocity_about_spine(

            throwing_elbow,

            throwing_wrist,

            spine_high,

            spine_low,

            dt,

            filter_win=41

        )


        #################################################################
        # Peak forearm rotational velocity before release
        #################################################################

        forearm_peak_frame = np.argmax(

            abs(
                throwing_forearm_rot_speed[p4:p7]
            )

        ) + p4


        peak_forearm_velocity = (

            throwing_forearm_rot_speed[
                forearm_peak_frame
            ]

        )
    if is_pitcher:


        #################################################################
        # Trunk rotational velocity (handedness neutral)
        #################################################################

        trunk_rot_velocity = abs(
            UT_rot_speed
        )


        #################################################################
        # Peak trunk rotation velocity
        #################################################################

        trunk_peak_idx = np.argmax(

            trunk_rot_velocity[p1:p8 + 1]

        ) + p1


        trunk_peak = {


            "peak_velocity_deg_s":

                float(
                    trunk_rot_velocity[trunk_peak_idx]
                ),


            "time_to_release_ms":

                float(

                    1000
                    *
                    (trunk_peak_idx - p7)
                    /
                    frame_rate

                )

        }


        #################################################################
        # Trunk rotation range
        #################################################################

        trunk_rotation_angle = np.cumsum(

            UT_rot_speed * dt

        )


        trunk_rotation_window = (

            trunk_rotation_angle[p1:p8 + 1]

        )


        trunk_rotation_range = float(

            np.max(
                trunk_rotation_window
            )

            -

            np.min(
                trunk_rotation_window
            )

        )


        Pitcher_performance_metrics["trunk_rotation"] = {


            "peak_velocity_deg_s":

                trunk_peak["peak_velocity_deg_s"],


            "peak_time_ms":

                trunk_peak["time_to_release_ms"],


            "rotation_range_deg":

                trunk_rotation_range

        }
    if is_pitcher:


        #################################################################
        # Throwing shoulder rotational velocity
        #################################################################

        throwing_arm_rot_speed = rotational_velocity_about_spine(

            throwing_shoulder,

            throwing_elbow,

            spine_high,

            spine_low,

            dt,

            filter_win=41

        )


        #################################################################
        # Peak shoulder rotation velocity
        #################################################################

        shoulder_peak_idx = np.argmax(

            abs(
                throwing_arm_rot_speed[p1:p8+1]
            )

        ) + p1


        shoulder_peak_velocity = (

            throwing_arm_rot_speed[
                shoulder_peak_idx
            ]

        )


        Pitcher_performance_metrics["shoulder_rotation"] = {


            "peak_velocity_deg_s":

                float(

                    abs(
                        shoulder_peak_velocity
                    )

                ),


            "peak_time_ms":

                float(

                    1000
                    *
                    (shoulder_peak_idx-p7)
                    /
                    frame_rate

                ),


            "peak_frame":

                int(
                    shoulder_peak_idx
                )

        }
    if is_pitcher:

        upper_arm = safe_normalize(
            throwing_shoulder
            -
            throwing_elbow
        )

        forearm = safe_normalize(
            throwing_wrist
            -
            throwing_elbow
        )

        elbow_angle = np.degrees(

            np.arccos(

                np.clip(

                    np.sum(
                        upper_arm *
                        forearm,
                        axis=1
                    ),

                    -1,
                    1

                )

            )

        )

        elbow_velocity = np.gradient(
            elbow_angle,
            dt
        )

        idx = np.argmax(
            np.abs(
                elbow_velocity[p4:p7]
            )
        ) + p4

        Pitcher_performance_metrics["elbow_extension_velocity"] = {

            "peak_velocity_deg_s":
                float(
                    elbow_velocity[idx]
                ),

            "peak_time_ms":
                float(
                    (idx-p7)
                    /
                    frame_rate
                    *
                    1000
                ),

            "extension_range_deg":
                float(
                    np.max(elbow_angle)
                    -
                    np.min(elbow_angle)
                )

        }
    if is_pitcher:

        if lefty == 1:

            lead_knee = knee_r
            lead_ankle = ankle_r
            lead_hip = hip_r

        else:

            lead_knee = knee_l
            lead_ankle = ankle_l
            lead_hip = hip_l


        thigh = lead_hip - lead_knee
        shank = lead_ankle - lead_knee

        thigh = thigh / (
            np.linalg.norm(thigh,axis=1,keepdims=True)+1e-8
        )

        shank = shank / (
            np.linalg.norm(shank,axis=1,keepdims=True)+1e-8
        )

        knee_angle = np.degrees(

            np.arccos(

                np.clip(

                    np.sum(

                        thigh*shank,

                        axis=1

                    ),

                    -1,

                    1

                )

            )

        )

        knee_extension_velocity = np.gradient(
            knee_angle,
            dt
        )

        knee_extension_acceleration = np.gradient(
            knee_extension_velocity,
            dt
        )

        release_frame = p7
        start_frame = max(p5, 0)

        window = slice(start_frame, release_frame + 1)

        knee_accel_window = knee_extension_acceleration[window]

        if knee_accel_window.size == 0:
            # Invalid/inverted event frames (e.g. p5 >= p7) - skip metric for this trial
            print(f"Warning: empty knee bracing window (p5={p5}, p7={p7}) - skipping lead_leg_bracing")
            peak_bracing = np.nan
            lead_leg_bracing = np.nan
        else:
            peak_bracing = np.max(knee_accel_window)

            lead_leg_bracing = np.clip(

                100
                *
                peak_bracing
                /
                15000,

                0,

                100

            )

        Pitcher_performance_metrics["lead_leg_bracing"] = {

            "score":

                float(lead_leg_bracing),

            "peak_extension_acceleration_deg_s2":

                float(peak_bracing)

        }
    if is_pitcher:


        #################################################################
        # Throwing forearm rotational velocity (handedness corrected)
        #################################################################

        throwing_hand_rot_speed = rotational_velocity_about_spine(

            throwing_elbow,

            throwing_wrist,

            spine_high,

            spine_low,

            dt,

            filter_win=41

        )


        #################################################################
        # Angular acceleration
        #################################################################

        arm_acceleration = np.gradient(

            throwing_hand_rot_speed,

            dt

        )


        #################################################################
        # Late acceleration window before release
        #################################################################

        late_window = slice(

            max(
                release_frame - 12,
                p1
            ),

            release_frame + 1

        )


        #################################################################
        # Peak acceleration magnitude
        #################################################################

        peak_idx = np.argmax(

            abs(
                arm_acceleration[late_window]
            )

        ) + late_window.start


        peak_acceleration = (

            arm_acceleration[
                peak_idx
            ]

        )


        Pitcher_performance_metrics["late_arm_acceleration"] = {


            "peak_deg_s2":

                float(

                    abs(
                        peak_acceleration
                    )

                ),


            "time_to_release_ms":

                float(

                    1000
                    *
                    (peak_idx-release_frame)
                    /
                    frame_rate

                ),


            "peak_frame":

                int(
                    peak_idx
                )

        }
    if is_pitcher:

        pelvis_peak=np.max(
            np.abs(
                pelvis_rot_speed[p1:p8+1]
            )
        )

        trunk_peak=np.max(
            np.abs(
                UT_rot_speed[p1:p8+1]
            )
        )

        arm_peak=np.max(
            np.abs(
                arm_rot_speed[p1:p8+1]
            )
        )

        hand_peak=np.max(
            np.abs(
                hand_rot_speed[p1:p8+1]
            )
        )

        transfer=np.mean([

            trunk_peak/pelvis_peak,

            arm_peak/trunk_peak,

            hand_peak/arm_peak

        ])

        energy_transfer=np.clip(

            100*transfer,

            0,

            100

        )

        Pitcher_performance_metrics["energy_transfer"]={

            "score":

                float(energy_transfer)

        }
    if is_pitcher:


        #################################################################
        # Throwing hand trajectory
        # Already dominance corrected:
        # throwing_wrist = wrist_l or wrist_r
        #################################################################

        hand_path = throwing_wrist


        #################################################################
        # Pitching acceleration phase
        # P4 = maximum leg lift / start of drive
        # P7 = ball release
        #################################################################

        hand_segment = hand_path[p4:p7+1]


        #################################################################
        # Actual hand path travelled
        #################################################################

        segment_lengths = np.linalg.norm(
            np.diff(
                hand_segment,
                axis=0
            ),
            axis=1
        )


        actual_path = np.sum(
            segment_lengths
        )


        #################################################################
        # Direct displacement
        #################################################################

        direct_path = np.linalg.norm(

            hand_segment[-1]
            -
            hand_segment[0]

        )


        #################################################################
        # Efficiency calculation
        #################################################################

        if actual_path > 1e-6:

            efficiency = (
                direct_path
                /
                actual_path
            )

        else:

            efficiency = 0.0



        #################################################################
        # Additional interpretation values
        #################################################################

        path_waste = (

            1.0
            -
            efficiency

        )


        Pitcher_performance_metrics["hand_path_efficiency"] = {


            "value":

                float(
                    efficiency
                ),


            "percent":

                float(
                    efficiency * 100
                ),


            "actual_path_length_m":

                float(
                    actual_path
                ),


            "direct_displacement_m":

                float(
                    direct_path
                ),


            "path_waste_percent":

                float(
                    path_waste * 100
                ),


            "window":

                {

                    "start_frame":
                        int(p4),

                    "release_frame":
                        int(p7)

                }

        }
    if is_pitcher:


        #################################################################
        # Throwing forearm rotational velocity (handedness corrected)
        #################################################################

        throwing_hand_rot_speed = rotational_velocity_about_spine(

            throwing_elbow,

            throwing_wrist,

            spine_high,

            spine_low,

            dt,

            filter_win=41

        )


        #################################################################
        # Release window stability
        #################################################################

        release_window = slice(

            max(release_frame - 3, p1),

            min(release_frame + 4, p8)

        )


        release_velocity_variability = np.std(

            throwing_hand_rot_speed[
                release_window
            ]

        )


        #################################################################
        # Stability score
        #################################################################

        stability = np.clip(

            100

            -

            release_velocity_variability / 10,

            0,

            100

        )


        if stability > 90:

            label = "Excellent"

        elif stability > 75:

            label = "Good"

        elif stability > 60:

            label = "Average"

        else:

            label = "Poor"


        Pitcher_performance_metrics["release_stability"] = {


            "score":

                float(
                    stability
                ),


            "classification":

                label,


            "release_velocity_sd_deg_s":

                float(
                    release_velocity_variability
                ),


            "release_window_frames":

                [

                    int(max(release_frame - 3, p1)),

                    int(min(release_frame + 4, p8))

                ]

        }
    if is_pitcher:



                # =========================================================
        # RIGHT HAND FORCE
        # =========================================================

        hand_r_f = np.zeros(hand_r.shape)
        hand_l_f = np.zeros(hand_l.shape)

        fc = 3  # Hz
        N = 4  # Order of Butterworth filter

        c, d = signal.butter(N, fc / (frame_rate / 2))

        hand_r_f[:, 0] = signal.filtfilt(
            c, d, hand_r[:, 0]
        )

        hand_r_f[:, 1] = signal.filtfilt(
            c, d, hand_r[:, 1]
        )

        hand_l_f[:, 0] = signal.filtfilt(
            c, d, hand_l[:, 0]
        )

        hand_l_f[:, 1] = signal.filtfilt(
            c, d, hand_l[:, 1]
        )


        fc = 5  # Hz
        N = 4

        c, d = signal.butter(N, fc / (frame_rate / 2))

        hand_r_f[:, 2] = signal.filtfilt(
            c, d, hand_r[:, 2]
        )

        hand_l_f[:, 2] = signal.filtfilt(
            c, d, hand_l[:, 2]
        )

        # ---------------------------------------------------------
        # Right hand displacement
        # ---------------------------------------------------------

        hand_r_disp_x = hand_r_f[:, 0] - hand_r_f[p1, 0]
        hand_r_disp_y = hand_r_f[:, 1] - hand_r_f[p1, 1]
        hand_r_disp_z = hand_r_f[:, 2] - hand_r_f[p1, 2]

        # ---------------------------------------------------------
        # Left hand displacement
        # ---------------------------------------------------------

        hand_l_disp_x = hand_l_f[:, 0] - hand_l_f[p1, 0]
        hand_l_disp_y = hand_l_f[:, 1] - hand_l_f[p1, 1]
        hand_l_disp_z = hand_l_f[:, 2] - hand_l_f[p1, 2]


        # ---------------------------------------------------------
        # Right hand speed
        # ---------------------------------------------------------

        hand_r_speed_x = np.gradient(hand_r_disp_x, dt)
        hand_r_speed_y = np.gradient(hand_r_disp_y, dt)
        hand_r_speed_z = np.gradient(hand_r_disp_z, dt)

        # ---------------------------------------------------------
        # Left hand speed
        # ---------------------------------------------------------

        hand_l_speed_x = np.gradient(hand_l_disp_x, dt)
        hand_l_speed_y = np.gradient(hand_l_disp_y, dt)
        hand_l_speed_z = np.gradient(hand_l_disp_z, dt)
        hand_r_speed_z = np.gradient(hand_r_disp_z, dt)


        # ---------------------------------------------------------
        # Right hand acceleration
        # ---------------------------------------------------------

        hand_r_acc_x = np.gradient(hand_r_speed_x, dt)
        hand_r_acc_y = np.gradient(hand_r_speed_y, dt)
        hand_r_acc_z = np.gradient(hand_r_speed_z, dt)

        # ---------------------------------------------------------
        # Left hand acceleration
        # ---------------------------------------------------------

        hand_l_acc_x = np.gradient(hand_l_speed_x, dt)
        hand_l_acc_y = np.gradient(hand_l_speed_y, dt)
        hand_l_acc_z = np.gradient(hand_l_speed_z, dt)


        # ---------------------------------------------------------
        # Right hand FORCE COMPONENTS
        # ---------------------------------------------------------
        hand_r_m = 0.61 / 100
        hand_l_m = 0.61 / 100

        Fx_t_est_hand_r = W * hand_r_m * hand_r_acc_x
        Fy_t_est_hand_r = W * hand_r_m * hand_r_acc_y
        Fz_t_est_hand_r = W * hand_r_m * hand_r_acc_z

        Fx_t_est_hand_l = W * hand_l_m * hand_l_acc_x
        Fy_t_est_hand_l = W * hand_l_m * hand_l_acc_y
        Fz_t_est_hand_l = W * hand_l_m * hand_l_acc_z


        # ---------------------------------------------------------
        # Combined right hand force magnitude
        # ---------------------------------------------------------

        F_hand_r = np.sqrt(
            Fx_t_est_hand_r ** 2
            + Fy_t_est_hand_r ** 2
            + Fz_t_est_hand_r ** 2
        )


        # ---------------------------------------------------------
        # Right hand velocity components
        # ---------------------------------------------------------

        hand_r_speed_X = np.diff(hand_r[:, 0]) * frame_rate
        hand_r_speed_X = np.insert(hand_r_speed_X, 0, 0)

        hand_r_speed_Y = np.diff(hand_r[:, 1]) * frame_rate
        hand_r_speed_Y = np.insert(hand_r_speed_Y, 0, 0)

        hand_r_speed_Z = np.diff(hand_r[:, 2]) * frame_rate
        hand_r_speed_Z = np.insert(hand_r_speed_Z, 0, 0)


        # ---------------------------------------------------------
        # Right hand power
        # ---------------------------------------------------------

        hand_r_power_x = Fx_t_est_hand_r * hand_r_speed_X
        hand_r_power_y = Fy_t_est_hand_r * hand_r_speed_Y
        hand_r_power_z = Fz_t_est_hand_r * hand_r_speed_Z

        hand_r_power = (
            hand_r_power_x
            + hand_r_power_y
            + hand_r_power_z
        )


        # ---------------------------------------------------------
        # Combined right hand speed
        # ---------------------------------------------------------

        hand_r_speed = np.sqrt(
            hand_r_speed_X ** 2
            + hand_r_speed_Y ** 2
            + hand_r_speed_Z ** 2
        )

        # =========================================================
        # THROWING HAND FORCE VECTORS - 3D
        # Handles right-handed and left-handed pitchers
        # =========================================================

        if lefty == 1:

            # =====================================================
            # LEFT HAND / LEFT ELBOW
            # =====================================================

            # -----------------------------------------------------
            # 1. Tangential direction
            #    Direction of instantaneous left-hand movement
            # -----------------------------------------------------

            F_tan_hand_vec_norm = hand_l_f[1:] - hand_l_f[:-1]

            norms = np.linalg.norm(F_tan_hand_vec_norm, axis=1)

            nonzero = norms > 0

            F_tan_hand_vec_norm[nonzero] /= (
                norms[nonzero, np.newaxis]
            )

            # Keep same number of frames
            F_tan_hand_vec_norm = np.vstack([
                np.zeros((1, 3)),
                F_tan_hand_vec_norm
            ])


            # -----------------------------------------------------
            # 2. Axial direction
            #    Left forearm axis: elbow -> hand
            # -----------------------------------------------------

            F_axial_hand_vec_norm = hand_l_f - elbow_l

            norms = np.linalg.norm(
                F_axial_hand_vec_norm,
                axis=1
            )

            nonzero = norms > 0

            F_axial_hand_vec_norm[nonzero] /= (
                norms[nonzero, np.newaxis]
            )


            # -----------------------------------------------------
            # 3. Force magnitudes
            # -----------------------------------------------------

            F_hand_vec = np.column_stack([
                Fx_t_est_hand_l,
                Fy_t_est_hand_l,
                Fz_t_est_hand_l
            ])


            # Tangential force magnitude
            F_tan_hand = np.sum(
                F_hand_vec * F_tan_hand_vec_norm,
                axis=1
            )


            # Axial force magnitude
            F_axial_hand = np.sum(
                F_hand_vec * F_axial_hand_vec_norm,
                axis=1
            )


            # -----------------------------------------------------
            # 4. Construct the three actual force vectors
            # -----------------------------------------------------

            # Tangential
            F_tan_hand_vec = (
                F_tan_hand_vec_norm
                * F_tan_hand[:, np.newaxis]
            )


            # Axial
            F_axial_hand_vec = (
                F_axial_hand_vec_norm
                * F_axial_hand[:, np.newaxis]
            )


            # Combined / resultant left-hand force
            F_combined_hand_vec = F_hand_vec


            # =====================================================
            # FRONTEND VISUALIZATION SCALING ONLY
            # =====================================================

            hand_force_arrow_scale = 0.01

            F_tan_hand_vec_vis = (
                F_tan_hand_vec
                * hand_force_arrow_scale
            )

            F_axial_hand_vec_vis = (
                F_axial_hand_vec
                * hand_force_arrow_scale
            )

            F_combined_hand_vec_vis = (
                F_combined_hand_vec
                * hand_force_arrow_scale
            )


            # =====================================================
            # LEFT HAND FORCE VECTOR START / END POINTS
            # =====================================================

            # Start all three arrows at the left hand
            start_x_hand_3d = hand_l[:, 0]
            start_y_hand_3d = hand_l[:, 1]
            start_z_hand_3d = hand_l[:, 2]


            # -----------------------------------------------------
            # Tangential force arrow end point
            # -----------------------------------------------------

            end_x_tan_force_hand_3d = (
                start_x_hand_3d + F_tan_hand_vec_vis[:, 0]
            )

            end_y_tan_force_hand_3d = (
                start_y_hand_3d + F_tan_hand_vec_vis[:, 1]
            )

            end_z_tan_force_hand_3d = (
                start_z_hand_3d + F_tan_hand_vec_vis[:, 2]
            )


            # -----------------------------------------------------
            # Axial force arrow end point
            # -----------------------------------------------------

            end_x_axial_force_hand_3d = (
                start_x_hand_3d + F_axial_hand_vec_vis[:, 0]
            )

            end_y_axial_force_hand_3d = (
                start_y_hand_3d + F_axial_hand_vec_vis[:, 1]
            )

            end_z_axial_force_hand_3d = (
                start_z_hand_3d + F_axial_hand_vec_vis[:, 2]
            )


            # -----------------------------------------------------
            # Combined force arrow end point
            # -----------------------------------------------------

            end_x_combined_force_hand_3d = (
                start_x_hand_3d + F_combined_hand_vec_vis[:, 0]
            )

            end_y_combined_force_hand_3d = (
                start_y_hand_3d + F_combined_hand_vec_vis[:, 1]
            )

            end_z_combined_force_hand_3d = (
                start_z_hand_3d + F_combined_hand_vec_vis[:, 2]
            )


        else:

            # =====================================================
            # RIGHT HAND / RIGHT ELBOW
            # =====================================================

            # -----------------------------------------------------
            # 1. Tangential direction
            # -----------------------------------------------------

            F_tan_hand_vec_norm = hand_r_f[1:] - hand_r_f[:-1]

            norms = np.linalg.norm(F_tan_hand_vec_norm, axis=1)

            nonzero = norms > 0

            F_tan_hand_vec_norm[nonzero] /= (
                norms[nonzero, np.newaxis]
            )

            F_tan_hand_vec_norm = np.vstack([
                np.zeros((1, 3)),
                F_tan_hand_vec_norm
            ])


            # -----------------------------------------------------
            # 2. Axial direction
            #    Right forearm axis: elbow -> hand
            # -----------------------------------------------------

            F_axial_hand_vec_norm = hand_r_f - elbow_r

            norms = np.linalg.norm(
                F_axial_hand_vec_norm,
                axis=1
            )

            nonzero = norms > 0

            F_axial_hand_vec_norm[nonzero] /= (
                norms[nonzero, np.newaxis]
            )


            # -----------------------------------------------------
            # 3. Force magnitudes
            # -----------------------------------------------------

            F_hand_vec = np.column_stack([
                Fx_t_est_hand_r,
                Fy_t_est_hand_r,
                Fz_t_est_hand_r
            ])


            F_tan_hand = np.sum(
                F_hand_vec * F_tan_hand_vec_norm,
                axis=1
            )


            F_axial_hand = np.sum(
                F_hand_vec * F_axial_hand_vec_norm,
                axis=1
            )


            # -----------------------------------------------------
            # 4. Construct force vectors
            # -----------------------------------------------------

            F_tan_hand_vec = (
                F_tan_hand_vec_norm
                * F_tan_hand[:, np.newaxis]
            )

            F_axial_hand_vec = (
                F_axial_hand_vec_norm
                * F_axial_hand[:, np.newaxis]
            )

            F_combined_hand_vec = F_hand_vec


            # =====================================================
            # FRONTEND VISUALIZATION SCALING ONLY
            # =====================================================

            hand_force_arrow_scale = 0.01

            F_tan_hand_vec_vis = (
                F_tan_hand_vec
                * hand_force_arrow_scale
            )

            F_axial_hand_vec_vis = (
                F_axial_hand_vec
                * hand_force_arrow_scale
            )

            F_combined_hand_vec_vis = (
                F_combined_hand_vec
                * hand_force_arrow_scale
            )


            # =====================================================
            # RIGHT HAND FORCE VECTOR START / END POINTS
            # =====================================================

            start_x_hand_3d = hand_r[:, 0]
            start_y_hand_3d = hand_r[:, 1]
            start_z_hand_3d = hand_r[:, 2]


            # Tangential
            end_x_tan_force_hand_3d = (
                start_x_hand_3d + F_tan_hand_vec_vis[:, 0]
            )

            end_y_tan_force_hand_3d = (
                start_y_hand_3d + F_tan_hand_vec_vis[:, 1]
            )

            end_z_tan_force_hand_3d = (
                start_z_hand_3d + F_tan_hand_vec_vis[:, 2]
            )


            # Axial
            end_x_axial_force_hand_3d = (
                start_x_hand_3d + F_axial_hand_vec_vis[:, 0]
            )

            end_y_axial_force_hand_3d = (
                start_y_hand_3d + F_axial_hand_vec_vis[:, 1]
            )

            end_z_axial_force_hand_3d = (
                start_z_hand_3d + F_axial_hand_vec_vis[:, 2]
            )


            # Combined
            end_x_combined_force_hand_3d = (
                start_x_hand_3d + F_combined_hand_vec_vis[:, 0]
            )

            end_y_combined_force_hand_3d = (
                start_y_hand_3d + F_combined_hand_vec_vis[:, 1]
            )

            end_z_combined_force_hand_3d = (
                start_z_hand_3d + F_combined_hand_vec_vis[:, 2]
            )


        # =========================================================
        # =========================================================
        # JSON RESPONSE
        # =========================================================

        M2C_force_vectors_hand_3d = {
            "start_3d": [
                [x, y, z]
                for x, y, z in zip(
                    start_x_hand_3d.tolist(),
                    start_y_hand_3d.tolist(),
                    start_z_hand_3d.tolist()
                )
            ],

            "end_rad_force_3d": [
                [x, y, z]
                for x, y, z in zip(
                    end_x_axial_force_hand_3d.tolist(),
                    end_y_axial_force_hand_3d.tolist(),
                    end_z_axial_force_hand_3d.tolist()
                )
            ],

            "end_tan_force_3d": [
                [x, y, z]
                for x, y, z in zip(
                    end_x_tan_force_hand_3d.tolist(),
                    end_y_tan_force_hand_3d.tolist(),
                    end_z_tan_force_hand_3d.tolist()
                )
            ],

            "end_combined_force_3d": [
                [x, y, z]
                for x, y, z in zip(
                    end_x_combined_force_hand_3d.tolist(),
                    end_y_combined_force_hand_3d.tolist(),
                    end_z_combined_force_hand_3d.tolist()
                )
            ],
        }
    out = locals().copy()
    out.pop("state", None)
    return out
