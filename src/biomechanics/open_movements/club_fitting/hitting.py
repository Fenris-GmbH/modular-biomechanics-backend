from .math_helpers import *

def calculate(state):
    json3d = state["json3d"]
    """Compute hitting stage with cached dependencies."""
    A_l = state.get('A_l')
    B_center = state.get('B_center')
    B_hat = state.get('B_hat')
    B_hat_flipped = state.get('B_hat_flipped')
    B_l = state.get('B_l')
    C = state.get('C')
    COM = state.get('COM')
    COM_f = state.get('COM_f')
    COM_f_acc_x = state.get('COM_f_acc_x')
    COM_f_acc_y = state.get('COM_f_acc_y')
    COM_f_acc_z = state.get('COM_f_acc_z')
    COM_f_disp_x = state.get('COM_f_disp_x')
    COM_f_disp_y = state.get('COM_f_disp_y')
    COM_f_disp_z = state.get('COM_f_disp_z')
    COM_f_speed_x = state.get('COM_f_speed_x')
    COM_f_speed_y = state.get('COM_f_speed_y')
    COM_f_speed_z = state.get('COM_f_speed_z')
    C_in = state.get('C_in')
    C_in_plane = state.get('C_in_plane')
    C_l = state.get('C_l')
    DEBUG = state.get('DEBUG')
    F_MH_normal = state.get('F_MH_normal')
    F_MH_swing_axis = state.get('F_MH_swing_axis')
    F_MH_tangential = state.get('F_MH_tangential')
    F_MH_vec = state.get('F_MH_vec')
    F_club_head = state.get('F_club_head')
    F_comb = state.get('F_comb')
    F_hand = state.get('F_hand')
    F_rad_shaft = state.get('F_rad_shaft')
    F_rad_vec = state.get('F_rad_vec')
    F_rad_vec_norm = state.get('F_rad_vec_norm')
    F_rad_vec_vis = state.get('F_rad_vec_vis')
    F_shaft_X = state.get('F_shaft_X')
    F_shaft_Y = state.get('F_shaft_Y')
    F_shaft_Z = state.get('F_shaft_Z')
    F_tan_shaft = state.get('F_tan_shaft')
    F_tan_vec = state.get('F_tan_vec')
    F_tan_vec_norm = state.get('F_tan_vec_norm')
    F_tan_vec_vis = state.get('F_tan_vec_vis')
    Fx_t_est = state.get('Fx_t_est')
    Fx_t_est_hand = state.get('Fx_t_est_hand')
    Fy_t_est = state.get('Fy_t_est')
    Fy_t_est_hand = state.get('Fy_t_est_hand')
    Fz_t_est = state.get('Fz_t_est')
    Fz_t_est_hand = state.get('Fz_t_est_hand')
    Grip = state.get('Grip')
    I_trans = state.get('I_trans')
    MH_force_moment_vec = state.get('MH_force_moment_vec')
    MH_point = state.get('MH_point')
    MHforce_moment_normal = state.get('MHforce_moment_normal')
    MHforce_moment_swing_axis = state.get('MHforce_moment_swing_axis')
    MHforce_moment_tangential = state.get('MHforce_moment_tangential')
    M_MH_normal = state.get('M_MH_normal')
    M_MH_swing_axis = state.get('M_MH_swing_axis')
    M_MH_tangential = state.get('M_MH_tangential')
    M_MH_vec = state.get('M_MH_vec')
    M_in = state.get('M_in')
    M_in_plane = state.get('M_in_plane')
    N = state.get('N')
    N_hat = state.get('N_hat')
    N_hat_safe = state.get('N_hat_safe')
    N_norm = state.get('N_norm')
    N_valid = state.get('N_valid')
    Nb = state.get('Nb')
    P_shaft = state.get('P_shaft')
    T_arr = state.get('T_arr')
    T_hat = state.get('T_hat')
    T_hat_valid = state.get('T_hat_valid')
    T_in_plane = state.get('T_in_plane')
    T_net = state.get('T_net')
    T_net_normal = state.get('T_net_normal')
    T_net_swing_axis = state.get('T_net_swing_axis')
    T_net_tangential = state.get('T_net_tangential')
    T_net_vec = state.get('T_net_vec')
    V = state.get('V')
    W = state.get('W')
    W_C_vec = state.get('W_C_vec')
    Wrist_torque = state.get('Wrist_torque')
    _butter_filt_cols = state.get('_butter_filt_cols')
    _closest = state.get('_closest')
    _cross_terms = state.get('_cross_terms')
    _debug_print = state.get('_debug_print')
    _fill_degenerate = state.get('_fill_degenerate')
    _hi = state.get('_hi')
    _i = state.get('_i')
    _lo = state.get('_lo')
    _local_com = state.get('_local_com')
    _mean_B_over_window = state.get('_mean_B_over_window')
    _p4c = state.get('_p4c')
    _p7c = state.get('_p7c')
    _plane_com = state.get('_plane_com')
    _plane_v = state.get('_plane_v')
    _pts = state.get('_pts')
    _rel = state.get('_rel')
    _signed_area_vec = state.get('_signed_area_vec')
    _sweep_sign_val = state.get('_sweep_sign_val')
    _t_cross_ms = state.get('_t_cross_ms')
    a = state.get('a')
    ankle_l = state.get('ankle_l')
    ankle_r = state.get('ankle_r')
    arc_width = state.get('arc_width')
    arc_width_inch = state.get('arc_width_inch')
    arm_l_COM = state.get('arm_l_COM')
    arm_l_m = state.get('arm_l_m')
    arm_r_COM = state.get('arm_r_COM')
    arm_r_m = state.get('arm_r_m')
    arr = state.get('arr')
    b = state.get('b')
    barrel_stability_index = state.get('barrel_stability_index')
    c = state.get('c')
    cb = state.get('cb')
    club_COM = state.get('club_COM')
    club_COM_2d = state.get('club_COM_2d')
    club_COM_2d_pixel = state.get('club_COM_2d_pixel')
    club_COM_3d = state.get('club_COM_3d')
    club_COM_acc_X = state.get('club_COM_acc_X')
    club_COM_acc_Y = state.get('club_COM_acc_Y')
    club_COM_acc_Z = state.get('club_COM_acc_Z')
    club_COM_acc_vec = state.get('club_COM_acc_vec')
    club_COM_hand_dist = state.get('club_COM_hand_dist')
    club_COM_speed = state.get('club_COM_speed')
    club_COM_speed_X = state.get('club_COM_speed_X')
    club_COM_speed_X_filt = state.get('club_COM_speed_X_filt')
    club_COM_speed_Y = state.get('club_COM_speed_Y')
    club_COM_speed_Y_filt = state.get('club_COM_speed_Y_filt')
    club_COM_speed_Z = state.get('club_COM_speed_Z')
    club_COM_speed_Z_filt = state.get('club_COM_speed_Z_filt')
    club_disp_X = state.get('club_disp_X')
    club_disp_Y = state.get('club_disp_Y')
    club_disp_Z = state.get('club_disp_Z')
    club_head = state.get('club_head')
    club_head_2d = state.get('club_head_2d')
    club_head_acc = state.get('club_head_acc')
    club_head_mass = state.get('club_head_mass')
    club_head_power = state.get('club_head_power')
    club_main = state.get('club_main')
    club_main_2d = state.get('club_main_2d')
    club_main_2d_pixel = state.get('club_main_2d_pixel')
    club_main_3d = state.get('club_main_3d')
    club_main_end_x = state.get('club_main_end_x')
    club_main_end_y = state.get('club_main_end_y')
    club_mass = state.get('club_mass')
    club_mid = state.get('club_mid')
    club_mid_2d = state.get('club_mid_2d')
    club_speed = state.get('club_speed')
    club_speed_X = state.get('club_speed_X')
    club_speed_Y = state.get('club_speed_Y')
    club_speed_Z = state.get('club_speed_Z')
    club_start_2d = state.get('club_start_2d')
    col = state.get('col')
    cos_angle = state.get('cos_angle')
    crossings = state.get('crossings')
    d = state.get('d')
    db = state.get('db')
    denom = state.get('denom')
    dotNB = state.get('dotNB')
    dotTB = state.get('dotTB')
    dotTN = state.get('dotTN')
    ds_dt = state.get('ds_dt')
    dt = state.get('dt')
    elbow_l = state.get('elbow_l')
    elbow_l_3d = state.get('elbow_l_3d')
    elbow_r = state.get('elbow_r')
    element = state.get('element')
    end_x_rad_force_3d = state.get('end_x_rad_force_3d')
    end_x_tan_force_3d = state.get('end_x_tan_force_3d')
    end_y_rad_force_3d = state.get('end_y_rad_force_3d')
    end_y_tan_force_3d = state.get('end_y_tan_force_3d')
    end_z_rad_force_3d = state.get('end_z_rad_force_3d')
    end_z_tan_force_3d = state.get('end_z_tan_force_3d')
    factor = state.get('factor')
    fc = state.get('fc')
    fc_hz = state.get('fc_hz')
    fit_plane = state.get('fit_plane')
    foot_l = state.get('foot_l')
    foot_l_COM = state.get('foot_l_COM')
    foot_l_m = state.get('foot_l_m')
    foot_r = state.get('foot_r')
    foot_r_COM = state.get('foot_r_COM')
    foot_r_m = state.get('foot_r_m')
    force_direction_interpretation = state.get('force_direction_interpretation')
    force_direction_ratio = state.get('force_direction_ratio')
    forearm_l_COM = state.get('forearm_l_COM')
    forearm_l_m = state.get('forearm_l_m')
    forearm_r_COM = state.get('forearm_r_COM')
    forearm_r_m = state.get('forearm_r_m')
    frame_rate = state.get('frame_rate')
    frame_window_ms = state.get('frame_window_ms')
    friction = state.get('friction')
    h = state.get('h')
    h_C_vec = state.get('h_C_vec')
    h_C_vec_filt = state.get('h_C_vec_filt')
    h_dot_C_vec = state.get('h_dot_C_vec')
    h_test = state.get('h_test')
    half_win = state.get('half_win')
    hand_COM_acc_x = state.get('hand_COM_acc_x')
    hand_COM_acc_y = state.get('hand_COM_acc_y')
    hand_COM_acc_z = state.get('hand_COM_acc_z')
    hand_l = state.get('hand_l')
    hand_l_3d = state.get('hand_l_3d')
    hand_l_COM = state.get('hand_l_COM')
    hand_l_disp_x = state.get('hand_l_disp_x')
    hand_l_disp_y = state.get('hand_l_disp_y')
    hand_l_disp_z = state.get('hand_l_disp_z')
    hand_l_f = state.get('hand_l_f')
    hand_l_m = state.get('hand_l_m')
    hand_l_speed_x = state.get('hand_l_speed_x')
    hand_l_speed_y = state.get('hand_l_speed_y')
    hand_l_speed_z = state.get('hand_l_speed_z')
    hand_power = state.get('hand_power')
    hand_power_x = state.get('hand_power_x')
    hand_power_y = state.get('hand_power_y')
    hand_power_z = state.get('hand_power_z')
    hand_r = state.get('hand_r')
    hand_r_3d = state.get('hand_r_3d')
    hand_r_COM = state.get('hand_r_COM')
    hand_r_m = state.get('hand_r_m')
    hand_speed = state.get('hand_speed')
    hand_speed_X = state.get('hand_speed_X')
    hand_speed_Y = state.get('hand_speed_Y')
    hand_speed_Z = state.get('hand_speed_Z')
    head_COM = state.get('head_COM')
    head_m = state.get('head_m')
    hide_COM = state.get('hide_COM')
    hip_l = state.get('hip_l')
    hip_r = state.get('hip_r')
    i = state.get('i')
    idx = state.get('idx')
    impact_frame = state.get('impact_frame')
    inPlaneMetrics = state.get('inPlaneMetrics')
    in_plane_rms = state.get('in_plane_rms')
    index_max_F_club_head = state.get('index_max_F_club_head')
    index_max_F_rad_shaft = state.get('index_max_F_rad_shaft')
    index_max_F_tan_shaft = state.get('index_max_F_tan_shaft')
    index_max_Grip = state.get('index_max_Grip')
    index_max_Wrist_torque = state.get('index_max_Wrist_torque')
    index_min_club_COM_speed = state.get('index_min_club_COM_speed')
    index_sorted = state.get('index_sorted')
    kinetic_order_frontend = state.get('kinetic_order_frontend')
    knee_l = state.get('knee_l')
    knee_r = state.get('knee_r')
    label = state.get('label')
    last_valid = state.get('last_valid')
    late_mask = state.get('late_mask')
    late_oop = state.get('late_oop')
    lead_forearm_shaft_angle_2d = state.get('lead_forearm_shaft_angle_2d')
    lead_forearm_shaft_angle_deg_2d = state.get('lead_forearm_shaft_angle_deg_2d')
    lefty = state.get('lefty')
    lever_arm_res = state.get('lever_arm_res')
    max_F_club_head = state.get('max_F_club_head')
    max_F_rad_shaft = state.get('max_F_rad_shaft')
    max_F_tan_shaft = state.get('max_F_tan_shaft')
    max_Grip = state.get('max_Grip')
    max_Wrist_torque = state.get('max_Wrist_torque')
    mean_club_power1 = state.get('mean_club_power1')
    midhand_3d = state.get('midhand_3d')
    nDigits = state.get('nDigits')
    n_frames_total = state.get('n_frames_total')
    nan_count = state.get('nan_count')
    neck = state.get('neck')
    nonzero_norms = state.get('nonzero_norms')
    norms = state.get('norms')
    omega_perp = state.get('omega_perp')
    oop_total = state.get('oop_total')
    other_index = state.get('other_index')
    out = state.get('out')
    outOfPlaneMetrics = state.get('outOfPlaneMetrics')
    out_norm_rms = state.get('out_norm_rms')
    out_tan_rms = state.get('out_tan_rms')
    p = state.get('p')
    p1 = state.get('p1')
    p3 = state.get('p3')
    p4 = state.get('p4')
    p5 = state.get('p5')
    p7 = state.get('p7')
    p8 = state.get('p8')
    peak_club_mechanics = state.get('peak_club_mechanics')
    peak_club_power = state.get('peak_club_power')
    peak_force_order = state.get('peak_force_order')
    peak_force_order_F_rad_shaft = state.get('peak_force_order_F_rad_shaft')
    peak_force_order_F_tan_shaft = state.get('peak_force_order_F_tan_shaft')
    peak_force_order_Wrist_torque = state.get('peak_force_order_Wrist_torque')
    peak_norm_force = state.get('peak_norm_force')
    peak_norm_idx = state.get('peak_norm_idx')
    peak_norm_time_ms = state.get('peak_norm_time_ms')
    peak_oop = state.get('peak_oop')
    peak_oop_idx = state.get('peak_oop_idx')
    peak_oop_time = state.get('peak_oop_time')
    peak_tan_force = state.get('peak_tan_force')
    peak_tan_idx = state.get('peak_tan_idx')
    peak_tan_time_ms = state.get('peak_tan_time_ms')
    peak_torque = state.get('peak_torque')
    peak_torque_idx = state.get('peak_torque_idx')
    peak_torque_time_ms = state.get('peak_torque_time_ms')
    pelvis = state.get('pelvis')
    point_to_line_distance = state.get('point_to_line_distance')
    point_to_line_projection = state.get('point_to_line_projection')
    pose = state.get('pose')
    power_index2 = state.get('power_index2')
    proj_club_COM = state.get('proj_club_COM')
    proj_club_COM_3d = state.get('proj_club_COM_3d')
    proj_club_COM_pixel = state.get('proj_club_COM_pixel')
    r_CH_vec = state.get('r_CH_vec')
    rms_inplane = state.get('rms_inplane')
    rms_net = state.get('rms_net')
    rms_oop = state.get('rms_oop')
    rms_sources = state.get('rms_sources')
    s_hat = state.get('s_hat')
    s_hat_filt = state.get('s_hat_filt')
    search_hi = state.get('search_hi')
    search_lo = state.get('search_lo')
    shaft_COM = state.get('shaft_COM')
    shaft_force_arrow_scale = state.get('shaft_force_arrow_scale')
    shaft_len = state.get('shaft_len')
    shaft_mass = state.get('shaft_mass')
    shaft_vec = state.get('shaft_vec')
    shank_l_COM = state.get('shank_l_COM')
    shank_l_m = state.get('shank_l_m')
    shank_r_COM = state.get('shank_r_COM')
    shank_r_m = state.get('shank_r_m')
    shoulder_l = state.get('shoulder_l')
    shoulder_r = state.get('shoulder_r')
    skull = state.get('skull')
    skullbase = state.get('skullbase')
    start_x_3d = state.get('start_x_3d')
    start_y_3d = state.get('start_y_3d')
    start_z_3d = state.get('start_z_3d')
    std_club_power1 = state.get('std_club_power1')
    swing_type = state.get('swing_type')
    t_ms = state.get('t_ms')
    taper_deg = state.get('taper_deg')
    taper_rad = state.get('taper_rad')
    th = state.get('th')
    thigh_l_COM = state.get('thigh_l_COM')
    thigh_l_m = state.get('thigh_l_m')
    thigh_r_COM = state.get('thigh_r_COM')
    thigh_r_m = state.get('thigh_r_m')
    time_to_impact = state.get('time_to_impact')
    time_to_impact_F_club_head = state.get('time_to_impact_F_club_head')
    time_to_impact_F_rad_shaft = state.get('time_to_impact_F_rad_shaft')
    time_to_impact_F_tan_shaft = state.get('time_to_impact_F_tan_shaft')
    time_to_impact_Grip = state.get('time_to_impact_Grip')
    time_to_impact_Wrist_torque = state.get('time_to_impact_Wrist_torque')
    torque_N = state.get('torque_N')
    torque_T = state.get('torque_T')
    torque_axis = state.get('torque_axis')
    torque_cancellation = state.get('torque_cancellation')
    torque_efficiency = state.get('torque_efficiency')
    trunk_COM = state.get('trunk_COM')
    trunk_m = state.get('trunk_m')
    u = state.get('u')
    v = state.get('v')
    v_MH = state.get('v_MH')
    v_proj = state.get('v_proj')
    v_proj_norm = state.get('v_proj_norm')
    valid_mask = state.get('valid_mask')
    vecs = state.get('vecs')
    vertical_power = state.get('vertical_power')
    wrist_angle_l = state.get('wrist_angle_l')
    wrist_angle_l_deg = state.get('wrist_angle_l_deg')
    wrist_l = state.get('wrist_l')
    wrist_l_3d = state.get('wrist_l_3d')
    wrist_lift = state.get('wrist_lift')
    wrist_r = state.get('wrist_r')
    wrist_speed = state.get('wrist_speed')
    wrist_speed_X = state.get('wrist_speed_X')
    wrist_speed_Y = state.get('wrist_speed_Y')
    wrist_speed_Z = state.get('wrist_speed_Z')
    wrist_speed_mph = state.get('wrist_speed_mph')
    wrist_sway = state.get('wrist_sway')
    wrist_thrust = state.get('wrist_thrust')
    wrist_vel_l_deg = state.get('wrist_vel_l_deg')
    x = state.get('x')
    x_2d = state.get('x_2d')
    x_factor = state.get('x_factor')
    x_factor_stretch = state.get('x_factor_stretch')
    y = state.get('y')
    z = state.get('z')
    if swing_type == 0:
        C = json3d["data_3d_bio"]["data"]["club"]


        ## Loading club data (3d input in meters, keeping old "_2d"-suffixed variable naming for compatibility)
        club_head_2d = []
        club_start_2d = []
        club_mid_2d = []

        for element in C:
            club_head_2d.append(C[element]["end"])
            club_start_2d.append(C[element]["start"])
            club_mid_2d.append(C[element]["mid"])

        club_head_2d = np.array(club_head_2d)
        club_start_2d = np.array(club_start_2d)
        club_mid_2d = np.array(club_mid_2d)

        club_start_2d_raw = club_start_2d

        # No PPMM multiplication needed - club input is already in meters:
        # club_start_2d = club_start_2d * json3d["configuration"]["swing"]["PPMM"]
        # club_mid_2d = club_mid_2d * json3d["configuration"]["swing"]["PPMM"]

        ## --- PIXEL-SPACE hand/wrist/elbow (unchanged from Script A) ---
        ## Kept for anything still consuming genuine 2D pixel coordinates, e.g. M2C_force_vectors_shaft
        ## used for frontend video overlay.
        # elbow_l_2d = []
        # hand_l_2d = []
        # hand_r_2d = []
        # wrist_l_2d = []
        # if lefty == 0:
        #     for element in json2d["data"]:
        #         elbow_l_2d.append(element[19])
        #         wrist_l_2d.append(element[20])
        #         hand_l_2d.append(element[21])
        #         hand_r_2d.append(element[17])
        # else:
        #     for element in json2d["data"]:
        #         elbow_l_2d.append(element[15])
        #         wrist_l_2d.append(element[16])
        #         hand_l_2d.append(element[17])
        #         hand_r_2d.append(element[21])
        # elbow_l_2d = np.array(elbow_l_2d)
        # wrist_l_2d = np.array(wrist_l_2d)
        # hand_l_2d = np.array(hand_l_2d)
        # hand_r_2d = np.array(hand_r_2d)

        ## --- 3D-METERS hand/wrist/elbow (new source: json3d pose) ---
        ## Used for lead_forearm_shaft_angle (needs to be paired with the 3D club data) and for
        ## any new 3D-only outputs (e.g. future stick-figure / avatar integration).
        elbow_l_3d = []
        hand_l_3d = []
        hand_r_3d = []
        wrist_l_3d = []

        pose = np.squeeze(np.array(json3d["data_3d_bio"]["data"]["pose"]))

        if lefty == 0:
            for idx in range(pose.shape[1]):
                elbow_l_3d.append(pose[19, idx])
                wrist_l_3d.append(pose[20, idx])
                hand_l_3d.append(pose[21, idx])
                hand_r_3d.append(pose[17, idx])
        else:
            for idx in range(pose.shape[1]):
                elbow_l_3d.append(pose[15, idx])
                wrist_l_3d.append(pose[16, idx])
                hand_l_3d.append(pose[17, idx])
                hand_r_3d.append(pose[21, idx])

        elbow_l_3d = np.array(elbow_l_3d)
        wrist_l_3d = np.array(wrist_l_3d)
        hand_l_3d = np.array(hand_l_3d)
        hand_r_3d = np.array(hand_r_3d)

        # Estimating the lead wrist shaft angle
        # Uses the 3D-meters wrist/elbow (paired with the 3D-meters club data) - intentional per 3D club update.
        # 03.06.24: Updated the lead forearm shaft angle with the wrist marker instead of the club start marker

        lead_forearm_shaft_angle_2d = []

        for idx in range(0, len(elbow_l_3d)):
            # Wrist -> club midpoint
            A_l = np.sqrt(
                (club_mid_2d[idx, 0] - wrist_l_3d[idx, 0]) ** 2
                + (club_mid_2d[idx, 1] - wrist_l_3d[idx, 1]) ** 2
                + (club_mid_2d[idx, 2] - wrist_l_3d[idx, 2]) ** 2
            )

            # Elbow -> club midpoint
            C_l = np.sqrt(
                (elbow_l_3d[idx, 0] - club_mid_2d[idx, 0]) ** 2
                + (elbow_l_3d[idx, 1] - club_mid_2d[idx, 1]) ** 2
                + (elbow_l_3d[idx, 2] - club_mid_2d[idx, 2]) ** 2
            )

            # Elbow -> wrist
            B_l = np.sqrt(
                (elbow_l_3d[idx, 0] - wrist_l_3d[idx, 0]) ** 2
                + (elbow_l_3d[idx, 1] - wrist_l_3d[idx, 1]) ** 2
                + (elbow_l_3d[idx, 2] - wrist_l_3d[idx, 2]) ** 2
            )

            # Numerical protection for acos
            cos_angle = (A_l ** 2 + B_l ** 2 - C_l ** 2) / (2 * A_l * B_l)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            lead_forearm_shaft_angle_2d.append(
                math.acos(cos_angle)
            )

        lead_forearm_shaft_angle_deg_2d = (
                np.array(lead_forearm_shaft_angle_2d) * 360 / (2 * np.pi)
        )

        ## --------------------------------------------- Club - Force & Power ------------------------------------------------

        # % Compute center of mass of each segment
        hide_COM = 1
        if hide_COM == 1:
            V = skull - skullbase
            factor = 0.5976
            head_COM = skull - V * factor
            V = neck - pelvis
            factor = 0.4486
            trunk_COM = neck - V * factor
            V = shoulder_r - elbow_r
            factor = 0.5772
            arm_r_COM = shoulder_r - V * factor
            V = elbow_r - wrist_r
            factor = 0.4574
            forearm_r_COM = elbow_r - V * factor
            V = wrist_r - hand_r
            factor = 0.79
            hand_r_COM = wrist_r - V * factor
            V = shoulder_l - elbow_l
            factor = 0.5772
            arm_l_COM = shoulder_l - V * factor
            V = elbow_l - wrist_l
            factor = 0.4574
            forearm_l_COM = elbow_l - V * factor
            V = wrist_l - hand_l
            factor = 0.79
            hand_l_COM = wrist_l - V * factor
            V = hip_r - knee_r
            factor = 0.4095
            thigh_r_COM = hip_r - V * factor
            V = knee_r - ankle_r
            factor = 0.4459
            shank_r_COM = knee_r - V * factor
            V = ankle_r - foot_r
            factor = 0.4415
            foot_r_COM = ankle_r - V * factor
            V = hip_l - knee_l
            factor = 0.4095
            thigh_l_COM = hip_l - V * factor
            V = knee_l - ankle_l
            factor = 0.4459
            shank_l_COM = knee_l - V * factor
            V = ankle_l - foot_l
            factor = 0.4415
            foot_l_COM = ankle_l - V * factor

            # Compute mass (%) of each segment
            head_m = 6.94 / 100
            trunk_m = 43.36 / 100
            arm_r_m = 2.71 / 100
            forearm_r_m = 1.62 / 100
            hand_r_m = 0.61 / 100
            thigh_r_m = 14.16 / 100
            shank_r_m = 4.33 / 100
            foot_r_m = 1.37 / 100
            arm_l_m = 2.71 / 100
            forearm_l_m = 1.62 / 100
            hand_l_m = 0.61 / 100
            thigh_l_m = 14.16 / 100
            shank_l_m = 4.33 / 100
            foot_l_m = 1.37 / 100

            COM = (
                head_COM * head_m
                + trunk_COM * trunk_m
                + arm_r_COM * arm_r_m
                + forearm_r_COM * forearm_r_m
                + hand_r_COM * hand_r_m
                + thigh_r_COM * thigh_r_m
                + shank_r_COM * shank_r_m
                + foot_r_COM * foot_r_m
                + arm_l_COM * arm_l_m
                + forearm_l_COM * forearm_l_m
                + hand_l_COM * hand_l_m
                + thigh_l_COM * thigh_l_m
                + shank_l_COM * shank_l_m
                + foot_l_COM * foot_l_m
            )

            # COM filtering
            dt = 1 / frame_rate
            COM_f = np.zeros(COM.shape)
            fc = 3  # Hz
            N = 4  # Order of butterworth filter

            c, d = signal.butter(N, fc / (frame_rate / 2))
            COM_f[:, 0] = signal.filtfilt(
                c, d, COM[:, 0]
            )  # Vfilt_array3(COM[:, 0], frame_rate, 3)
            COM_f[:, 1] = signal.filtfilt(
                c, d, COM[:, 1]
            )  # Vfilt_array3(COM[:, 1], frame_rate, 3)

            fc = 5  # Hz
            N = 4  # Order of butterworth filter
            c, d = signal.butter(N, fc / (frame_rate / 2))
            COM_f[:, 2] = signal.filtfilt(
                c, d, COM[:, 2]
            )  # Vfilt_array3(COM[:, 2], frame_rate, 5)

            # COM filtered displacements (m)
            COM_f_disp_x = COM_f[:, 0] - COM_f[p1, 0]
            COM_f_disp_y = COM_f[:, 1] - COM_f[p1, 1]
            COM_f_disp_z = COM_f[:, 2] - COM_f[p1, 2]

            # COM filtered speeds (m/s)
            COM_f_speed_x = np.gradient(COM_f_disp_x, dt)
            COM_f_speed_y = np.gradient(COM_f_disp_y, dt)
            COM_f_speed_z = np.gradient(COM_f_disp_z, dt)
            COM_f_speed = np.sqrt(COM_f_speed_x**2 + COM_f_speed_y**2 + COM_f_speed_z**2)

            # COM filtered accelerations (m/s^2)
            COM_f_acc_x = np.gradient(COM_f_speed_x, dt)
            COM_f_acc_y = np.gradient(COM_f_speed_y, dt)
            COM_f_acc_z = np.gradient(COM_f_speed_z, dt)

            # Estimated combined GRF
            Fx_t_est = (W) * COM_f_acc_x
            Fy_t_est = (W) * COM_f_acc_y
            Fz_t_est = (W) * (COM_f_acc_z + 9.81)
            Fz_t_est[Fz_t_est < 0] = 0

        F_comb = []
        for i in range(0, len(Fx_t_est)):
            F_comb.append(math.sqrt(Fx_t_est[i] ** 2 + Fy_t_est[i] ** 2 + Fz_t_est[i] ** 2))
        F_comb = np.array(F_comb)

        # vertical power / ground power
        # 06.07.2026: Updated to true instantaneous power (dot product of force and velocity vectors)
        # to introduce direction, replacing the magnitude-product formulation. Intentional per update.
        vertical_power = (
                Fx_t_est * COM_f_speed_x
                + Fy_t_est * COM_f_speed_y
                + Fz_t_est * COM_f_speed_z
        )

        hand_l_f = np.zeros(hand_l.shape)
        fc = 3  # Hz
        N = 4  # Order of butterworth filter

        c, d = signal.butter(N, fc / (frame_rate / 2))
        hand_l_f[:, 0] = signal.filtfilt(
            c, d, hand_l[:, 0]
        )  # Vfilt_array3(COM[:, 0], frame_rate, 3)
        hand_l_f[:, 1] = signal.filtfilt(
            c, d, hand_l[:, 1]
        )  # Vfilt_array3(COM[:, 1], frame_rate, 3)

        fc = 5  # Hz
        N = 4  # Order of butterworth filter
        c, d = signal.butter(N, fc / (frame_rate / 2))
        hand_l_f[:, 2] = signal.filtfilt(
            c, d, hand_l[:, 2]
        )  # Vfilt_array3(COM[:, 2], frame_rate, 5)

        # Left hand displacement
        hand_l_disp_x = hand_l_f[:, 0] - hand_l_f[p1, 0]
        hand_l_disp_y = hand_l_f[:, 1] - hand_l_f[p1, 1]
        hand_l_disp_z = hand_l_f[:, 2] - hand_l_f[p1, 2]

        # Left hand speed (m/s)
        hand_l_speed_x = np.gradient(hand_l_disp_x, dt)
        hand_l_speed_y = np.gradient(hand_l_disp_y, dt)
        hand_l_speed_z = np.gradient(hand_l_disp_z, dt)

        # Left hand acceleration (m/s^2)
        hand_COM_acc_x = np.gradient(hand_l_speed_x, dt)
        hand_COM_acc_y = np.gradient(hand_l_speed_y, dt)
        hand_COM_acc_z = np.gradient(hand_l_speed_z, dt)

        # Estimate the Left hand FORCE COMPONENTS
        # 07.11.23: Adding: W* --> to actually using the weight for the hand and not only the weight percent
        Fx_t_est_hand = W * hand_l_m * hand_COM_acc_x
        Fy_t_est_hand = W * hand_l_m * hand_COM_acc_y
        Fz_t_est_hand = W * hand_l_m * hand_COM_acc_z

        # 03.06.2024: Updated the combination formula for the combined hand force
        F_hand = []
        for i in range(0, len(Fx_t_est_hand)):
            F_hand.append(
                math.sqrt(
                    Fx_t_est_hand[i] ** 2 + Fy_t_est_hand[i] ** 2 + Fz_t_est_hand[i] ** 2
                )
            )
        F_hand = np.array(F_hand)

        # Three components of hand speed
        hand_speed_X = np.diff(hand_l[:, 0]) * frame_rate  # m/s
        hand_speed_X = np.insert(hand_speed_X, 0, 0)
        hand_speed_Y = np.diff(hand_l[:, 1]) * frame_rate
        hand_speed_Y = np.insert(hand_speed_Y, 0, 0)
        hand_speed_Z = np.diff(hand_l[:, 2]) * frame_rate
        hand_speed_Z = np.insert(hand_speed_Z, 0, 0)

        # Calculate hand power
        hand_power_x = Fx_t_est_hand * hand_speed_X
        hand_power_y = Fy_t_est_hand * hand_speed_Y
        hand_power_z = Fz_t_est_hand * hand_speed_Z  

        # 03.06.2024: Updated the combination formula for the combined hand speed
        hand_speed = []
        for i in range(0, len(hand_speed_X)):
            hand_speed.append(
                math.sqrt(
                    hand_speed_X[i] ** 2 + hand_speed_Y[i] ** 2 + hand_speed_Z[i] ** 2
                )
            )
        hand_speed = np.array(hand_speed)

        # 06.07.2026: Updated to true instantaneous power (dot product of force and velocity components),
        # to introduce direction. Intentional per update.
        hand_power = (
                hand_power_x
                + hand_power_y
                + hand_power_z
        )


        # ------------------------------------------------- club mechanics ------------------------------------------------------
        # Club coordinates are now supplied directly in meters (3D), so no PPMM pixel-to-mm conversion is needed.
        nDigits = club_head_2d.shape
        nDigits = nDigits[1]
        # ballSizePixel = []
        # if len(json3d['data'].keys()) == 3:
        #     ballData = json3d['data']['ball']
        #     for element in ballData:
        #         if ballSizePixel == []:
        #             ballSizePixel = (ballData[element][2] + ballData[element][3]) / 2  # pix
        #     ballSizeReal = 42.67  # mm
        #     PPMM = ballSizeReal / ballSizePixel
        # else:
        #     PPMM = 1
        # PPMM = 1
        # nDigits = 3
        club_head = np.zeros((len(hand_power_z), nDigits))
        club_mid = np.zeros((len(hand_power_z), nDigits))
        club_main = np.zeros((len(hand_power_z), nDigits))

        # Loading club marker data
        other_index = 0
        for element in C:  # i in range(0, len(hand_power_z)):
            if other_index < len(hand_power_z):
                club_main[other_index] = C[element]["start"]  # * PPMM / 1000
                club_head[other_index] = C[element]["end"]  # * PPMM / 1000
                club_mid[other_index] = C[element]["mid"]  # * PPMM / 1000

            other_index += 1

        # Club coordinates already in meters - no pixel scaling needed
        club_head = np.array(club_head)  # * PPMM / 1080
        club_main = np.array(club_main)  # * PPMM / 1080
        club_mid = np.array(club_mid)  # * PPMM / 1080
        club_main_2d = np.array(club_main)  # * json3d["configuration"]["swing"]["PPMM"]

        # Three different methods to determine the hand club speed
        club_disp_X = club_head_2d[1:, 0] - club_head_2d[:-1, 0]
        club_disp_Y = club_head_2d[1:, 1] - club_head_2d[:-1, 1]
        club_disp_Z = club_head_2d[1:, 2] - club_head_2d[:-1, 2]  # 06.07.2026: 3D club input now available
        club_disp_X = np.insert(club_disp_X, 0, 0)
        club_disp_Y = np.insert(club_disp_Y, 0, 0)
        club_disp_Z = np.insert(club_disp_Z, 0, 0)  # 06.07.2026: 3D club input now available

        club_speed_X = np.gradient(club_head[:, 0], dt)
        club_speed_Y = np.gradient(club_head[:, 1], dt)
        club_speed_Z = np.gradient(club_head[:, 2], dt)
        club_speed = np.sqrt(club_speed_X**2 + club_speed_Y**2 + club_speed_Z**2)  # m/s

        if json3d["data_3d_bio"]["configuration"]["swing"]["BatType"] == "driver":
            fc = 20
        else:
            fc = 5

        # Defining shaft and club head masses, based on values from the real world
        if json3d["data_3d_bio"]["configuration"]["swing"]["BatType"] == "driver":  # == 1:
            shaft_mass = 0.059  # in kg
            club_head_mass = 0.192  # in kg
        else:
            shaft_mass = 0.120  # in kg
            club_head_mass = 0.256  # in kg

        club_mass = shaft_mass + club_head_mass  # in kg
        shaft_COM = club_mid

        # Compute club COM
        club_COM = (shaft_COM * shaft_mass + club_head * club_head_mass) / club_mass

        # Compute club COM speed
        club_COM_speed_X = np.gradient(club_COM[:, 0], dt)
        club_COM_speed_Y = np.gradient(club_COM[:, 1], dt)
        club_COM_speed_Z = np.gradient(club_COM[:, 2], dt)
        club_COM_speed = np.sqrt(club_COM_speed_X**2 + club_COM_speed_Y**2 + club_COM_speed_Z**2)  # m/s

        if json3d["data_3d_bio"]["configuration"]["swing"]["swingType"] == "pitching":
            index_min_club_COM_speed = np.argmin(club_COM_speed[p3:p5]) + p3
        else:
            index_min_club_COM_speed = np.argmin(club_COM_speed[p1 + 10 : p7]) + p1 + 10
        # Invert values up to index_min_club_COM_speed
        club_COM_speed[:index_min_club_COM_speed] = -club_COM_speed[
            :index_min_club_COM_speed
        ]

        # Apply filtering to club_COM_speed
        fc = 5  # Hz
        N = 4  # Order of butterworth filter
        c, d = signal.butter(N, fc / (frame_rate / 2))
        club_COM_speed = signal.filtfilt(c, d, club_COM_speed)

        club_COM_acc = np.gradient(club_COM_speed, dt)

        club_speed = np.array(club_speed)/ 2.23693629
        club_speed = signal.filtfilt(c, d, club_speed)
        club_head_acc = np.gradient(club_speed, dt)

        # Calculating the club head force
        F_club_head = club_head_mass * club_head_acc  # N

        # Calculating the club head power
        club_head_power = F_club_head * club_speed

        peak_club_power = max(club_head_power[p4 : p7 + 1])
        mean_club_power = np.mean(club_head_power[p4 : p7 + 1])
        std_club_power = np.std(club_head_power[p4 : p7 + 1])

        # third method to calculate the power index cv speed
        if json3d["data_3d_bio"]["configuration"]["swing"]["BatType"] == "driver":
            mean_club_power1 = 4457
            std_club_power1 = 497
        else:
            mean_club_power1 = 7342
            std_club_power1 = 2863

        # 29.01.2024: this is the new calculations for the power index
        power_index2 = peak_club_power / (mean_club_power1 + std_club_power1 * 4) * 100

        # Defining over all club COM
        club_COM = (shaft_COM * shaft_mass + club_head * club_head_mass) / club_mass

        # Calculating Club COM velocity
        club_COM_speed_X = np.gradient(club_COM[:, 0], dt)
        club_COM_speed_Y = np.gradient(club_COM[:, 1], dt)
        club_COM_speed_Z = np.gradient(club_COM[:, 2], dt)

        fc = 4  # Hz
        N = 4  # Order of butterworth filter
        c, d = signal.butter(N, fc / (frame_rate / 2))

        # Filtering Club COM velocity
        club_COM_speed_X_filt = signal.filtfilt(c, d, club_COM_speed_X)
        club_COM_speed_Y_filt = signal.filtfilt(c, d, club_COM_speed_Y)
        club_COM_speed_Z_filt = signal.filtfilt(c, d, club_COM_speed_Z)
        club_COM_speed1 = np.sqrt(club_COM_speed_X_filt**2 + club_COM_speed_Y_filt**2 + club_COM_speed_Z_filt**2)

        club_COM_acc_X = np.gradient(club_COM_speed_X_filt, dt)
        club_COM_acc_Y = np.gradient(club_COM_speed_Y_filt, dt)
        club_COM_acc_Z = np.gradient(club_COM_speed_Z_filt, dt)

        # Calculating the Forces at the shaft in 3D
        F_shaft_X = club_mass * club_COM_acc_X
        F_shaft_Y = club_mass * club_COM_acc_Y
        F_shaft_Z = club_mass * club_COM_acc_Z
        F_shaft1 = np.sqrt(F_shaft_X**2 + F_shaft_Y**2 + F_shaft_Z**2)

        # 06.07.2026: Updated Shaft power to true instantaneous power (dot product), to introduce direction.
        # Intentional per update.
        P_shaft = (
                F_shaft_X * club_COM_speed_X_filt
                + F_shaft_Y * club_COM_speed_Y_filt
                + F_shaft_Z * club_COM_speed_Z_filt
        )

        # Distance between lead hand and club COM
        club_COM_hand_dist = []
        for i in range(0, len(club_main)):
            club_COM_hand_dist.append(
                np.sqrt(
                    (club_main[i, 0] - club_COM[i, 0]) ** 2 +
                    (club_main[i, 1] - club_COM[i, 1]) ** 2 +
                    (club_main[i, 2] - club_COM[i, 2]) ** 2
                )
            )
        club_COM_hand_dist = np.array(club_COM_hand_dist)

        ## ====================================================
        ## KWON HAND-CLUB INVERSE DYNAMICS
        ## Y.-H. Kwon, "Hand-Club Interaction: 1. Inverse Dynamics" (Dec 2017 / rev. Feb 2018)
        ## Point of interaction (MH) = actual mid-hand landmark (hand_l_3d/hand_r_3d)
        ## Swing plane = continuously moving frame, re-fit every frame from the MH path (Frenet-Serret
        ## tangential/normal/binormal) - matches Kwon's Figs. 3-6 "moving MH frame".
        ## ============================================================================================

        DEBUG = False

        def _debug_print(label, arr):
            if not DEBUG:
                return
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0:
                print(f"[DEBUG] {label:40s} EMPTY ARRAY")
                return
            nan_count = int(np.isnan(arr).sum())
            print(
                f"[DEBUG] {label:40s} shape={str(arr.shape):14s} "
                f"min={np.nanmin(arr):10.4g} max={np.nanmax(arr):10.4g} "
                f"mean={np.nanmean(arr):10.4g} nan_count={nan_count}"
            )

        MH_point = (hand_l_3d + hand_r_3d) / 2  # Kwon's mid-hand (MH) point - actual hand landmarks
        _debug_print("MH_point", MH_point)

        def _butter_filt_cols(x_2d, fc_hz):
            Nb = 4
            cb, db = signal.butter(Nb, fc_hz / (frame_rate / 2))
            out = np.zeros_like(x_2d)
            for col in range(x_2d.shape[1]):
                out[:, col] = signal.filtfilt(cb, db, x_2d[:, col])
            return out

        def _fill_degenerate(vecs, valid_mask):
            # Forward-fill then back-fill frames where a unit vector could not be reliably defined
            # (near-zero MH speed / near-zero path curvature - e.g. near the top of the backswing).
            out = vecs.copy()
            last_valid = None
            for i in range(len(out)):
                if valid_mask[i]:
                    last_valid = out[i].copy()
                elif last_valid is not None:
                    out[i] = last_valid
            last_valid = None
            for i in range(len(out) - 1, -1, -1):
                if valid_mask[i]:
                    last_valid = out[i].copy()
                elif not valid_mask[i] and last_valid is not None:
                    out[i] = last_valid
            return out

        # --- Net MH force (Eq. 12): F_MH = m_C * a_C - W_C (full 3D vector, gravity handled properly) ---
        club_COM_acc_vec = np.column_stack([club_COM_acc_X, club_COM_acc_Y, club_COM_acc_Z])
        W_C_vec = np.tile(np.array([0.0, 0.0, -9.81 * club_mass]), (len(club_COM), 1))
        F_MH_vec = club_mass * club_COM_acc_vec - W_C_vec
        _debug_print("F_MH_vec", F_MH_vec)

        # --- Club angular momentum about its own COM, via shaft-direction vector ---
        # Slender axisymmetric-rod assumption (same assumption already implicit in the legacy I=mL^2/12
        # formula): transverse MOI is identical about any axis perpendicular to the shaft, so the shaft's
        # own roll/twist about its long axis - unmeasurable from a 3-collinear-marker club track - is not
        # required to compute h_C this way.
        shaft_vec = club_head - club_main
        shaft_len = np.linalg.norm(shaft_vec, axis=1)
        s_hat = shaft_vec / shaft_len[:, np.newaxis]
        s_hat_filt = _butter_filt_cols(s_hat, fc_hz=10)
        s_hat_filt = s_hat_filt / np.linalg.norm(s_hat_filt, axis=1)[:, np.newaxis]
        ds_dt = np.gradient(s_hat_filt, dt, axis=0)
        omega_perp = np.cross(s_hat_filt, ds_dt)  # club angular velocity, perpendicular to the shaft

        I_trans = club_mass * shaft_len**2 / 12  # transverse MOI (slender rod about its own COM)
        h_C_vec = I_trans[:, np.newaxis] * omega_perp
        h_C_vec_filt = _butter_filt_cols(h_C_vec, fc_hz=8)
        h_dot_C_vec = np.gradient(h_C_vec_filt, dt, axis=0)
        _debug_print("shaft_len", shaft_len)
        _debug_print("omega_perp", omega_perp)
        _debug_print("h_C_vec", h_C_vec)
        _debug_print("h_dot_C_vec", h_dot_C_vec)

        # --- Point of interaction: r_CH = MH position relative to club COM ---
        r_CH_vec = MH_point - club_COM
        _debug_print("r_CH_vec", r_CH_vec)

        # --- Moment produced by the net MH force about club's COM (Kwon Fig. 5 quantity) ---
        MH_force_moment_vec = np.cross(r_CH_vec, F_MH_vec)
        _debug_print("MH_force_moment_vec", MH_force_moment_vec)

        # --- Net MH moment (Eq. 13) - couple action of the hands (Kwon Fig. 4 quantity) ---
        M_MH_vec = h_dot_C_vec - MH_force_moment_vec
        _debug_print("M_MH_vec", M_MH_vec)

        # --- Net club moment (Kwon Fig. 6 quantity) - equals h_dot_C_vec by construction ---
        T_net_vec = M_MH_vec + MH_force_moment_vec
        _debug_print("T_net_vec", T_net_vec)

        # --- Moving MH reference frame - windowed local plane fit ---
        # The swing axis (B_hat) is now derived from a small spatial window of nearby MH positions via
        # fit_plane() (same SVD approach already used for BS_plane/DS_plane), rather than from instantaneous
        # acceleration (2nd derivative of noisy marker position). This is far more stable exactly where hand
        # speed passes through ~zero (e.g. top of backswing), which is where the previous acceleration-based
        # curvature estimate produced spurious high-frequency spikes once tested against real data.
        frame_window_ms = 100.0  # +/- half-window in ms - tunable
        half_win = max(3, int(round(frame_window_ms / 1000.0 * frame_rate)))

        B_hat = np.zeros((len(MH_point), 3))
        B_center = np.zeros((len(MH_point), 3))  # local windowed centroid, reused below for N_hat's sign
        for _i in range(len(MH_point)):
            _lo = max(0, _i - half_win)
            _hi = min(len(MH_point), _i + half_win + 1)
            if _hi - _lo < 3:
                B_hat[_i] = np.array([0.0, 0.0, 1.0])
                B_center[_i] = MH_point[_i]
                continue
            _plane_v, _plane_com = fit_plane(MH_point[_lo:_hi, :])
            B_hat[_i] = _plane_v
            B_center[_i] = _plane_com

        # Eigenvector sign is arbitrary per-window - enforce frame-to-frame continuity first.
        for _i in range(1, len(B_hat)):
            if np.dot(B_hat[_i], B_hat[_i - 1]) < 0:
                B_hat[_i] = -B_hat[_i]

        # Anchor the OVERALL sign using the POSITION geometry of the downswing sweep itself (P4:P7) - the
        # signed area swept by the MH path around its own local centroid, projected onto B_hat. This uses
        # only positions (no velocity/acceleration/angular-velocity derivatives at all), which proved more
        # reliable than the earlier omega_perp-based anchor (that one is contaminated by wrist cock/release,
        # which can rotate the club shaft in a way that doesn't track the overall arm-swing rotation, and
        # produced a regression on real data). Every downswing sweeps the hands in the same physical
        # rotational sense, so this anchor is swing-independent.
        _p4c, _p7c = min(p4, p7), max(p4, p7)
        _pts = MH_point[_p4c:_p7c + 1]
        _local_com = np.mean(_pts, axis=0)
        _rel = _pts - _local_com
        _cross_terms = np.cross(_rel[:-1], _rel[1:])
        _signed_area_vec = np.sum(_cross_terms, axis=0)
        _mean_B_over_window = np.mean(B_hat[_p4c:_p7c + 1], axis=0)
        _sweep_sign_val = np.dot(_signed_area_vec, _mean_B_over_window)
        B_hat_flipped = bool(_sweep_sign_val < 0)
        if B_hat_flipped:
            B_hat = -B_hat

        # Handedness correction: MH_point (from hand_l_3d/hand_r_3d) and the club data are deliberately kept
        # in the RAW, untransformed capture frame (paired with each other, per earlier agreement) - meaning
        # this frame is NOT mirrored for lefty golfers the way the rest of the script's pose-derived metrics
        # are. A lefty and a righty swing sweep the hands in opposite rotational senses within that same raw
        # frame, so the position-based sweep anchor above is only self-consistent WITHIN one handedness.
        # This additional correction makes "positive" consistently mean "promotes the golfer's own swing
        # direction" for both - confirmed as the likely cause of Fig.4-7 flipping between two real swings.
        if lefty == 1:
            B_hat = -B_hat
        if DEBUG:
            print(
                f"[DEBUG] B_hat swing-direction sign correction applied: {B_hat_flipped} "
                f"(anchored to P4:P7 MH-path signed-area sweep, position-only, no derivatives) | "
                f"lefty={lefty} (additional handedness correction {'APPLIED' if lefty == 1 else 'not needed'})"
            )

        v_MH = _butter_filt_cols(np.gradient(MH_point, dt, axis=0), fc_hz=6)
        v_proj = v_MH - np.sum(v_MH * B_hat, axis=1)[:, np.newaxis] * B_hat  # velocity projected into the plane
        v_proj_norm = np.linalg.norm(v_proj, axis=1)
        T_hat_valid = v_proj_norm > 1e-4
        T_hat = np.zeros_like(v_MH)
        T_hat[T_hat_valid] = v_proj[T_hat_valid] / v_proj_norm[T_hat_valid, np.newaxis]
        T_hat = _fill_degenerate(T_hat, T_hat_valid)

        N_hat = np.cross(B_hat, T_hat)  # completes the right-handed frame (T x N = B)
        N_norm = np.linalg.norm(N_hat, axis=1)
        N_valid = N_norm > 1e-6
        N_hat_safe = np.zeros_like(N_hat)
        N_hat_safe[N_valid] = N_hat[N_valid] / N_norm[N_valid, np.newaxis]
        N_hat = _fill_degenerate(N_hat_safe, N_valid)

        # 15.07.2026: The per-frame "outward-positive" N_hat correction that was here has been REMOVED.
        # It assumed Kwon's normal axis was dynamically re-oriented to always mean "outward," but empirical
        # testing against real swings (both players) showed the NATURAL N_hat = B_hat x T_hat triad (no
        # extra correction) reproduces Kwon's Fig. 3 pattern correctly on its own. N_hat is left as the
        # plain cross product below.

        if DEBUG:
            n_frames_total = len(T_hat_valid)
            print(
                f"[DEBUG] windowed plane fit: half_win={half_win} frames (~{frame_window_ms:.0f}ms), "
                f"T_hat valid {T_hat_valid.sum()}/{n_frames_total} (degenerate frames forward/back-filled)"
            )
            dotTN = np.abs(np.sum(T_hat * N_hat, axis=1))
            dotTB = np.abs(np.sum(T_hat * B_hat, axis=1))
            dotNB = np.abs(np.sum(N_hat * B_hat, axis=1))
            print(
                f"[DEBUG] frame orthonormality check (should be ~0): "
                f"max|T.N|={np.nanmax(dotTN):.4g} max|T.B|={np.nanmax(dotTB):.4g} max|N.B|={np.nanmax(dotNB):.4g}"
            )

        # NOTE (15.07.2026): validated against two real (righthanded) swings - Fig.3-6 now match Kwon's
        # qualitative pattern and are cross-swing consistent. Not yet validated for lefty or down-the-line
        # swings, since hand_l_3d/hand_r_3d/MH_point come from the RAW, untransformed pose data.

        # --- Decompose net MH force: in-plane (tangential/normal) + out-of-plane (swing-axis) ---
        F_MH_tangential = np.sum(F_MH_vec * T_hat, axis=1)
        F_MH_normal = np.sum(F_MH_vec * N_hat, axis=1)
        F_MH_swing_axis = np.sum(F_MH_vec * B_hat, axis=1)  # out-of-plane force component (new)

        # Moments are pseudovectors (defined via a cross product / right-hand rule), so their positive-
        # rotation-direction convention is inherently tied to "which side you view the rotation from" - a
        # convention the force-decomposition frame (T_hat, N_hat, B_hat), built purely from position
        # geometry, has no way to know on its own. Empirically, Kwon's positive in-plane (swing-axis) torque
        # convention comes out opposite to B_hat's geometric orientation. Since this is a property of moments
        # in general (not specific to the swing axis), the SAME negation is applied to all three torque axes
        # uniformly - one consistent rule for every moment quantity (in-plane AND out-of-plane), rather than
        # a special case for B_hat alone. This is what keeps Fig.3 (force) and Figs.4-7 (torque) internally
        # consistent with each other.
        torque_T = -T_hat
        torque_N = -N_hat
        torque_axis = -B_hat

        # --- Decompose net MH moment (the "couple") ---
        M_MH_swing_axis = np.sum(M_MH_vec * torque_axis, axis=1)  # in-plane driving torque (Kwon Fig. 4)
        M_MH_tangential = np.sum(M_MH_vec * torque_T, axis=1)  # out-of-plane (new)
        M_MH_normal = np.sum(M_MH_vec * torque_N, axis=1)      # out-of-plane (new)

        # --- Decompose MH force moment (r_CH x F_MH) ---
        MHforce_moment_swing_axis = np.sum(MH_force_moment_vec * torque_axis, axis=1)  # in-plane (Kwon Fig. 5)
        MHforce_moment_tangential = np.sum(MH_force_moment_vec * torque_T, axis=1)  # out-of-plane (new)
        MHforce_moment_normal = np.sum(MH_force_moment_vec * torque_N, axis=1)      # out-of-plane (new)

        # --- Decompose net club moment (total) ---
        # T_net = h_dot_C_vec is the club's own kinematic response (not an externally-applied moment from
        # the hands like M_MH/MH_force_moment), so - per empirical validation against real swing data -
        # it stays on the plain geometric axes (B_hat/T_hat/N_hat), NOT the flipped torque_axis/torque_T/
        # torque_N used for the two hand-applied moment components above.
        T_net_swing_axis = np.sum(T_net_vec * B_hat, axis=1)  # in-plane (Kwon Fig. 6)
        T_net_tangential = np.sum(T_net_vec * T_hat, axis=1)  # out-of-plane (new)
        T_net_normal = np.sum(T_net_vec * N_hat, axis=1)      # out-of-plane (new)


        _debug_print("F_MH_tangential (Fig.3 green)", F_MH_tangential)
        _debug_print("F_MH_normal (Fig.3 blue, signed)", F_MH_normal)
        _debug_print("F_MH_swing_axis (out-of-plane force)", F_MH_swing_axis)
        _debug_print("M_MH_swing_axis (Fig.4 red)", M_MH_swing_axis)
        _debug_print("MHforce_moment_swing_axis (Fig.5 green)", MHforce_moment_swing_axis)
        _debug_print("T_net_swing_axis (Fig.6 blue)", T_net_swing_axis)
        _debug_print("M_MH_tangential/normal (out-of-plane)", np.column_stack([M_MH_tangential, M_MH_normal]))
        _debug_print("MHforce_moment_tan/normal (out-of-plane)", np.column_stack([MHforce_moment_tangential, MHforce_moment_normal]))
        ## ============================================================================================

        # 06.07.2026: Tangential_Force / Centripetal_Force now sourced from the Kwon net MH force
        # decomposition (F_MH = m*a_C - W_C, projected onto the moving swing-plane frame), replacing the
        # previous scalar-speed-based approximation. Gravity is now embedded properly via W_C rather than
        # added ad hoc to the radial term.
        F_tan_shaft = F_MH_tangential
        F_rad_shaft = np.abs(F_MH_normal)  # magnitude - grip must resist the radial component regardless of sign

        # Grip force
        taper_deg = 4
        # degrees
        taper_rad = taper_deg * 2 * np.pi / 360
        # rad
        friction = 0.7
        # coefficient
        Grip = F_rad_shaft / friction * (1 - np.sin(taper_rad))
        _debug_print("F_tan_shaft (reassigned)", F_tan_shaft)
        _debug_print("F_rad_shaft (reassigned, abs)", F_rad_shaft)
        _debug_print("Grip", Grip)

        # Build shaft force vectors normalized (direction) - now full 3D
        F_tan_vec_norm = club_head[1:] - club_head[:-1]

        # Add a check for zero norm to avoid division by zero
        norms = np.linalg.norm(F_tan_vec_norm, axis=1)
        nonzero_norms = norms != 0  # Find elements with non-zero norms
        F_tan_vec_norm[nonzero_norms] /= norms[nonzero_norms, np.newaxis]

        F_tan_vec_norm = np.vstack([[0, 0, 0], F_tan_vec_norm])

        shaft_COM = (club_head + club_main) / 2

        F_rad_vec_norm = club_main - club_COM
        F_rad_vec_norm = (
            F_rad_vec_norm / np.linalg.norm(F_rad_vec_norm, axis=1)[:, np.newaxis]
        )

        # Shaft force vectors (3D)
        # Shaft force vectors (3D)
        F_tan_vec = F_tan_vec_norm * F_tan_shaft[:, np.newaxis]
        F_rad_vec = F_rad_vec_norm * F_rad_shaft[:, np.newaxis]

        # ---------------------------------------------------------
        # Scale vectors for frontend visualization only
        # (does NOT affect biomechanics calculations)
        # ---------------------------------------------------------
        shaft_force_arrow_scale = 0.01      # Adjust as needed

        F_tan_vec_vis = F_tan_vec * shaft_force_arrow_scale
        F_rad_vec_vis = F_rad_vec * shaft_force_arrow_scale

        # 20.02.25 Resultant force from Radial and Tangential forces
        F_res_vec = (
            F_rad_vec + F_tan_vec
        )  # Adding F_rad_vec and F_tan_vec to get the resultant force vector

        ## --- PIXEL-SPACE shaft force vectors (unchanged from Script A) ---
        ## Uses the restored pixel-space hand_l_2d/hand_r_2d. Only the X/Y components of the (now 3D)
        ## force vectors are used, matching the original 2D visualization logic exactly.
        # midhand_2d = (np.array(hand_l_2d) + np.array(hand_r_2d)) / 2
        # start_x = midhand_2d[:, 0]
        # start_y = midhand_2d[:, 1]
        # end_x_rad_force = midhand_2d[:, 0] + F_rad_vec[:, 0]
        # end_y_rad_force = midhand_2d[:, 1] + F_rad_vec[:, 1]
        # end_x_tan_force = []
        # end_y_tan_force = []
        # for i in range(0, len(midhand_2d)):
        #     if i < index_min_club_COM_speed:
        #         end_x_tan_force = midhand_2d[:, 0] - F_tan_vec[:, 0]
        #         end_y_tan_force = midhand_2d[:, 1] - F_tan_vec[:, 1]
        #     else:
        #         end_x_tan_force = midhand_2d[:, 0] + F_tan_vec[:, 0]
        #         end_y_tan_force = midhand_2d[:, 1] + F_tan_vec[:, 1]

        ## --- NEW: 3D shaft force vectors (for future stick-figure/avatar integration) ---
        midhand_3d = (hand_l_3d + hand_r_3d) / 2
        start_x_3d = hand_r_3d[:, 0]
        start_y_3d = hand_r_3d[:, 1]
        start_z_3d = hand_r_3d[:, 2]
        end_x_rad_force_3d = hand_r_3d[:, 0] + F_rad_vec_vis[:, 0]
        end_y_rad_force_3d = hand_r_3d[:, 1] + F_rad_vec_vis[:, 1]
        end_z_rad_force_3d = hand_r_3d[:, 2] + F_rad_vec_vis[:, 2]
        end_x_tan_force_3d = []
        end_y_tan_force_3d = []
        end_z_tan_force_3d = []
        for i in range(len(midhand_3d)):
            if i < index_min_club_COM_speed:
                end_x_tan_force_3d = hand_r_3d[:, 0] - F_tan_vec_vis[:, 0]
                end_y_tan_force_3d = hand_r_3d[:, 1] - F_tan_vec_vis[:, 1]
                end_z_tan_force_3d = hand_r_3d[:, 2] - F_tan_vec_vis[:, 2]
            else:
                end_x_tan_force_3d = hand_r_3d[:, 0] + F_tan_vec_vis[:, 0]
                end_y_tan_force_3d = hand_r_3d[:, 1] + F_tan_vec_vis[:, 1]
                end_z_tan_force_3d = hand_r_3d[:, 2] + F_tan_vec_vis[:, 2]

        # Compute club COM (pixel-labeled dict, computed from the 3D-meters club data - see note below)
        club_COM_2d = (shaft_COM * shaft_mass + club_head_2d * club_head_mass) / club_mass
        club_COM_2d = club_COM_2d

        def point_to_line_projection(x, u, v):
            x = np.asarray(x, dtype=float)
            u = np.asarray(u, dtype=float)
            v = np.asarray(v, dtype=float)

            th = np.dot(x - v, u) / np.dot(u, u)

            h = v + th * u

            return h

        def point_to_line_distance(p, a, b):

            p = np.asarray(p, dtype=float).reshape(-1)
            a = np.asarray(a, dtype=float).reshape(-1)
            b = np.asarray(b, dtype=float).reshape(-1)

            if p.size == 2:
                p = np.append(p, 0.0)

            if a.size == 2:
                a = np.append(a, 0.0)

            if b.size == 2:
                b = np.append(b, 0.0)

            denom = np.dot(a, a)
            if denom < 1e-12:
                return np.linalg.norm(p - b)

            th = np.dot(p - b, a) / denom
            h_test = b + th * a

            return np.linalg.norm(p - h_test)

        club_main_end_x = []
        club_main_end_y = []

        proj_club_COM = []
        for i in range(0, len(club_COM_2d)):
            if i < index_min_club_COM_speed:
                proj_club_COM.append(
                    point_to_line_projection(
                        club_COM_2d[i, :],
                        F_rad_vec[i] - F_tan_vec[i],
                        club_main_2d[i, :],
                    )
                )
                club_main_end_x.append(
                    club_main_2d[i, 0] + (F_rad_vec[i, 0] - F_tan_vec[i, 0])
                )
                club_main_end_y.append(
                    club_main_2d[i, 1] + (F_rad_vec[i, 1] - F_tan_vec[i, 1])
                )
            else:
                proj_club_COM.append(
                    point_to_line_projection(
                        club_COM_2d[i, :],
                        F_rad_vec[i] + F_tan_vec[i],
                        club_main_2d[i, :],
                    )
                )
                club_main_end_x.append(
                    club_main_2d[i, 0] + (F_rad_vec[i, 0] + F_tan_vec[i, 0])
                )
                club_main_end_y.append(
                    club_main_2d[i, 1] + (F_rad_vec[i, 1] + F_tan_vec[i, 1])
                )

        lever_arm_res = []
        for i in range(0, len(club_COM_2d)):
            if i < index_min_club_COM_speed:
                # 06.07.2026: fixed to pass the full 3D force-direction vector (was previously only
                # X/Y, inconsistent with proj_club_COM directly above which already uses the full vector).
                lever_arm_res.append(
                    point_to_line_distance(
                        club_COM_2d[i, :],
                        F_rad_vec[i] - F_tan_vec[i],
                        club_main_2d[i, :],
                    )
                )
                club_main_end_x.append(
                    club_main_2d[i, 0] + (F_rad_vec[i, 0] - F_tan_vec[i, 0])
                )
                club_main_end_y.append(
                    club_main_2d[i, 1] - F_tan_vec[i, 1] + F_rad_vec[i, 1]
                )
            else:
                lever_arm_res.append(
                    point_to_line_distance(
                        club_COM_2d[i, :],
                        F_rad_vec[i] + F_tan_vec[i],
                        club_main_2d[i, :],
                    )
                )
                club_main_end_x.append(
                    club_main_2d[i, 0] + (F_rad_vec[i, 0] + F_tan_vec[i, 0])
                )
                club_main_end_y.append(
                    club_main_2d[i, 1] + (F_rad_vec[i, 1] + F_tan_vec[i, 1])
                )
        # Lever arms from point_to_line_project and point_to_line_distance
        proj_club_COM = np.array(proj_club_COM)  # point_to_line_project
        lever_arm_res = np.array(lever_arm_res)  # point_to_line_distance

        # 06.07.2026: M_in_plane now sourced from the Kwon "MH force moment" (r_CH x F_MH), swing-axis
        # component - replacing the previous world-frame-X-component * 3D-lever-arm approximation.
        M_in_plane = MHforce_moment_swing_axis

        # 06.07.2026: Wrist_torque (the couple) now sourced from the Kwon net MH moment (Eq. 13),
        # swing-axis component - replacing the previous planar-circular-motion approximation
        # (I=mL^2/12 * a/r), which assumed a fixed pivot and discarded genuine 3D angular momentum.
        Wrist_torque = M_MH_swing_axis
        _debug_print("M_in_plane (reassigned)", M_in_plane)
        _debug_print("Wrist_torque / C_in_plane (reassigned)", Wrist_torque)

        # Compute MAX values and time to impact
        max_F_tan_shaft, index_max_F_tan_shaft = np.max(
            F_tan_shaft[p1 : p8 + 1]
        ), np.argmax(F_tan_shaft[p1 : p8 + 1])
        time_to_impact_F_tan_shaft = 1000 * (index_max_F_tan_shaft - p7 + p1) / frame_rate

        max_F_rad_shaft, index_max_F_rad_shaft = np.max(
            F_rad_shaft[p1 : p8 + 1]
        ), np.argmax(F_rad_shaft[p1 : p8 + 1])
        time_to_impact_F_rad_shaft = 1000 * (index_max_F_rad_shaft - p7 + p1) / frame_rate

        max_Wrist_torque, index_max_Wrist_torque = np.max(
            Wrist_torque[p1 : p8 + 1]
        ), np.argmax(Wrist_torque[p1 : p8 + 1])
        time_to_impact_Wrist_torque = 1000 * (index_max_Wrist_torque - p7 + p1) / frame_rate

        max_F_club_head, index_max_F_club_head = np.max(
            F_club_head[p1 : p8 + 1]
        ), np.argmax(F_club_head[p1 : p8 + 1])
        time_to_impact_F_club_head = 1000 * (index_max_F_club_head - p7 + p1) / frame_rate

        max_Grip, index_max_Grip = np.max(Grip[p1 : p8 + 1]), np.argmax(Grip[p1 : p8 + 1])
        time_to_impact_Grip = 1000 * (index_max_Grip - p7 + p1) / frame_rate

        # Create a list of maximum values in the desired order
        peak_force_order = [
            index_max_F_tan_shaft,
            index_max_F_rad_shaft,
            index_max_Wrist_torque,
        ]

        # Sort the list and get the indices that represent the original order
        index_sorted = np.argsort(peak_force_order)

        # Find the positions of each measurement in the sorted order
        peak_force_order_F_tan_shaft = np.where(index_sorted == 0)[0]
        peak_force_order_F_rad_shaft = np.where(index_sorted == 1)[0]
        peak_force_order_Wrist_torque = np.where(index_sorted == 2)[0]

        ##################################### In plane metrics ######################################################
        # In plane couple
        C_in_plane = Wrist_torque

        # In plane torque (total torque)
        # T_in_plane now uses T_net_swing_axis directly (T_net_vec projected onto the plain B_hat, not
        # torque_axis) - NOT C_in_plane + M_in_plane. That old sum was a leftover from the original script
        # and never actually reflected the T_net/B_hat sign-convention fix, even after T_net_swing_axis
        # itself was corrected - this was the real, final bug in the chain.
        T_in_plane = T_net_swing_axis

        inPlaneMetrics = {
            "M_in_plane": M_in_plane.tolist(),
            "C_in_plane": C_in_plane.tolist(),
            "T_in_plane": T_in_plane.tolist(),
        }

        # NEW: out-of-plane counterparts, from the same Kwon vector decomposition above.
        # Each in-plane moment (swing-axis component) has two out-of-plane siblings (tangential-axis and
        # normal-axis components, i.e. the two axes that lie IN the swing plane) representing torque that
        # would push the club above/below the swing plane rather than driving its rotation within it.
        outOfPlaneMetrics = {
            "F_out_of_plane": F_MH_swing_axis.tolist(),                     # net MH force, out-of-plane component
            "M_out_of_plane_tangential": MHforce_moment_tangential.tolist(),
            "M_out_of_plane_normal": MHforce_moment_normal.tolist(),
            "C_out_of_plane_tangential": M_MH_tangential.tolist(),
            "C_out_of_plane_normal": M_MH_normal.tolist(),
            "T_out_of_plane_tangential": T_net_tangential.tolist(),
            "T_out_of_plane_normal": T_net_normal.tolist(),
        }

        # NEW: supplementary raw 3D kinetics vectors + moving swing-plane frame axes, for future
        # stick-figure/avatar visualization and debugging. Not required for the scalar metrics above.
        M2C_kinetics_3d = {
            "MH_point_3d": MH_point.tolist(),
            "F_MH_vec_3d": F_MH_vec.tolist(),
            "M_MH_vec_3d": M_MH_vec.tolist(),
            "MH_force_moment_vec_3d": MH_force_moment_vec.tolist(),
            "T_net_vec_3d": T_net_vec.tolist(),
            "swing_frame_tangential_3d": T_hat.tolist(),
            "swing_frame_normal_3d": N_hat.tolist(),
            "swing_frame_axis_3d": B_hat.tolist(),
            "F_MH_tangential_signed": F_MH_tangential.tolist(),  # same as Tangential_Force, kept here for plotting
            "F_MH_normal_signed": F_MH_normal.tolist(),          # signed version (Centripetal_Force is abs())
        }

        if DEBUG:
            print("[DEBUG] ---- Final in-plane / out-of-plane metric summary ----")
            _debug_print("inPlaneMetrics.C_in_plane", inPlaneMetrics["C_in_plane"])
            _debug_print("inPlaneMetrics.M_in_plane", inPlaneMetrics["M_in_plane"])
            _debug_print("inPlaneMetrics.T_in_plane", inPlaneMetrics["T_in_plane"])
            _debug_print("outOfPlaneMetrics.C_out_of_plane_tangential", outOfPlaneMetrics["C_out_of_plane_tangential"])
            _debug_print("outOfPlaneMetrics.C_out_of_plane_normal", outOfPlaneMetrics["C_out_of_plane_normal"])
            _debug_print("outOfPlaneMetrics.M_out_of_plane_tangential", outOfPlaneMetrics["M_out_of_plane_tangential"])
            _debug_print("outOfPlaneMetrics.M_out_of_plane_normal", outOfPlaneMetrics["M_out_of_plane_normal"])
            _debug_print("outOfPlaneMetrics.T_out_of_plane_tangential", outOfPlaneMetrics["T_out_of_plane_tangential"])
            _debug_print("outOfPlaneMetrics.T_out_of_plane_normal", outOfPlaneMetrics["T_out_of_plane_normal"])
            # sanity check: out-of-plane should generally be small relative to in-plane for a well-planed swing
            in_plane_rms = np.sqrt(np.mean(np.array(inPlaneMetrics["T_in_plane"])**2))
            out_tan_rms = np.sqrt(np.mean(np.array(outOfPlaneMetrics["T_out_of_plane_tangential"])**2))
            out_norm_rms = np.sqrt(np.mean(np.array(outOfPlaneMetrics["T_out_of_plane_normal"])**2))
            print(
                f"[DEBUG] RMS check - in-plane T: {in_plane_rms:.4g} Nm | "
                f"out-of-plane T tangential: {out_tan_rms:.4g} Nm ({100*out_tan_rms/max(in_plane_rms,1e-9):.1f}% of in-plane) | "
                f"out-of-plane T normal: {out_norm_rms:.4g} Nm ({100*out_norm_rms/max(in_plane_rms,1e-9):.1f}% of in-plane)"
            )
            # Kwon: T-Net "largely remains positive... but turns to negative right before the impact."
            # Find the positive->negative crossing nearest P7 to check this precisely, rather than by eye.
            T_arr = np.array(inPlaneMetrics["T_in_plane"])
            search_lo, search_hi = max(0, p4), min(len(T_arr), p8 + 1)
            crossings = [
                _i for _i in range(search_lo + 1, search_hi)
                if T_arr[_i - 1] >= 0 and T_arr[_i] < 0
            ]
            if crossings:
                _closest = min(crossings, key=lambda idx: abs(idx - p7))
                _t_cross_ms = (_closest - p7) / frame_rate * 1000.0
                print(
                    f"[DEBUG] T_in_plane +->- crossing nearest P7: frame {_closest}, {_t_cross_ms:+.1f} ms "
                    f"relative to P7 (negative ms = before impact, matches Kwon's 'turns negative right "
                    f"before impact' if small and negative)"
                )
            else:
                print("[DEBUG] T_in_plane: no positive->negative crossing found in the P4:P8 window")

        # In Plane Metrics "Pixel" Values for frontend - UNCHANGED FROM PRIOR CODE.
        # NOTE (06.07.2026): club_COM_2d / club_main_2d / proj_club_COM are now derived from the
        # 3D-meters club input (Script B no longer receives a 2D pixel club track), so the *1080
        # scaling below no longer converts to genuine pixel coordinates - it is left as-is pending
        # confirmation of whether a separate 2D club track is still available for this dict.
        proj_club_COM_pixel = (
            proj_club_COM.squeeze()[:, :2] * 1080
        )  # 2D Coordinates of projected COM
        club_COM_2d_pixel = club_COM_2d * 1080
        club_main_2d_pixel = club_main_2d * 1080

        M2C_inplane_vectors = {
            "proj_club_COM_pixel": proj_club_COM_pixel.tolist(),
            "club_COM_2d_pixel": club_COM_2d_pixel.tolist(),
            "club_main_2d_pixel": club_main_2d_pixel.tolist(),
        }

        # NEW: 3D (meters) in-plane vectors, unscaled, for future stick-figure/avatar integration.
        proj_club_COM_3d = proj_club_COM.squeeze()
        club_COM_3d = club_COM_2d
        club_main_3d = club_main_2d

        M2C_inplane_vectors_3d = {
            "proj_club_COM_3d": proj_club_COM_3d.tolist(),
            "club_COM_3d": club_COM_3d.tolist(),
            "club_main_3d": club_main_3d.tolist(),
        }
        ####################################### lead wrist metrics ####################################################################

        # Wrist displacements
        wrist_sway = 100 * (wrist_l[:, 1] - wrist_l[0, 1])
        wrist_thrust = 100 * (wrist_l[:, 0] - wrist_l[0, 0])
        wrist_lift = 100 * (wrist_l[:, 2] - wrist_l[0, 2])

        # Imperial


        # LEFT WRIST ANGLE
        A_l = np.sqrt(
            (hand_l[:, 0] - wrist_l[:, 0]) ** 2
            + (hand_l[:, 1] - wrist_l[:, 1]) ** 2
            + (hand_l[:, 2] - wrist_l[:, 2]) ** 2
        )
        B_l = np.sqrt(
            (elbow_l[:, 0] - hand_l[:, 0]) ** 2
            + (elbow_l[:, 1] - hand_l[:, 1]) ** 2
            + (elbow_l[:, 2] - hand_l[:, 2]) ** 2
        )
        C_l = np.sqrt(
            (elbow_l[:, 0] - wrist_l[:, 0]) ** 2
            + (elbow_l[:, 1] - wrist_l[:, 1]) ** 2
            + (elbow_l[:, 2] - wrist_l[:, 2]) ** 2
        )

        wrist_angle_l = np.arccos((A_l**2 + C_l**2 - B_l**2) / (2 * A_l * C_l))
        wrist_angle_l_deg = np.degrees(wrist_angle_l)

        # Wrist velocity in degrees per second
        wrist_vel_l_deg = np.gradient(wrist_angle_l_deg, dt)

        # Three components of hand speed
        wrist_speed_X = (wrist_l[1:, 0] - wrist_l[:-1, 0]) * frame_rate  # m/s
        wrist_speed_Y = (wrist_l[1:, 1] - wrist_l[:-1, 1]) * frame_rate
        wrist_speed_Z = (wrist_l[1:, 2] - wrist_l[:-1, 2]) * frame_rate

        # Resultant wrist speed
        wrist_speed = np.concatenate(
            [[0], np.sqrt(wrist_speed_X**2 + wrist_speed_Y**2 + wrist_speed_Z**2)]
        )  # m/s

        # Convert to MPH if requested
        factor = 2.23693629
        wrist_speed_mph = wrist_speed * factor


        kinetic_order_frontend = {
            "peak_force_order_F_tangential_force": peak_force_order_F_tan_shaft[0] + 1,
            "peak_force_order_F_centripetal_force": peak_force_order_F_rad_shaft[0] + 1,
            "peak_force_order_Wrist_torque": peak_force_order_Wrist_torque[0] + 1,
        }

        time_to_impact = {
            "time_to_impact_tangential_force": time_to_impact_F_tan_shaft,
            "time_to_impact_centripetal_force": time_to_impact_F_rad_shaft,
            "time_to_impact_wrist_torque": time_to_impact_Wrist_torque,
            "time_to_impact_Grip_force": time_to_impact_Grip,
            "time_to_impact_F_club_head": time_to_impact_F_club_head,
        }

        peak_club_mechanics = {
            "peak_tangential_force": max_F_tan_shaft,
            "peak_centripetal_force": max_F_rad_shaft,
            "peak_wrist_torque": max_Wrist_torque,
            "peak_Grip_force": max_Grip,
            "peak_F_club_head": max_F_club_head,
        }

        M2C_insights = {
            "arc_width": arc_width.tolist(),
            "arc_width_inch": arc_width_inch.tolist(),
            "lead_forearm_shaft_angle": lead_forearm_shaft_angle_deg_2d.tolist(),
            "x_factor": x_factor.tolist(),
            "x_factor_stretch": x_factor_stretch,
        }



        M2C_power = {
            "vertical_power": vertical_power.tolist(),
            "hand_power": hand_power.tolist(),
            "Shaft_power": P_shaft.tolist(),
            "club_head_power": club_head_power.tolist(),
            "power_index": power_index2,
        }



        M2C_club_mechanics = {
            "Tangential_Force": F_tan_shaft.tolist(),
            "Centripetal_Force": F_rad_shaft.tolist(),
            "Grip_Force": Grip.tolist(),
            "F_club_head": F_club_head.tolist(),
            "Wrist_torque": Wrist_torque.tolist(),
        }



        M2C_club_mechanics_highlights = {
            "peak_club_mechanics": peak_club_mechanics,
            "time_to_impact": time_to_impact,
            "kinetic_order": kinetic_order_frontend,
        }

        # UNCHANGED from Script A - pixel-space, for video overlay
        # M2C_force_vectors_shaft = {
        #     "start_x": start_x.tolist(),
        #     "start_y": start_y.tolist(),
        #     "end_x_rad_force": end_x_rad_force.tolist(),
        #     "end_y_rad_force": end_y_rad_force.tolist(),
        #     "end_x_tan_force": end_x_tan_force.tolist(),
        #     "end_y_tan_force": end_y_tan_force.tolist(),
        # }

        # NEW: 3D (meters) shaft force vectors, for future stick-figure/avatar integration
        # M2C_force_vectors_shaft_3d = {
        #     "start_x_3d": start_x_3d.tolist(),
        #     "start_y_3d": start_y_3d.tolist(),
        #     "start_z_3d": start_z_3d.tolist(),
        #     "end_x_rad_force_3d": end_x_rad_force_3d.tolist(),
        #     "end_y_rad_force_3d": end_y_rad_force_3d.tolist(),
        #     "end_z_rad_force_3d": end_z_rad_force_3d.tolist(),
        #     "end_x_tan_force_3d": end_x_tan_force_3d.tolist(),
        #     "end_y_tan_force_3d": end_y_tan_force_3d.tolist(),
        #     "end_z_tan_force_3d": end_z_tan_force_3d.tolist(),
        # }

        M2C_force_vectors_shaft_3d = {
            "start_3d": [
                [x, y, z]
                for x, y, z in zip(
                    start_x_3d.tolist(),
                    start_y_3d.tolist(),
                    start_z_3d.tolist()
                )
            ],
            "end_rad_force_3d": [
                [x, y, z]
                for x, y, z in zip(
                    end_x_rad_force_3d.tolist(),
                    end_y_rad_force_3d.tolist(),
                    end_z_rad_force_3d.tolist()
                )
            ],
            "end_tan_force_3d": [
                [x, y, z]
                for x, y, z in zip(
                    end_x_tan_force_3d.tolist(),
                    end_y_tan_force_3d.tolist(),
                    end_z_tan_force_3d.tolist()
                )
            ],
        }



        M2C_wrist = {
            "lead_forearm_shaft_angle": lead_forearm_shaft_angle_deg_2d.tolist(),
            "lead_wrist_extension_flexion": wrist_angle_l_deg.tolist(),
            "lead_wrist_extension_flexion_vel": wrist_vel_l_deg.tolist(),
            "lead_wrist_speed": wrist_speed_mph.tolist(),
            "lead_wrist_thrust": wrist_thrust.tolist(),
            "lead_wrist_sway": wrist_sway.tolist(),
            "lead_wrist_lift": wrist_lift.tolist(),

        }


            ############################################################
        # PERFORMANCE METRICS
        ############################################################

        impact_frame = p7  

        # ----------------------------------------------------------
        # 1. Peak Tangential Force
        # ----------------------------------------------------------
        peak_tan_idx = np.argmax(np.abs(F_tan_shaft))
        peak_tan_force = float(F_tan_shaft[peak_tan_idx])

        peak_tan_time_ms = (
            (peak_tan_idx - impact_frame)
            / frame_rate
            * 1000
        )

        # ----------------------------------------------------------
        # 2. Peak Normal Force
        # ----------------------------------------------------------
        peak_norm_idx = np.argmax(np.abs(F_rad_shaft))
        peak_norm_force = float(F_rad_shaft[peak_norm_idx])

        peak_norm_time_ms = (
            (peak_norm_idx - impact_frame)
            / frame_rate
            * 1000
        )

        # ----------------------------------------------------------
        # 3. Peak Net Torque
        # ----------------------------------------------------------
        T_net = np.array(inPlaneMetrics["T_in_plane"])

        peak_torque_idx = np.argmax(np.abs(T_net))
        peak_torque = float(T_net[peak_torque_idx])

        peak_torque_time_ms = (
            (peak_torque_idx - impact_frame)
            / frame_rate
            * 1000
        )

        # ----------------------------------------------------------
        # 4. Torque Efficiency
        # ----------------------------------------------------------
        C_in = np.array(inPlaneMetrics["C_in_plane"])
        M_in = np.array(inPlaneMetrics["M_in_plane"])

        rms_net = rms(T_net)

        rms_sources = (
            rms(C_in) +
            rms(M_in)
        )

        torque_efficiency = (
            rms_net /
            rms_sources
            if rms_sources > 0 else 0
        )

        torque_cancellation = 1.0 - torque_efficiency

        # ----------------------------------------------------------
        # 5. Force Direction Ratio
        # ----------------------------------------------------------
        force_direction_ratio = (
            rms(F_tan_shaft) /
            rms(F_rad_shaft)
            if rms(F_rad_shaft) > 0 else 0
        )

        if force_direction_ratio < 0.6:
            force_direction_interpretation = "Steering dominant"

        elif force_direction_ratio < 1.0:
            force_direction_interpretation = "Balanced"

        else:
            force_direction_interpretation = "Acceleration dominant"

        # ----------------------------------------------------------
        # 6. Barrel Stability Index
        # ----------------------------------------------------------

        rms_oop = rms(
            np.concatenate([
                outOfPlaneMetrics["T_out_of_plane_tangential"],
                outOfPlaneMetrics["T_out_of_plane_normal"]
            ])
        )

        rms_inplane = rms(T_net)

        barrel_stability_index = (
            rms_oop /
            rms_inplane
            if rms_inplane > 0 else 0
        )

        # ----------------------------------------------------------
        # 7. Peak Out-of-Plane Torque
        # Late swing only
        # ----------------------------------------------------------

        t_ms = (
            (np.arange(len(T_net)) - impact_frame)
            / frame_rate
            * 1000
        )

        late_mask = (
            (t_ms >= -50) &
            (t_ms <= 20)
        )

        oop_total = np.sqrt(
            np.array(outOfPlaneMetrics["T_out_of_plane_tangential"])**2 +
            np.array(outOfPlaneMetrics["T_out_of_plane_normal"])**2
        )

        late_oop = oop_total[late_mask]

        peak_oop = np.max(late_oop)

        peak_oop_idx = np.where(
            oop_total == peak_oop
        )[0][0]

        peak_oop_time = t_ms[peak_oop_idx]

        Performance_metrics = {

            "peak_tangential_force": {
                "value_N": peak_tan_force,
                "time_ms": peak_tan_time_ms,
            },

            "peak_normal_force": {
                "value_N": peak_norm_force,
                "time_ms": peak_norm_time_ms,
            },

            "peak_net_torque": {
                "value_Nm": peak_torque,
                "time_ms": peak_torque_time_ms,
            },

            "torque_efficiency": {
                "efficiency_percent":
                    100 * torque_efficiency,

                "cancellation_percent":
                    100 * torque_cancellation,
            },

            "force_direction_ratio": {
                "ratio":
                    force_direction_ratio,

                "interpretation":
                    force_direction_interpretation,
            },

            "barrel_stability_index": {
                "ratio":
                    barrel_stability_index,

                "percent":
                    100 * barrel_stability_index,
            },

            "peak_out_of_plane_torque": {
                "value_Nm":
                    peak_oop,

                "time_ms":
                    peak_oop_time,
            }
        }
    out = locals().copy()
    out.pop("state", None)
    return out
