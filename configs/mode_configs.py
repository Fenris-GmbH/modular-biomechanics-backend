MODE_CONFIG = {
    Mode.BASEBALL_HITTING: {
        "constructor": baseball,
        "inputs": ["pose3d", "frame_rate", "weight", "orientation", "handedness"],
        "kwargs": {"movement_type": "hitting"},
        "setup": {
            "forces": {
                "fun": "initialize_forces",
                "optional_inputs": ["critical_events"],
                "kwargs": {"include_club": False},
            },
            "club_fitting": {"fun": "initialize_club_fitting"},
        },
        "funcs": [
            # Basic movement
            {"fun": "get_basic_time"},
            {"fun": "calculate_kinematic_sequence"},
            {"fun": "calculate_COM"},
            {"fun": "calculate_segment_angles"},
            {"fun": "calculate_segment_displacements"},

            # Forces
            {"fun": "calculate_force_vectors", "setup": "forces", "optional_inputs": ["pose2d"]},
            {"fun": "calculate_weight_shift", "setup": "forces", "optional_inputs": ["pose2d"]},
            {"fun": "calculate_forces", "setup": "forces"},
            {"fun": "calculate_force_time", "setup": "forces"},
            {"fun": "calculate_force_vectors_3d", "setup": "forces"},
            {"fun": "calculate_values_for_COP_trajectory", "setup": "forces"},
            {"fun": "calculate_COM_MODELS", "setup": "forces", "optional_inputs": ["pose2d"]},
            {"fun": "calculate_power", "setup": "forces"},
            {"fun": "calculate_energy", "setup": "forces"},
            {"fun": "calculate_impulse", "setup": "forces"},
            {"fun": "calculate_small_force_vectors", "setup": "forces"},
            {"fun": "calculate_COM_COP_traj_back_view", "setup": "forces"},
            {"fun": "calculate_critical_events", "setup": "forces"},
            {"fun": "calculate_frame_offset", "setup": "forces"},
            {"fun": "calculate_all_forces", "setup": "forces"},

            # Club fitting
            {"fun": "calculate_movement_type", "setup": "club_fitting"},
            {
                "fun": "calculate_M2C_insights",
                "setup": "club_fitting",
                "inputs": ["club_3d", "critical_events"],
            },
            {
                "fun": "calculate_M2C_wrist",
                "setup": "club_fitting",
                "inputs": ["club_3d"],
            },
            {
                "fun": "calculate_Performance_metrics",
                "setup": "club_fitting",
                "inputs": ["club_3d", "bat_type", "critical_events"],
            },
            {
                "fun": "calculate_M2C_power",
                "setup": "club_fitting",
                "inputs": ["club_3d", "bat_type", "critical_events"],
            },
            {
                "fun": "calculate_M2C_club_mechanics",
                "setup": "club_fitting",
                "inputs": ["club_3d", "bat_type", "critical_events"],
            },
            {
                "fun": "calculate_M2C_club_mechanics_highlights",
                "setup": "club_fitting",
                "inputs": ["club_3d", "bat_type", "critical_events"],
            },
            {
                "fun": "calculate_M2C_force_vectors_bat_3d",
                "setup": "club_fitting",
                "inputs": ["club_3d", "bat_type", "critical_events"],
            },
            {
                "fun": "calculate_inPlaneMetrics",
                "setup": "club_fitting",
                "inputs": ["club_3d", "bat_type", "critical_events"],
            },
            {
                "fun": "calculate_outOfPlaneMetrics",
                "setup": "club_fitting",
                "inputs": ["club_3d", "bat_type", "critical_events"],
            },
            {
                "fun": "calculate_M2C_inplane_vectors",
                "setup": "club_fitting",
                "inputs": ["club_3d", "bat_type", "critical_events"],
            },
            {
                "fun": "calculate_M2C_inplane_vectors_3d",
                "setup": "club_fitting",
                "inputs": ["club_3d", "bat_type", "critical_events"],
            },
            {
                "fun": "calculate_M2C_kinetics_3d",
                "setup": "club_fitting",
                "inputs": ["club_3d", "bat_type", "critical_events"],
            },
            {
                "fun": "calculate_all_club_fitting",
                "setup": "club_fitting",
                "inputs": ["club_3d", "bat_type", "critical_events"],
            },
        ],
    },

    Mode.BASEBALL_PITCHING: {
        "constructor": baseball,
        "inputs": ["pose3d", "frame_rate", "weight", "orientation", "handedness"],
        "kwargs": {"movement_type": "pitching"},
        "setup": {
            "forces": {
                "fun": "initialize_forces",
                "optional_inputs": ["critical_events"],
                "kwargs": {"include_club": False},
            },
            "club_fitting": {"fun": "initialize_club_fitting"},
        },
        "funcs": [
            # Basic movement
            {"fun": "get_basic_time"},
            {"fun": "calculate_kinematic_sequence"},
            {"fun": "calculate_COM"},
            {"fun": "calculate_segment_angles"},
            {"fun": "calculate_segment_displacements"},

            # Forces
            {"fun": "calculate_force_vectors", "setup": "forces", "optional_inputs": ["pose2d"]},
            {"fun": "calculate_weight_shift", "setup": "forces", "optional_inputs": ["pose2d"]},
            {"fun": "calculate_forces", "setup": "forces"},
            {"fun": "calculate_force_time", "setup": "forces"},
            {"fun": "calculate_force_vectors_3d", "setup": "forces"},
            {"fun": "calculate_values_for_COP_trajectory", "setup": "forces"},
            {"fun": "calculate_COM_MODELS", "setup": "forces", "optional_inputs": ["pose2d"]},
            {"fun": "calculate_power", "setup": "forces"},
            {"fun": "calculate_energy", "setup": "forces"},
            {"fun": "calculate_impulse", "setup": "forces"},
            {"fun": "calculate_small_force_vectors", "setup": "forces"},
            {"fun": "calculate_COM_COP_traj_back_view", "setup": "forces"},
            {"fun": "calculate_critical_events", "setup": "forces"},
            {"fun": "calculate_frame_offset", "setup": "forces"},
            {"fun": "calculate_all_forces", "setup": "forces"},

            # Club fitting
            {"fun": "calculate_movement_type", "setup": "club_fitting"},
            {
                "fun": "calculate_M2C_insights",
                "setup": "club_fitting",
                "inputs": ["critical_events"],
            },
            {
                "fun": "calculate_Performance_metrics",
                "setup": "club_fitting",
                "inputs": ["critical_events"],
            },
            {
                "fun": "calculate_M2C_power",
                "setup": "club_fitting",
                "inputs": ["critical_events"],
            },
            {"fun": "calculate_M2C_club_mechanics", "setup": "club_fitting"},
            {
                "fun": "calculate_M2C_club_mechanics_highlights",
                "setup": "club_fitting",
                "inputs": ["critical_events"],
            },
            {"fun": "calculate_M2C_wrist", "setup": "club_fitting"},
            {"fun": "calculate_M2C_force_vectors_hand_r_3d", "setup": "club_fitting"},
            {"fun": "calculate_pitcher_metric_equivalents", "setup": "club_fitting"},
            {
                "fun": "calculate_all_club_fitting",
                "setup": "club_fitting",
                "inputs": ["critical_events"],
            },
        ],
    },

    Mode.GOLF: {
        "constructor": golf,
        "inputs": ["pose3d", "frame_rate", "weight", "orientation", "handedness"],
        "kwargs": {"movement_type": "full swing"},
        "setup": {
            "forces": {
                "fun": "initialize_forces",
                "inputs": ["club_3d", "club_type"],
                "optional_inputs": ["critical_events"],
                "kwargs": {"include_club": True},
            },
            "club_fitting": {"fun": "initialize_club_fitting"},
        },
        "funcs": [
            # Basic movement
            {"fun": "get_basic_time"},
            {"fun": "calculate_kinematic_sequence"},
            {"fun": "calculate_COM"},
            {"fun": "calculate_segment_angles"},
            {"fun": "calculate_segment_displacements"},

            # Forces
            {"fun": "calculate_force_vectors", "setup": "forces", "optional_inputs": ["pose2d"]},
            {"fun": "calculate_weight_shift", "setup": "forces", "optional_inputs": ["pose2d"]},
            {"fun": "calculate_forces", "setup": "forces"},
            {"fun": "calculate_force_time", "setup": "forces"},
            {"fun": "calculate_force_vectors_3d", "setup": "forces"},
            {"fun": "calculate_values_for_COP_trajectory", "setup": "forces"},
            {"fun": "calculate_COM_MODELS", "setup": "forces", "optional_inputs": ["pose2d"]},
            {"fun": "calculate_power", "setup": "forces"},
            {"fun": "calculate_energy", "setup": "forces"},
            {"fun": "calculate_impulse", "setup": "forces"},
            {"fun": "calculate_small_force_vectors", "setup": "forces"},
            {"fun": "calculate_COM_COP_traj_back_view", "setup": "forces"},
            {"fun": "calculate_critical_events", "setup": "forces"},
            {"fun": "calculate_frame_offset", "setup": "forces"},
            {"fun": "calculate_all_forces", "setup": "forces"},

            # Club fitting, using the current generic graph
            {"fun": "calculate_movement_type", "setup": "club_fitting"},
            {"fun": "calculate_M2C_power", "setup": "club_fitting"},
            {"fun": "calculate_M2C_club_mechanics", "setup": "club_fitting"},
            {"fun": "calculate_M2C_club_mechanics_highlights", "setup": "club_fitting"},
            {"fun": "calculate_M2C_wrist", "setup": "club_fitting"},
            {
                "fun": "calculate_M2C_power_left",
                "setup": "club_fitting",
                "available_key": "M2C_power_left",
            },
            {
                "fun": "calculate_M2C_power_right",
                "setup": "club_fitting",
                "available_key": "M2C_power_right",
            },
            {
                "fun": "calculate_M2C_club_mechanics_left",
                "setup": "club_fitting",
                "available_key": "M2C_club_mechanics_left",
            },
            {
                "fun": "calculate_M2C_club_mechanics_right",
                "setup": "club_fitting",
                "available_key": "M2C_club_mechanics_right",
            },
            {
                "fun": "calculate_M2C_club_mechanics_highlights_left",
                "setup": "club_fitting",
                "available_key": "M2C_club_mechanics_highlights_left",
            },
            {
                "fun": "calculate_M2C_club_mechanics_highlights_right",
                "setup": "club_fitting",
                "available_key": "M2C_club_mechanics_highlights_right",
            },
            {
                "fun": "calculate_M2C_wrist_left",
                "setup": "club_fitting",
                "available_key": "M2C_wrist_left",
            },
            {
                "fun": "calculate_M2C_wrist_right",
                "setup": "club_fitting",
                "available_key": "M2C_wrist_right",
            },
            {"fun": "calculate_all_club_fitting", "setup": "club_fitting"},
        ],
    },

    # Any non-hitting/non-pitching swing type (for example "jump") uses the
    # generic club-fitting graph. Keep movement_type as an input so the mode
    # can be used with different generic movement payloads.
    Mode.GENERIC: {
        "constructor": movement,
        "inputs": [
            "pose3d",
            "movement_type",
            "frame_rate",
            "weight",
            "orientation",
            "handedness",
        ],
        "setup": {
            "forces": {
                "fun": "initialize_forces",
                "optional_inputs": ["critical_events"],
                "kwargs": {"include_club": False},
            },
            "club_fitting": {"fun": "initialize_club_fitting"},
        },
        "funcs": [
            # Basic movement
            {"fun": "get_basic_time"},
            {"fun": "calculate_kinematic_sequence"},
            {"fun": "calculate_COM"},
            {"fun": "calculate_segment_angles"},
            {"fun": "calculate_segment_displacements"},

            # Body-only forces
            {"fun": "calculate_force_vectors", "setup": "forces", "optional_inputs": ["pose2d"]},
            {"fun": "calculate_weight_shift", "setup": "forces", "optional_inputs": ["pose2d"]},
            {"fun": "calculate_forces", "setup": "forces"},
            {"fun": "calculate_force_time", "setup": "forces"},
            {"fun": "calculate_force_vectors_3d", "setup": "forces"},
            {"fun": "calculate_values_for_COP_trajectory", "setup": "forces"},
            {"fun": "calculate_COM_MODELS", "setup": "forces", "optional_inputs": ["pose2d"]},
            {"fun": "calculate_power", "setup": "forces"},
            {"fun": "calculate_energy", "setup": "forces"},
            {"fun": "calculate_impulse", "setup": "forces"},
            {"fun": "calculate_small_force_vectors", "setup": "forces"},
            {"fun": "calculate_COM_COP_traj_back_view", "setup": "forces"},
            {"fun": "calculate_critical_events", "setup": "forces"},
            {"fun": "calculate_frame_offset", "setup": "forces"},
            {"fun": "calculate_all_forces", "setup": "forces"},

            # Generic club fitting. The aggregate includes primary and
            # secondary side widgets, skeleton, diagnostics, and equivalents.
            {"fun": "calculate_movement_type", "setup": "club_fitting"},
            {"fun": "calculate_all_club_fitting", "setup": "club_fitting"},
        ],
    },
}
