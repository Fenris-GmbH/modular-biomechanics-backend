from biomechanics.constants import SEGMENT_MASS_FRACTIONS


def com(ctx, option=1):
    m = ctx.markers
    pelvis = m["pelvis"]
    hip_l = m["hip_l"]
    knee_l = m["knee_l"]
    ankle_l = m["ankle_l"]
    foot_l = m["foot_l"]
    midfoot_l = m["midfoot_l"]
    hip_r = m["hip_r"]
    knee_r = m["knee_r"]
    ankle_r = m["ankle_r"]
    foot_r = m["foot_r"]
    midfoot_r = m["midfoot_r"]
    nose = m["nose"]
    spine_low = m["spine_low"]
    spine_high = m["spine_high"]
    shoulder_l = m["shoulder_l"]
    elbow_l = m["elbow_l"]
    wrist_l = m["wrist_l"]
    hand_l = m["hand_l"]
    shoulder_r = m["shoulder_r"]
    elbow_r = m["elbow_r"]
    wrist_r = m["wrist_r"]
    hand_r = m["hand_r"]
    neck = m["neck"]
    skullbase = m["skullbase"]
    skull = m["skull"]
    club_main = m["club_main"]
    club_head = m["club_head"]
    ear_l = m["ear_l"]
    ear_r = m["ear_r"]
    # % Compute center of mass of each segment
    calculate_COM = 1
    if calculate_COM == 1:
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

        # Compute mass fractions of each segment
        head_m = SEGMENT_MASS_FRACTIONS["head"]
        trunk_m = SEGMENT_MASS_FRACTIONS["trunk"]
        arm_r_m = SEGMENT_MASS_FRACTIONS["arm_r"]
        forearm_r_m = SEGMENT_MASS_FRACTIONS["forearm_r"]
        hand_r_m = SEGMENT_MASS_FRACTIONS["hand_r"]
        thigh_r_m = SEGMENT_MASS_FRACTIONS["thigh_r"]
        shank_r_m = SEGMENT_MASS_FRACTIONS["shank_r"]
        foot_r_m = SEGMENT_MASS_FRACTIONS["foot_r"]
        arm_l_m = SEGMENT_MASS_FRACTIONS["arm_l"]
        forearm_l_m = SEGMENT_MASS_FRACTIONS["forearm_l"]
        hand_l_m = SEGMENT_MASS_FRACTIONS["hand_l"]
        thigh_l_m = SEGMENT_MASS_FRACTIONS["thigh_l"]
        shank_l_m = SEGMENT_MASS_FRACTIONS["shank_l"]
        foot_l_m = SEGMENT_MASS_FRACTIONS["foot_l"]

        # Weight normalization (for partial COM)
        W_no_arms = 1 - 2 * (arm_l_m + forearm_l_m + hand_l_m)
        W_lower_body = 1 - 2 * (arm_l_m + forearm_l_m + hand_l_m) - trunk_m - head_m
        W_upper_body = 1 - 2 * (thigh_l_m + shank_l_m + foot_l_m)

        if option == 1:  # Compute normal center of mass
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

        if option == 4:  # COM without arms
            COM = (
                head_COM * head_m
                + trunk_COM * trunk_m
                + thigh_r_COM * thigh_r_m
                + shank_r_COM * shank_r_m
                + foot_r_COM * foot_r_m
                + thigh_l_COM * thigh_l_m
                + shank_l_COM * shank_l_m
                + foot_l_COM * foot_l_m
            ) / W_no_arms

        if option == 5:  # COM of Lower Body
            COM = (
                thigh_r_COM * thigh_r_m
                + shank_r_COM * shank_r_m
                + foot_r_COM * foot_r_m
                + thigh_l_COM * thigh_l_m
                + shank_l_COM * shank_l_m
                + foot_l_COM * foot_l_m
            ) / W_lower_body

        if option == 6:  # COM of Upper Body
            COM = (
                head_COM * head_m
                + trunk_COM * trunk_m
                + arm_r_COM * arm_r_m
                + forearm_r_COM * forearm_r_m
                + hand_r_COM * hand_r_m
                + arm_l_COM * arm_l_m
                + forearm_l_COM * forearm_l_m
                + hand_l_COM * hand_l_m
            ) / W_upper_body

    return COM
