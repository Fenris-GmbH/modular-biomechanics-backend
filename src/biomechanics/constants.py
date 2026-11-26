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


SEGMENT_MASS_FRACTIONS = {
    "head": head_m,
    "trunk": trunk_m,
    "arm_r": arm_r_m,
    "forearm_r": forearm_r_m,
    "hand_r": hand_r_m,
    "thigh_r": thigh_r_m,
    "shank_r": shank_r_m,
    "foot_r": foot_r_m,
    "arm_l": arm_l_m,
    "forearm_l": forearm_l_m,
    "hand_l": hand_l_m,
    "thigh_l": thigh_l_m,
    "shank_l": shank_l_m,
    "foot_l": foot_l_m,
}

class Camera:
    ORIENTATION_DOWN_THE_LINE = "DownTheLine"
    ORIENTATION_FACEON = "FaceOn"

class BaseballMovements:
    HITTING = "hitting"
    PITCHING = "pitching"

class GolfSwings:
    FULL_SWING = "full swing"

class GolfClubs:
    IRON = "iron"
    DRIVER = "driver"

class Player:
    HANDEDNESS_LEFT = "lefthanded"
    HANDEDNESS_RIGHT = "righthanded"

class ErrorMessages:
    MISSING_P_POS = "Missing P positions in config"
    MISSING_FORCE_DATA = "Missing force data"

class Filters:
    CLUB_SPEED_FC = 5
    CLUB_SPEED_N = 4

class Conversions:
    MPS_TO_MPH = 2.23693629
    CM_TO_INCH = 0.3937007874

