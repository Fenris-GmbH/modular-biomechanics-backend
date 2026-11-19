# from config_loader import load_config
# from pose3d_data_loader import load_pose3d_data
# from pose2d_data_loader import load_pose2d_data
# from force_data_loader import load_force_data
# from club_data_loader import load_club_data
# from ball_club_speed_data_loader import load_ball_club_speed_data

from data_ingestion.club_data_loader import load_club_data
from data_ingestion.config_loader import load_config
from data_ingestion.ball_data_loader import load_ball_position_data
from data_ingestion.pose3d_data_loader import load_pose3d_data
from data_ingestion.pose2d_data_loader import load_pose2d_data
from data_ingestion.force_data_loader import load_force_data
from data_ingestion.club_data_loader import load_club_data
from data_ingestion.ball_club_speed_data_loader import load_ball_club_speed_data
from utils.logging_utils import setup_logging
from numpy.linalg import norm
import numpy as np
from dataclasses import dataclass

logger = setup_logging()


class config:
    def __init__(self):
        self.config = None

    def load_config(self, config_path):
        self.info = load_config(config_path)
        if self.info is None:
            raise Exception("No config data")

        self.video = self.info.get("video", [])
        self.swing = self.info.get("swing", [])
        self.processing = self.info.get("processing", [])
        self.user = self.info.get("user", [])

        if self.processing.get("isMultiCamResults") == True:
            isMultiCamResults = 1
        else:
            isMultiCamResults = 0

        # Assign None to P8 and P9 for downtheline single camera swings
        if self.swing["orientation"] == "DownTheLine" and isMultiCamResults == 0:
            logger.info("Assigning Artificial P8 and P9 for Single Camera DTL Swing")
            self.swing["critical_events"]["P8"] = (
                self.swing["critical_events"]["P7"] + 5
            )  # Dummy Values
            self.swing["critical_events"]["P9"] = (
                self.swing["critical_events"]["P10"] - 5
            )  # Dummy Values

        # Check if P positions for Full-Swings are complete
        if self.swing["swingType"] == "full swing":
            unique_P = set(
                v
                for v in self.swing["critical_events"].values()
                if isinstance(v, (int, float))
            )
            if len(unique_P) < 10:
                logger.warning(
                    f"Only {len(unique_P)} unique P positions found in critical_events: {unique_P}"
                )


class pose3d:
    def __init__(self, config):
        self.markers = None
        self.config = config

    def load_pose3d(self, json3d_path):
        self.markers = load_pose3d_data(json3d_path)
        if self.markers is None:
            logger.error("Missing Pose3d data")
            raise Exception("Missing Pose3d data")

    def assign_markers(self):
        if self.markers is None:
            logger.error("Pose3d data not available for marker assignment.")
            raise ValueError("Pose3d data not available for marker assignment.")
        if self.config.swing["orientation"] == "DownTheLine":
            downtheLine = 1
        else:
            downtheLine = 0

        if self.config.swing["handedness"] == "leftHanded":
            lefty = 1
        else:
            lefty = 0

        # Tracking data
        if downtheLine == 1:
            if lefty == 0:
                temp = np.squeeze(self.markers)
                new_x = temp[:, :, 1]
                new_y = -temp[:, :, 0]
                temp[:, :, 0] = new_x  # change x to -y
                temp[:, :, 1] = new_y  # change y to -x
                self.markers = temp
            else:
                temp = np.squeeze(self.markers)
                new_x = -temp[:, :, 1]
                new_y = -temp[:, :, 0]
                temp[:, :, 0] = new_x  # change x to y
                temp[:, :, 1] = new_y  # change y to -x
                self.markers = temp
        if lefty == 1:
            if downtheLine == 0:
                temp = np.squeeze(self.markers)
                new_y = -temp[:, :, 1]
                temp[:, :, 1] = new_y  # change y to -y    # mirroring
                self.markers = temp

        # Loading all markers depending on the handedness of the player
        if lefty == 1:

            self.pelvis = np.squeeze(self.markers[0])
            self.hip_l = np.squeeze(self.markers[1])
            self.knee_l = np.squeeze(self.markers[2])
            self.ankle_l = np.squeeze(self.markers[3])
            self.foot_l = np.squeeze(self.markers[4])  # changed
            self.midfoot_l = np.squeeze(self.markers[5])
            self.hip_r = np.squeeze(self.markers[6])
            self.knee_r = np.squeeze(self.markers[7])
            self.ankle_r = np.squeeze(self.markers[8])
            self.foot_r = np.squeeze(self.markers[9])  # changed
            self.midfoot_r = np.squeeze(self.markers[10])
            self.nose = np.squeeze(self.markers[11])  # only for calculations
            self.spine_low = np.squeeze(self.markers[12])
            self.spine_high = np.squeeze(self.markers[13])
            self.shoulder_l = np.squeeze(self.markers[14])
            self.elbow_l = np.squeeze(self.markers[15])
            self.wrist_l = np.squeeze(self.markers[16])
            self.hand_l = np.squeeze(self.markers[17])
            self.shoulder_r = np.squeeze(self.markers[18])
            self.elbow_r = np.squeeze(self.markers[19])
            self.wrist_r = np.squeeze(self.markers[20])
            self.hand_r = np.squeeze(self.markers[21])
            self.neck = np.squeeze(self.markers[22])
            self.skullbase = np.squeeze(self.markers[23])
            self.skull = np.squeeze(self.markers[24])
            self.club_main = np.squeeze(self.markers[25])  # NaN
            self.club_head = np.squeeze(self.markers[26])  # NaN
            self.ear_l = np.squeeze(self.markers[27])
            self.ear_r = np.squeeze(self.markers[28])
        else:
            self.pelvis = np.squeeze(self.markers[0])
            self.hip_r = np.squeeze(self.markers[1])
            self.knee_r = np.squeeze(self.markers[2])
            self.ankle_r = np.squeeze(self.markers[3])
            self.foot_r = np.squeeze(self.markers[4])  # changed
            self.midfoot_r = np.squeeze(self.markers[5])
            self.hip_l = np.squeeze(self.markers[6])
            self.knee_l = np.squeeze(self.markers[7])
            self.ankle_l = np.squeeze(self.markers[8])
            self.foot_l = np.squeeze(self.markers[9])  # changed
            self.midfoot_l = np.squeeze(self.markers[10])
            self.nose = np.squeeze(self.markers[11])  # only for calculations
            self.spine_low = np.squeeze(self.markers[12])
            self.spine_high = np.squeeze(self.markers[13])
            self.shoulder_r = np.squeeze(self.markers[14])
            self.elbow_r = np.squeeze(self.markers[15])
            self.wrist_r = np.squeeze(self.markers[16])
            self.hand_r = np.squeeze(self.markers[17])
            self.shoulder_l = np.squeeze(self.markers[18])
            self.elbow_l = np.squeeze(self.markers[19])
            self.wrist_l = np.squeeze(self.markers[20])
            self.hand_l = np.squeeze(self.markers[21])
            self.neck = np.squeeze(self.markers[22])
            self.skullbase = np.squeeze(self.markers[23])
            self.skull = np.squeeze(self.markers[24])
            self.club_main = np.squeeze(self.markers[25])  # NaN
            self.club_head = np.squeeze(self.markers[26])  # NaN
            self.ear_r = np.squeeze(self.markers[27])
            self.ear_l = np.squeeze(self.markers[28])


class pose2d:
    def __init__(
        self,
        config,
    ):
        self.markers = None
        self.config = config

    def load_pose2d(self, json2d_path):
        self.markers = load_pose2d_data(json2d_path)
        if self.markers is None:
            logger.error("Missing Pose2d data")
            raise Exception("No pose2d data")

    def assign_markers(self):
        if self.markers is None:
            logger.error("Pose2d data not available for marker assignment")
            raise ValueError("Pose2d data not available for marker assignment")

        if self.config.swing["handedness"] == "leftHanded":
            lefty = 1
        else:
            lefty = 0

        pelvis = []
        hip_r = []
        knee_r = []
        ankle_r = []
        foot_r = []
        hip_l = []
        knee_l = []
        ankle_l = []
        foot_l = []
        nose = []
        midfoot_l = []
        spine_low = []
        spine_high = []
        shoulder_r = []
        elbow_r = []
        wrist_r = []
        hand_r = []
        midfoot_r = []
        shoulder_l = []
        elbow_l = []
        wrist_l = []
        hand_l = []
        neck = []
        skullbase = []
        skull = []
        ear_r = []
        ear_l = []

        if lefty == 1:
            for index in range(0, len(self.markers)):
                pelvis.append(np.squeeze(self.markers[index][0]))
                hip_l.append(np.squeeze(self.markers[index][1]))
                knee_l.append(np.squeeze(self.markers[index][2]))
                ankle_l.append(np.squeeze(self.markers[index][3]))
                foot_l.append(np.squeeze(self.markers[index][4]))
                midfoot_l.append(np.squeeze(self.markers[index][5]))
                hip_r.append(np.squeeze(self.markers[index][6]))
                knee_r.append(np.squeeze(self.markers[index][7]))
                ankle_r.append(np.squeeze(self.markers[index][8]))
                foot_r.append(np.squeeze(self.markers[index][9]))
                midfoot_r.append(np.squeeze(self.markers[index][10]))
                nose.append(np.squeeze(self.markers[index][11]))
                spine_low.append(np.squeeze(self.markers[index][12]))
                spine_high.append(np.squeeze(self.markers[index][13]))
                shoulder_l.append(np.squeeze(self.markers[index][14]))
                elbow_l.append(np.squeeze(self.markers[index][15]))
                wrist_l.append(np.squeeze(self.markers[index][16]))
                hand_l.append(np.squeeze(self.markers[index][17]))
                shoulder_r.append(np.squeeze(self.markers[index][18]))
                elbow_r.append(np.squeeze(self.markers[index][19]))
                wrist_r.append(np.squeeze(self.markers[index][20]))
                hand_r.append(np.squeeze(self.markers[index][21]))
                neck.append(np.squeeze(self.markers[index][22]))
                skullbase.append(np.squeeze(self.markers[index][23]))
                skull.append(np.squeeze(self.markers[index][24]))
                ear_l.append(np.squeeze(self.markers[index][27]))
                ear_r.append(np.squeeze(self.markers[index][28]))
        else:
            for index in range(0, len(self.markers)):
                pelvis.append(np.squeeze(self.markers[index][0]))
                hip_r.append(np.squeeze(self.markers[index][1]))
                knee_r.append(np.squeeze(self.markers[index][2]))
                ankle_r.append(np.squeeze(self.markers[index][3]))
                foot_r.append(np.squeeze(self.markers[index][4]))  # changed
                midfoot_r.append(np.squeeze(self.markers[index][5]))
                hip_l.append(np.squeeze(self.markers[index][6]))
                knee_l.append(np.squeeze(self.markers[index][7]))
                ankle_l.append(np.squeeze(self.markers[index][8]))
                foot_l.append(np.squeeze(self.markers[index][9]))  # changed
                midfoot_l.append(np.squeeze(self.markers[index][10]))
                nose.append(
                    np.squeeze(self.markers[index][11])
                )  # only for calculations
                spine_low.append(np.squeeze(self.markers[index][12]))
                spine_high.append(np.squeeze(self.markers[index][13]))
                shoulder_r.append(np.squeeze(self.markers[index][14]))
                elbow_r.append(np.squeeze(self.markers[index][15]))
                wrist_r.append(np.squeeze(self.markers[index][16]))
                hand_r.append(np.squeeze(self.markers[index][17]))
                shoulder_l.append(np.squeeze(self.markers[index][18]))
                elbow_l.append(np.squeeze(self.markers[index][19]))
                wrist_l.append(np.squeeze(self.markers[index][20]))
                hand_l.append(np.squeeze(self.markers[index][21]))
                neck.append(np.squeeze(self.markers[index][22]))
                skullbase.append(np.squeeze(self.markers[index][23]))
                skull.append(np.squeeze(self.markers[index][24]))
                ear_r.append(np.squeeze(self.markers[index][27]))
                ear_l.append(np.squeeze(self.markers[index][28]))

        self.midfoot_l = np.array(midfoot_l)
        self.midfoot_r = np.array(midfoot_r)
        self.pelvis = np.array(pelvis)
        self.hip_r = np.array(hip_r)
        self.knee_r = np.array(knee_r)
        self.ankle_r = np.array(ankle_r)
        self.hip_l = np.array(hip_l)
        self.knee_l = np.array(knee_l)
        self.ankle_l = np.array(ankle_l)
        self.foot_l = np.array(foot_l)
        self.midfoot_l = np.array(midfoot_l)
        self.nose = np.array(nose)
        self.spine_low = np.array(spine_low)
        self.spine_high = np.array(spine_high)
        self.shoulder_r = np.array(shoulder_r)
        self.elbow_r = np.array(elbow_r)
        self.wrist_r = np.array(wrist_r)
        self.hand_r = np.array(hand_r)
        self.shoulder_l = np.array(shoulder_l)
        self.elbow_l = np.array(elbow_l)
        self.wrist_l = np.array(wrist_l)
        self.hand_l = np.array(hand_l)
        self.neck = np.array(neck)
        self.skullbase = np.array(skullbase)
        self.skull = np.array(skull)
        self.ear_r = np.array(ear_r)
        self.ear_l = np.array(ear_l)
        self.foot_r = np.array(foot_r)


class forces:
    def __init__(self):
        self.forces = None

    def load_forces(self, force_path):
        try:
            self.forces = load_force_data(force_path)
            if self.forces is None:
                raise Exception("Missing force data")
        except Exception as e:
            logger.error(f"Failed to load force data from {force_path}: {e}")
            self.forces = None

    def assign_forces(self):
        F = self.forces
        if F is not None:
            Fz1_left = []
            Fz2_left = []
            Fz3_left = []
            Fz4_left = []
            Fx12_left = []
            Fx34_left = []
            Fy14_left = []
            Fy23_left = []

            Fz1_right = []
            Fz2_right = []
            Fz3_right = []
            Fz4_right = []
            Fx12_right = []
            Fx34_right = []
            Fy14_right = []
            Fy23_right = []

            # maybe you need to load the forces file only from p1 to p10 , look in the comment below.
            # 28.09.2023: changed this loop from p1 to p10 --> for all tracking frames
            for index in range(0, len(F[0]["Data"])):  # range(P1, P10+1)
                Fz1_left.append(F[0]["Data"][index]["forcesvalues"]["Fz1"])
                Fz2_left.append(F[0]["Data"][index]["forcesvalues"]["Fz2"])
                Fz3_left.append(F[0]["Data"][index]["forcesvalues"]["Fz3"])
                Fz4_left.append(F[0]["Data"][index]["forcesvalues"]["Fz4"])
                Fx12_left.append(F[0]["Data"][index]["forcesvalues"]["Fx12"])
                Fx34_left.append(F[0]["Data"][index]["forcesvalues"]["Fx34"])
                Fy14_left.append(F[0]["Data"][index]["forcesvalues"]["Fy14"])
                Fy23_left.append(F[0]["Data"][index]["forcesvalues"]["Fy23"])

                Fz1_right.append(F[1]["Data"][index]["forcesvalues"]["Fz1"])
                Fz2_right.append(F[1]["Data"][index]["forcesvalues"]["Fz2"])
                Fz3_right.append(F[1]["Data"][index]["forcesvalues"]["Fz3"])
                Fz4_right.append(F[1]["Data"][index]["forcesvalues"]["Fz4"])
                Fx12_right.append(F[1]["Data"][index]["forcesvalues"]["Fx12"])
                Fx34_right.append(F[1]["Data"][index]["forcesvalues"]["Fx34"])
                Fy14_right.append(F[1]["Data"][index]["forcesvalues"]["Fy14"])
                Fy23_right.append(F[1]["Data"][index]["forcesvalues"]["Fy23"])

            self.Fz1_left = np.array(Fz1_left)
            self.Fz2_left = np.array(Fz2_left)
            self.Fz3_left = np.array(Fz3_left)
            self.Fz4_left = np.array(Fz4_left)
            self.Fx12_left = np.array(Fx12_left)
            self.Fx34_left = np.array(Fx34_left)
            self.Fy14_left = np.array(Fy14_left)
            self.Fy23_left = np.array(Fy23_left)

            self.Fz1_right = np.array(Fz1_right)
            self.Fz2_right = np.array(Fz2_right)
            self.Fz3_right = np.array(Fz3_right)
            self.Fz4_right = np.array(Fz4_right)
            self.Fx12_right = np.array(Fx12_right)
            self.Fx34_right = np.array(Fx34_right)
            self.Fy14_right = np.array(Fy14_right)
            self.Fy23_right = np.array(Fy23_right)


class club:
    def __init__(self):
        self.markers = None
        self.speed = None

    def load_club(self, club_path):
        self.markers = load_club_data(club_path)
        if self.markers is None:
            raise Exception("No club data")

    def load_speed(self, ball_club_speed_path):
        speed = load_ball_club_speed_data(ball_club_speed_path)
        try:
            self.speed = speed["club_speed"]
        except:
            logger.info("No Club Speed Data")
            self.speed = None


class ball:
    def __init__(self):
        self.speed = None

    def load_speed(self, ball_club_speed_path):
        speed = load_ball_club_speed_data(ball_club_speed_path)
        try:
            self.speed = speed["ball_speed"]
        except:
            logger.info("No Ball Speed Data Available")
            self.speed = None


class swing_session:
    def __init__(self, cv_path):
        self.isClubmarkers3D = True  # Club 3D markers are in production
        self.cv_path = cv_path
        self.load_config()
        self.load_pose3d()
        self.assign_3D_markers()
        self.load_ball_position()
        self.load_club_markers()

    def load_config(self):
        self.config = load_config(self.cv_path)

    def load_pose3d(self):
        self.markers3D = load_pose3d_data(self.cv_path)
        if self.markers3D is None:
            logger.error("Missing Pose3d data")
            raise Exception("Missing Pose3d data")

    def load_ball_position(self):
        self.ball_position = load_ball_position_data(self.cv_path)
        if self.ball_position is None:
            logger.error("Missing ball position data")
            # raise Exception("Missing ball position data")

    def assign_3D_markers(self):
        if self.markers3D is None:
            logger.error("Pose3d data not available for marker assignment.")
            raise ValueError("Pose3d data not available for marker assignment.")

        if self.config["swing"]["orientation"] == "DownTheLine":
            downtheLine = 1
        else:
            downtheLine = 0

        if self.config["swing"]["handedness"] == "leftHanded":
            lefty = 1
        else:
            lefty = 0

        if self.config["processing"].get("isMultiCamResults") == True:
            isMultiCamResults = 1
        else:
            isMultiCamResults = 0

        logger.info(f"lefty: {lefty}")
        logger.info(f"downtheLine: {downtheLine}")
        logger.info(f"isMultiCamResults: {isMultiCamResults}")

        # Assign None to P8 and P9 for downtheline single camera swings
        if downtheLine == 1 and isMultiCamResults == 0:
            self.config["swing"]["critical_events"]["P8"] = 0  # Dummy Values
            self.config["swing"]["critical_events"]["P9"] = 0  # Dummy Values

        # Force Screen Markers (Not flipped or reversed as in CLub/Basic Screen Markers)
        if lefty == 1:
            # print(f"Assiging lefty 3D markers") #debug
            self.pelvis_fs = np.squeeze(self.markers3D[0])
            self.hip_l_fs = np.squeeze(self.markers3D[1])
            self.knee_l_fs = np.squeeze(self.markers3D[2])
            self.ankle_l_fs = np.squeeze(self.markers3D[3])
            self.foot_l_fs = np.squeeze(self.markers3D[4])  # changed
            self.midfoot_l_fs = np.squeeze(self.markers3D[5])
            self.hip_r_fs = np.squeeze(self.markers3D[6])
            self.knee_r_fs = np.squeeze(self.markers3D[7])
            self.ankle_r_fs = np.squeeze(self.markers3D[8])
            self.foot_r_fs = np.squeeze(self.markers3D[9])  # changed
            self.midfoot_r_fs = np.squeeze(self.markers3D[10])
            self.nose_fs = np.squeeze(self.markers3D[11])  # only for calculations
            self.spine_low_fs = np.squeeze(self.markers3D[12])
            self.spine_high_fs = np.squeeze(self.markers3D[13])
            self.shoulder_l_fs = np.squeeze(self.markers3D[14])
            self.elbow_l_fs = np.squeeze(self.markers3D[15])
            self.wrist_l_fs = np.squeeze(self.markers3D[16])
            self.hand_l_fs = np.squeeze(self.markers3D[17])
            self.shoulder_r_fs = np.squeeze(self.markers3D[18])
            self.elbow_r_fs = np.squeeze(self.markers3D[19])
            self.wrist_r_fs = np.squeeze(self.markers3D[20])
            self.hand_r_fs = np.squeeze(self.markers3D[21])
            self.neck_fs = np.squeeze(self.markers3D[22])
            self.skullbase_fs = np.squeeze(self.markers3D[23])
            self.skull_fs = np.squeeze(self.markers3D[24])
            self.club_main_fs = np.squeeze(self.markers3D[25])  # NaN
            self.club_head_fs = np.squeeze(self.markers3D[26])  # NaN
            self.ear_l_fs = np.squeeze(self.markers3D[27])
            self.ear_r_fs = np.squeeze(self.markers3D[28])

        else:
            self.pelvis_fs = np.squeeze(self.markers3D[0])
            self.hip_r_fs = np.squeeze(self.markers3D[1])
            self.knee_r_fs = np.squeeze(self.markers3D[2])
            self.ankle_r_fs = np.squeeze(self.markers3D[3])
            self.foot_r_fs = np.squeeze(self.markers3D[4])  # changed
            self.midfoot_r_fs = np.squeeze(self.markers3D[5])
            self.hip_l_fs = np.squeeze(self.markers3D[6])
            self.knee_l_fs = np.squeeze(self.markers3D[7])
            self.ankle_l_fs = np.squeeze(self.markers3D[8])
            self.foot_l_fs = np.squeeze(self.markers3D[9])  # changed
            self.midfoot_l_fs = np.squeeze(self.markers3D[10])
            self.nose_fs = np.squeeze(self.markers3D[11])  # only for calculations
            self.spine_low_fs = np.squeeze(self.markers3D[12])
            self.spine_high_fs = np.squeeze(self.markers3D[13])
            self.shoulder_r_fs = np.squeeze(self.markers3D[14])
            self.elbow_r_fs = np.squeeze(self.markers3D[15])
            self.wrist_r_fs = np.squeeze(self.markers3D[16])
            self.hand_r_fs = np.squeeze(self.markers3D[17])
            self.shoulder_l_fs = np.squeeze(self.markers3D[18])
            self.elbow_l_fs = np.squeeze(self.markers3D[19])
            self.wrist_l_fs = np.squeeze(self.markers3D[20])
            self.hand_l_fs = np.squeeze(self.markers3D[21])
            self.neck_fs = np.squeeze(self.markers3D[22])
            self.skullbase_fs = np.squeeze(self.markers3D[23])
            self.skull_fs = np.squeeze(self.markers3D[24])
            self.club_main_fs = np.squeeze(self.markers3D[25])  # NaN
            self.club_head_fs = np.squeeze(self.markers3D[26])  # NaN
            self.ear_r_fs = np.squeeze(self.markers3D[27])
            self.ear_l_fs = np.squeeze(self.markers3D[28])

        # Tracking data
        if downtheLine == 1:
            if lefty == 0:
                temp = np.squeeze(self.markers3D)
                new_x = temp[:, :, 1]
                new_y = -temp[:, :, 0]
                temp[:, :, 0] = new_x  # change x to -y
                temp[:, :, 1] = new_y  # change y to -x
                self.markers3D = temp
            else:
                temp = np.squeeze(self.markers3D)
                new_x = -temp[:, :, 1]
                new_y = -temp[:, :, 0]
                temp[:, :, 0] = new_x  # change x to y
                temp[:, :, 1] = new_y  # change y to -x
                self.markers3D = temp
        if lefty == 1:
            if downtheLine == 0:
                temp = np.squeeze(self.markers3D)
                new_y = -temp[:, :, 1]
                temp[:, :, 1] = new_y  # change y to -y    # mirroring
                self.markers3D = temp

        # Loading all markers depending on the handedness of the player
        if lefty == 1:
            # print(f"Assiging lefty 3D markers") #debug

            self.pelvis = np.squeeze(self.markers3D[0])
            self.hip_l = np.squeeze(self.markers3D[1])
            self.knee_l = np.squeeze(self.markers3D[2])
            self.ankle_l = np.squeeze(self.markers3D[3])
            self.foot_l = np.squeeze(self.markers3D[4])  # changed
            self.midfoot_l = np.squeeze(self.markers3D[5])
            self.hip_r = np.squeeze(self.markers3D[6])
            self.knee_r = np.squeeze(self.markers3D[7])
            self.ankle_r = np.squeeze(self.markers3D[8])
            self.foot_r = np.squeeze(self.markers3D[9])  # changed
            self.midfoot_r = np.squeeze(self.markers3D[10])
            self.nose = np.squeeze(self.markers3D[11])  # only for calculations
            self.spine_low = np.squeeze(self.markers3D[12])
            self.spine_high = np.squeeze(self.markers3D[13])
            self.shoulder_l = np.squeeze(self.markers3D[14])
            self.elbow_l = np.squeeze(self.markers3D[15])
            self.wrist_l = np.squeeze(self.markers3D[16])
            self.hand_l = np.squeeze(self.markers3D[17])
            self.shoulder_r = np.squeeze(self.markers3D[18])
            self.elbow_r = np.squeeze(self.markers3D[19])
            self.wrist_r = np.squeeze(self.markers3D[20])
            self.hand_r = np.squeeze(self.markers3D[21])
            self.neck = np.squeeze(self.markers3D[22])
            self.skullbase = np.squeeze(self.markers3D[23])
            self.skull = np.squeeze(self.markers3D[24])
            self.club_main = np.squeeze(self.markers3D[25])  # NaN
            self.club_head = np.squeeze(self.markers3D[26])  # NaN
            self.ear_l = np.squeeze(self.markers3D[27])
            self.ear_r = np.squeeze(self.markers3D[28])
        else:
            # print(f"Assiging Right handed 3D markers") #debug
            self.pelvis = np.squeeze(self.markers3D[0])
            self.hip_r = np.squeeze(self.markers3D[1])
            self.knee_r = np.squeeze(self.markers3D[2])
            self.ankle_r = np.squeeze(self.markers3D[3])
            self.foot_r = np.squeeze(self.markers3D[4])  # changed
            self.midfoot_r = np.squeeze(self.markers3D[5])
            self.hip_l = np.squeeze(self.markers3D[6])
            self.knee_l = np.squeeze(self.markers3D[7])
            self.ankle_l = np.squeeze(self.markers3D[8])
            self.foot_l = np.squeeze(self.markers3D[9])  # changed
            self.midfoot_l = np.squeeze(self.markers3D[10])
            self.nose = np.squeeze(self.markers3D[11])  # only for calculations
            self.spine_low = np.squeeze(self.markers3D[12])
            self.spine_high = np.squeeze(self.markers3D[13])
            self.shoulder_r = np.squeeze(self.markers3D[14])
            self.elbow_r = np.squeeze(self.markers3D[15])
            self.wrist_r = np.squeeze(self.markers3D[16])
            self.hand_r = np.squeeze(self.markers3D[17])
            self.shoulder_l = np.squeeze(self.markers3D[18])
            self.elbow_l = np.squeeze(self.markers3D[19])
            self.wrist_l = np.squeeze(self.markers3D[20])
            self.hand_l = np.squeeze(self.markers3D[21])
            self.neck = np.squeeze(self.markers3D[22])
            self.skullbase = np.squeeze(self.markers3D[23])
            self.skull = np.squeeze(self.markers3D[24])
            self.club_main = np.squeeze(self.markers3D[25])  # NaN
            self.club_head = np.squeeze(self.markers3D[26])  # NaN
            self.ear_r = np.squeeze(self.markers3D[27])
            self.ear_l = np.squeeze(self.markers3D[28])

        #     check_marker_dict = {  #NOTE: debug
        #         "pelvis": self.pelvis.tolist(),  # 0
        #         "hip_r": self.hip_r.tolist(),  # 1
        #         "knee_r": self.knee_r.tolist(),  # 2
        #         "ankle_r": self.ankle_r.tolist(),  # 3
        #         "foot_r": self.foot_r.tolist(),  # 4
        #         "midfoot_r": self.midfoot_r.tolist(),  # 5
        #         "hip_l": self.hip_l.tolist(),  # 6
        #         "knee_l": self.knee_l.tolist(),  # 7
        #         "ankle_l": self.ankle_l.tolist(),  # 8
        #         "foot_l": self.foot_l.tolist(),  # 9
        #         "midfoot_l": self.midfoot_l.tolist(),  # 10
        #         "nose": self.nose.tolist(),  # 11
        #         "spine_low": self.spine_low.tolist(),  # 12
        #         "spine_high": self.spine_high.tolist(),  # 13
        #         "shoulder_r": self.shoulder_r.tolist(),  # 14
        #         "elbow_r": self.elbow_r.tolist(),  # 15
        #         "wrist_r": self.wrist_r.tolist(),  # 16
        #         "hand_r": self.hand_r.tolist(),  # 17
        #         "shoulder_l": self.shoulder_l.tolist(),  # 18
        #         "elbow_l": self.elbow_l.tolist(),  # 19
        #         "wrist_l": self.wrist_l.tolist(),  # 20
        #         "hand_l": self.hand_l.tolist(),  # 21
        #         "neck": self.neck.tolist(),  # 22
        #         "skullbase": self.skullbase.tolist(),  # 23
        #         "skull": self.skull.tolist(),  # 24
        #         "club_main": self.club_main.tolist(),  # 25
        #         "club_head": self.club_head.tolist(),  # 26
        #         "ear_r": self.ear_r.tolist(),  # 27
        #         "ear_l": self.ear_l.tolist(),  # 28
        #     }

        # for i, (k, v) in enumerate(check_marker_dict.items()):
        #     print(f"{k}, {np.array_equal(np.array(v), np.squeeze(self.markers3D[i]))}")

    def load_club_markers(self):
        self.club_markers = load_club_data(self.cv_path)
        if self.club_markers is None:
            raise Exception("No club marker data")

    def compute_club_speed(self):
        try:
            self.ball_size = self.config["swing"]["launch_data"]["ball"]["ball_size"]
            self.ball_size = self.ball_size - self.ball_size * 0.165
            self.pix_m_conversion = 0.04267 / self.ball_size
        except:
            logger.exception("Exception occurred while fetching ball size")
            # raise ValueError("Ball size is required in config but missing.")  #NOTE implement when new config is available -> All perhaps everything will be in meters

        if self.ball_size == 0:
            logger.exception("Exception occurred while fetching ball size")
            # raise ValueError("Ball size is set to zero.") #NOTE implement when new config is available -> All perhaps everything will be in meters

        # TODO whether it is 2D / 3D value should be based on IOS/ uncalibrated Studio (for 2D) or calibrated Studio (3D), a tag will be provided in new jsons
        club_end_values = np.array([val["end"] for val in self.club_markers.values()])

        if club_end_values.shape[1] == 2:
            self.isClubmarkers3D = False
            logger.info(f"Computing 2D Club Speed")
            zero_array = np.array([0, 0])
        else:
            logger.info(f"Computing 3D Club Speed")
            self.isClubmarkers3D = True
            zero_array = np.array([0, 0, 0])

            # convert 3D markers into pixels
            for frame_dict in self.club_markers.values():
                for key, value in frame_dict.items():
                    frame_dict[key] = [int(v / self.pix_m_conversion) for v in value]
                    frame_dict[key][-1] = -1 * frame_dict[key][-1]
            club_end_values = np.array(
                [val["end"] for val in self.club_markers.values()]
            )

        clubhead_speed = (
            (club_end_values[1:] - club_end_values[:-1])
            * self.config["video"]["frame_rate"]
            * self.pix_m_conversion
        )

        # Pad with zero vector at the start
        clubhead_speed_vec = np.insert(clubhead_speed, 0, zero_array, axis=0)

        # Compute speed magnitude (m/s), then convert to mph
        clubhead_speed = norm(clubhead_speed_vec, axis=-1)
        clubhead_speed_mph = clubhead_speed * 2.23694

        self.club_speed = clubhead_speed_mph.tolist()

    def load_club_speed(self, ball_club_speed_path):
        speed = load_ball_club_speed_data(ball_club_speed_path)
        try:
            self.club_speed = speed["club_speed"]
        except:
            # raise Exception("No club speed data")
            logger.info("No Club Speed Data Available")
            self.club_speed = None

    # def get_2D_club_markers(self): #NOTE Might render unneccassary if 2D and 3D club markers cases are completely separated
    #     # updated_data = copy.deepcopy(data)  # To avoid modifying the original if needed

    #     try:
    #         self.ball_size = self.config["swing"]["launch_data"]["ball"]["ball_size"]
    #         self.ball_size = self.ball_size - self.ball_size * 0.165
    #         self.pix_m_conversion = 0.04267 / self.ball_size
    #     except:
    #         # logger.info("Ball Size is missing from config, assigning 24.5")
    #         logger.exception("Exception occurred while fetching ball size")
    #         # raise ValueError("Ball size is required in config but missing.")  #NOTE implement when new config is available -> All perhaps everything will be in meters
    #     for frame_dict in self.club_markers.values():
    #         for key, value in frame_dict.items():
    #             # Keep last 2 entries and multiply each by 10
    #             frame_dict[key] = [int(v / self.pix_m_conversion) for v in value[-2:]]


if __name__ == "__main__":
    import os
    from pathlib import Path

    current_dir = Path.cwd()
    parent_dir = current_dir.parent

    inputs_folder = os.path.join(parent_dir, r"inputs/")
    outputs_folder = os.path.join(parent_dir, r"outputs/")

    swing_id = r"m2c-dc1-1393-1750069325"

    # Json Paths
    cv_path = os.path.join(inputs_folder, swing_id + "_cv.json")
    ball_club_speed_path = os.path.join(inputs_folder, swing_id + "_ball.json")
    json3d_path = os.path.join(inputs_folder, swing_id + "_3d.json")  # 3d data
    print(json3d_path)
    json2d_path = os.path.join(inputs_folder, swing_id + "_2d.json")  # 2d data
    force_path = os.path.join(
        inputs_folder, swing_id + "_measured_force.json"
    )  # 2d data

    config = config()
    config.load_config(cv_path)

    # pose3d = pose3d()
    # pose3d.load_pose3d(json3d_path)
    # pose3d.assign_markers(config)

    pose2d = pose2d()  # NOTE: View FO.DTL
    pose2d.load_pose2d(json2d_path)
    pose2d.assign_markers()

    forces = forces()
    forces.load_forces(force_path)
    forces.assign_forces()

    # club = club()
    # club.load_club(cv_path)
    # club.load_speed(ball_club_speed_path)

    ball = ball()
    ball.load_speed(ball_club_speed_path)

    swing_session = swing_session()
    swing_session.load_config(cv_path)
    swing_session.load_pose3d(cv_path)
    swing_session.assign_3D_markers()
    swing_session.load_club_markers(cv_path)
    swing_session.load_club_speed(ball_club_speed_path)
