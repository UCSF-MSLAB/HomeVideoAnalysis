import cv2
import mediapipe as mp
import sys
import pandas as pd
import os

poseDict = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index"
}

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def renameCols(col):
    lmNum = col.split('_')[0]
    lmVal = poseDict[int(lmNum)]
    return col.replace(lmNum, lmVal)

mp4_file = os.getcwd() + "/tests/fixtures/Athletic Male Standard Walk Animation Reference Body Mechanics.mp4"
cap = cv2.VideoCapture(mp4_file)

pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

success, image = cap.read()
image.flags.writeable = False
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image)

tLndMrks = list(map(lambda lndMrk: (lndMrk.x, lndMrk.y,
                                    lndMrk.z, lndMrk.visibility,
                                    lndMrk.presence) if lndMrk else None,
                    results.pose_landmarks.landmark))

opose_res = pd.DataFrame(tLndMrks,
                         columns=['X', 'Y', 'Z', 'vis', 'pres'])
opose_res['frame'] = [0] * len(poseDict)
opose_res['label'] = poseDict.values()
