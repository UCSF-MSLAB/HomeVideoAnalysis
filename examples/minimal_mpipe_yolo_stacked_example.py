import os
import cv2
import mediapipe as mp
from ultralytics import YOLO
import pandas as pd

import sys
sys.path.append(os.getcwd())

from src.yolo_result_handler import YoloResultHandler as yrh
from src.mpipe_result_handler import MPipeResultHandler as mprh

MARGIN = 10

MP4_PATH = os.getcwd() + "/tests/fixtures/Athletic Male Standard Walk Animation Reference Body Mechanics.mp4"

# MP4_PATH = os.getcwd() + "/tests/fixtures/Phases of Walking Gait.mp4"

yolo_pose = YOLO(os.getcwd() + '/models/yolov8m-pose.pt')
mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

yolo_data = []
mpipe_data = []

cap = cv2.VideoCapture(MP4_PATH)
i = 0
bad = []
good = []
while cap.isOpened():

    success, image = cap.read()
    if not success:
        break
    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    yolo_results = yolo_pose(image)
    yolo_handle = yrh(i, yolo_results)

    yolo_data.append(yolo_handle.dfs)
    
    # get bounds for the mediapipe image
    mpipe_dfs = []
    for (xmin, ymin, xmax, ymax) in yolo_handle.boxes:
        img_crop = image[int(ymin)+MARGIN:int(ymax)+MARGIN,
                         int(xmin)+MARGIN:int(xmax)+MARGIN:]
        mpipe_results = mp_pose.process(img_crop)
        if not mpipe_results.pose_landmarks:
            mpipe_results = mp_pose.process(img_crop)
        mpipe_handle = mprh(i, mpipe_results)
        if isinstance(mpipe_handle.df, pd.DataFrame):
            mpipe_dfs.append(mpipe_handle.df)
            good.append(i)
        else:
            bad.append(i)
    mpipe_data.append(mpipe_dfs)
    i += 1

cap.release()
print("Good frames:\n")
print(good)
print("Bad frames:\n")
print(bad)
# cadence, step length, step width, velocity, velocity peak(s)
# total travel, displacement,

# from src.etl_funs import (extract_pose_data, transform_pose_data)

# data_sets = extract_pose_data(MP4_PATH)
# yolo = transform_pose_data(data_sets[0])
# mpipe = transform_pose_data(data_sets[1])

# ValueError: Model disagreement found in /home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tests/fixtures/Athletic Male Standard Walk Animation Reference Body Mechanics.mp4 frame 39 boxes: [tensor([432.,  40., 526., 331.]), tensor([ 29.,  39., 197., 325.])]
