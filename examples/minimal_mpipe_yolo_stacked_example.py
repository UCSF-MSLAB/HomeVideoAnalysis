import os
import sys
import cv2
import mediapipe as mp
from ultralytics import YOLO
import pandas as pd

from src.yolo_result_handler import YoloResultHandler as yrh
from src.mpipe_result_handler import MPipeResultHandler as mprh

MARGIN = 10

MP4_PATH = os.getcwd() + "/tests/fixtures/Athletic Male Standard Walk Animation Reference Body Mechanics.mp4"

# MP4_PATH = os.getcwd() + "/tests/fixtures/Phases of Walking Gait.mp4"

yolo_pose = YOLO(os.getcwd() + '/models/yolov8m-pose.pt')
mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

cap = cv2.VideoCapture(MP4_PATH)

success, image = cap.read()
image.flags.writeable = False
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

yolo_results = yolo_pose(image)
yolo_handle = yrh(0, yolo_results)

tmp = [yolo_handle.dfs] * 10
tmp3 = [pd.concat(dfs) for dfs in list(zip(*tmp))]

# get bounds for the mediapipe image
mpipe_dfs = []
for (xmin, ymin, xmax, ymax) in yolo_handle.boxes:
    img_crop = image[int(ymin)+MARGIN:int(ymax)+MARGIN,
                     int(xmin)+MARGIN:int(xmax)+MARGIN:]
    mpipe_results = mp_pose.process(img_crop)
    mpipe_handle = mprh(0, mpipe_results)
    mpipe_dfs.append(mpipe_handle.df)

# cadence, step length, step width, velocity, velocity peak(s)
# total travel, displacement,
