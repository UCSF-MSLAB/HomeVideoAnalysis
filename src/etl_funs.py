import os
import cv2
import mediapipe as mp
from ultralytics import YOLO
import pandas as pd
from src.yolo_result_handler import YoloResultHandler as init_yolo_hndl
from src.mpipe_result_handler import MPipeResultHandler as init_mpipe_hndl

MARGIN = 10

yolo_pose = YOLO(os.getcwd() + '/models/yolov8m-pose.pt')
mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)


def extract_pose_data(vid_in_path):

    cap = cv2.VideoCapture(vid_in_path)
    yolo_pose_data = []
    mpipe_pose_data = []

    print(f"Processing {vid_in_path}...")
    while cap.isOpened():

        success, image = cap.read()
        if not success:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yolo_results = yolo_pose(image)
        yolo_handler = init_yolo_hndl(0, yolo_results)

        yolo_pose_data.append(yolo_handler.dfs)

        # get bounds for the mediapipe image
        mpipe_dfs = []
        for (xmin, ymin, xmax, ymax) in yolo_handler.boxes:
            img_crop = image[int(ymin)+MARGIN:int(ymax)+MARGIN,
                             int(xmin)+MARGIN:int(xmax)+MARGIN:]
            mpipe_results = mp_pose.process(img_crop)
            mpipe_handler = init_mpipe_hndl(0, mpipe_results)
            mpipe_dfs.append(mpipe_handler.df)

        mpipe_pose_data.append(mpipe_dfs)

    cap.release()
    print("Processing complete...")

    return (yolo_pose_data, mpipe_pose_data)


def transform_pose_data(raw_pose_data):
    """
    Note: this will return an error if the figure count isn't consistent
    between frames. Also, tracking may be necessary if more complicated
    figure/movement videos are used.

    Otherwise, a list of dataframes with length equal to the number of
    figures is returned.
    """
    # check that each frame has the same number of figures:
    fig_count = [len(figures) for figures in raw_pose_data]

    if fig_count.count(fig_count[0]) != len(fig_count):
        raise ValueError('Figure count was not consistent between frames')

    indv_pose_dfs = [pd.concat(dfs) for dfs in list(zip(*raw_pose_data))]

    return indv_pose_dfs


def load_pose_data(pose_data, data_name, data_out_path):

    for i, df in enumerate(pose_data):
        df.to_csv(data_out_path + f'/{data_name}_{i}.csv',
                  sep=",")

    pass
