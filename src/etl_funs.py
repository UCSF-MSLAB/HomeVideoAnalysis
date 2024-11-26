import os
import cv2
import mediapipe as mp
from ultralytics import YOLO
import pandas as pd
from src.yolo_result_handler import YoloResultHandler as init_yolo_hndl
from src.mpipe_result_handler import MPipeResultHandler as init_mpipe_hndl
from Marigold.marigold import MarigoldPipeline
import logging
# import itertools
logger = logging.getLogger(__name__)

# encounter errors when box includes image edge
MARGIN = 0

yolo_pose = YOLO(os.getcwd() + '/models/yolov8m-pose.pt')
mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
marigold_pipe = MarigoldPipeline.from_pretrained("Bingxin/Marigold")


def extract_pose_data(vid_in_path):

    cap = cv2.VideoCapture(vid_in_path)
    yolo_pose_data = []
    mpipe_pose_data = []
    mpipe_world_data = []

    print(f"Processing {vid_in_path}...")
    frame_i = 0
    while cap.isOpened():

        success, image = cap.read()
        if not success:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yolo_results = yolo_pose(image)
        if len(yolo_results[0].boxes) > 0:
            yolo_handler = init_yolo_hndl(frame_i, yolo_results)
            if yolo_handler.box is not None:
                yolo_pose_data.append(yolo_handler.dfs[0])
                # get bounds for the mediapipe image

                mpipe_pose_df, mpipe_world_df = \
                    extract_mpipe_pose_data(image,
                                            yolo_handler.box,
                                            frame_i,
                                            False)
                
                # perform depth estimation
                depth_est = marigold_pipe(image)['depth_colored']
                
                
                mpipe_pose_data.append(mpipe_pose_df)
                mpipe_world_data.append(mpipe_world_df)

        frame_i += 1

    cap.release()
    print("Processing complete...")
    logger.info(f'Processed {frame_i} frames in {vid_in_path}')
    return [yolo_pose_data, mpipe_pose_data, mpipe_world_data]


def mpipe_process(image, frame):

    mpipe_results = mp_pose.process(image)
    if not mpipe_results.pose_landmarks:
        # rerun if nothing was found (this is a bug, I think)
        mpipe_results = mp_pose.process(image)
    return init_mpipe_hndl(frame, mpipe_results)


def extract_mpipe_pose_data(image, box, frame, deref):

    if len(box) > 0:
        xmin, ymin, xmax, ymax = box

    else:
        xmin, ymin = (0, 0)
        ymax, xmax = image.shape[:2]

    mpipe_handler = mpipe_process(image[int(ymin)-MARGIN:int(ymax)+MARGIN,
                                        int(xmin)-MARGIN:int(xmax)+MARGIN:],
                                  frame)

    if deref:
        mpipe_handler.denormalize((int(xmin), int(ymin),
                                   int(xmax), int(ymax)),
                                  MARGIN)
    return [mpipe_handler.df, mpipe_handler.world_df]


def transform_pose_data(raw_pose_data):
    # if any of the models are missing data, the output will be empty
    # indv_pose_dfs = [pd.concat(dfs) for dfs in list(zip(*raw_pose_data))]
    indv_pose_dfs = pd.concat(raw_pose_data)

    return indv_pose_dfs


def load_pose_data(pose_data, data_out_prefix):

    pose_data.to_csv(data_out_prefix + ".csv", sep=",")

    pass
