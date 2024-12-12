import os
import cv2
from PIL import Image
import mediapipe as mp
from ultralytics import YOLO
import pandas as pd
from src.yolo_result_handler import YoloResultHandler as init_yolo_hndl
from src.mpipe_result_handler import MPipeResultHandler as init_mpipe_hndl
from src.marigold_result_handler import MarigoldResultHandler as init_mari_hndl
from Marigold.marigold import MarigoldPipeline
import logging
# import torch
# import itertools
logger = logging.getLogger(__name__)

# encounter errors when box includes image edge
MARGIN = 0
FPS = 30

yolo_pipe = YOLO(os.getcwd() + '/models/yolov8m-pose.pt')
mp_pipe = mp.solutions.pose.Pose(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
marigold_pipe = MarigoldPipeline \
    .from_pretrained("prs-eth/marigold-depth-lcm-v1-0")
# , variant='fp16', torch_dtype=torch.float16
marigold_pipe.set_progress_bar_config(disable=True)


def extract_pose_data(vid_in_path):

    cap = cv2.VideoCapture(vid_in_path)
    yolo_pose_data = []
    mpipe_pose_data = []
    mpipe_world_data = []
    mari_depth_data = []

    print(f"Processing {vid_in_path}...")
    frame_i = 0
    while cap.isOpened():

        success, image = cap.read()
        if not success:
            break

        pipe_results = run_pipes_on_frame(image, frame_i)
        if pipe_results is not None:
            yolo_pose_data.append(pipe_results[0])
            mpipe_pose_data.append(pipe_results[1])
            mpipe_world_data.append(pipe_results[2])
            mari_depth_data.append(pipe_results[3])

        frame_i += 1

    cap.release()
    print("Processing complete...")
    logger.info(f'Processed {frame_i} frames in {vid_in_path}')
    return [yolo_pose_data, mpipe_pose_data, mpipe_world_data, mari_depth_data]


def run_pipes_on_frame(image, frame_i):

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yolo_results = yolo_pipe(image)
    if len(yolo_results[0].boxes) > 0:
        yolo_handler = init_yolo_hndl(frame_i, yolo_results)
        if yolo_handler.box is not None:

            # get bounds for the mediapipe image
            mpipe_pose_df, mpipe_world_df = \
                extract_mpipe_pose_data(image,
                                        yolo_handler.box,
                                        frame_i,
                                        True)
            # perform depth estimation
            # since this takes so long, let's only do it 1x per second
            if frame_i % FPS == 0:
                mari_output = marigold_pipe(Image.fromarray(image))
            else:
                mari_output = None
            mari_depth_df = init_mari_hndl(frame_i, mari_output) \
                .extract(mpipe_pose_df)
            return [yolo_handler.dfs[0], mpipe_pose_df,
                    mpipe_world_df, mari_depth_df]
    return None


def mpipe_process(image, frame):

    mpipe_results = mp_pipe.process(image)
    if not mpipe_results.pose_landmarks:
        # rerun if nothing was found (this is a bug, I think)
        mpipe_results = mp_pipe.process(image)
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
