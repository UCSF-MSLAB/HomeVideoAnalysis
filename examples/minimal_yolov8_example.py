from ultralytics import YOLO
import pandas as pd
import os

model = YOLO(os.getcwd() + '/models/yolov8m-pose.pt')
vid_path = os.getcwd() + "/tests/fixtures/gait_vertical_left.mov"

LABELS = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle"
    ]

results = model(source=vid_path, show=True, conf=0.3)

yolo_res = pd.DataFrame(results[0].keypoints[0].xyn[0].numpy(),
                        columns=['X', 'Y'])
yolo_res['Y'] = yolo_res['Y'] * (-1) + 1
yolo_res['label'] = LABELS
yolo_res['frame'] = [0] * len(LABELS)

results[0].boxes.xyxy
