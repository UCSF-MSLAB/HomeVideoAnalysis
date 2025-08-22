import cv2
import os
import mediapipe as mp
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from PIL import Image
from Marigold.marigold import MarigoldPipeline
import diffusers
import torch
import imageio

mp4_file = os.getcwd() + "/tests/fixtures/gait_vertical_left.mov"
cap = cv2.VideoCapture(mp4_file)

i = 0
while i < 100:
    success, image = cap.read()
    i += 1

# -####################################
# - Mediapipe
# -####################################

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

pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)


def denormalize(df, box, margin):
    df.X = df.X * (box[2] - box[0] + 2*margin) + box[0] - margin
    df.Y = df.Y * (box[3] - box[1] + 2*margin) + box[1] - margin
    return df


def renameCols(col):
    lmNum = col.split('_')[0]
    lmVal = poseDict[int(lmNum)]
    return col.replace(lmNum, lmVal)


image.flags.writeable = False
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image)

tLndMrks = list(map(lambda lndMrk: (lndMrk.x, lndMrk.y,
                                    lndMrk.z, lndMrk.visibility,
                                    lndMrk.presence) if lndMrk else None,
                    results.pose_landmarks.landmark)) # pose_world_landmarks

opose_res = pd.DataFrame(tLndMrks,
                         columns=['X', 'Y', 'Z', 'vis', 'pres'])

opose_res['frame'] = [i] * len(poseDict)
opose_res['label'] = poseDict.values()

opose_res = denormalize(opose_res, [0, 0, image.shape[1], image.shape[0]], 0)

ax = sb.scatterplot(data=opose_res, x='X', y='Y')


def label_points(df):
    for i, point in df.iterrows():
        if point['X'] != 0 and point['Y'] != 0:
            ax.text(point['X'] + .001,
                    point['Y'] + .001,
                    str(point['label']))


label_points(opose_res)

# -####################################
# - Marigold
# -####################################

# pipe = MarigoldPipeline.from_pretrained("prs-eth/marigold-depth-lcm-v1-0")
# output = pipe(Image.fromarray(image))
# depth_image = output['depth_colored']
# depth_data = output['depth_np']

# plt.imshow(np.asarray(depth_image))
# plt.show()

# with latents

reader = imageio.get_reader(mp4_file)
size = reader.get_meta_data()['size']
last_frame_latent = None
latent_common = torch.randn(
        (1, 4, 768 * size[1] // (8 * max(size)), 768 * size[0] // (8 * max(size)))
    )

latents = latent_common
if last_frame_latent is not None:
    latents = .9 * latents + .1*last_frame_latent

pipe = diffusers.MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-1")
pipe.vae = diffusers.AutoencoderTiny.from_pretrained("madebyollin/taesd")

output = pipe(Image.fromarray(image),
              match_input_resolution=False,
              latents=latents,
              output_latent=True)

np_out = output.prediction.reshape(size[1], size[0])

plt.imshow(np_out)
plt.show()

# -####################################
# - together
# -####################################

lndmrk_px_vals = []

for i, row in opose_res.iterrows():
    if row['X'] != 0 and row['Y'] != 0:
        depth_est = np_out[int(row.Y), int(row.X)]
        lndmrk_px_vals.append({'frame': i, 'label': row['label'],
                               'depth_est': depth_est})

pd.DataFrame(lndmrk_px_vals)
