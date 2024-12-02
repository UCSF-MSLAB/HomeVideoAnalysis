
# https://towardsdatascience.com/how-to-estimate-depth-from-a-single-image-7f421d86b22d

import cv2
import os
# import diffusers
# import torch
from PIL import Image
from Marigold.marigold import MarigoldPipeline
import numpy as np
import matplotlib.pyplot as plt

mp4_file = os.getcwd() + "/tests/fixtures/gait_vertical_left.mov"
cap = cv2.VideoCapture(mp4_file)

i = 0
while i < 100:
    success, image = cap.read()
    i += 1

image.flags.writeable = False
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.imshow(image)
# plt.show()

pipe = MarigoldPipeline.from_pretrained("prs-eth/marigold-depth-lcm-v1-0")
output = pipe(Image.fromarray(image))
depth_image = output['depth_colored']
depth_data = np.asarray(depth_image)
depth_data[0:10, 0:10]

depth_image.size

plt.imshow(output['depth_np'])
plt.show()


class MarigoldResultHandler():

    def __init__(self, i, marigold_output):

        self.frame = i
        self.data = marigold_output['depth_np']

    def extract(self, mpipe_lndmrks):

        lndmrk_px_vals = []

        for i, row in mpipe_lndmrks.iterrows():
            if row['X'] != 0 and row['Y'] != 0:
                depth_est = self.data[int(row.Y), int(row.X)]
                lndmrk_px_vals.append({'frame': self.i, 'label': row['label'],
                                       'depth_est': depth_est})

        return pd.DataFrame(lndmrk_px_vals)
