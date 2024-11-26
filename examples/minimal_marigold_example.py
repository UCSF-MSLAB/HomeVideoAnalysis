
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
depth_image = np.asarray(output['depth_colored'])
depth_image[0:10, 0:10]

plt.imshow(depth_image)
plt.show()
