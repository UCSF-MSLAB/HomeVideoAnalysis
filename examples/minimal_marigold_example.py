
# https://towardsdatascience.com/how-to-estimate-depth-from-a-single-image-7f421d86b22d

import cv2
import os
# import diffusers
# import torch
from PIL import Image
from Marigold.marigold import MarigoldPipeline
import numpy as np
import matplotlib.pyplot as plt
import torch

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

pipe = MarigoldPipeline.from_pretrained("prs-eth/marigold-depth-lcm-v1-0", half_precision=True)
start = time.time()
output = pipe(Image.fromarray(image), ensemble_size=1, denoising_steps=1)
end = time.time()
print(end - start)
depth_image = output['depth_colored']
depth_data = np.asarray(depth_image)
depth_data[0:10, 0:10]

depth_image.size

plt.imshow(output['depth_np'])
plt.show()

# ########################
# multiprocessing
# ########################

import multiprocessing

cap = cv2.VideoCapture(mp4_file)
i = 0
images = []
while i < 100:
    success, image = cap.read()
    if success:
        images.append((i, image))
        i += 1
