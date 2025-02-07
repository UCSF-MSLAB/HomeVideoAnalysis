
# https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage#frame-by-frame-video-processing-with-temporal-consistency
import cv2
import os
# import diffusers
# import torch
import imageio
from PIL import Image
# from Marigold.marigold import MarigoldPipeline
import numpy as np
import matplotlib.pyplot as plt
import torch
import diffusers


pipe = diffusers.MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-lcm-v1-0")
pipe.vae = diffusers.AutoencoderTiny.from_pretrained("madebyollin/taesd")

mp4_file = os.getcwd() + "/tests/fixtures/gait_vertical_left.mov"
cap = cv2.VideoCapture(mp4_file)

i = 0
while i < 100:
    success, image = cap.read()
    i += 1

image.flags.writeable = False
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

reader = imageio.get_reader(mp4_file)
size = reader.get_meta_data()['size']
last_frame_latent = None
latent_common = torch.randn(
        (1, 4, 768 * size[1] // (8 * max(size)), 768 * size[0] // (8 * max(size)))
    )

latents = latent_common
if last_frame_latent is not None:
    latents = .9 * latents + .1*last_frame_latent

output = pipe(Image.fromarray(image),
              match_input_resolution=False,
              latents=latents,
              output_latent=True)

depth_image = output['depth_colored']
depth_data = np.asarray(depth_image)
depth_data[0:10, 0:10]

depth_image.size

plt.imshow(output['depth_np'])
plt.show()

