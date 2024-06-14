import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from src.analysis_funs import get_landmark_pairs
from scipy.ndimage import gaussian_filter1d

# ############################################
# upper body speed
# #############################################

mpipe_lateral = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/gait_horizontal_left_mediapipe_0.csv")

# Constants
# #############################################

# feet
FOREARM_DIST = 1

# this is the default for iphones (source?)
FPS = 30

# determine which side is closest to camera:
# #############################################

hip_df = get_landmark_pairs(mpipe_lateral, 'left_hip', 'right_hip')
hip_df['Left_Hip'] = hip_df.Z_x.rolling(20).median()
hip_df['Right_Hip'] = hip_df.Z_y.rolling(20).median()

sb.lineplot(hip_df[['Left_Hip', 'Right_Hip']])
plt.ylabel("Distance (PX) from Camera")
plt.xlabel("Frame Number")
plt.title("Plot 1. Determining Model Side")
plt.show()

hip_df['left_closer'] = hip_df.Left_Hip < hip_df.Right_Hip

sb.lineplot(hip_df, x='frame', y='left_closer')
plt.show()

frame_to_side = hip_df[['frame', 'left_closer']].copy()
frame_to_side["right_closer"] = ~frame_to_side.left_closer

# determine turns (label with id)
# #############################################

frame_to_side['turn_right_id'] = frame_to_side.left_closer.cumsum() * frame_to_side.right_closer
frame_to_side['turn_left_id'] = frame_to_side.right_closer.cumsum() * frame_to_side.left_closer
frame_to_side['turn_id'] = (frame_to_side.turn_right_id * frame_to_side.right_closer +
                            frame_to_side.turn_left_id * frame_to_side.left_closer)

sb.lineplot(frame_to_side, x='frame', y='turn_id')
plt.xlabel("Frame Number")
plt.ylabel("Between-Turn Grouping")
plt.title("Plot 2. Labeling Which Frames to Group Between Turns")
plt.show()

# merge sides / turns into dataset
# #############################################

mpipe_lateral = mpipe_lateral.merge(frame_to_side, on='frame', how='left')

# calculate inches per pixel
# #############################################

right_forearm_df = get_landmark_pairs(mpipe_lateral[mpipe_lateral.right_closer],
                                      'right_wrist', 'right_elbow')
right_forearm_df['dist'] = np.sqrt((right_forearm_df.X_x - right_forearm_df.X_y)**2 +
                                   (right_forearm_df.Y_x - right_forearm_df.Y_y)**2)

left_forearm_df = get_landmark_pairs(mpipe_lateral[mpipe_lateral.left_closer],
                                     'left_wrist', 'left_elbow')
left_forearm_df['dist'] = np.sqrt((left_forearm_df.X_x - left_forearm_df.X_y)**2 +
                                  (left_forearm_df.Y_x - left_forearm_df.Y_y)**2)

INCH_PER_PX = (FOREARM_DIST / right_forearm_df.dist.median() +
               FOREARM_DIST / left_forearm_df.dist.median()) / 2

# calculate position of upper body
# #############################################

upr_bdy = ['_hip', '_shoulder', '_ear']


def calc_sided_pos(df):
    if all(df.left_closer):
        return df[df.label.isin(['left'+part for part in upr_bdy])].X.mean()
    return df[df.label.isin(['right'+part for part in upr_bdy])].X.mean()


frame_to_side["upr_body_X"] = mpipe_lateral.groupby(['frame']).apply(calc_sided_pos, include_groups=False).values

sb.lineplot(frame_to_side, x='frame', y='upr_body_X')
plt.xlabel("Frame Number")
plt.ylabel("X (PX)")
plt.title("Plot 3. Torso Position (Note Discontinuity at Turns)")
plt.show()

# calculate change in X within turn-groups
# #############################################

turns = frame_to_side.turn_id.unique()

def calc_abs_diff(turn_id):
    return np.abs(frame_to_side[frame_to_side.turn_id == turn_id]
                  .upr_body_X
                  .diff())

dXs = [calc_abs_diff(turn_id).values[1:] for turn_id in turns]
dX = np.concatenate(dXs) * INCH_PER_PX * FPS

sb.lineplot(dX)
plt.xlabel("Frame Number")
plt.ylabel("Change in X (Feet/Second)")
plt.title("Plot 4. Raw Horizontal Torso Speed")
plt.show()

# apply a basic filter:

dX = [dx for dx in dX if dx < 7.33]

sb.lineplot(dX)
plt.xlabel("Frame Number")
plt.ylabel("Change in X (Feet/Second)")
plt.title("Plot 5. First Pass Filtered Horizontal Torso Speed")
plt.show()

# apply gaussian filter
dX_gf = gaussian_filter1d(dX, 10)

sb.lineplot(dX_gf)
plt.xlabel("Frame Number")
plt.ylabel("Change in X (Feet/Second)")
plt.title("Plot 6. Gaussian Filtered Horizontal Torso Speed")
plt.show()


max(dX_gf)
