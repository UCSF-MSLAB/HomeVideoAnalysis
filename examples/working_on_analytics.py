import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from whittaker_eilers import WhittakerSmoother
from src.analysis_funs import calc_stride, get_landmark_pairs
from scipy.fftpack import fftfreq, fft, ifft
# #############################################
# STRIDE LENGTH
# #############################################

# mpipe_lateral = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/Athletic Male Standard Walk Animation Reference Body Mechanics_mediapipe_1.csv")

# mpipe_lateral['model'] = ['mpipe'] * mpipe_lateral.shape[0]

# #############################################
# MPIPE
# #############################################

mpipe_lateral = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/Mature Aged Woman Walk Casual - Slow Motion Animation Reference Body Mechanics_mediapipe_1.csv")
mpipe_lateral = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/Phases of Walking Gait_mediapipe_0.csv")
mpipe_lateral = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/gait_horizontal_left_mediapipe_0.csv")
yolo_lateral = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/gait_horizontal_left_yolo_0.csv")


sb.scatterplot(mpipe_lateral[mpipe_lateral['frame'] == 10], x='X', y='Y')
plt.show()

SIDE = 'right'

# #############################################
# Distinguish which side is closer to cam
# #############################################

heel_heel_df = get_landmark_pairs(mpipe_lateral, 'left_heel', 'right_heel')
heel_heel_df['Z_x_md'] = heel_heel_df.Z_x.rolling(20).median()
heel_heel_df['Z_y_md'] = heel_heel_df.Z_y.rolling(20).median()

sb.lineplot(heel_heel_df[['Z_x_md', 'Z_y_md']])
plt.show()

heel_heel_df['left_closer'] = heel_heel_df.Z_x_md < heel_heel_df.Z_y_md

sb.lineplot(heel_heel_df, x='frame', y='left_closer')
plt.show()

frame_to_side = heel_heel_df[['frame', 'left_closer']].copy()

# #############################################
# calculate reasonable errors ?
# #############################################

# left

left_heel_ankle_df = get_landmark_pairs(mpipe_lateral, 'left_heel', 'left_ankle')
left_heel_ankle_df = left_heel_ankle_df.merge(frame_to_side, on='frame', how='left')
left_heel_ankle_df = left_heel_ankle_df[left_heel_ankle_df['left_closer']]

sb.lineplot(left_heel_ankle_df[['X_y', 'X_x']])
plt.show()

left_heel_ankle_df['dist'] = np.sqrt((left_heel_ankle_df.X_x - left_heel_ankle_df.X_y)**2 +
                                     (left_heel_ankle_df.Y_x - left_heel_ankle_df.Y_y)**2)

sb.lineplot(left_heel_ankle_df[['dist']])
plt.show()

left_heel_ankle_df['dist'].std()

right_heel_ankle_df = get_landmark_pairs(mpipe_lateral, 'right_heel', 'right_ankle')
right_heel_ankle_df = right_heel_ankle_df.merge(frame_to_side, on='frame', how='right')
right_heel_ankle_df = right_heel_ankle_df[~right_heel_ankle_df['left_closer']]

sb.lineplot(right_heel_ankle_df[['X_y', 'X_x']])
plt.show()

right_heel_ankle_df['dist'] = np.sqrt((right_heel_ankle_df.X_x - right_heel_ankle_df.X_y)**2 +
                                      (right_heel_ankle_df.Y_x - right_heel_ankle_df.Y_y)**2)

sb.lineplot(right_heel_ankle_df[['dist']])
plt.show()

right_heel_ankle_df['dist'].std()

# #############################################
# can yolo provide bounds?
# #############################################

yolo_lateral = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/gait_horizontal_left_yolo_0.csv")

right_heel_df = mpipe_lateral[(mpipe_lateral.label == 'right_heel')].copy()

right_heel_df = right_heel_df.reset_index(drop=True)

right_heel_df = right_heel_df.merge(yolo_lateral[yolo_lateral.label == 'right_ankle'],
                                    on='frame', how='left')

sb.lineplot(right_heel_df[['X_x', 'X_y']])
plt.show()

## FFT smoothing

# sig_fft = fft(heel_df.Y.values)
# sig_fft_filt = sig_fft.copy()
# freq = fftfreq(heel_df.shape[0])

# sig_fft_filt[np.abs(freq) > 1/40] = 0
# filtered = ifft(sig_fft_filt)
# sb.lineplot(filtered)
# plt.show()

# #############################################
# find floor?
# #############################################

# idea:
vals = []
band = (heel_ankle_df['dist'].mean() / 2) + heel_ankle_df['dist'].std()*2
for i in range(0, int(max(heel_df.Y))):
    vals.append((i, heel_df[((i - band) < heel_df.Y) &
                            (heel_df.Y < (i+band))].shape[0]))

vals_df = pd.DataFrame(vals, columns=['Y', 'count'])

sb.lineplot(vals_df, x='Y', y='count')
plt.show()

floor_Y = vals_df[vals_df['count'] == np.max(vals_df['count'])].Y.values[0]

heel_df['heel_on_floor'] = ((heel_df['Y'] > floor_Y - band) &
                            (heel_df['Y'] < floor_Y + band))
heel_df['heel_on_floor_md'] = heel_df.heel_on_floor.rolling(10).median()

sb.lineplot(heel_df, x='frame', y='heel_on_floor_md')
plt.show()

# #############################################
# dx?
# #############################################

band = right_heel_ankle_df['dist'].std()
right_heel_df['dX_x'] = right_heel_df.X_x.diff()
right_heel_df['dX_y'] = right_heel_df.X_y.diff()

sb.lineplot(data=right_heel_df[['dX_x', 'dX_y']])
plt.show()


right_heel_df = right_heel_df.merge(frame_to_side,
                                    on="frame",
                                    how='left')


## rolling quantile

heel_df['dX_roll_q95'] = heel_df.dX.rolling(20).quantile(.95)

sb.lineplot(data=heel_df, x='frame', y='dX_roll_q95')
plt.show()


right_heel_df['dX_0'] = ((right_heel_df['dX_x'] > 0 - band) &
                         (right_heel_df['dX_x'] < 0 + band))
heel_df['heel_stopped'] = heel_df.dX_0 * heel_df.heel_on_floor_md

heel_df['heel_stopped_roll'] = heel_df.heel_stopped.rolling(5).max()
sb.lineplot(heel_df, x='frame', y='dX_0')
plt.show()

# #############################################
# upper body speed
# #############################################

# Constants
FOREARM_DIST = 12
# this is the default for iphones (source?)
FPS = 30

# determine which side is closest to camera:
hip_df = get_landmark_pairs(mpipe_lateral, 'left_hip', 'right_hip')
hip_df['Z_x_md'] = hip_df.Z_x.rolling(20).median()
hip_df['Z_y_md'] = hip_df.Z_y.rolling(20).median()

sb.lineplot(hip_df[['Z_x_md', 'Z_y_md']])
plt.show()

hip_df['left_closer'] = hip_df.Z_x_md < hip_df.Z_y_md

sb.lineplot(hip_df, x='frame', y='left_closer')
plt.show()

frame_to_side = hip_df[['frame', 'left_closer']].copy()
frame_to_side["right_closer"] = ~frame_to_side.left_closer

# determine turns

frame_to_side['turn_right_id'] = frame_to_side.left_closer.cumsum() * frame_to_side.right_closer
frame_to_side['turn_left_id'] = frame_to_side.right_closer.cumsum() * frame_to_side.left_closer

sb.lineplot(frame_to_side, x='frame', y ='turn_left')
plt.show()

mpipe_lateral = mpipe_lateral.merge(frame_to_side, on='frame', how='left')

# determine a conversion of pixels to feet (inches)
# say forearm to elbow is 12''
FOREARM_DIST = 12

right_forearm_df = get_landmark_pairs(mpipe_lateral[mpipe_lateral.right_closer],
                                      'right_wrist', 'right_elbow')
right_forearm_df['dist'] = np.sqrt((right_forearm_df.X_x - right_forearm_df.X_y)**2 +
                                   (right_forearm_df.Y_x - right_forearm_df.Y_y)**2)

left_forearm_df = get_landmark_pairs(mpipe_lateral[mpipe_lateral.left_closer],
                                     'left_wrist', 'left_elbow')
left_forearm_df['dist'] = np.sqrt((left_forearm_df.X_x - left_forearm_df.X_y)**2 +
                                  (left_forearm_df.Y_x - left_forearm_df.Y_y)**2)

sb.lineplot(right_forearm_df, x='frame', y='dist')
plt.show()

sb.lineplot(left_forearm_df, x='frame', y='dist')
plt.show()

INCH_PER_PX = (FOREARM_DIST / right_forearm_df.dist.median() +
               FOREARM_DIST / left_forearm_df.dist.median()) / 2

# calculate average speed of upper body when left is closer

from scipy.ndimage import gaussian_filter1d
upr_bdy_labs = ['left_hip', 'left_shoulder', 'left_ear']

upr_bdy_lat_left = mpipe_lateral[mpipe_lateral.left_closer &
                                 mpipe_lateral.label.isin(upr_bdy_labs)]

upr_body_lat_left_x = upr_bdy_lat_left.groupby(['frame'])['X'].mean().values
upr_body_lat_left = pd.DataFrame({'frame': })

upr_body_lat_left_x.rename({''})

sb.lineplot(upr_body_lat_left_x)
plt.show()

gaussian_filter
