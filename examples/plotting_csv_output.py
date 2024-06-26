import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
# from whittaker_eilers import WhittakerSmoother


# mpipe_frontal = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/Athletic Male Standard Walk Animation Reference Body Mechanics_mediapipe_0.csv")


mpipe_lateral = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/Athletic Male Standard Walk Animation Reference Body Mechanics_mediapipe_1.csv")

mpipe_lateral['model'] = ['mpipe'] * mpipe_lateral.shape[0]

yolo_lateral = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/Athletic Male Standard Walk Animation Reference Body Mechanics_yolo_1.csv")

yolo_lateral['model'] = ['yolo'] * yolo_lateral.shape[0]

sb.lineplot(data=mpipe_lateral[mpipe_lateral.label == 'right_wrist'],
            x="frame", y="Y")

plt.show()

ankle_diff_df = pd.concat([mpipe_lateral[mpipe_lateral.label == 'left_ankle'],
                           yolo_lateral[yolo_lateral.label == 'left_ankel']])

sb.lineplot(data=ankle_diff_df, x="frame", y="Y", hue='model')

plt.show()

FRAME = 150
frame_diff_df = pd.concat([mpipe_lateral[mpipe_lateral.frame == FRAME],
                           yolo_lateral[yolo_lateral.frame == FRAME]])

ax = sb.scatterplot(data=frame_diff_df,
                    x='X', y='Y', hue='model')


def label_points(df):
    for i, point in df.iterrows():
        if point['X'] != 0 and point['Y'] != 0:
            ax.text(point['X'] + .001,
                    point['Y'] + .001,
                    str(point['model']+"_"+point['label']))


label_points(frame_diff_df)

plt.show()

# MSE across all shared points:

joined_dfs = yolo_lateral.merge(mpipe_lateral,
                                on=['label', 'frame'],
                                how='left')
joined_dfs['err'] = np.sqrt((joined_dfs['X_x'] - joined_dfs['X_y'])**2 +
                            (joined_dfs['Y_x'] - joined_dfs['Y_y'])**2)

frame_err = joined_dfs[joined_dfs.vis > .90].groupby('frame')['err'].agg(['min','max','mean','median'])
label_err = joined_dfs[joined_dfs.vis > .90].groupby('label')['err'].agg(['min','max','mean','median'])

sb.lineplot(data=frame_err[['mean','median']])

sb.barplot(data=label_err[['mean','median']],
           x="label", y="median")


# #############################################
# STRIDE LENGTH
# #############################################

sb.lineplot(data=mpipe_lateral[mpipe_lateral.label == 'right_heel'],
            x="frame", y="Y")
