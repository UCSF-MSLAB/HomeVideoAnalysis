import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
# from whittaker_eilers import WhittakerSmoother


# mpipe_frontal = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/Athletic Male Standard Walk Animation Reference Body Mechanics_mediapipe_0.csv")

mpipe_frontal = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/gait_vertical_left_mediapipe.csv")

mpipe_frontal['model'] = ['mpipe'] * mpipe_frontal.shape[0]
tmp_df = mpipe_frontal[mpipe_frontal.frame == 90]
ax = sb.scatterplot(data=tmp_df, x="X", y="Y")

def label_points(df):
    for i, point in df.iterrows():
        if point['X'] != 0 and point['Y'] != 0:
            ax.text(point['X'] + .001,
                    point['Y'] + .001,
                    str(point['model']+"_"+point['label']))


label_points(tmp_df)

plt.show()

# MSE across all shared points:

joined_dfs = yolo_lateral.merge(mpipe_frontal,
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

sb.lineplot(data=mpipe_frontal[mpipe_frontal.label == 'right_heel'],
            x="frame", y="Y")
