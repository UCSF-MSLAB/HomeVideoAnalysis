import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
# from statistics import linear_regression
from src.analysis_funs import get_all_unique_pairs, rm_nulls, \
    est_landmark_dist
# from src.keypoint_labels import keypoint_labels
from sklearn import linear_model

PATH = "./tmp/csv_output/"

# keypoint_labels.mpipe_labels
# PARTS_LIST = keypoint_labels.mpipe_labels
PARTS_LIST = ["left_ankle", "right_ankle"]

# What criteria should we use to get a good label pairing?

FILE = "Yoni_10ft_Dark_"

mpipe = pd.read_csv(PATH + FILE + "mediapipe_world.csv")
mpipe_filt = rm_nulls(mpipe)
mpipe_filt = mpipe_filt[mpipe_filt['label'].isin(PARTS_LIST)]
mpipe_filt = mpipe_filt[mpipe_filt['vis'] > .7]
mpipe_pairs = get_all_unique_pairs(mpipe_filt)
mpipe_pairs["world_diff"] = abs(mpipe_pairs["Z_x"] - mpipe_pairs["Z_y"])

# ax = sb.lineplot(data=mpipe_filt,
#                     x="frame", y="vis", hue="label")
# plt.show()

# ax = sb.lineplot(data=mpipe_pairs,
#                     x="frame", y="world_diff", hue="labpair_id")
# plt.show()

mari = pd.read_csv(PATH + FILE + "marigold.csv")
mari_filt = rm_nulls(mari)
mari_filt = mari_filt[mari_filt['label'].isin(PARTS_LIST)]
mari_pairs = get_all_unique_pairs(mari_filt)
mari_pairs["depth_diff"] = abs(mari_pairs["depth_est_x"] -
                               mari_pairs["depth_est_y"])

# ax = sb.lineplot(data=mari_filt,
#                     x="frame", y="depth_est", hue="label")
# plt.show()

# ax = sb.lineplot(data=mari_pairs,
#                     x="frame", y="depth_diff", hue="labpair_id")
# plt.show()

mari_mpipe = mari_pairs.merge(mpipe_pairs,
                              on=['frame', 'row_id', 'labpair_id'])

# ax = sb.scatterplot(data=mari_mpipe,
#                     x="depth_diff", y="world_diff", hue="labpair_id")
# plt.show()

# testing linear models
alpha = np.array(mari_mpipe.apply(lambda row: min(row.vis_x, row.vis_y), axis=1))
X = np.array(mari_mpipe.depth_diff).reshape(-1, 1)
y = np.array(mari_mpipe.world_diff)

# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(X, y)

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor(random_state=0)
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.array(np.arange(X.min(), X.max(), .005)).reshape(-1, 1)
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

print("Estimated coefficients (true, linear regression, RANSAC):")
print(lr.coef_, ransac.estimator_.coef_)

lw = 2
plt.scatter(
    X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
)
plt.scatter(
    X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
)

for i, vis in enumerate(alpha):
    plt.annotate(str(vis)[:3], (X[i], y[i]))

plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor")
plt.plot(
    line_X,
    line_y_ransac,
    color="cornflowerblue",
    linewidth=lw,
    label="RANSAC regressor",
)
plt.legend(loc="lower right")
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()

def check_row(labxx, labyx, labxy, labyy):
    return set([labxx, labyx]) == set([labxy, labyy])

mari_mpipe["label_agreement"] = mari_mpipe.apply(lambda row: check_row(row["label_x_x"],
                                                                       row["label_y_x"],
                                                                       row["label_x_y"],
                                                                       row["label_y_y"]),
                                                 axis=1)
mari_mpipe["label_agreement"].all()

mari_filt["depth_est_ft"] = lr.predict(np.array(mari_filt["depth_est"]).reshape(-1, 1)) * 3.28084

ax = sb.scatterplot(data=mari_filt,
                    x="frame", y="depth_est_ft", hue="label")
plt.show()

mari_filt.groupby(['label']).agg({'depth_est_ft': [np.min, np.max]})

# look at multiple files
# ---------------------
# ---------------------
# ---------------------

files = [
    "2024_09_12_02_41_PWS_1_gait_vertical_left_",
    "MM_HC_10steps_away_1_",
    "MM_HC_17ft_gait_vertical_right_",
    "2024_09_12_02_43_PWS_1_gait_vertical_right_",
    "MM_HC_10steps_towards_0_",
    "MM_HC_10steps_away_0_",
    "MM_HC_10steps_towards_1_",
    "Yoni_10ft_Light_", "Yoni_10ft_Dark_"
         ]

# files = []

for f_name in files:
    print(f_name)
    mpipe = pd.read_csv(PATH + f_name + "mediapipe_world.csv")
    mp_world_df = rm_nulls(mpipe)
    mari = pd.read_csv(PATH + f_name + "marigold.csv")
    marigold_df = rm_nulls(mari)
    print(est_landmark_dist(mp_world_df, marigold_df))
    print("\n")
