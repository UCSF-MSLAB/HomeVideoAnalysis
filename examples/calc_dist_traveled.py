import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from src.analysis_funs import get_landmark_pairs, get_all_unique_pairs

mpipe = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/fixtures_gait_vertical_left_mediapipe_world.csv")

mpipe = mpipe.rename(columns={'Unnamed: 0': 'subframe_id'})
mpipe_filt = mpipe[mpipe['label'].notnull()]

mpipe_pairs = get_all_unique_pairs(mpipe_filt)

mari = pd.read_csv("/home/yoni/Projects/UCSF_Pose/HomeVideoAnalysis/tmp/csv_output/fixtures_gait_vertical_left_marigold.csv")

mari = mari.rename(columns={'Unnamed: 0': 'subframe_id'})
mari_filt = mari[mari['label'].notnull()]

mari_pairs = get_all_unique_pairs(mari_filt)

mari_mpipe = mari_pairs.merge(mpipe_pairs, on=['frame', 'row_id'])

def check_row(labxx, labyx, labxy, labyy):
    return (labxx == labxy) and (labyx == labyy)


mari_mpipe["label_agreement"] = mari_mpipe.apply(lambda row: check_row(row["label_x_x"],
                                                                       row["label_y_x"],
                                                                       row["label_x_y"],
                                                                       row["label_y_y"]),
                                                 axis=1)


mari_mpipe["depth_to_m"] = np.abs(mari_mpipe["Z_x"] - mari_mpipe["Z_y"]) / np.abs(mari_mpipe["depth_est_x"] - mari_mpipe["depth_est_y"])

mari_mpipe.groupby(["frame"])["depth_to_m"].median()

ax = sb.lineplot(data=mari_mpipe.groupby(["frame"])["depth_to_m"].median())
plt.show() # cool!?
