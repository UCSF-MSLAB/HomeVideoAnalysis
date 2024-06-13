import pandas as pd
import numpy as np


def calc_stride(heel_loc_df, band=5):

    vals = []
    for i in range(0, int(max(heel_loc_df.Y))):
        vals.append((i, heel_loc_df[((i - band) < heel_loc_df.Y) &
                                    (heel_loc_df.Y < (i+band))].shape[0]))

    vals_df = pd.DataFrame(vals, columns=['Y', 'count'])
    floor_Y = vals_df[vals_df['count'] == np.max(vals_df['count'])].Y.values[0]

    heel_loc_df['heel_on_floor'] = ((heel_loc_df['Y'] > floor_Y - band) &
                                    (heel_loc_df['Y'] < floor_Y + band))


def get_landmark_pairs(landmark_df, label_x, label_y):

    tmp = landmark_df.merge(landmark_df, on='frame', how='left')
    return tmp[(tmp.label_x == label_x) & (tmp.label_y == label_y)].copy()
