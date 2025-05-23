import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d


def get_all_unique_pairs(df):
    df_pairs = df.merge(df, on='frame', how='left')
    df_pairs = df_pairs[df_pairs['label_x'] != df_pairs['label_y']]

    def make_id(x, y, i):
        return str(i) + "-" + str(max(x, y)) + "-" + str(min(x, y))

    df_pairs["row_id"] = df_pairs.apply(lambda row: make_id(row["subframe_id_x"],
                                                            row["subframe_id_y"],
                                                            row["frame"]),
                                        axis=1)
    return df_pairs.drop_duplicates(subset="row_id", keep="first")


def get_landmark_pairs(landmark_df, label_x, label_y):

    tmp = landmark_df.merge(landmark_df, on='frame', how='left')
    return tmp[(tmp.label_x == label_x) & (tmp.label_y == label_y)].copy()


def calculate_pair_dist():
    pass

# def calc_max_horizontal_speed(landmark_df, forearm_len, fps):
#     """
#     return a point estimate of the maximum speed of the torso.
#     """

#     hip_df = get_landmark_pairs(landmark_df, 'left_hip', 'right_hip')
#     hip_df['Z_x_md'] = hip_df.Z_x.rolling(20).median()
#     hip_df['Z_y_md'] = hip_df.Z_y.rolling(20).median()

#     # sb.lineplot(hip_df[['Z_x_md', 'Z_y_md']])
#     # plt.show()

#     hip_df['left_closer'] = hip_df.Z_x_md < hip_df.Z_y_md

#     # sb.lineplot(hip_df, x='frame', y='left_closer')
#     # plt.show()

#     frame_to_side = hip_df[['frame', 'left_closer']].copy()
#     frame_to_side["right_closer"] = ~frame_to_side.left_closer

#     # determine turns (label with id)
#     # #############################################

#     frame_to_side['turn_right_id'] = (frame_to_side.left_closer.cumsum() *
#                                       frame_to_side.right_closer)
#     frame_to_side['turn_left_id'] = (frame_to_side.right_closer.cumsum() *
#                                      frame_to_side.left_closer)
#     frame_to_side['turn_id'] = (frame_to_side.turn_right_id *
#                                 frame_to_side.right_closer +
#                                 frame_to_side.turn_left_id *
#                                 frame_to_side.left_closer)

#     landmark_df = landmark_df.merge(frame_to_side, on='frame', how='left')

#     # calculate inches per pixel
#     # #############################################

#     right_forearm_df = get_landmark_pairs(landmark_df[landmark_df.right_closer],
#                                           'right_wrist', 'right_elbow')
#     right_forearm_df['dist'] = np.sqrt((right_forearm_df.X_x - right_forearm_df.X_y)**2 +
#                                        (right_forearm_df.Y_x - right_forearm_df.Y_y)**2)

#     left_forearm_df = get_landmark_pairs(landmark_df[landmark_df.left_closer],
#                                          'left_wrist', 'left_elbow')
#     left_forearm_df['dist'] = np.sqrt((left_forearm_df.X_x - left_forearm_df.X_y)**2 +
#                                       (left_forearm_df.Y_x - left_forearm_df.Y_y)**2)

#     inch_per_px = (forearm_len / right_forearm_df.dist.median() +
#                    forearm_len / left_forearm_df.dist.median()) / 2

#     # calculate position of upper body each frame
#     # #############################################

#     upr_bdy = ['_hip', '_shoulder', '_ear']

#     def calc_sided_pos(df):
#         if all(df.left_closer):
#             return df[df.label.isin(['left'+part for part in upr_bdy])].X.mean()
#         return df[df.label.isin(['right'+part for part in upr_bdy])].X.mean()

#     frame_to_side["upr_body_X"] = (landmark_df
#                                    .groupby(['frame'])
#                                    .apply(calc_sided_pos, include_groups=False)
#                                    .values)

#     # sb.lineplot(frame_to_side, x='frame', y='upr_body_X')
#     # plt.show()

#     # calculate change in X within turn-groups
#     # #############################################

#     turns = frame_to_side.turn_id.unique()

#     def calc_abs_diff(turn_id):
#         return np.abs(frame_to_side[frame_to_side.turn_id == turn_id]
#                       .upr_body_X
#                       .diff())

#     dXs = [calc_abs_diff(turn_id).values[1:] for turn_id in turns]
#     dX = np.concatenate(dXs) * inch_per_px * fps

#     # sb.lineplot(dX)
#     # plt.show()

#     # apply a basic filter based on max reasonable value
#     dX = [dx for dx in dX if dx < 7.33]

#     # apply a 1d gaussian filter to smooth
#     dX_gf = gaussian_filter1d(dX, 10)

#     # sb.lineplot(dX_gf)
#     # plt.show()

#     return max(dX_gf)

# TODO
# def calc_stride(heel_loc_df, band=5):

#     # vals = []
#     # for i in range(0, int(max(heel_loc_df.Y))):
#     #     vals.append((i, heel_loc_df[((i - band) < heel_loc_df.Y) &
#     #                                 (heel_loc_df.Y < (i+band))].shape[0]))

#     # vals_df = pd.DataFrame(vals, columns=['Y', 'count'])
#     # floor_Y = vals_df[vals_df['count'] == np.max(vals_df['count'])].Y.values[0]

#     # heel_loc_df['heel_on_floor'] = ((heel_loc_df['Y'] > floor_Y - band) &
#     #                                 (heel_loc_df['Y'] < floor_Y + band))
#     pass
