#!/usr/bin/env python
# coding: utf-8

# stride time script 
#Stenum et al: Frontal Videos - We identified events of left and right gait cycles by local maxima 
#and minima ofthe vertical distance between the left and right ankle keypoints. 
#Gait events on the left limb were detected at positive peaks and gait events on the right 
#limb were detected at nega- tive peaks in trials where the participants walked away from the 
#frontal plane camera; and vice versa in trials where the participants walked toward the camera. 
#In order to unify the nomenclature ofgait events across motion capture data and sagittal 
#and fron- tal plane video data, we refer to the gait events ofthe frontal plane analysis as heel- strikes.

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sig 
from raw_pose_analysis_funs.filter_interpolate_funs import (interpolate_landmark_single_axis, 
filter_landmark_single_axis)
import os 


# interpolate ankle markers Y axis data 
def stride_time_interp(mp_all_df,video_id_date_name, dir_out_prefix, max_gap, fps): 
    # KEEP THIS ORDER of landmark and pose THE SAME!! 
    # if change order of variables or for loop, need to update in later steps
    stride_time_landmarks = ['right_ankle', 'left_ankle'] 
    stride_time_axes = ['Y_pose']

    # interpolate yolo and mp data 
    mp_stride_time_interp_dfs = []
    dataset = 'mediapipe'
    df = mp_all_df
    
    for landmark_i, current_landmark in enumerate(stride_time_landmarks): 
        for axis_i, current_axis in enumerate(stride_time_axes):
            # interpolate 
            current_interp_dfs = interpolate_landmark_single_axis(df, # mediapipe or yolo data frame 
                                                                  current_landmark, # marker to interpolate 
                                                                  current_axis, # axis to interpolate
                                                                  max_gap, # seconds, maximum gap to interpolate over
                                                                  fps,
                                                                  video_id_date_name,
                                                                  dir_out_prefix,
                                                                  mediapipe_or_yolo = dataset)
            
            # add interpolated data for both landmark to one list 
            mp_stride_time_interp_dfs = mp_stride_time_interp_dfs + [current_interp_dfs]

    return(mp_stride_time_interp_dfs)


def calculate_stride_time(mp_ankle_Y_interp, fps, vid_in_path, output_parent_folder, find_peaks_distance, find_peaks_prominence): 
    # difference between Y values of left and right ankles 
        # min and max of this plot are gait events 
    ank_r_mp_y_interp = mp_ankle_Y_interp[0]
    ank_l_mp_y_interp = mp_ankle_Y_interp[1]

    # vertical distance between l and r ankle; 2 = Y column 
    ank_y_diff_0 = ank_l_mp_y_interp['left_ankle_Y_pose_interpolated'] - ank_r_mp_y_interp['right_ankle_Y_pose_interpolated']

    # moving mean of Y difference - Stenum et al paper window = 10
    ank_y_diff = pd.Series(ank_y_diff_0).rolling(window=15, min_periods=1).mean()

    # find index of local minimum and maximum of distance between right and left ankle  
    ank_y_diff_peaks_byFrame, _ = sig.find_peaks(ank_y_diff, distance = find_peaks_distance, prominence = (find_peaks_prominence, None))
    ank_y_diff_valleys_byFrame, _ = sig.find_peaks(-ank_y_diff, distance = find_peaks_distance, prominence = (find_peaks_prominence, None))

    # divide by Hz to get position of peaks in seconds, use for plots below 
    ank_y_diff_peaks_bySecond = ank_y_diff_peaks_byFrame/fps
    ank_y_diff_valleys_bySecond = ank_y_diff_valleys_byFrame/fps

    # Time between local max, seconds - R or L gait event (TBD)
    stride_times_peaks = pd.Series(ank_y_diff_peaks_bySecond).diff()
    
    # Time between local min, seconds - R or L  gait event (TBD) 
    stride_times_valleys = pd.Series(ank_y_diff_valleys_bySecond).diff()
   
    # Stride time stats 
    stats = ['mean_sec', 'median_sec', 'std', 'cv']
    stride_time_stats_df = pd.DataFrame(index = stats, columns = ['leg_1_peaks', 'leg_2_valleys', 'all_strides'])

    stride_time_stats_df.loc['mean_sec', 'leg_1_peaks'] = stride_times_peaks.mean()
    stride_time_stats_df.loc['median_sec', 'leg_1_peaks'] = stride_times_peaks.median()
    stride_time_stats_df.loc['std', 'leg_1_peaks'] = stride_times_peaks.std()
    stride_time_stats_df.loc['cv', 'leg_1_peaks'] = (stride_times_peaks.std()/stride_times_peaks.mean()) * 100

    stride_time_stats_df.loc['mean_sec', 'leg_2_valleys'] = stride_times_valleys.mean()
    stride_time_stats_df.loc['median_sec', 'leg_2_valleys'] = stride_times_valleys.median()
    stride_time_stats_df.loc['std', 'leg_2_valleys'] = stride_times_valleys.std()
    stride_time_stats_df.loc['cv', 'leg_2_valleys'] = (stride_times_valleys.std()/stride_times_valleys.mean()) * 100

    stride_time_stats_df.loc['mean_sec', 'all_strides'] = pd.concat([stride_times_peaks, stride_times_valleys]).mean()
    stride_time_stats_df.loc['median_sec', 'all_strides'] = pd.concat([stride_times_peaks, stride_times_valleys]).median()
    stride_time_stats_df.loc['std', 'all_strides'] = pd.concat([stride_times_peaks, stride_times_valleys]).std()
    stride_time_stats_df.loc['cv', 'all_strides'] = (pd.concat([stride_times_peaks, stride_times_valleys]).std()/
                                                     pd.concat([stride_times_peaks, stride_times_valleys]).mean()) * 100
    

    #save outputs 
    output_folder = os.path.join(output_parent_folder, '005_gait_metrics', 'stride_time')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]
    
    # save stasts
    stride_time_stats_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_stride_time_stats.csv')))
    stride_time_stats_df.to_csv(stride_time_stats_path)
    
    # save peaks diff 
    leg_1_peaks_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_stride_time_leg_1_all.csv')))
    stride_times_peaks.to_csv(leg_1_peaks_path)
    
    # save valleys diff 
    leg_2_valleys_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + 'stride_time_leg_2_all.csv')))
    stride_times_valleys.to_csv(leg_2_valleys_path)

    # --------------------------------------------
    # plot and save plots 
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.suptitle(os.path.splitext(os.path.basename(vid_in_path_no_ext))[0] + ': Stride Time')
    # plot y difference in ankles and add labels on local min and max 
    ax1.plot(ank_r_mp_y_interp['frame'], ank_y_diff, color = 'black', label='Y Difference between Ankles')
    ax1.plot(ank_y_diff_peaks_byFrame, ank_y_diff.iloc[ank_y_diff_peaks_byFrame], "x", color = 'orange', label='Peak')
    ax1.plot(ank_y_diff_valleys_byFrame, ank_y_diff.iloc[ank_y_diff_valleys_byFrame], ".", label='Local Minima')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel("L Ankle Y - R Ankle Y (Pose)")
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig1.tight_layout()  # avoid plot overlap

    output_plot_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_stride_time.png')))
    fig1.savefig(output_plot_path, bbox_inches = 'tight')
    plt.close(fig1)
    plt.close()


    return([stride_time_stats_df, stride_times_peaks, stride_times_valleys])

