#!/usr/bin/env python
# coding: utf-8

# segment video into times when person is walking away from the camera, toward the camera, and turning 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sig 
from raw_pose_analysis_funs.filter_interpolate_funs import (interpolate_landmark_single_axis, 
filter_landmark_single_axis)
import os 

def segment_video_interp_filter(mp_all_df, yolo_df,vid_in_path, output_parent_folder, max_gap, fps, cutoff, order): 
    # KEEP THIS ORDER of landmark and pose THE SAME!! 
    # if change order of variables or for loop, need to update in later steps
    segment_walk_landmarks = ['right_hip', 'left_hip'] 
    segment_walk_axes = ['Z_pose', 'X_yolo']

    # interpolate yolo and mp data 
    mp_segement_walk_interp_dfs = []
    yolo_segement_walk_interp_dfs = []
    for landmark_i, current_landmark in enumerate(segment_walk_landmarks): 
        for axis_i, current_axis in enumerate(segment_walk_axes): 
            if 'pose' in current_axis: 
                dataset = 'mediapipe'
                df = mp_all_df
            elif 'yolo' in current_axis: 
                dataset = 'yolo'
                df = yolo_df

            # interpolate 
            current_interp_dfs = interpolate_landmark_single_axis(df, # mediapipe or yolo data frame 
                                                                  current_landmark, # marker to interpolate 
                                                                  current_axis, # axis to interpolate
                                                                  max_gap, # seconds, maximum gap to interpolate over
                                                                  fps,
                                                                  vid_in_path,
                                                                  output_parent_folder,
                                                                  mediapipe_or_yolo = dataset)
            # add interpolation to mp or yolo dfs 
            if 'pose' in current_axis: 
                mp_segement_walk_interp_dfs = mp_segement_walk_interp_dfs + [current_interp_dfs]
            elif 'yolo' in current_axis: 
                yolo_segement_walk_interp_dfs = yolo_segement_walk_interp_dfs + [current_interp_dfs]     

    # filter mp_pose interpolated data 
    mp_segment_walk_filt_dfs = []
    for mp_interp_i, mp_interp_df in enumerate(mp_segement_walk_interp_dfs): 

        # filter interpolated data (3rd column in df)
        current_mp_segment_walk_filt = filter_landmark_single_axis(
            mp_segement_walk_interp_dfs[mp_interp_i].iloc[:, 2],
            fps, # video HZ
            cutoff, # filter cutoff 
            order, # butterworth filter order
            vid_in_path,
            output_parent_folder
            )
        mp_segment_walk_filt_dfs = mp_segment_walk_filt_dfs + [current_mp_segment_walk_filt]

    # save interpolated yolo X hip positions and filtered mp hip Z positions 
    return([mp_segment_walk_filt_dfs, yolo_segement_walk_interp_dfs])


#using yolo x hip and mp z hip positions, ID when person is walking or turning 
def segment_video_walks_turn(mp_hip_z_filt, yolo_hip_x_interp, vid_in_path, output_parent_folder, fps,
                             find_peaks_distance, find_peaks_prominence, flattening_point_atol, 
                             dist_turn_mid_to_flattening, mean_rolling_window_size): 
    # -----------------------------------------------------------------------------
    # use hip z position to ID start, stop, and midpoint of turns in vertical videos 
    hip_r_mp_z_filt = mp_hip_z_filt[0]
    hip_l_mp_z_filt = mp_hip_z_filt[1]
    hip_l_mp_z_frames = hip_l_mp_z_filt.index
    
    # distance between l and r z and smooth
    hip_z_diff_mp_filt = hip_l_mp_z_filt - hip_r_mp_z_filt
    hip_z_diff_mp_filt = pd.Series(hip_z_diff_mp_filt).rolling(window=mean_rolling_window_size, min_periods=1).mean()
    hip_z_diff_mp_filt.index = hip_l_mp_z_frames

    # find max and min of hip distance filtered 
    # max and min = frame of midpoint of turn 
    hip_z_diff_mp_filt_peak_frames, _ = sig.find_peaks(hip_z_diff_mp_filt, distance = find_peaks_distance, prominence = (find_peaks_prominence, None))
    hip_z_diff_mp_filt_peak_frames = hip_z_diff_mp_filt.index[hip_z_diff_mp_filt_peak_frames] # set to index, accounts for missing data where frame doesn't equal row index
    
    hip_z_diff_mp_filt_valley_frames, _ = sig.find_peaks(-hip_z_diff_mp_filt, distance = find_peaks_distance, prominence = (find_peaks_prominence, None))
    hip_z_diff_mp_filt_valley_frames = hip_z_diff_mp_filt.index[hip_z_diff_mp_filt_valley_frames]

    # merge together peaks and valleys of hip z diff df -> frames of each turn, ordered 
    hip_z_diff_mp_filt_turn_midpoints = np.concatenate((hip_z_diff_mp_filt_peak_frames, hip_z_diff_mp_filt_valley_frames), axis = None)
    hip_z_diff_mp_filt_turn_midpoints = np.sort(hip_z_diff_mp_filt_turn_midpoints)

    # rate of change of z hip distance 
    hip_z_diff_mp_filt_gradient = hip_z_diff_mp_filt.diff()

    # Identify where the slope is within absolute tolerance value (atol) away from zero 
    flattening_points = np.where(np.isclose(hip_z_diff_mp_filt_gradient, 0, atol=flattening_point_atol))[0]
    flattening_points = hip_z_diff_mp_filt_gradient.index[flattening_points]
   
    # Find first flattening point prior to turn midpoint
    turn_start_frames = np.array([], dtype='int16')
    for midpoint_i, current_midpoint in enumerate(hip_z_diff_mp_filt_turn_midpoints):
        # flattening points that are before current midpoint and at least 10 frames away from midpoint (exclude midpoint itself)
        before_peak_flattening_all = flattening_points[(flattening_points < current_midpoint) & (abs(current_midpoint - flattening_points) >= dist_turn_mid_to_flattening)]
        # select last element (closest to turn midpoint)
        if len(before_peak_flattening_all) > 0: 
            before_peak_flattening_last = before_peak_flattening_all[-1]
        else: 
            before_peak_flattening_last = np.nan
        # save 
        turn_start_frames = np.append(turn_start_frames, before_peak_flattening_last)

    #Find first flattening point after hip midpoint 
    turn_stop_frames = np.array([], dtype='int16')
    for midpoint_i, current_midpoint in enumerate(hip_z_diff_mp_filt_turn_midpoints):
        # flattening points that are after current midpoint and at least 10 frames away from midpoint (exclude midpoint itself)
        after_peak_flattening_all = flattening_points[(flattening_points > current_midpoint) & (abs(current_midpoint - flattening_points) >= dist_turn_mid_to_flattening)]

        # select first element (closest to turn midpoint)
        if len(after_peak_flattening_all) > 0: 
            after_peak_flattening_first = after_peak_flattening_all[0]
        else: 
            after_peak_flattening_first = np.nan
     
        # save 
        turn_stop_frames = np.append(turn_stop_frames, after_peak_flattening_first)

    # save all turn info as one df 
    turn_data = {'turn_num' : np.arange(0, len(hip_z_diff_mp_filt_turn_midpoints), step = 1),
                 'turn_start_frame' : turn_start_frames, 
                 'turn_midpoint' : hip_z_diff_mp_filt_turn_midpoints,
                 'turn_stop_frame' : turn_stop_frames,
                 'turn_time_frames' : turn_stop_frames - turn_start_frames, 
                 'turn_time_seconds' : (turn_stop_frames - turn_start_frames) / fps
                } 

    turn_df = pd.DataFrame(turn_data)
    # -----------------------------------------------------------------------------
    # Use distance between hips in pixels (yolo) to determine direction subject is moving 
    # hip width increasing = walking toward camera  
    #  hip width decreasing = walking away from camera
    # use start and stop of turns from hip z distance to ID walking times

    # create one df for r hip, one for l 
    hip_r_yolo_df = yolo_hip_x_interp[0]
    hip_l_yolo_df = yolo_hip_x_interp[1]

    # hip width 
    hip_width_yolo = abs(hip_r_yolo_df['right_hip_X_yolo_interpolated'] - hip_l_yolo_df['left_hip_X_yolo_interpolated'])
    hip_width_yolo_smooth = pd.Series(hip_width_yolo).rolling(window=15, min_periods=1).mean()

    # frames 
    frames = hip_r_yolo_df.index
    # walk start - start one second in to account for time for model to fit to person
        # start of entire video 
    first_walk_start_frame = frames[0] 
    # end of last walk 
    last_walk_end_frame = frames[-1]

    # create walk_df with start and stop of each walk, time per walk, and direction 
    walks_df = pd.DataFrame(index=range(len(turn_df) + 1), 
                            columns = ['walk_num', 
                                       'walk_start_frame', 
                                       'walk_end_frame', 
                                       'walk_time_frames', 
                                       'walk_time_turns', 
                                       'walk_direction'])

    number_of_walks = np.arange(0, len(walks_df), step = 1)

    # set start and stop frames from turns df 
    for current_walk_num in number_of_walks: 
        # walk_num
        walks_df.iloc[current_walk_num, 0] = current_walk_num
    
        #walk_start_frame 
        # if walk 1 - start = first_walk_start_frame
        # all other walks = walk start = end of previous turn 
        if current_walk_num == 0:
            current_walk_start = first_walk_start_frame
        else:   
            turn_stop_frame = turn_df['turn_stop_frame'] 
            current_walk_start = turn_stop_frame[current_walk_num - 1]

        walks_df.iloc[current_walk_num, 1] = current_walk_start

        # walk end frame 
        # if current walk is the last walk, stop frame = last walk stop 
        if current_walk_num == max(number_of_walks): 
             current_walk_stop = last_walk_end_frame

        # middle walks 
        else:
            turn_start_frame = turn_df['turn_start_frame'] 
            current_walk_stop = turn_start_frame[current_walk_num]
        
        walks_df.iloc[current_walk_num, 2] = current_walk_stop

        # walk_time_frames 
        walks_df.iloc[current_walk_num, 3] = current_walk_stop - current_walk_start

        # walk_time_seconds 
        walks_df.iloc[current_walk_num, 4] = (current_walk_stop - current_walk_start) / fps

        # walk direction 
        # if hip width is bigger at walk stop than walk start, person is moving toward camera 
    
        # new loop for toward or away ------------- 
        if (np.isnan(current_walk_start) | np.isnan(current_walk_stop)): 
            walks_df.iloc[current_walk_num, 5] = 'unknown'

        else: 
            # is current_walk_start in hip_width index? 
            if current_walk_start in hip_width_yolo_smooth.index:
                walk_start_w = hip_width_yolo_smooth.loc[current_walk_start]
            # if not - get closest index 
            else: 
                walk_start_w_i = [(walks_df.index - current_walk_start).abs().argmin()]
                walk_start_w = hip_width_yolo_smooth.loc[walk_start_w_i]

            # is current_walk_stop in hip width index?
            if current_walk_stop in hip_width_yolo_smooth.index: 
                walk_stop_w = hip_width_yolo_smooth.loc[current_walk_stop]
            # if not, closest index 
            else: 
                walk_stop_w_i = [(walks_df.index - current_walk_stop).abs().argmin()]
                walk_stop_w = hip_width_yolo_smooth.loc[walk_stop_w_i]

            # if width greater at stop than start, 'toward, otherwise away 
            if (walk_stop_w > walk_start_w): 
               walks_df.iloc[current_walk_num, 5] = 'toward' 
            elif (walk_stop_w < walk_start_w): 
                walks_df.iloc[current_walk_num, 5] = 'away'
            else: 
                walks_df.iloc[current_walk_num, 5] = 'unknown'
                

                
    ## plots  -------------------------------------------
    # plot #1 - hip and hip positions 
    fig1, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
    fig1.suptitle(os.path.splitext(os.path.basename(vid_in_path))[0])

    # subplot 1 - mp z for each hip 
    ax1.scatter(hip_r_mp_z_filt.index, hip_r_mp_z_filt, label = 'r_hip_z_filt', color = 'blue', marker = 'o', s = 1)
    ax1.scatter(hip_l_mp_z_filt.index, hip_l_mp_z_filt, label = 'l_hip_z_filt', color = 'red', marker = 'o', s = 1)
    ax1.set_ylabel('MP Pose')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # subplot 2 - yolo x for each hip 
    ax2.scatter(hip_r_yolo_df['frame'], hip_r_yolo_df['right_hip_X_yolo_interpolated'], label = 'r_hip_yolo_x', color = 'orange', marker = 'o', s = 1)
    ax2.scatter(hip_l_yolo_df['frame'], hip_l_yolo_df['left_hip_X_yolo_interpolated'], label = 'l_hip_yolo_x', color = 'green', marker = 'o', s = 1)
    ax2.set_ylabel('Yolo Pixels')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig1.tight_layout()  # avoid plot overlap

    # plot 2 - hip and hip dist 
    # set plot with two subplots 
    fig2, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
    fig2.suptitle(os.path.splitext(os.path.basename(vid_in_path))[0])

    # subplot 1 - distance between right and left hip, use peaks and mins as turns 
    ax1.set_title('Turns')
    ax1.scatter(hip_l_mp_z_frames, hip_z_diff_mp_filt, label = 'l_hip_z_filt - r_hip_z_filt', color = 'black', marker = 'o', s = 1)
    ax1.vlines(x = turn_start_frames, ymin = -1, ymax = 1, color = 'green', alpha = 0.5, label = 'turn_start_calculated')
    ax1.vlines(x=turn_df['turn_midpoint'], ymin = -1, ymax = 1, color = 'yellow',  alpha = 0.5, label = 'turn_midpoint calculated')
    ax1.vlines(x = turn_stop_frames, ymin = -1, ymax = 1, color = 'red', alpha = 0.5,  label = 'turn_stop_calculated')
    ax1.set_ylabel('MP Pose')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # sublot 2 - yolo hip width 
    ax2.set_title('Walks')
    ax2.scatter(hip_r_yolo_df['frame'], hip_width_yolo_smooth, label = "hip_x_width, abs, smooth", color = 'black', marker = 'o', s = 1) 
    ax2.vlines(x = walks_df['walk_start_frame'], ymin = 0, ymax = max(hip_width_yolo_smooth), color = 'green', alpha = 0.5, label = 'walk_start_calculated')
    ax2.vlines(x = walks_df['walk_end_frame'], ymin = 0, ymax = max(hip_width_yolo_smooth), color = 'red', alpha = 0.5,  label = 'walk_stop_calculated')
    ax2.vlines(x = first_walk_start_frame, ymin = 0, ymax = max(hip_width_yolo_smooth),  color = 'green', linestyle = '--', label = 'first_walk_start_frame')
    ax2.vlines(x = last_walk_end_frame, ymin = 0, ymax = max(hip_width_yolo_smooth), color = 'red', linestyle = '--', label = 'last_walk_end_frame')
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Yolo Pixels')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig2.tight_layout()  # avoid plot overlap

    # save outputs ------------  
    output_folder = os.path.join(output_parent_folder, '004_segment_towards_away_turn')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]

    # save plots 
    # plot 1 
    output_plot_1 = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_hip_z_mp_hip_x_yolo.png')))
    fig1.savefig(output_plot_1, bbox_inches = 'tight')
    plt.close(fig1)
    plt.close()

    # plot 2 
    output_plot_2 = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_turn_walk_start_stop.png')))
    fig2.savefig(output_plot_2, bbox_inches = 'tight')
    plt.close(fig2)
    plt.close()

    # save turn and walk data frames 
    turn_df_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_turn_start_stop_frames.csv')))
    turn_df.to_csv(turn_df_path)

    walk_df_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_walk_start_stop_frames.csv'))) 
    walks_df.to_csv(walk_df_path)
    
    # outputs 
    return([turn_df, walks_df])






