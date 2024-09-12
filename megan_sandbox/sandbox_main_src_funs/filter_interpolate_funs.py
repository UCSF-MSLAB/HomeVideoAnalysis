#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig 
import os


# INTERPOLATE 
# From Stenum et al Clinical Gait Analysis using video based pose estimation: 
# multiple perspectives, clinical populations, and measuring change 
# "We gap-filled keypoint trajectories using linear interpolation for gaps spanning to up 0.12 s."#

# inputs 
    # df = full data frame with all landmarks and axis (XYZ, etc), either mediapipe or yolo 
# output 
    # df with column for frames, original data, and interpolated data 
    # plots saved 
def interpolate_landmark_singe_axis(df, landmark, axis_to_interpolate, max_gap_sec, fps, vid_in_path, output_parent_folder, mediapipe_or_yolo):
    if mediapipe_or_yolo == 'mediapipe': 
        df_landmark = df.loc[(df['label'] == landmark)|(df['label'] == 'no_labels_tracked')]
    elif mediapipe_or_yolo == 'yolo': 
        df_landmark = df.loc[df['label'] == landmark]
        
    # identify where position data is missing 
    df_landmark.index = df_landmark['frame'] # set index to frames, keep missing data  
    position_data = df_landmark[axis_to_interpolate] # select position data for axis to interpolate 
    missing_indices = np.where(np.isnan(position_data))

    # find consecutive frames of missing data 
    all_frames = df_landmark['frame']
    missing_frames = all_frames.iloc[missing_indices]
    frame_diff = np.diff(missing_frames) # Find differences between missing frames 
    frame_breaks = np.where(frame_diff != 1)[0] # Identify indices where the difference is not 1 (i.e., the sequence breaks)
    frame_breaks = np.concatenate(([ -1 ], frame_breaks, [ len(missing_frames) - 1 ])) # mark the start and end of sequences

    # Group consecutive sequences into a list 
    consecutive_frames = [missing_frames[frame_breaks[i] + 1 : frame_breaks[i + 1] + 1] for i in range(len(frame_breaks) - 1)]

    # convert max gap in seconds to frames  
    max_frame_gap = max_gap_sec/(1/fps)
    max_frame_gap = round(max_frame_gap) # maximum frames to interpolate over 

    # set blank lists and df to populated below 
    frames_to_interpolate = []
    interp_values = [] 
    current_col_name = landmark + '_' + axis_to_interpolate  
    position_interp_data = {'frame' : df_landmark['frame'],
                            (current_col_name + '_raw') : df_landmark[axis_to_interpolate],
                            (current_col_name + '_interpolated') : df_landmark[axis_to_interpolate] 
                            # start with original data, will replace with interpolated below
                           }  
    position_interp_df = pd.DataFrame(data = position_interp_data, index = df_landmark.index)
    
    # if the length of current set of consecutive frames is below the maximum frame gap, 
        # interpolate over the current set of frames and update df interpolated column with new values 
    for i, current_frame_set in enumerate(consecutive_frames):
        if len(current_frame_set) <= max_frame_gap: 
            # save frames to interpolate over 
            frames_to_interpolate = frames_to_interpolate + [current_frame_set] 
            current_interp_values = np.interp(current_frame_set, # frames to interpolate 
                                              all_frames[~np.isnan(position_data)], # all frames of data 
                                              position_data[~np.isnan(position_data)]) # landmark position data 
            # save interpolated position values
            interp_values = interp_values + [current_interp_values]  
            # insert interpolated values into position data 
            position_interp_df.loc[current_frame_set, (current_col_name + '_interpolated')] = current_interp_values


    # plot original data vs linterpolated data 
    fig1, ax1 = plt.subplots()
    fig1.suptitle('Interpolated Data')
    ax1.plot(position_interp_df['frame'], position_interp_df[current_col_name + '_raw'], color = 'red', alpha = 0.5, label = 'raw')
    ax1.plot(position_interp_df['frame'], position_interp_df[current_col_name + '_interpolated'], color = 'blue', alpha = 0.5, label = 'interpolated')
    ax1.legend()
    ax1.set_xlabel('Frame')
    ax1.set_ylabel(current_col_name)
    plt.close(fig1)
    plt.close()

    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]
    fig1.savefig(os.path.join(output_parent_folder, 'interpolated_data_plots', (vid_in_path_no_ext + '_' + current_col_name + '.png')))
    
    return(position_interp_df)
    

# FILTER 
# filter interpolated data 
# intputs 
    # interpolated data for one landmark over one axis (either yolo or mediapipe), panda Series 
# outputs 
    # filtered data with all frames, included NaN missing values 
    # plots saved 
def filter_landmark_single_axis(original_data, video_fps, cutoff_hz, filter_order, vid_in_path, output_parent_folder): 
    # Normalized cutoff frequency (cutoff frequency divided by the Nyquist frequency)
    nyquist = 0.5 * video_fps
    normal_cutoff = cutoff_hz / nyquist

    # Design a Butterworth low-pass filter
    b, a = sig.butter(filter_order, normal_cutoff, btype='low', analog=False)

    # remove NaN Values from original data to filter 
    original_data_no_na = original_data.dropna()
    print(original_data_no_na.isna().sum())
    
    # filter data 
    filtered_data = sig.filtfilt(b, a, original_data_no_na)
    filtered_data = pd.Series(filtered_data)
    filtered_data.index = original_data_no_na.index

    # add missing data back into filtered 
    filtered_data_w_nan = original_data
    filtered_data_w_nan = filtered_data_w_nan.copy()
    non_nan_indices = original_data[original_data.notna()].index
    filtered_data_w_nan.loc[non_nan_indices] = filtered_data
    
    # plot filtered vs original 
    original_data_name = original_data.name
    frame_original_data = original_data.index
    frame_filtered_data = filtered_data.index
    frame_filtered_data_w_nan = filtered_data_w_nan.index
    
    fig2, ax2 = plt.subplots()
    fig2.suptitle('Filtered Data')
    ax2.plot(frame_original_data, original_data, color = 'red', alpha = 0.5, label = original_data_name)
    ax2.scatter(frame_filtered_data, filtered_data, color = 'blue', alpha = 0.5, marker = 'o', s = 1, label = original_data_name + '_filtered')
    ax2.scatter(frame_filtered_data_w_nan, filtered_data_w_nan, color = 'green', alpha = 0.5, marker = 'o', s = 1, label = original_data_name + '_filtered_w_nan')
    ax2.legend()
    ax2.set_xlabel('Frame')
    
    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]
    fig2.savefig(os.path.join(output_parent_folder, 'filtered_data_plots', (vid_in_path_no_ext + '_'+  original_data_name + '_filtered.png')))

    plt.close(fig2)
    plt.close()
    
    return (filtered_data_w_nan)



