#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sig 
from raw_pose_analysis_funs.filter_interpolate_funs import (interpolate_landmark_single_axis, 
filter_landmark_single_axis)
import os 


# In[ ]:


# interpolate ankle markers Y axis data 
def support_interp(yolo_df, vid_in_path, output_parent_folder, max_gap, fps): 
    # KEEP THIS ORDER of landmark and pose THE SAME!! 
    # if change order of variables or for loop, need to update in later steps
    stride_time_landmarks = ['right_ankle', 'left_ankle'] 
    stride_time_axes = ['Y_yolo_negative']

    # interpolate yolo and mp data 
    yolo_support_interp_dfs = []
    dataset = 'yolo'
    df = yolo_df
    
    for landmark_i, current_landmark in enumerate(stride_time_landmarks): 
        for axis_i, current_axis in enumerate(stride_time_axes):
            # interpolate 
            current_interp_dfs = interpolate_landmark_single_axis(df, # mediapipe or yolo data frame 
                                                                  current_landmark, # marker to interpolate 
                                                                  current_axis, # axis to interpolate
                                                                  max_gap, # seconds, maximum gap to interpolate over
                                                                  fps,
                                                                  vid_in_path,
                                                                  output_parent_folder,
                                                                  mediapipe_or_yolo = dataset)
            
            # add interpolated data for both landmark to one list 
            yolo_support_interp_dfs = yolo_support_interp_dfs + [current_interp_dfs]


    return(yolo_support_interp_dfs)

# filter gradient of yolo right and left ankle 
def support_filter_lowpass(yolo_support_interp_dfs, vid_in_path, output_parent_folder, fps, cutoff, order):
    yolo_support_filt_dfs = []
    for yolo_interp_i, yolo_interp_df in enumerate(yolo_support_interp_dfs): 
        current_yolo_support_filt = filter_landmark_single_axis(yolo_support_interp_dfs[yolo_interp_i].iloc[:,0],
                                                               fps, # video HZ
                                                               cutoff, # filter cutoff 
                                                               order, # butterworth filter order
                                                               vid_in_path,
                                                               output_parent_folder
                                                               )

        yolo_support_filt_dfs = yolo_support_filt_dfs + [current_yolo_support_filt]

    

    # save interpolated yolo X hip positions and filtered mp hip Z positions 
    return(yolo_support_filt_dfs)

def id_toe_off_heel_strike(yolo_df, video_id_date_name, dir_out_prefix, max_gap, fps, cutoff, order): 
    # interpolate 
    yolo_support_interp_dfs = support_interp(yolo_df, video_id_date_name, dir_out_prefix, max_gap, fps)

    right_ankle_y = yolo_support_interp_dfs[0]
    left_ankle_y = yolo_support_interp_dfs[1]

    # change of ankle y positions (velocity)
    right_ankle_y_gradient = np.gradient(right_ankle_y['right_ankle_Y_yolo_negative_interpolated'])
    right_ankle_y_gradient = pd.DataFrame(data = {'right_ankle_Y_yolo_negative_gradient' : right_ankle_y_gradient})
    right_ankle_y_gradient.index = right_ankle_y['frame']

    left_ankle_y_gradient = np.gradient(left_ankle_y['left_ankle_Y_yolo_negative_interpolated'])
    left_ankle_y_gradient = pd.DataFrame(data = {'left_ankle_Y_yolo_negative_gradient' : left_ankle_y_gradient}) 
    left_ankle_y_gradient.index = left_ankle_y['frame']

    ankle_y_gradients = [right_ankle_y_gradient] + [left_ankle_y_gradient]

    # filter change in y position (velocity)
    ankle_y_gradient_filt = support_filter_lowpass(ankle_y_gradients, video_id_date_name, dir_out_prefix, fps, cutoff, order)
    right_ankle_y_gradient_filt = ankle_y_gradient_filt[0]
    #right_ankle_y_gradient_filt_abs = abs(right_ankle_y_gradient_filt)

    left_ankle_y_gradient_filt = ankle_y_gradient_filt[1]
    #left_ankle_y_gradient_filt_abs = abs(left_ankle_y_gradient_filt)

    # find local min and max of filtered gradient (velocity) data 
    right_peak_frames, _ = sig.find_peaks(right_ankle_y_gradient_filt, distance = 20)
    right_valley_frames, _ = sig.find_peaks(-right_ankle_y_gradient_filt, distance = 20)

    left_peak_frames, _ = sig.find_peaks(left_ankle_y_gradient_filt, distance = 20)
    left_valley_frames, _ = sig.find_peaks(-left_ankle_y_gradient_filt, distance = 20) 

    # gradient of y gradient (acceleration)
    right_ankle_y_gradient_2 = np.gradient(right_ankle_y_gradient_filt)
    right_ankle_y_gradient_2 = pd.Series(right_ankle_y_gradient_2).rolling(window=20, min_periods=1).mean()

    left_ankle_y_gradient_2 = np.gradient(left_ankle_y_gradient_filt)
    left_ankle_y_gradient_2 = pd.Series(left_ankle_y_gradient_2).rolling(window=20, min_periods=1).mean()

    # find local min and max of filtered gradient of gradient (acceleration) data 
    right_peak_frames_2, _ = sig.find_peaks(right_ankle_y_gradient_2, distance = 20, prominence = .05)
    right_valley_frames_2, _ = sig.find_peaks(-right_ankle_y_gradient_2, distance = 20, prominence = .05)

    left_peak_frames_2, _ = sig.find_peaks(left_ankle_y_gradient_2, distance = 20, prominence = .05)
    left_valley_frames_2, _ = sig.find_peaks(-left_ankle_y_gradient_2, distance = 20, prominence = .05) 

    # right
    positive_r_ankle_grad = right_ankle_y_gradient_filt.loc[right_ankle_y_gradient_filt > 0]
    positive_r_ankle_grad_frames = positive_r_ankle_grad.index

    r_peak_df = pd.DataFrame(index = range(len(right_peak_frames_2)), columns = ['count', 'frame', 'gait_event'])
    for right_peak_i, current_right_peak in enumerate(right_peak_frames_2):
        r_peak_df.loc[right_peak_i, 'count'] = right_peak_i
        r_peak_df.loc[right_peak_i, 'frame'] = current_right_peak 

        if current_right_peak in positive_r_ankle_grad_frames: 
            r_peak_df.loc[right_peak_i, 'gait_event'] = 'right_toe_off'
        # if vel at that frame is negative, peak = heel strike
        else: 
            r_peak_df.loc[right_peak_i, 'gait_event'] = 'right_heel_strike'

    r_valley_df = pd.DataFrame(index = range(len(right_valley_frames_2)), columns = ['count', 'frame', 'gait_event'])
    for right_valley_i, current_right_valley in enumerate(right_valley_frames_2):
        r_valley_df.loc[right_valley_i, 'count'] = right_valley_i
        r_valley_df.loc[right_valley_i, 'frame'] = current_right_valley
    
        # if vel at that frame is positive, valley = heel strike
        if current_right_valley in positive_r_ankle_grad_frames: 
            r_valley_df.loc[right_valley_i, 'gait_event'] = 'right_heel_strike'
        # if vel at that frame is negative, valley = toe off 
        else: 
            r_valley_df.loc[right_valley_i, 'gait_event'] = 'right_toe_off'

    # left
    positive_l_ankle_grad = left_ankle_y_gradient_filt.loc[left_ankle_y_gradient_filt > 0]
    positive_l_ankle_grad_frames = positive_l_ankle_grad.index

    l_peak_df = pd.DataFrame(index = range(len(left_peak_frames_2)), columns = ['count', 'frame', 'gait_event'])
    for left_peak_i, current_left_peak in enumerate(left_peak_frames_2):
        l_peak_df.loc[left_peak_i, 'count'] = left_peak_i
        l_peak_df.loc[left_peak_i, 'frame'] = current_left_peak
        # if vel at that frame is positive, peak = toe off
        if current_left_peak in positive_l_ankle_grad_frames: 
            l_peak_df.loc[left_peak_i, 'gait_event'] = 'left_toe_off'
        # if vel at that frame is negative, peak = heel strike
        else: 
            l_peak_df.loc[left_peak_i, 'gait_event'] = 'left_heel_strike'

    l_valley_df = pd.DataFrame(index = range(len(left_valley_frames_2)), columns = ['count', 'frame', 'gait_event'])
    for left_valley_i, current_left_valley in enumerate(left_valley_frames_2):
        l_valley_df.loc[left_valley_i, 'count'] = left_valley_i
        l_valley_df.loc[left_valley_i, 'frame'] = current_left_valley
    
        # if vel at that frame is positive, valley = heel strike
        if current_left_valley in positive_l_ankle_grad_frames: 
            l_valley_df.loc[left_valley_i, 'gait_event'] = 'left_heel_strike'
        # if vel at that frame is negative, valley = toe off 
        else: 
            l_valley_df.loc[left_valley_i, 'gait_event'] = 'left_toe_off'

    # concatenate data frames and order 
    all_df = pd.concat([r_peak_df, r_valley_df, l_peak_df, l_valley_df])
    all_df = all_df.reset_index(drop=True)

    all_df_pivoted = all_df.pivot_table(index='count', columns='gait_event', values='frame', aggfunc='first')

    # if first event is right_toe off, order columns as 
    if all_df_pivoted.loc[0,'right_toe_off'] < all_df_pivoted.loc[0,'left_toe_off']:
        col_order = ['right_toe_off', 'right_heel_strike', 'left_toe_off', 'left_heel_strike']
    else: 
        col_order = ['left_toe_off', 'left_heel_strike', 'right_toe_off', 'right_heel_strike']

    all_events = all_df_pivoted[col_order]


    # #get strides one and two
    all_events_start = all_events.iloc[1:4, :]
    print('all_events_start')
    print(all_events_start)
 
    first_frame = all_events_start.iloc[0,0]
    if np.isnan(all_events_start.iloc[2,3]): 
        last_frame = all_events.iloc[0,4]
    else: 
        last_frame = all_events_start.iloc[2,3]
    

    # y position 
    r_ank_y_start = right_ankle_y.loc[first_frame:last_frame, 'right_ankle_Y_yolo_negative_interpolated']
    l_ank_y_start = left_ankle_y.loc[first_frame:last_frame, 'left_ankle_Y_yolo_negative_interpolated']

    # gradient 2 (acceleration)
    right_ankle_y_gradient_2_start = right_ankle_y_gradient_2[first_frame:last_frame]
    left_ankle_y_gradient_2_start = left_ankle_y_gradient_2[first_frame:last_frame]


    # plots 
    # plots # set y max for vertical lines 
    r_y_max = max(right_ankle_y['right_ankle_Y_yolo_negative_interpolated'])
    r_y_min = min(right_ankle_y['right_ankle_Y_yolo_negative_interpolated'])

    r_grad_max = max(right_ankle_y_gradient_filt)
    r_grad_min = min(right_ankle_y_gradient_filt)

    r_grad2_max = max(right_ankle_y_gradient_2)
    r_grad2_min = min(right_ankle_y_gradient_2)

    l_y_max = max(left_ankle_y['left_ankle_Y_yolo_negative_interpolated'])
    l_y_min = min(left_ankle_y['left_ankle_Y_yolo_negative_interpolated'])

    l_grad_max = max(left_ankle_y_gradient_filt)
    l_grad_min = min(left_ankle_y_gradient_filt)

    l_grad2_max = max(left_ankle_y_gradient_2)
    l_grad2_min = min(left_ankle_y_gradient_2)

    # right 
    vid_in_path_no_ext = os.path.splitext(os.path.basename(video_id_date_name))[0]
    fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 6))
    fig1.suptitle(vid_in_path_no_ext)
    ax1.plot(right_ankle_y['right_ankle_Y_yolo_negative_interpolated'], color = 'black', alpha = 0.5, label = 'right_ankle_y')
    ax1.set_ylabel('-Yolo Y (pixels)')
    ax1.set_xlabel('Frame')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # gradient 1 
    ax2.plot(right_ankle_y_gradient_filt, color = 'black', alpha = 0.5, label = 'right_ankle_y_gradient_filt')
    ax2.scatter(right_peak_frames, 
                right_ankle_y_gradient_filt[right_peak_frames], 
                color = 'green',  
                marker = "|", 
                s = 100,
                label = 'right_gradient_peak')
    ax2.scatter(right_valley_frames, 
                right_ankle_y_gradient_filt[right_valley_frames], 
                color = 'purple', 
                marker = "|",
                s = 100,
                label = 'right_gradient_valley')
    ax2.set_ylabel('Pixel/Frame')
    ax2.set_xlabel('Frame')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # gradient 2 
    ax3.plot(right_ankle_y_gradient_2, color = 'black', alpha = 0.5, label = 'right_gradient_2')
    ax3.scatter(right_peak_frames_2, 
                right_ankle_y_gradient_2[right_peak_frames_2], 
                color = 'blue',  
                marker = "|", 
                s = 100,
                label = 'right_gradient2_peak')
    ax3.scatter(right_valley_frames_2, 
                right_ankle_y_gradient_2[right_valley_frames_2], 
                color = 'red', 
                marker = "|",
                s = 100,
                label = 'right_gradient2_valley')
    ax3.set_ylabel('(Pixel/Frame) / Frame')
    ax3.set_xlabel('Frame')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # fig 2 = left 
    fig2, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 6))
    fig2.suptitle(vid_in_path_no_ext)
    ax1.plot(left_ankle_y['left_ankle_Y_yolo_negative_interpolated'], color = 'black', alpha = 0.5, label = 'left_ankle_y')
    ax1.set_ylabel('-Yolo Y (pixels)')
    ax1.set_xlabel('Frame')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # gradient 1 
    ax2.plot(left_ankle_y_gradient_filt, color = 'black', alpha = 0.5, label = 'left_ankle_y_gradient_filt')
    ax2.scatter(left_peak_frames, 
                left_ankle_y_gradient_filt[left_peak_frames], 
                color = 'green', 
                marker = "|",
                s = 100,
                label = 'left_gradient_peak')
    ax2.scatter(left_valley_frames, 
                left_ankle_y_gradient_filt[left_valley_frames], 
                color = 'purple',
                marker = "|",
                s = 100,
                label = 'left_gradient_valley')
    ax2.set_ylabel('Pixel/Frame')
    ax2.set_xlabel('Frame')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # gradient 2
    ax3.plot(left_ankle_y_gradient_2, color = 'black', alpha = 0.5, label = 'left_gradient_2')
    ax3.scatter(left_peak_frames_2, 
                left_ankle_y_gradient_2[left_peak_frames_2], 
                color = 'blue',  
                marker = "|", 
                s = 100,
                label = 'left_gradient2_peak')
    ax3.scatter(left_valley_frames_2, 
                left_ankle_y_gradient_2[left_valley_frames_2], 
                color = 'red', 
                marker = "|",
                s = 100,
                label = 'left_gradient2_valley')
    ax3.set_ylabel('(Pixel/Frame) / Frame')
    ax3.set_xlabel('Frame')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plot calculated and true toe off and heel strike 
    fig3, ax1 = plt.subplots(figsize=(10, 6))
    fig3.suptitle(vid_in_path_no_ext)
    ax1.plot(right_ankle_y_gradient_2_start, color = 'orange', alpha = 0.5, label = 'right_ankle_grad2')
    ax1.plot(left_ankle_y_gradient_2_start, color = 'black', alpha = 0.5, label = 'left_ankle_grad2')

    ax1.vlines(x = all_events_start['right_heel_strike'], ymax = l_grad2_max, ymin = l_grad2_min, 
               color = 'blue',
               alpha = 0.3, 
               linestyle = 'dotted', 
               label = 'r_calc_heel_strike')

    ax1.vlines(x = all_events_start['right_toe_off'], ymax = l_grad2_max, ymin = l_grad2_min, 
               color = 'red',
               alpha = 0.3, 
               linestyle = 'dotted', 
               label = 'r_calc_toe_off')

    ax1.vlines(x = all_events_start['left_heel_strike'], ymax = l_grad2_max, ymin = l_grad2_min, 
               color = 'green',
               alpha = 0.3, 
               linestyle = 'dotted', 
               label = 'l_calc_heel_strike')

    ax1.vlines(x = all_events_start['left_toe_off'], ymax = l_grad2_max, ymin = l_grad2_min, 
               color = 'purple',
               alpha = 0.3, 
               linestyle = 'dotted', 
               label = 'l_calc_toe_off')

    ax1.set_ylabel('-Yolo Y (pixels)')
    ax1.set_xlabel('Frame')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    # save outputs ------------  
    output_folder = os.path.join(dir_out_prefix, '005_gait_metrics', 'double_single_support')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # plot 1 
    output_plot_1 = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_right_support.png')))
    fig1.savefig(output_plot_1, bbox_inches = 'tight')
    plt.close(fig1)
    plt.close()

    # plot 2 
    output_plot_2 = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_left_support.png')))
    fig2.savefig(output_plot_2, bbox_inches = 'tight')
    plt.close(fig2)
    plt.close()

    # plot 3 
    output_plot_3 = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + 'right_and_left.png')))
    fig3.savefig(output_plot_3, bbox_inches = 'tight')
    plt.close(fig3)
    plt.close()

    return([right_peak_frames_2, right_valley_frames_2, left_peak_frames_2, left_valley_frames_2, all_events, all_events_start])

