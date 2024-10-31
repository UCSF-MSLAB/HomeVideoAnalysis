#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import os 
import numpy as np 
import scipy.signal as sig
import matplotlib.pyplot as plt 
from raw_pose_analysis_funs.filter_interpolate_funs import (interpolate_landmark_single_axis)

#  double support 
## NOTE!! assumes person starts walking away from the camera. Will not be correct if they start walking towards camera !!! 

# functions ------------------------------------------------------
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

# -----------------------------------------------------------
# identify peaks and valleys of ankle y velocity and acceleration 
def ankle_y_vel_accel_peak_min(ank_y_df, diff_period, peaks_distance, peaks_prominence_percent_max, valleys_prominence_percent_max):
    ank_y_df['diff1_vel'] = ank_y_df.iloc[: ,2].diff(periods = diff_period) # interpolated data 
    ank_y_df['diff2_accel'] = ank_y_df['diff1_vel'].diff(periods = diff_period)

    # find local min and max of velocity and acceleration  
    peak_vel_prominence = peaks_prominence_percent_max * ank_y_df['diff1_vel'].max(skipna = True)
    valley_vel_prominence = valleys_prominence_percent_max * ank_y_df['diff1_vel'].max(skipna = True)

    peak_accel_prominence = peaks_prominence_percent_max * ank_y_df['diff2_accel'].max(skipna = True)
    valley_accel_prominence = valleys_prominence_percent_max * ank_y_df['diff2_accel'].max(skipna = True)

    
    vel_peak_i, _ = sig.find_peaks(ank_y_df['diff1_vel'], 
                                   distance = peaks_distance, 
                                   prominence = peak_vel_prominence)
    vel_valley_i, _ = sig.find_peaks(-ank_y_df['diff1_vel'], 
                                     distance = peaks_distance, 
                                     prominence = valley_vel_prominence)
    accel_peak_i, _ = sig.find_peaks(ank_y_df['diff2_accel'],
                                     distance = peaks_distance,  
                                     prominence = peak_accel_prominence)
    accel_valley_i, _ = sig.find_peaks(-ank_y_df['diff2_accel'], 
                                       distance = peaks_distance, 
                                       prominence = valley_accel_prominence)

    # find_peaks uses positional indices, not value of index column. Account for gaps in frame with nan values
    vel_peak_df = pd.DataFrame(data = {'frame' :  ank_y_df.iloc[vel_peak_i].index, 
                                       'diff1_vel' : ank_y_df.iloc[vel_peak_i]['diff1_vel']}) 
    vel_valley_df = pd.DataFrame(data = {'frame' :  ank_y_df.iloc[vel_valley_i].index, 
                                        'diff1_vel' : ank_y_df.iloc[vel_valley_i]['diff1_vel']}) 
    accel_peak_df = pd.DataFrame(data = {'frame' :  ank_y_df.iloc[accel_peak_i].index, 
                                        'diff2_accel' : ank_y_df.iloc[accel_peak_i]['diff2_accel']}) 
    accel_valley_df = pd.DataFrame(data = {'frame' :  ank_y_df.iloc[accel_valley_i].index, 
                                        'diff2_accel' : ank_y_df.iloc[accel_valley_i]['diff2_accel']})
    # save lists 
    # 0 = data for each frame 
    # 1 = vel peak data frame 
    # 2 = vel valley data frame 
    # 3 = accel peak data frame 
    # 4 = accel valley data frame
    return([ank_y_df, vel_peak_df, vel_valley_df, accel_peak_df, accel_valley_df])

# -----------------------------------------------------------------------------------
# get gait events from position, velocity, and acceleration data 

# find where R y becomes greater than L y (right toe off) and vice versa (left toe off)
def find_cross_frames(right_ank_y_data, left_ank_y_data):
    right_ank_y_df = right_ank_y_data[0] 
    #right_ank_y_df.index = range(len(right_ank_y_df))
    right_ank_y_df = right_ank_y_df.drop(['frame', 'diff1_vel', 'diff2_accel'], axis = 1)
    
    left_ank_y_df = left_ank_y_data[0]
    #left_ank_y_df.index = range(len(left_ank_y_df))
    left_ank_y_df = left_ank_y_df.drop(['frame', 'diff1_vel', 'diff2_accel'], axis = 1)

    # join based on common index (frames) 
    df = right_ank_y_df.join(left_ank_y_df)
    
    # Create a new column to identify whether col1 is greater than col2
    df['r_greater'] = df['right_ankle_Y_yolo_negative_interpolated'] > df['left_ankle_Y_yolo_negative_interpolated']

    # Find the places where the value of 'col1_greater' changes
    df['change'] = df['r_greater'].ne(df['r_greater'].shift())

    # Get the frames where the change happens (both switch cases)
    switch_frames = df.loc[df['change'] == True].index
    
    # Filter df to just include dropped 
    y_cross_df = df.loc[df['change'] == True]

    # new data frame - only y crossing values 
    y_cross_df = pd.DataFrame(data = {'frame' : df.loc[df['change'] == True].index,
                                      'r_ank_y_neg_interp' : df.loc[switch_frames]['right_ankle_Y_yolo_negative_interpolated'], 
                                      'l_ank_y_neg_interp' : df.loc[switch_frames]['left_ankle_Y_yolo_negative_interpolated'], 
                                      'r_greater' : df.loc[switch_frames]['r_greater'], 
                                      'change' : df.loc[switch_frames]['change']
                                     }) 

    # right greater than left = right toe off (? double check)
    r_toe_off_df = y_cross_df[y_cross_df['r_greater'] == True]
    l_toe_off_df = y_cross_df[y_cross_df['r_greater'] == False]

    # mean of l and r ankle y position at each cross 
    y_cross_df['ank_y_mean'] =  y_cross_df[['r_ank_y_neg_interp', 
                                            'l_ank_y_neg_interp']].mean(axis=1)
    y_cross_df['y_mean_diff'] = y_cross_df['ank_y_mean'].diff()
    # y_mean_diff_shift: change in y position after y_cross_df['frame'] in that row 
    y_cross_df['y_mean_diff_shift'] = y_cross_df['y_mean_diff'].shift(-1)
    
    return([y_cross_df, r_toe_off_df, l_toe_off_df, df])

# get max vel peak, over first 10 local maxima  
def find_max_peak_frame(ank_vel_data): 
    vel_max_df = ank_vel_data[ank_vel_data['diff1_vel'] == ank_vel_data['diff1_vel'].max(skipna = True)]
    vel_max_frame = vel_max_df['frame'].iloc[0]    
    return(vel_max_frame)


def id_toe_off_heel_strike(right_ank_y_data, left_ank_y_data, video_id_date_name, dir_out_prefix):

    # find where right and left y position crosses and vice versa (heel strike + toe off) 
    crossing_results = find_cross_frames(right_ank_y_data, left_ank_y_data)

    # greatest peak of first ten velocity local max 
    r_vel_max_frame = find_max_peak_frame(right_ank_y_data[1][0:15])  
    l_vel_max_frame = find_max_peak_frame(left_ank_y_data[1][0:15])

    # find crossing value closest to max vel peak (either left or right, whichver occurs first) 
    #first_max_vel = max(r_vel_max_frame, l_vel_max_frame)
    #y_cross_df = crossing_results[0]
    #closest_index = [(y_cross_df['frame'] - first_max_vel).abs().argmin()]
   # closest_row = y_cross_df.iloc[closest_index]
    # start frame = first toe off (crossing) closest to the first velocity max 
    #start_frame = closest_row['frame'].iloc[0]

    # second crossing 
   # y_cross_df = crossing_results[0]
   # second_row = y_cross_df.iloc[[1]]
   # print(second_row)
  #  start_frame = second_row['frame'].iloc[0] # second crossing 
   # print(start_frame)


    # new select first 10 values where l and r y values cross 
    y_cross_df = crossing_results[0]
    
    # select first 30 crossing points 
    first_15_cross = y_cross_df.iloc[0:15, :]
    # select only positive mean y position between cross 
    first_15_cross_pos = first_15_cross[first_15_cross['y_mean_diff_shift'] >= 0]
    # of the three largest crossings, select the first crossing  
    top_3_diffs = first_15_cross_pos['y_mean_diff_shift'].nlargest(3).values
    top_3_diff_df = first_15_cross_pos[first_15_cross_pos['y_mean_diff_shift'].isin(top_3_diffs)]
    # start = second of top three largest crossings (first step little funky if starting walk)
    start_frame_row = top_3_diff_df.iloc[[1]]
    start_frame = start_frame_row['frame'].iloc[0]

    # if right max frame occurs first, start gait cyle at right toe off 
        # 1 and 2 labels below: either right or left foot 
    if (start_frame_row['r_greater'].iloc[0] == True): 
        gait_cycle_events = ['right_toe_off1', 'right_heel_strike1', 'left_toe_off1', 'left_heel_strike1', 
                             'right_toe_off2', 'right_heel_strike2', 'left_toe_off2', 'left_heel_strike2'] 
     
        ank_data_1 = right_ank_y_data
        crossing_df_1 = crossing_results[1] # right 
    
        ank_data_2 = left_ank_y_data
        crossing_df_2 = crossing_results[2] # left 
    
    # if left  max frame occurs first, start gait cyle at left toe off     
    elif (start_frame_row['r_greater'].iloc[0] == False):
        gait_cycle_events = ['left_toe_off1', 'left_heel_strike1', 'right_toe_off1', 'right_heel_strike1', 
                             'left_toe_off2', 'left_heel_strike2', 'right_toe_off2', 'right_heel_strike2'] 
    
        ank_data_1 = left_ank_y_data
        crossing_df_1 = crossing_results[2] # left 
    
        ank_data_2 = right_ank_y_data
        crossing_df_2 = crossing_results[1] # right
      
    # blank df to populate with gait events 
    gait_events_df = pd.DataFrame(index = range(8),
                                  columns = ['event', 'frame'])

    # event 0 - foot 1 toe off; either right or left toe off identified from crossing over point of y position 
    gait_events_df.loc[0, 'event'] = gait_cycle_events[0]
    gait_events_df.loc[0, 'frame'] = start_frame

    # event 1 = foot 1 heel strike; accel valley followed by peak -> peak = heel strike 
    # get first accel valley after last velocity crossing point
    accel_valley_1 = ank_data_1[4] # accel valleys 
    accel_valley_1 = accel_valley_1[accel_valley_1['frame'] > gait_events_df.loc[0, 'frame']] 
    next_accel_valley_1 = accel_valley_1['frame'].iloc[0]

    accel_peak_1 = ank_data_1[3] # accel peaks 
    accel_peak_1 = accel_peak_1[accel_peak_1['frame'] > next_accel_valley_1] # peaks after next_accel_valley_1 
    next_accel_peak_1 = accel_peak_1['frame'].iloc[0] # select frame from first row in df 

    gait_events_df.loc[1, 'event'] = gait_cycle_events[1]
    gait_events_df.loc[1, 'frame'] = next_accel_peak_1

    # event 2 = foot 2 toe off; toe off = next crossing point   
    next_crosses_df_2 = crossing_df_2[crossing_df_2['frame'] > gait_events_df.loc[1, 'frame']] # highest velocity peaks after other foot heel strike 
    next_cross_2 = next_crosses_df_2['frame'].iloc[0] # select frame from first row in df 

    gait_events_df.loc[2, 'event'] = gait_cycle_events[2]
    gait_events_df.loc[2, 'frame'] = next_cross_2

    # event 3 = foot 2 heel strike; accel valley followed by peak -> peak = heel strike 
    # get first accel valley after last velocity crossing point
    accel_valley_2 = ank_data_2[4] # accel valleys 
    accel_valley_2 = accel_valley_2[accel_valley_2['frame'] > gait_events_df.loc[2, 'frame']]
    next_accel_valley_2 = accel_valley_2['frame'].iloc[0] # select frame from first row in df 

    accel_peak_2 = ank_data_2[3] # accel peaks 
    accel_peak_2 = accel_peak_2[accel_peak_2['frame'] > next_accel_valley_2] # peaks after next_accel_valley_2 
    next_accel_peak_2 = accel_peak_2['frame'].iloc[0] # select frame from first row in df 

    gait_events_df.loc[3, 'event'] = gait_cycle_events[3]
    gait_events_df.loc[3, 'frame'] = next_accel_peak_2

    # event 4 - foot 1 toe off #2;  next crossing point
    next_crosses_df_1b = crossing_df_1[crossing_df_1['frame'] > gait_events_df.loc[3, 'frame']] # highest velocity peaks after other foot heel strike
    next_cross_1b = next_crosses_df_1b['frame'].iloc[0] # select frame from first row in df

    gait_events_df.loc[4, 'event'] = gait_cycle_events[4]
    gait_events_df.loc[4, 'frame'] = next_cross_1b

    # event 5 - foot 1 heel strike #2;  accel valley followed by peak -> peak = heel strike
    # get first accel valley after last velocity crossing point
    accel_valley_1b = ank_data_1[4] # accel valleys 
    accel_valley_1b = accel_valley_1b[accel_valley_1b['frame'] > gait_events_df.loc[4, 'frame']]
    next_accel_valley_1b = accel_valley_1b['frame'].iloc[0]

    accel_peak_1b = ank_data_1[3] # accel peaks 
    accel_peak_1b = accel_peak_1b[accel_peak_1b['frame'] > next_accel_valley_1b] # peaks after next_accel_valley_1 
    next_accel_peak_1b = accel_peak_1b['frame'].iloc[0] # select frame from first row in df 

    gait_events_df.loc[5, 'event'] = gait_cycle_events[5]
    gait_events_df.loc[5, 'frame'] = next_accel_peak_1b

    # event 6 - foot 2 toe off #2; toe off = next crossing point
    next_crosses_df_2b = crossing_df_2[crossing_df_2['frame'] > gait_events_df.loc[5, 'frame']] # highest velocity peaks after other foot heel strike 
    next_cross_2b = next_crosses_df_2b['frame'].iloc[0] # select frame from first row in df 

    gait_events_df.loc[6, 'event'] = gait_cycle_events[6]
    gait_events_df.loc[6, 'frame'] = next_cross_2b

    # event 7 - foot 2 heel strike #2; accel valley followed by peak -> peak = heel strike
    # get first accel valley after last velocity crossing point
    accel_valley_2b = ank_data_2[4] # accel valleys 
    accel_valley_2b = accel_valley_2b[accel_valley_2b['frame'] > gait_events_df.loc[6, 'frame']]
    next_accel_valley_2b = accel_valley_2b['frame'].iloc[0] # select frame from first row in df 

    accel_peak_2b = ank_data_2[3] # accel peaks 
    accel_peak_2b = accel_peak_2b[accel_peak_2b['frame'] > next_accel_valley_2b] # peaks after next_accel_valley_2 
    next_accel_peak_2b = accel_peak_2b['frame'].iloc[0] # select frame from first row in df 

    gait_events_df.loc[7, 'event'] = gait_cycle_events[7]
    gait_events_df.loc[7, 'frame'] = next_accel_peak_2b

    calc_right_toe_df = gait_events_df[gait_events_df['event'].str.contains('right_toe', case=False, na=False)]
    calc_right_heel_df = gait_events_df[gait_events_df['event'].str.contains('right_heel', case=False, na=False)]
    calc_left_toe_df = gait_events_df[gait_events_df['event'].str.contains('left_toe', case=False, na=False)]
    calc_left_heel_df = gait_events_df[gait_events_df['event'].str.contains('left_heel', case=False, na=False)]

    # -----------------------------------------
    # plots 
    # set min max 
    x_max = max(gait_events_df['frame']) + 50

    r_y_max = right_ank_y_data[0]['right_ankle_Y_yolo_negative_interpolated'].max(skipna = True)
    r_y_min = right_ank_y_data[0]['right_ankle_Y_yolo_negative_interpolated'].min(skipna = True)

    r_vel_max = right_ank_y_data[0]['diff1_vel'].max(skipna = True)
    r_vel_min = right_ank_y_data[0]['diff1_vel'].min(skipna = True)

    r_accel_max = right_ank_y_data[0]['diff2_accel'].max(skipna = True)
    r_accel_min = right_ank_y_data[0]['diff2_accel'].min(skipna = True)

    l_y_max = left_ank_y_data[0]['left_ankle_Y_yolo_negative_interpolated'].max(skipna = True)
    l_y_min = left_ank_y_data[0]['left_ankle_Y_yolo_negative_interpolated'].min(skipna = True)

    l_vel_max = left_ank_y_data[0]['diff1_vel'].max(skipna = True)
    l_vel_min = left_ank_y_data[0]['diff1_vel'].min(skipna = True)

    l_accel_max = left_ank_y_data[0]['diff2_accel'].max(skipna = True)
    l_accel_min = left_ank_y_data[0]['diff2_accel'].min(skipna = True)


    # plots 
    # right 
    fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 6))
    fig1.suptitle(video_id_date_name)
    ax1.plot(right_ank_y_data[0]['frame'],
             right_ank_y_data[0]['right_ankle_Y_yolo_negative_interpolated'], color = 'black', alpha = 0.5, label = 'right_ankle_y')
    ax1.set_ylabel('-Yolo Y (pixels)')
    ax1.set_xlabel('Frame')
    ax1.set_xlim(0, x_max)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Diff 1 - velocity 
    ax2.plot(right_ank_y_data[0]['frame'], right_ank_y_data[0]['diff1_vel'], color = 'black', alpha = 0.5, label = 'right_vel')
    ax2.scatter(right_ank_y_data[1].index, 
                right_ank_y_data[1]['diff1_vel'], 
                color = 'green',  
                marker = "|", 
                s = 100,
                label = 'right_vel_peak')
    ax2.scatter(right_ank_y_data[2]['frame'], 
                right_ank_y_data[2]['diff1_vel'], 
                color = 'purple', 
                marker = "|",
                s = 100,
                label = 'right_vel_valley')
    ax2.set_ylabel('Pixel/Frame')
    ax2.set_xlabel('Frame')
    ax2.set_xlim(0, x_max)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # diff 2 - acceleration 
    ax3.plot(right_ank_y_data[0]['frame'], right_ank_y_data[0]['diff2_accel'], color = 'black', alpha = 0.5, label = 'right_accel')
    ax3.scatter(right_ank_y_data[3]['frame'], 
                right_ank_y_data[3]['diff2_accel'], 
                color = 'blue',  
                marker = "|", 
                s = 100,
                label = 'right_accel_peak')
    ax3.scatter(right_ank_y_data[4]['frame'], 
                right_ank_y_data[4]['diff2_accel'], 
                color = 'red', 
                marker = "|",
                s = 100,
                label = 'right_accel_valley')
    ax3.set_ylabel('(Pixel/Frame) / Frame')
    ax3.set_xlabel('Frame')
    ax3.set_xlim(0, x_max)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    
    # fig 2 = left 
    fig2, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 6))
    fig2.suptitle(video_id_date_name)
    ax1.plot(left_ank_y_data[0]['frame'], 
             left_ank_y_data[0]['left_ankle_Y_yolo_negative_interpolated'], color = 'black', alpha = 0.5, label = 'left_ankle_y')
    ax1.set_ylabel('-Yolo Y (pixels)')
    ax1.set_xlabel('Frame')
    ax1.set_xlim(0, x_max)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # diff 1 = velocity 
    ax2.plot(left_ank_y_data[0]['frame'],
             left_ank_y_data[0]['diff1_vel'], color = 'black', alpha = 0.5, label = 'left_vel')
    ax2.scatter(left_ank_y_data[1]['frame'], 
                left_ank_y_data[1]['diff1_vel'], 
                color = 'green', 
                marker = "|",
                s = 100,
                label = 'left_vel_peak')
    ax2.scatter(left_ank_y_data[2]['frame'], 
                left_ank_y_data[2]['diff1_vel'], 
                color = 'purple',
                marker = "|",
                s = 100,
                label = 'left_vel_valley')
    ax2.set_ylabel('Pixel/Frame')
    ax2.set_xlabel('Frame')
    ax2.set_xlim(0, x_max)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # diff 2 = acceleration 
    ax3.plot(left_ank_y_data[0]['frame'],
             left_ank_y_data[0]['diff2_accel'], color = 'black', alpha = 0.5, label = 'left_accel')
    ax3.scatter(left_ank_y_data[3]['frame'], 
                left_ank_y_data[3]['diff2_accel'], 
                color = 'blue',  
                marker = "|", 
                s = 100,
                label = 'left_accel_peak')
    ax3.scatter(left_ank_y_data[4]['frame'], 
                left_ank_y_data[4]['diff2_accel'], 
                color = 'red', 
                marker = "|",
                s = 100,
                label = 'left_accel_valley')
    ax3.set_ylabel('(Pixel/Frame) / Frame')
    ax3.set_xlabel('Frame')
    ax3.set_xlim(0, x_max)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plot calculated and true toe off and heel strike 
    fig3, ax1 = plt.subplots(figsize=(10, 6))
    fig3.suptitle(video_id_date_name)
    ax1.plot(right_ank_y_data[0]['frame'],
             right_ank_y_data[0]['right_ankle_Y_yolo_negative_interpolated'], color = 'orange', alpha = 1, label = 'right_ankle_y')
    ax1.plot(left_ank_y_data[0]['frame'],
             left_ank_y_data[0]['left_ankle_Y_yolo_negative_interpolated'], color = 'black', alpha = 1, label = 'left_ankle_y')

    ax1.vlines(x = calc_right_toe_df['frame'], ymax = l_y_max, ymin = l_y_min, 
               color = 'orange',
               alpha = 0.7, 
               linestyle = 'dashed', 
               label = 'calculated_right_toe_off')
    ax1.vlines(x = calc_right_heel_df['frame'], ymax = l_y_max, ymin = l_y_min, 
               color = 'orange',
               alpha = 0.7, 
               linestyle = 'dotted', 
               label = 'calculated_right_heel_strike')
    ax1.vlines(x = calc_left_toe_df['frame'], ymax = l_y_max, ymin = l_y_min, 
               color = 'black',
               alpha = 0.7, 
               linestyle = 'dashed', 
               label = 'calculated_left_toe_off')
    ax1.vlines(x = calc_left_heel_df['frame'], ymax = l_y_max, ymin = l_y_min, 
               color = 'black',
               alpha = 0.7, 
               linestyle = 'dotted', 
               label = 'calculated_left_heel_strike')

    ax1.set_ylabel('-Yolo Y (pixels)')
    ax1.set_xlabel('Frame')
    ax1.set_xlim(0, x_max)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    
    # save plots and .csv 
    output_folder = os.path.join(dir_out_prefix, '005_gait_metrics', 'support')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_in_path_no_ext = os.path.splitext(os.path.basename(video_id_date_name))[0]

    # save gait events as .csv 
    gait_events_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_gait_events.csv')))
    gait_events_df.to_csv(gait_events_path)
 
    output_plot_1 = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_right_support.png')))
    fig1.savefig(output_plot_1, bbox_inches = 'tight')
    plt.close(fig1)
    plt.close()

    output_plot_2 = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_left_support.png')))
    fig2.savefig(output_plot_2, bbox_inches = 'tight')
    plt.close(fig2)
    plt.close()
    
    output_plot_3 = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_left_right_support.png')))
    fig3.savefig(output_plot_3, bbox_inches = 'tight')
    plt.close(fig3)
    plt.close()

    return(gait_events_df)


### Calculate single and double support % from gait event frames + save excel 
def calculate_single_double_support(gait_events_df, fps, video_id_date_name, dir_out_prefix):

    # foot 1 = foot in first row of data frame (first toe off)
    # foot 2 = other foot 

    # gait cycle time
    # The period of time from first contact of one foot, to the following first contact of the same foot (sec).
    foot1_gait_cycle_time = (gait_events_df.loc[5, 'frame'] - gait_events_df.loc[1, 'frame']) / fps 
    foot2_gait_cycle_time = (gait_events_df.loc[7, 'frame'] - gait_events_df.loc[3, 'frame']) / fps

    # stance time 
    # The period of time when the foot is in contact with the ground (secs).
    foot1_stance_time = (gait_events_df.loc[4, 'frame'] - gait_events_df.loc[1, 'frame']) / fps 
    foot2_stance_time = (gait_events_df.loc[6, 'frame'] - gait_events_df.loc[3, 'frame']) / fps 

    # stance % 
    # Stance Percentage is Stance Time presented as a percentage of the Gait Cycle Time (%). 
    foot1_stance_per = (foot1_stance_time / foot1_gait_cycle_time) * 100 
    foot2_stance_per = (foot2_stance_time / foot2_gait_cycle_time) * 100

    # swing time 
    # The period of time the foot is not in contact with the ground (secs); 𝐒𝐰𝐢𝐧𝐠 𝐓𝐢𝐦𝐞=𝐆𝐚𝐢𝐭 𝐂𝐲𝐜𝐥𝐞 𝐓𝐢𝐦𝐞−𝐒𝐭𝐚𝐧𝐜𝐞 𝐓𝐢𝐦𝐞
    foot1_swing_time = foot1_gait_cycle_time - foot1_stance_time 
    foot2_swing_time = foot2_gait_cycle_time - foot2_stance_time

    # swing % 
    # Swing Percentage is Swing Time presented as a percentage of the Gait Cycle Time (%). 
    foot1_swing_per = (foot1_swing_time / foot1_gait_cycle_time) * 100
    foot2_swing_per = (foot2_swing_time / foot2_gait_cycle_time) * 100

    # initial double support time 
    # The period of time when both feet are in contact with the ground at the beginning of the stance phase (secs).
    foot1_ini_double_support_time = (gait_events_df.loc[2, 'frame'] - gait_events_df.loc[1, 'frame']) / fps 
    foot2_ini_double_support_time = (gait_events_df.loc[4, 'frame'] - gait_events_df.loc[3, 'frame']) / fps 

    # terminal double support time 
    # The period of time when both feet are in contact with the ground at the end of the stance phase (secs).
    foot1_term_double_support_time = (gait_events_df.loc[4, 'frame'] - gait_events_df.loc[3, 'frame']) / fps
    foot2_term_double_support_time = (gait_events_df.loc[6, 'frame'] - gait_events_df.loc[5, 'frame']) / fps

    # total double support time 
    # The sum of all periods when both feet are in contact with the ground during stance phase (secs)
    foot1_tot_double_support_time = foot1_ini_double_support_time + foot1_term_double_support_time
    foot2_tot_double_support_time = foot2_ini_double_support_time + foot2_term_double_support_time

    # total double support % 
    # Total Double Support Time presented as a percentage of the Gait Cycle Time (%)
    foot1_double_support_per = (foot1_tot_double_support_time / foot1_gait_cycle_time) * 100
    foot2_double_support_per = (foot2_tot_double_support_time / foot2_gait_cycle_time) * 100

    # single support time 
    # The period of time when only the current foot is in contact with the ground (secs
    foot1_single_support_time = (gait_events_df.loc[3, 'frame'] - gait_events_df.loc[2, 'frame']) / fps
    foot2_single_support_time = (gait_events_df.loc[5, 'frame'] - gait_events_df.loc[4, 'frame']) / fps

    # single support % 
    # Single Support Time expressed as a percentage of the Gait Cycle Time (%). 
    foot1_single_support_per = (foot1_single_support_time / foot1_gait_cycle_time) * 100
    foot2_single_support_per = (foot2_single_support_time / foot2_gait_cycle_time) * 100 

    # save in data frame 
    df = pd.DataFrame(data = {'foot1_gait_cycle_time' : [foot1_gait_cycle_time],
                              'foot1_stance_time' : [foot1_stance_time],
                              'foot1_stance_per' : [foot1_stance_per],
                              'foot1_swing_time' : [foot1_swing_time],
                              'foot1_swing_per' : [foot1_swing_per], 
                              'foot1_ini_double_support_time' : [foot1_ini_double_support_time],
                              'foot1_term_double_support_time' : [foot1_term_double_support_time],
                              'foot1_tot_double_support_time' : [foot1_tot_double_support_time], 
                              'foot1_double_support_per' : [foot1_double_support_per],
                              'foot1_single_support_time' : [foot1_single_support_time], 
                              'foot1_single_support_per' : [foot1_single_support_per],
                              'foot2_gait_cycle_time' : [foot2_gait_cycle_time],
                              'foot2_stance_time' : [foot2_stance_time],
                              'foot2_stance_per' : [foot2_stance_per],
                              'foot2_swing_time' : [foot2_swing_time],
                              'foot2_swing_per' : [foot2_swing_per], 
                              'foot2_ini_double_support_time' : [foot2_ini_double_support_time],
                              'foot2_term_double_support_time' : [foot2_term_double_support_time],
                              'foot2_tot_double_support_time' : [foot2_tot_double_support_time], 
                              'foot2_double_support_per' : [foot2_double_support_per],
                              'foot2_single_support_time' : [foot2_single_support_time], 
                              'foot2_single_support_per' : [foot2_single_support_per]
                             })

    # if first row of gait_event_df is left toe off, foot 1 = left 
    if 'left_' in gait_events_df.loc[0, 'event']: 
        df.columns = df.columns.str.replace('foot1', 'left')
        df.columns = df.columns.str.replace('foot2', 'right')

    # if first row of gait_event_df is right toe off, foot 1 = right 
    elif 'right_' in gait_events_df.loc[0, 'event']: 
        df.columns = df.columns.str.replace('foot1', 'right')
        df.columns = df.columns.str.replace('foot2', 'left')

    # add single support and double support % mean to df 
    df['double_support_per_mean_l_r'] = df[['right_double_support_per', 'left_double_support_per']].mean(axis=1)
    df['single_support_per_mean_l_r'] = df[['right_single_support_per', 'left_single_support_per']].mean(axis=1)
    # add L/R double and single support ratio to df 
    df['double_support_per_ratio_l_r'] = df['left_double_support_per'] / df['right_double_support_per']
    df['single_support_per_ratio_l_r'] = df['left_single_support_per'] / df['right_single_support_per']

    # save . csv with metrics 
    output_folder = os.path.join(dir_out_prefix, '005_gait_metrics', 'support')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_in_path_no_ext = os.path.splitext(os.path.basename(video_id_date_name))[0]

    support_metrics_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_support_metrics.csv')))
    df.to_csv(support_metrics_path)
        
    return(df)

    

    

    
    
