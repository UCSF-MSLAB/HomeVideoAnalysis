#!/usr/bin/env python
# coding: utf-8

# In[1]:


#PKMAS - Stride Width is the perpendicular distance between the line connecting the two ipsilateral foot heel contacts (stride) with the contralateral heel contact between those events (cm).

# don't think we can calculate in same way without z distance in vertical videos.
    # from deID Brainwalk dataset: Zeno stride width - mean 9.3 cm, median 9.1 cm
    # Using  x value of heels when y difference between heels = 0, x value of world mp data (meters)


# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sig 
from raw_pose_analysis_funs.filter_interpolate_funs import (interpolate_landmark_single_axis, filter_landmark_single_axis)
import os 


# interpolate ankle markers Y axis data 
def stride_width_interp(mp_all_df,video_id_date_name, dir_out_prefix, max_gap, fps): 
    # KEEP THIS ORDER of landmark and pose THE SAME!! 
    # if change order of variables or for loop, need to update in later steps
    stride_width_landmarks = ['right_heel', 'left_heel'] 
    stride_width_axes = ['Y_world', 'X_world']

    # interpolate yolo and mp data 
    mp_stride_width_interp_dfs = []
    dataset = 'mediapipe'
    df = mp_all_df
    
    for landmark_i, current_landmark in enumerate(stride_width_landmarks): 
        for axis_i, current_axis in enumerate(stride_width_axes):
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
            mp_stride_width_interp_dfs = mp_stride_width_interp_dfs + [current_interp_dfs]

    return(mp_stride_width_interp_dfs)

# calculate stride width using interpolated data 
def calculate_stride_width(mp_stride_width_interp_dfs, vid_in_path, output_parent_folder, walk_num): 
    heel_right_y_df = mp_stride_width_interp_dfs[0]
    heel_right_x_df = mp_stride_width_interp_dfs[1]
    heel_left_y_df = mp_stride_width_interp_dfs[2]
    heel_left_x_df =  mp_stride_width_interp_dfs[3]

    # create data frame with heel y diff 
    heel_y_diff_df = pd.DataFrame(index = heel_right_y_df['frame'],
                            data = {'frame' : heel_right_y_df['frame'],
                                    'heel_y_diff_0' : heel_left_y_df['left_heel_Y_world_interpolated'] - heel_right_y_df['right_heel_Y_world_interpolated']
                                   })

    heel_y_diff_df['heel_y_diff_smooth'] = heel_y_diff_df['heel_y_diff_0'].rolling(window=15, min_periods=1).mean()

    # create data frame with heel x diff 
    heel_x_diff_df = pd.DataFrame(index = heel_right_x_df['frame'],
                                  data = {'frame' : heel_right_x_df['frame'],
                                          'heel_x_diff_0' : 
                                          heel_left_x_df['left_heel_X_world_interpolated'] - heel_right_x_df['right_heel_X_world_interpolated']
                                          })
                                                                      
    # frames when y diff crosses zero 
    # Determine the sign of each value (+1 for positive, -1 for negative, 0 for zero)
    signs = np.sign(heel_y_diff_df['heel_y_diff_smooth'])
    # find indices where sign changes (crosses zero) 
    cross_zero_frame =  signs.diff().ne(0).index[signs.diff().ne(0) & (signs != 0)]
    cross_zero_frame = cross_zero_frame[1:] # remove first value, frame = 0 

    # check if there is data at cross zero frames for x data 
    valid_r_cross_zero_indices = [idx for idx in cross_zero_frame if idx in heel_right_x_df.index]
    valid_l_cross_zero_indices = [idx for idx in cross_zero_frame if idx in heel_left_x_df.index]
    common_valid_cross_indices = list(set(valid_r_cross_zero_indices).intersection(valid_r_cross_zero_indices))
    
    # df of heel x differences at valid crossing points 
    heel_x_diff_at_cross = heel_x_diff_df.loc[common_valid_cross_indices]
    x_diff = abs(heel_x_diff_at_cross['heel_x_diff_0']) * 100 # convert m to cm 
    #x_diff_smooth = abs(heel_x_diff_at_cross['heel_x_diff_0']).rolling(window=15, min_periods=1).mean()

    # print frames x_diff is calculated at 
#    print('heel_x_diff_at_cross_df')
#    print(heel_x_diff_at_cross)
    
    x_diff_mean = x_diff.mean(skipna = True)
    x_diff_median = x_diff.median(skipna = True)
    x_diff_std = x_diff.std(skipna = True)
    x_diff_cv = (x_diff_std/x_diff_mean) * 100
    
    stride_width_stats_df = pd.DataFrame(data = {'stride_width_mean_cm' : [x_diff_mean],
                                                 'stride_width_median_cm' : [x_diff_median],
                                                 'stride_width_std' : [x_diff_std], 
                                                 'stride_width_cv' : [x_diff_cv]})

    # create and save data frame as .csv 
    output_folder = os.path.join(output_parent_folder, '005_gait_metrics', 'stride_width')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]


    # save each step wdith
   # x_diff_df = pd.DataFrame(data = {'step_width_m' : x_diff})
   # x_diff_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_' + walk_num + '_stride_width_per_step.csv')))
  #  x_diff.to_csv(x_diff_path)

    # plots --------------------------------------------------
    # plot y distance between heels, confirm zero crossing values are correct 
    fig1, ax1 = plt.subplots(figsize=(5.75, 3))
   # fig1.suptitle(os.path.splitext(os.path.basename(vid_in_path_no_ext))[0] + ': Stride Width')
    ax1.set_title('Vertical Distance Between Heels')
    ax1.plot(heel_y_diff_df['frame'], heel_y_diff_df['heel_y_diff_smooth'], color = 'black')
    ax1.axhline(y=0, color='grey', linestyle='--')
    ax1.plot(heel_y_diff_df.loc[common_valid_cross_indices, 'frame'], 
             heel_y_diff_df.loc[common_valid_cross_indices, 'heel_y_diff_smooth'], 
             "o", markersize = 7, color = 'black', label = 'Zero Crossing Frames')
    ax1.set_xlabel("Time (Frames)", fontsize = 11)
    ax1.set_ylabel("Meters", fontsize = 11)
    ax1.legend(loc='best', fontsize = 11)

    # for paper figure 
#    ax1.set_xlim([90, 145]) 
    

    output_plot_path_1 = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_' + walk_num + '_y_zero_cross.png')))
    fig1.savefig(output_plot_path_1, bbox_inches = 'tight')
#    plt.show()
    plt.close(fig1)
    plt.close()

    # plot x distance between heels, label zero crossing values 
    fig2, ax1 = plt.subplots(figsize=(10, 6))
    fig2.suptitle(os.path.splitext(os.path.basename(vid_in_path_no_ext))[0] + ': Stride Width')
    ax1.plot(heel_x_diff_df['frame'], heel_x_diff_df['heel_x_diff_0'], color = 'grey', label = 'X Difference')
    #ax1.plot(heel_x_diff_df['frame'], x_diff_smooth, color = 'grey', label = 'X Abs Difference Smooth')
    ax1.plot(heel_x_diff_at_cross['frame'], heel_x_diff_at_cross['heel_x_diff_0'], "x", color = 'orange', label='Zero Crossing')
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("L Heel X - R Heel X (meters)")
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    output_plot_path_2 = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_' + walk_num + '_x_width_zero_cross.png')))
    fig2.savefig(output_plot_path_2, bbox_inches = 'tight')
    plt.close(fig2)
    plt.close()

    return(stride_width_stats_df, x_diff)