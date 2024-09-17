#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np 
import os 

# input = mp_pose, mp_world, and yolo df for one video  
# output = one mediapipe df and one yolo df  
def merge_mp_pose_world(mp_pose_df, mp_world_df, yolo_df):
    
    # rename mp columns 
    mp_pose_df = mp_pose_df.rename(columns = {"X" : "X_pose", 
                                              "Y" : "Y_pose", 
                                              "Z" : "Z_pose", 
                                              "Unnamed: 0" : "label_num"})

    mp_world_df = mp_world_df.rename(columns = {"X" : "X_world", 
                                                 "Y" : "Y_world", 
                                                 "Z" : "Z_world", 
                                                 "Unnamed: 0" : "label_num"})

    yolo_df = yolo_df.rename(columns = {"Unnamed: 0" : "label_num",
                                       'X': 'X_yolo', 
                                        'Y': 'Y_yolo'})
    

    # merge mp world and pose (same vis, markers)
    mp_all_df = mp_pose_df
    mp_all_df['X_world'] = mp_world_df['X_world']
    mp_all_df['Y_world'] = mp_world_df['Y_world']
    mp_all_df['Z_world'] = mp_world_df['Z_world']
    
    # take negative of Y values - when Y is negative the "stick figure" plots right side up and is more intuitive for gait calculations   
    #mp_all_df.loc[:,'Y_pose_negative'] = -mp_all_df['Y_pose']
    #mp_all_df.loc[:,'Y_world_negative'] = -(mp_all_df['Y_world'])
    #yolo_df.loc[:,'Y_negative'] = -(yolo_df['Y'])
    
    # if y = inf, y negative = inf; otherwise, y_negative = negative value of y at that row 
    mp_all_df['Y_pose_negative'] = mp_all_df['Y_pose'].apply(lambda y: y if y == np.inf else -y)
    mp_all_df['Y_world_negative'] = mp_all_df['Y_world'].apply(lambda y: y if y == np.inf else -y)
    yolo_df['Y_yolo_negative'] = yolo_df['Y_yolo'].apply(lambda y: y if y == np.inf else -y)

    return([mp_all_df, yolo_df])


# input = merged mp_pose and world df, one yolo df 
# output = merged mp_pose and world df, one yolo df; 
    # cleaned up columns with no markers tracked 
    # add column for yolo landmark visibility 
def clean_mp_yolo_missing_data(mp_all_df, yolo_df):
    #

    # mediapipe 
    # replace inf values in vis score with 0
    mp_all_df['vis'] = mp_all_df['vis'].replace(np.inf, 0) 
    
    # add column: any_markers_tracked? y/n 
        # if XYZ and vis = inf -> no
        # use for interpolation 
    mp_all_df['any_markers_visible'] = np.where((mp_all_df[['X_pose', 'Y_pose', 'Z_pose']] == np.inf).all(axis=1), 'no', 'yes')

    # replace nan values in label with 
    mp_all_df['label'] = mp_all_df['label'].fillna('no_labels_tracked')

    # replace an inf with nan 
    mp_all_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    
    # yolo
    # add landmark_visible column
        # if X + Y == 0 -> landmark_visible = 0 (missing) 
        # else -> landmark_visible = 1 (present) 
    yolo_df['landmark_visible'] = np.where((yolo_df[['X_yolo', 'Y_yolo']] == 0).all(axis=1), 'no', 'yes')

    # if both x and Y = zero, replace with NaN
    yolo_df[['X_yolo', 'Y_yolo', 'Y_yolo_negative']] = yolo_df[['X_yolo', 'Y_yolo', 'Y_yolo_negative']].mask((yolo_df['X_yolo'] == 0) & 
                                                                               (yolo_df['Y_yolo'] == 0) & 
                                                                               (yolo_df['Y_yolo_negative'] == 0), np.nan)
    
    
    return([mp_all_df, yolo_df])


# input = string of video identifier (either video_id_date_name in analysis code or vid_in_path from run), 
# merged mp_pose and world df, one yolo df 

# output = merged mp_pose and world, one yolo df; both dfs with camera orientation and turn direction columns  
def add_orientation_and_turn_direction(vid_in_path, mp_all_df, yolo_df):

    vid_in_path_basename = os.path.basename(vid_in_path)
    
    # add camera orientation from file name 
    if 'horizon' in vid_in_path_basename:
        mp_all_df['camera_orientation'] = 'horizontal'
        yolo_df['camera_orientation'] = 'horizontal'
    elif 'vert' in vid_in_path_basename: 
        mp_all_df['camera_orientation'] = 'vertical'
        yolo_df['camera_orientation'] = 'vertical'
    else: 
        mp_all_df['camera_orientation'] = 'UNK'
        yolo_df['camera_orientation'] = 'UNK'

    # add turn direction from file name 
    if 'left' in vid_in_path_basename:
        mp_all_df['turn_direction'] = 'left'
        yolo_df['turn_direction'] = 'left'
    elif 'right' in vid_in_path_basename: 
        mp_all_df['turn_direction'] = 'right'
        yolo_df['turn_direction'] = 'right'
    else: 
        mp_all_df['turn_direction'] = 'UNK'
        yolo_df['turn_direction'] = 'UNK'
    
    return([mp_all_df, yolo_df])


# In[ ]:
# inputs: 
    # string of video identifier (either video_id_date_name in analysis code or vid_in_path from run)
    # output_parent_folder = dir_out_prefix

# save outputs 
# vid_in_path - string used to save new .csv file 
def save_merge_mp_yolo_df(mp_all_df, yolo_df, vid_in_path, output_parent_folder):

    output_folder = os.path.join(output_parent_folder, '001_merge_mp_yolo_dfs')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]
    
    # save merged mp df as csv
    output_file_name_1 = vid_in_path_no_ext + '_mediapipe_all.csv'
    output_file_1 = os.path.normpath(os.path.join(output_folder, output_file_name_1))
    mp_all_df.to_csv(output_file_1)
    
    # save yolo df as csv  
    output_file_name_2 = vid_in_path_no_ext + '_yolo.csv'
    output_file_2 = os.path.normpath(os.path.join(output_folder, output_file_name_2))
    yolo_df.to_csv(output_file_2)





