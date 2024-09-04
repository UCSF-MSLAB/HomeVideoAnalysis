#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 
import os 


# In[ ]:


# input 
# load mp, mp_world, and yolo .csv files for one video (from main branch, output of home video analysis run.py 8/29/2024)
#mp_pose_filepath = r'..\..\temp\main_branch_outputs\000_run\DS_HC_gait_vertical_left_mediapipe.csv'
#mp_world_filepath = r'..\..\temp\main_branch_outputs\000_run\DS_HC_gait_vertical_left_mediapipe_world.csv'
#yolo_filepath = r'..\..\temp\main_branch_outputs\000_run\DS_HC_gait_vertical_left_yolo.csv'

# path to video 
#vid_in_path = r'..\..\tests\fixtures\all_videos\DS_HC_practice videos\DS_HC_gait_vertical_left.mov'

# output folder 
#output_parent_folder = r'..\..\temp\test_sandbox_pipeline_outputs'

# read csv
#mp_pose_df = pd.read_csv(mp_pose_filepath)
#mp_world_df = pd.read_csv(mp_world_filepath)
#yolo_df = pd.read_csv(yolo_filepath)
    


# In[ ]:


# input = mp_pose, mp_world, and yolo df for one video  
# output = one mediapipe df and one yolo df, both with positive and negative Y values  
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

    yolo_df = yolo_df.rename(columns = {"Unnamed: 0" : "label_num"})
    

    # merge mp world and pose (same vis, markers)
    mp_all_df = mp_pose_df
    mp_all_df['X_world'] = mp_world_df['X_world']
    mp_all_df['Y_world'] = mp_world_df['Y_world']
    mp_all_df['Z_world'] = mp_world_df['Z_world']
    
    # take negative of Y values - when Y is negative the "stick figure" plots right side up and is more intuitive for gait calculations   
    mp_all_df.loc[:,'Y_pose_negative'] = -mp_all_df['Y_pose']
    mp_all_df.loc[:,'Y_world_negative'] = -(mp_all_df['Y_world'])
    yolo_df.loc[:,'Y_negative'] = -(yolo_df['Y'])

    return([mp_all_df, yolo_df])


# In[ ]:


## add video direction and turn direction to data frame 
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


# save outputs 
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


# In[ ]:


#[mp_all_df, yolo_df] = merge_mp_pose_world(mp_pose_df, mp_world_df, yolo_df)
#[mp_all_df, yolo_df] = add_orientation_and_turn_direction(vid_in_path, mp_all_df, yolo_df)
#save_merge_mp_yolo_df(mp_all_df, yolo_df, vid_in_path, output_parent_folder)


# In[ ]:


## convert to .py file so functions can be used in other scripts 
get_ipython().system('jupyter nbconvert --to script merge_mp_yolo_dfs.ipynb')

