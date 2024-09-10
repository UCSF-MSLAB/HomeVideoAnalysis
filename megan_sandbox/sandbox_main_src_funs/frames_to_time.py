#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pandas as pd
import os 


# In[13]:


# path to video 
#vid_in_path = r'..\tests\fixtures\all_videos\DS_HC_practice videos\DS_HC_gait_vertical_left.mov' # vid_in_path set during process_dir() of run.py


# In[14]:


# get frames 
def get_frames_per_second(vid_in_path): 
    video = cv2.VideoCapture(vid_in_path) 
    fps = video.get(cv2.CAP_PROP_FPS)
    fps = round(fps)
    
    return(fps)


# In[4]:


# load data frames from previous step (after merging two mp dfs and adding negative Y column)  
#mp_all_filepath = r'..\temp\test_sandbox_pipeline_outputs\001_merge_mp_yolo_dfs\DS_HC_gait_vertical_left_mediapipe_all.csv'
#yolo_filepath = r'..\temp\test_sandbox_pipeline_outputs\001_merge_mp_yolo_dfs\DS_HC_gait_vertical_left_yolo.csv'

#mp_all_df = pd.read_csv(mp_all_filepath, index_col = 0)
#yolo_df = pd.read_csv(yolo_filepath, index_col = 0)

# output folder
#output_parent_folder = r'..\temp\test_sandbox_pipeline_outputs'


# In[5]:


# add time column in seconds to each data frame 
def add_time_column(mp_all_df, yolo_df, fps):
    mp_all_df.loc[:,'time_seconds'] = mp_all_df['frame']/fps
    yolo_df.loc[:,'time_seconds'] = yolo_df['frame']/fps
    return([mp_all_df, yolo_df])


# In[11]:


# save csv 
def save_df_w_time(mp_all_df, yolo_df, vid_in_path, output_parent_folder): 

    output_folder = os.path.join(output_parent_folder, '002_frames_to_time')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # input file name 
    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]

    # save new mp df with time as .csv 
    output_file_1 = os.path.normpath(os.path.join(output_folder, vid_in_path_no_ext + '_mediapipe_all_sec.csv'))
    mp_all_df.to_csv(output_file_1)

    # save new yolo df with time as .csv 
    output_file_2 = os.path.normpath(os.path.join(output_folder, vid_in_path_no_ext + '_yolo_sec.csv'))
    yolo_df.to_csv(output_file_2)


# In[15]:


#fps = get_frames_per_second(vid_in_path)
#[mp_all_df, yolo_df] = add_time_column(mp_all_df, yolo_df, fps)
#save_df_w_time(mp_all_df, yolo_df, vid_in_path, output_parent_folder)


# In[12]:




