#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pandas as pd
import os 


# In[2]:


# path to video 
#vid_in_path = r'..\tests\fixtures\all_videos\DS_HC_practice videos\DS_HC_gait_vertical_left.mov' # vid_in_path set during process_dir() of run.py


# In[3]:


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


# In[6]:


# save csv 
def save_df_w_time(mp_all_df, yolo_df, mp_all_filepath, yolo_filepath, output_parent_folder): 

    output_folder = os.path.join(output_parent_folder, '002_frames_to_time')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # save new mp df with time as .csv 
    output_file_1 = os.path.normpath(os.path.join(output_folder, os.path.basename(mp_all_filepath)))
    mp_all_df.to_csv(output_file_1)

    # save new yolo df with time as .csv 
    output_file_2 = os.path.normpath(os.path.join(output_folder, os.path.basename(yolo_filepath)))
    yolo_df.to_csv(output_file_2)


# In[7]:


#fps = get_frames_per_second(vid_in_path):
#[mp_all_df, yolo_df] = add_time_column(mp_all_df, yolo_df, fps)
#save_df_w_time(mp_all_df, yolo_df, mp_all_filepath, yolo_filepath, output_parent_folder)


# In[ ]:


## convert to .py file so functions can be used in other scripts 
get_ipython().system('jupyter nbconvert --to script frames_to_time.ipynb')

