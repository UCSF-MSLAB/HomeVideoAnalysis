#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pandas as pd
import os 


# In[13]:

# get frames 
# vid_in_path = path to video 
def get_frames_per_second(path_to_video): 
    video = cv2.VideoCapture(path_to_video) 
    fps = video.get(cv2.CAP_PROP_FPS)
    fps = round(fps)
    
    return(fps)


# add time column in seconds to each data frame 
def add_time_column(mp_all_df, yolo_df, fps):
    mp_all_df.loc[:,'time_seconds'] = mp_all_df['frame']/fps
    yolo_df.loc[:,'time_seconds'] = yolo_df['frame']/fps
    return([mp_all_df, yolo_df])



# inputs: 
    # string of video identifier (either video_id_date_name in analysis code or vid_in_path from run)
    # output_parent_folder = dir_out_prefix

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





