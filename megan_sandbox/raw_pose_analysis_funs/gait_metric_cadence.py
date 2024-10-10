#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os 


# In[ ]:


# PKMAS - The number of footfalls minus one divided by the ambulation time, converted to minutes (steps/min). 
# Cadence is shown on the “Mean” row of the Gait Table

def calculate_cadence(stride_times_peaks, stride_times_valleys, fps, mp_all_df, vid_in_path, output_parent_folder): 
    total_steps = len(stride_times_peaks) + len(stride_times_valleys) # peaks + valleys
    video_length_sec = max(mp_all_df['frame']) / fps # video length 
    cadence = (total_steps/video_length_sec) * 60
    cadence_df = pd.DataFrame(data = {'cadence_step_per_min' : [cadence]})

    #save outputs 
    output_folder = os.path.join(output_parent_folder, '005_gait_metrics', 'cadence')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]
    
    # save stasts
    cadence_path_df = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_cadence.csv')))
    cadence_df.to_csv(cadence_path_df)

    return([total_steps, video_length_sec, cadence])


