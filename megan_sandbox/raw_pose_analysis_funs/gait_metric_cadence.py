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

def calculate_cadence(stride_times_peaks, stride_times_valleys, start_sec, end_sec, vid_in_path, output_parent_folder): 
    total_steps = len(stride_times_peaks) + len(stride_times_valleys) # peaks + valleys
    video_length_sec = end_sec - start_sec # video length 
    cadence = (total_steps/video_length_sec) * 60
    cadence_df = pd.DataFrame(index = range(1),
                              data = {'cadence_step_per_min' : [cadence]})

    return total_steps, video_length_sec, cadence_df


