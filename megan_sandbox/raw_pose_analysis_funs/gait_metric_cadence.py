#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os 

#  The number of footfalls minus one divided by the ambulation time, converted to minutes (steps/min). 

def calculate_cadence(stride_times_peaks, stride_times_valleys, fps, vid_in_path, output_parent_folder, too_few_strides): 
    # if too_few_strides from stride time calc - don't calculate cadence for this segment 
    if too_few_strides == True:
        cadence_df = pd.DataFrame(index = range(1),
                                  data = {'cadence_step_per_min' : [np.nan]})
    # if sufficient steps - calculate cadence 
    else:
        # total steps = number of "footfalls" - 1. Need two consecutive footfalls to equal 1 step 
        total_steps = (len(stride_times_peaks) + len(stride_times_valleys)) - 1 # peaks + valleys - 1 

        # calculate cadence over time from first to last "footfall" (ie peak or valley) 
        # last "footfall" 
        peaks_frame_max = stride_times_peaks.index.max()
        valleys_frame_max = stride_times_valleys.index.max() 
        last_frame = max(peaks_frame_max, valleys_frame_max) # valley or peak last 
        last_sec = last_frame / fps # conver from frames to second 
        
        # first "footfall" 
        peaks_frame_min = stride_times_peaks.index.min()
        valleys_frame_min = stride_times_valleys.index.min() 
        first_frame = min(peaks_frame_min, valleys_frame_min) # valley or peak first 
        first_sec = first_frame / fps # conver from frames to second 

        # time duration 
        video_length_sec = last_sec - first_sec # time from first to last "footfall" 
        cadence = (total_steps/video_length_sec) * 60
        cadence_df = pd.DataFrame(index = range(1),
                                  data = {'cadence_step_per_min' : [cadence]})

    return cadence_df


