#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as sig
import os 

# In[2]:

def pivot_merge_yolo_df(mp_all_df, yolo_df, fps): 
    # pivot mediapipe 
    mp_marker_subset = mp_all_df[(mp_all_df['label'] == 'right_ankle') | (mp_all_df['label'] == 'left_ankle') | 
    (mp_all_df['label'] == 'right_heel') | (mp_all_df['label'] == 'left_heel')]


    mp_long = mp_marker_subset.pivot(index = 'frame', columns = 'label', values = ['vis', 'any_markers_visible'])
    mp_long.columns = [f"{col[1]}_{col[0]}" for col in mp_long.columns]

    # pivot yolo df 
    yolo_marker_subset = yolo_df[(yolo_df['label'] == 'right_hip') | (yolo_df['label'] == 'left_hip')]

    yolo_long = yolo_marker_subset.pivot(index = 'frame', columns = 'label', values = ['X_yolo', 'landmark_visible', 'time_seconds'])
    yolo_long.columns = [f"{col[1]}_{col[0]}" for col in yolo_long.columns]

    # smooth hip width 
    yolo_long['hip_x_width_yolo'] = abs(yolo_long['left_hip_X_yolo'] - yolo_long['right_hip_X_yolo'])

    # filter hip width 
    # Butterworth filter parameters
    order = 1  # Filter order
    cutoff_frequency = 1  # less than 0.5 * fps 

    # Create a Butterworth filter
    nyquist_frequency = 0.5 * fps  # Nyquist frequency
    normalized_cutoff = cutoff_frequency / nyquist_frequency # between 0 and 1
    
    b, a = sig.butter(order, normalized_cutoff, btype='low', analog=False)

    # Mask NaN values
    mask = yolo_long['hip_x_width_yolo'].isna()
    filtered_hip_width_data = yolo_long['hip_x_width_yolo'].copy()

    # Replace NaNs with zero temporarily for filtering
    data_filled = yolo_long['hip_x_width_yolo'].fillna(0)

    # Apply the filter
    filtered_values = sig.filtfilt(b, a, data_filled)

    # Insert the filtered values back, preserving original NaNs
    filtered_hip_width_data[~mask] = filtered_values[~mask]

    # replace in df
    yolo_long['hip_x_width_yolo_filt'] = filtered_hip_width_data

    # merge dfs together
    mp_yolo_df = pd.merge(mp_long, yolo_long, left_index=True, right_index=True)
    mp_yolo_df = mp_yolo_df.drop(columns = ['right_hip_time_seconds'])
    mp_yolo_df = mp_yolo_df.rename(columns = {'left_hip_time_seconds' : 'time_seconds'})
    return mp_yolo_df


# In[3]:
def find_valid_segments(df):
    # Step 1: Calculate differences and create a pattern column for identifying increases or decreases in the filtered hip width data 
    df['width_diff'] = df['hip_x_width_yolo_filt'].diff()

    # if filtered hip width is increasing, label as increasing or decreasing 
    df['pattern'] = df['width_diff'].apply(lambda x: 'increasing' if x > 0 else ('decreasing' if x < 0 else None))

    # Step 2: Identify continuous segments of increasing or decreasing patterns
    df['pattern_change'] = (df['pattern'] != df['pattern'].shift()).cumsum()

    # Frame diff - gaps with missing hip data --> likeley too close to camera 
    df['seconds_diff'] = df['time_seconds'].diff()
    
    # Step 3: Group by segment and filter based on criteria
    valid_segments = []
    for _, segment_data in df.groupby('pattern_change'):
        duration = segment_data['time_seconds'].iloc[-1] - segment_data['time_seconds'].iloc[0]
        nans_in_segment = segment_data['hip_x_width_yolo'].isna().sum()
        
        if (duration >= 2 and  # greater than 2 seconds 
            (segment_data['seconds_diff'] <= 0.25).all() and # no missing hip data for more than 1/4 of a second 
            (segment_data.iloc[:, 0:3] > 0.25).all().all() and # no vis scores less than 0.25
            segment_data.iloc[:, 0:3].values.mean() >= 0.75): # mean vis score >= 0.75

            # make current segment data frame and append to list 
            segment_data_df = pd.DataFrame(data = segment_data)
            valid_segments.append(segment_data_df)


    if len(valid_segments) > 0: 
        print('include: valid segments found') 
        valid_segments_found = 1
    else: 
        print('exclude: no valid segments found')
        valid_segments_found = 0
        
    return valid_segments, valid_segments_found


# In[ ]:

def plot_valid_walking_segments(mp_yolo_df, mp_all_df, valid_segments, vid_in_path, output_parent_folder):

     # plot #1 - hip width 
    fig1, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
    fig1.suptitle(os.path.splitext(os.path.basename(vid_in_path))[0])

    # suplot 1 - hip x width 
    ax1.scatter(mp_yolo_df['time_seconds'], mp_yolo_df['hip_x_width_yolo'], label = 'raw hip_width_yolo_x', color = 'black', alpha = 0.25, s =1)
    ax1.scatter(mp_yolo_df['time_seconds'], mp_yolo_df['hip_x_width_yolo_filt'], label = 'filtered hip_width_yolo_x', color = 'grey', alpha = 0.3, s = 1)

    # subplot 2 - landmark visibility 
    # change label to string for future filtering 
    mp_all_df['label'] = mp_all_df['label'].astype(str)

    # set labels to plot and filter data frame by label column
    labels_to_plot = ['left_heel', 'right_heel',
                  'left_ankle', 'right_ankle',
                   'left_hip', 'right_hip']

    mp_all_filt_df = mp_all_df[mp_all_df['label'].str.contains('|'.join(labels_to_plot), case=False)]
    
    # plot 
    ax2 = sns.lineplot(data=mp_all_filt_df, x='time_seconds', y='vis', hue='label', markers=True, dashes=False, estimator = None)
    ax2.set_ylim(0, 1.2)

    if len(valid_segments) > 0: # if first for loop found any valid segments 
    
        for i, current_segment_df in enumerate(valid_segments):
            current_start_sec = current_segment_df['time_seconds'].iloc[0]
            current_end_sec = current_segment_df['time_seconds'].iloc[-1]

            ax1.vlines(x = current_start_sec, 
                       ymin = mp_yolo_df['hip_x_width_yolo'].min(), 
                       ymax = mp_yolo_df['hip_x_width_yolo'].max(),
                       color = 'green', alpha = 0.25, linewidth = 2.5)
            ax1.vlines(x = current_end_sec, 
                       ymin = mp_yolo_df['hip_x_width_yolo'].min(), 
                       ymax = mp_yolo_df['hip_x_width_yolo'].max(),
                       color = 'black', alpha = 0.25, linewidth = 2.5)

            ax2.vlines(x = current_start_sec, 
                       ymin = 0, 
                       ymax = 1.2,
                       color = 'green', alpha = 0.25, linewidth = 2.5)
            ax2.vlines(x = current_end_sec, 
                       ymin = 0, 
                       ymax = 1.2,
                       color = 'black', alpha = 0.25, linewidth = 2.5)


    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))   
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # save plot
    output_folder = os.path.join(output_parent_folder, '003_b_select_linear_walking')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # create file name 
    input_file_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]
    output_file = os.path.normpath(os.path.join(output_folder, input_file_no_ext +'_walking_segment_selected.png'))

    # save figure 
    fig1.savefig(output_file, bbox_inches = 'tight')
    # plt.close(fig1)
    # plt.close()
    plt.show(fig1)


# In[ ]:


# run on all 
def select_plot_linear_walking(mp_all_df, yolo_df, fps, vid_in_path, output_parent_folder):
    mp_yolo_df = pivot_merge_yolo_df(mp_all_df, yolo_df, fps)
    valid_segments, valid_segments_found = find_valid_segments(mp_yolo_df)
    plot_valid_walking_segments(mp_yolo_df, mp_all_df, valid_segments, vid_in_path, output_parent_folder) 

    return valid_segments, valid_segments_found
    


