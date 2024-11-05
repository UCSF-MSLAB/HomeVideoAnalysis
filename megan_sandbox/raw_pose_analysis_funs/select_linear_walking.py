#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# In[2]:

def pivot_merge_yolo_df(mp_all_df, yolo_df): 
    # pivot mediapipe 
    mp_marker_subset = mp_all_df[(mp_all_df['label'] == 'right_ankle') | (mp_all_df['label'] == 'left_ankle') | 
    (mp_all_df['label'] == 'right_heel') | (mp_all_df['label'] == 'left_heel')]


    mp_long = mp_marker_subset.pivot(index = 'frame', columns = 'label', values = ['vis', 'any_markers_visible'])
    mp_long.columns = [f"{col[1]}_{col[0]}" for col in mp_long.columns]

    # pivot yolo df 
    yolo_marker_subset = yolo_df[(yolo_df['label'] == 'right_hip') | (yolo_df['label'] == 'left_hip')]

    yolo_long = yolo_marker_subset.pivot(index = 'frame', columns = 'label', values = ['X_yolo', 'landmark_visible', 'time_seconds'])
    yolo_long.columns = [f"{col[1]}_{col[0]}" for col in yolo_long.columns]
    yolo_long['hip_x_width_yolo'] = abs(yolo_long['left_hip_X_yolo'] - yolo_long['right_hip_X_yolo'])

    # merge dfs together
    mp_yolo_df = pd.merge(mp_long, yolo_long, left_index=True, right_index=True)
    mp_yolo_df = mp_yolo_df.drop(columns = ['right_hip_time_seconds'])
    mp_yolo_df = mp_yolo_df.rename(columns = {'left_hip_time_seconds' : 'time_seconds'})
    return mp_yolo_df


# In[3]:
def find_valid_segments(df):
    # Step 1: Calculate differences and create a pattern column for identifying increases or decreases
    df['width_diff'] = df['hip_x_width_yolo'].diff(periods = 5)
    df['pattern'] = df['width_diff'].apply(lambda x: 'increasing' if x > 0 else ('decreasing' if x < 0 else None))

    # Step 2: Identify continuous segments of increasing or decreasing patterns
    df['pattern_change'] = (df['pattern'] != df['pattern'].shift()).cumsum()
    
    # Step 3: Group by segment and filter based on criteria
    valid_segments = []
    for _, segment_data in df.groupby('pattern_change'):
        duration = segment_data['time_seconds'].iloc[-1] - segment_data['time_seconds'].iloc[0]
        nans_in_segment = segment_data['hip_x_width_yolo'].isna().sum()
        
        if (duration >= 1.5 and  # Pattern lasts at least 2 seconds
            (segment_data['pattern'].iloc[0] == 'decreasing') and # person is walking away from camera 
            (segment_data.iloc[:, 0:3] > 0.25).all().all()):  # All vis values in in columns 1, 2, 3 > 0.25

            # make current segment data frame and append to list 
            segment_data_df = pd.DataFrame(data = segment_data)
            valid_segments.append(segment_data_df)

    return valid_segments


# In[4]:
def pick_best_vis_segment(all_valid_segments, mp_all_df, yolo_df): 
    # pick walk away with best visibility score 
    if len(all_valid_segments) > 0: # if valid segements exist
        print('include: valid segments exist') 

        mean_vis_scores = []
        for i, current_segment_df in enumerate(all_valid_segments):
            current_mean_vis_score = current_segment_df.iloc[:, 0:3].values.mean()
            mean_vis_scores.append(current_mean_vis_score)
            
        # find index of max visibility score 
        max_vis_i = mean_vis_scores.index(max(mean_vis_scores))
        # if mean vis score is > 0.75 --> use in analysis 
        if mean_vis_scores[max_vis_i] >= 0.75: 
            print('include: greater than 0.75') 
            segment_to_analyze = all_valid_segments[max_vis_i]
            start_sec = segment_to_analyze['time_seconds'].iloc[0]
            end_sec = segment_to_analyze['time_seconds'].iloc[-1]
            # select yolo and mediapipe df between end and start seconds 
            walk_segment_mp_all_df = mp_all_df[(mp_all_df['time_seconds'] >= start_sec) & (mp_all_df['time_seconds'] <= end_sec)]
            walk_segment_yolo_df = yolo_df[(yolo_df['time_seconds'] >= start_sec) & (yolo_df['time_seconds'] <= end_sec)]
            valid_segment_found = 1
        else: 
            print('no walking segment with mean vis greater than 0.75: exclude from analysis')
            valid_segment_found = 0
            start_sec = []
            end_sec = []
            walk_segment_mp_all_df = []
            walk_segment_yolo_df = []

    else: 
        print('no walking segments found: exclude from analysis')
        valid_segment_found = 0 # no valid segments exist 
        start_sec = []
        end_sec = []
        walk_segment_mp_all_df = []
        walk_segment_yolo_df = []
        
    return valid_segment_found, start_sec, end_sec, walk_segment_mp_all_df, walk_segment_yolo_df


# In[ ]:

def plot_valid_walking_segments(mp_yolo_df, mp_all_df, all_valid_segments, valid_segment_found, start_sec, end_sec, vid_in_path, output_parent_folder):

     # plot #1 - hip width 
    fig1, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
    fig1.suptitle(os.path.splitext(os.path.basename(vid_in_path))[0])

    # suplot 1 - hip x width 
    ax1.scatter(mp_yolo_df['time_seconds'], mp_yolo_df['hip_x_width_yolo'], label = 'hip_width_yolo_x', color = 'black', s =1)


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

    if len(all_valid_segments) > 0: # if first for loop found any valid segments 
    
        for i, current_segment_df in enumerate(all_valid_segments):
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
                       ymax = 1,
                       color = 'green', alpha = 0.25, linewidth = 2.5)
            ax2.vlines(x = current_end_sec, 
                       ymin = 0, 
                       ymax = 1,
                       color = 'black', alpha = 0.25, linewidth = 2.5)

    # if one segment was selected, label with dotted lines --> this segment will be analyzed 
    if valid_segment_found == 1:

        ax1.vlines(x = start_sec, 
                    ymin = mp_yolo_df['hip_x_width_yolo'].min(), 
                    ymax = mp_yolo_df['hip_x_width_yolo'].max(),
                    color = 'green', alpha = 0.7, linewidth = 2.5, linestyle = 'dashed', 
                    label = 'start_analysis')
        ax1.vlines(x = end_sec, 
                    ymin = mp_yolo_df['hip_x_width_yolo'].min(), 
                    ymax = mp_yolo_df['hip_x_width_yolo'].max(),
                    color = 'black', alpha = 0.7, linewidth = 2.5, linestyle = 'dashed', 
                    label = 'end_analysis')

        ax2.vlines(x = start_sec, 
                    ymin = 0, 
                    ymax = 1,
                    color = 'green', alpha = 0.7, linewidth = 2.5, linestyle = 'dashed', 
                    label = 'start_analysis')
        ax2.vlines(x = end_sec, 
                    ymin = 0, 
                    ymax = 1,
                    color = 'black', alpha = 0.7, linewidth = 2.5, linestyle = 'dashed', 
                    label = 'end_analysis')
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
    plt.close(fig1)
    plt.close()


# In[ ]:


# run on all 
def select_plot_linear_walking(mp_all_df, yolo_df, vid_in_path, output_parent_folder):
    mp_yolo_df = pivot_merge_yolo_df(mp_all_df, yolo_df)
    all_valid_segments = find_valid_segments(mp_yolo_df)
    valid_segment_found, start_sec, end_sec, walk_segment_mp_all_df, walk_segment_yolo_df = pick_best_vis_segment(all_valid_segments, mp_all_df, yolo_df)
    plot_valid_walking_segments(mp_yolo_df, mp_all_df, all_valid_segments, valid_segment_found, start_sec, end_sec, vid_in_path, output_parent_folder) 

    return valid_segment_found, start_sec, end_sec, walk_segment_mp_all_df, walk_segment_yolo_df
    


