#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np 
import os 
import seaborn as sns 
import matplotlib.pyplot as plt
import math
import scipy.signal as sig


## calculate change in y between hip and ankle as a proxy for velocity in 2D vertical videos. 
# If person is walking faster, hip will move up and down the frame faster 
# make change in pixel relative to start of each second increment 

def calc_pix_size_change(start_frame, end_frame, df, fps): 
    start_row = df.loc[df['frame'] == start_frame]
    end_row = df.loc[df['frame'] == end_frame]

    # height in pixel at start and end of time group 
    pix_h_start = start_row['mean_diff'].iloc[0]
    pix_h_end = end_row['mean_diff'].iloc[0]

    # for consistency, only calculate when pixel Y is increasing 
    if (pix_h_end > pix_h_start):
        # Change in pixel height from start to end 
        delta_pix_h = pix_h_end - pix_h_start
        # change in pixel height relative to start
        delta_pix_h_rel = delta_pix_h / pix_h_start
    else: 
        delta_pix_h_rel = np.nan

    return delta_pix_h_rel


# -------------------------------------------------------------------------
def calc_velocity_proxy(yolo_df, video_id_date_name, output_folder, fps): 
    
    # left hip 
    l_hip_yolo_df = yolo_df.loc[yolo_df['label'] == 'left_hip'] 
    l_hip_yolo_df.set_index('frame', inplace=True)

    # left ankle
    l_ankle_yolo_df = yolo_df.loc[yolo_df['label'] == 'left_ankle']
    l_ankle_yolo_df.set_index('frame', inplace=True)

    # right hip
    r_hip_yolo_df = yolo_df.loc[yolo_df['label'] == 'right_hip']
    r_hip_yolo_df.set_index('frame', inplace = True) 

    # right ankle 
    r_ankle_yolo_df = yolo_df.loc[yolo_df['label'] == 'right_ankle']
    r_ankle_yolo_df.set_index('frame', inplace = True) 

    # make one df with both hips and heels 
    yolo_hip_combined = l_hip_yolo_df[['Y_yolo']].join(r_hip_yolo_df[['Y_yolo']], how = 'inner', 
                                               lsuffix = '_left',
                                               rsuffix = '_right') 

    
    yolo_ankle_combined = l_ankle_yolo_df[['Y_yolo']].join(r_ankle_yolo_df[['Y_yolo']], how = 'inner', 
                                               lsuffix = '_left',
                                               rsuffix = '_right')

    yolo_combined = yolo_hip_combined.join(yolo_ankle_combined, how = 'inner', 
                                           lsuffix = '_hip', 
                                           rsuffix = '_ankle') 

    # smooth vertical position data 
    yolo_combined['Y_yolo_left_hip_smooth'] = yolo_combined['Y_yolo_left_hip'].rolling(window=10).mean()
    yolo_combined['Y_yolo_left_ankle_smooth'] = yolo_combined['Y_yolo_left_ankle'].rolling(window=10).mean()
    yolo_combined['Y_yolo_right_hip_smooth'] = yolo_combined['Y_yolo_right_hip'].rolling(window=10).mean()
    yolo_combined['Y_yolo_right_ankle_smooth'] = yolo_combined['Y_yolo_right_ankle'].rolling(window=10).mean()

    # take difference between each hip and heel 
    yolo_combined['right_hip_ankle_diff'] = abs(yolo_combined['Y_yolo_right_hip_smooth'] - yolo_combined['Y_yolo_right_ankle_smooth']) 
    yolo_combined['left_hip_ankle_diff'] = abs(yolo_combined['Y_yolo_left_hip_smooth'] - yolo_combined['Y_yolo_left_ankle_smooth'])

    # mean of right hip and heel difference 
    yolo_combined['mean_diff'] = yolo_combined[['right_hip_ankle_diff', 'left_hip_ankle_diff']].mean(axis=1)
    yolo_combined['mean_diff'] = yolo_combined['mean_diff'].rolling(window=25).mean()
    
    # diff between l hip and l ankle 
#    l_ank_to_hip_df = l_ankle_yolo_df.copy()
#    l_ank_to_hip_df.loc[:, 'approx_height_Y_pix'] = abs(l_hip_yolo_df['Y_yolo'] - l_ankle_yolo_df['Y_yolo'])
#    l_ank_to_hip_df.loc[:, 'smooth_approx_height_Y_pix'] = l_ank_to_hip_df['approx_height_Y_pix'].rolling(window=25, min_periods = 1).mean()
            
    # drop X columns 
#    l_ank_to_hip_df = l_ank_to_hip_df.drop(columns = ['X_yolo'])
#    l_ank_to_hip_df.reset_index(inplace=True)

    # add "time_group" label column, will use to group 
#    l_ank_to_hip_df['time_group'] = l_ank_to_hip_df['frame'] / (fps)
#    l_ank_to_hip_df['time_group'] = l_ank_to_hip_df['time_group'].apply(math.floor)

    yolo_combined.reset_index(inplace = True) 
    yolo_combined['time_group'] = yolo_combined['frame'] / (fps)
    yolo_combined['time_group'] = yolo_combined['time_group'].apply(math.floor) 
    
    # identify peaks and valleys in pixel Y position (approximate turn locations) 
    peaks, _ = sig.find_peaks(yolo_combined['mean_diff'], distance = fps)
    valleys, _ = sig.find_peaks(-yolo_combined['mean_diff'], distance = fps)
    peak_frames = yolo_combined.iloc[peaks]['frame']
    valley_frames = yolo_combined.iloc[valleys]['frame']
    peaks_valleys = np.concatenate((peak_frames, valley_frames))

    # plot pixel Y position with time groups in grey lines and peaks valleys as scatter plots 
    fig1, ax1 = plt.subplots(figsize=(5.75, 3))
#    sns.scatterplot(x = 'frame', y = 'approx_height_Y_pix', data = yolo_combined, label = 'Y Dist L Hip to L Ank')
    plt.plot(yolo_combined['frame'], yolo_combined['mean_diff'], color = 'black')
    plt.scatter(peak_frames, yolo_combined.iloc[peaks]['mean_diff'], color='red', label='Local Maxima')
    plt.scatter(valley_frames, yolo_combined.iloc[valleys]['mean_diff'], color='red', label='Local Minima')
    for time_group in yolo_combined['time_group'].unique():
        plt.axvline(x = time_group * (fps), color = 'grey', alpha = 0.5)

    ax1.set_ylabel('Pixels', fontsize = 11)
    ax1.set_xlabel('Time (Frames)', fontsize = 11)

    ax1.set_title('Vertical Distance Between Hip to Ankle')

    # axis tick labels 
    ax1.tick_params(labelsize=10) 

    # for paper figure - set x limits 
#    ax1.set_xlim([200, 400])
    
    # name and save plot 
#    plt.title(video_id_date_name)
    # save figure 
    fig1.tight_layout()
    
#    plt.show()
    fig_path = os.path.join(output_folder, video_id_date_name + '_pix_change.png') 
    fig1.savefig(fig_path) 
    plt.close()

    # calculate depth proxy values and summarize 
    depth_proxies_all = []

    for current_sec_group in yolo_combined['time_group'].unique():
        current_yolo_combined = yolo_combined.loc[yolo_combined['time_group'] == current_sec_group]

        current_start_frame = current_yolo_combined['frame'].iloc[0]
        current_end_frame = current_yolo_combined['frame'].iloc[-1]

        # if this current time_group group contains peak or valley, skip because likeley a turn 
        if current_yolo_combined.loc[current_start_frame: current_end_frame, 'frame'].isin(peaks_valleys).any():
#            print('Skipped - contains peak or valley') 
            delta_pix_h_rel = np.nan 
        # if any missing data in segment - skip 
        elif current_yolo_combined['mean_diff'].isna().sum() > 0:
            delta_pix_h_rel = np.nan
        else: 
            delta_pix_h_rel  = calc_pix_size_change(current_start_frame, current_end_frame, yolo_combined, fps)
                
        # combine all time groups into one array 
        depth_proxies_all.append({'start_frame' : current_start_frame,
                                  'delta_pix_h_rel' : delta_pix_h_rel})
                      
    # convert array to df 
    depth_proxies_all_df = pd.DataFrame(depth_proxies_all)
    depth_proxies_all_df = depth_proxies_all_df.replace([np.inf, -np.inf], np.nan) # replace inf with nan
    depth_proxies_all_df.to_csv(os.path.join(output_folder, video_id_date_name + '_pix_change.csv')) 
    
    # calculate median relative change in pixels 
    delta_pix_h_rel_median = round(depth_proxies_all_df['delta_pix_h_rel'].median(), 2)

    return delta_pix_h_rel_median



