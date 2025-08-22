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
    pix_h_start = start_row['smooth_approx_height_Y_pix'].iloc[0]
    pix_h_end = end_row['smooth_approx_height_Y_pix'].iloc[0]

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

    # diff between l hip and l ankle 
    l_ank_to_hip_df = l_ankle_yolo_df.copy()
    l_ank_to_hip_df.loc[:, 'approx_height_Y_pix'] = abs(l_hip_yolo_df['Y_yolo'] - l_ankle_yolo_df['Y_yolo'])
    l_ank_to_hip_df.loc[:, 'smooth_approx_height_Y_pix'] = l_ank_to_hip_df['approx_height_Y_pix'].rolling(window=25, min_periods = 1).mean()
            
    # drop X columns 
    l_ank_to_hip_df = l_ank_to_hip_df.drop(columns = ['X_yolo'])
    l_ank_to_hip_df.reset_index(inplace=True)

    # add "time_group" label column, will use to group 
    l_ank_to_hip_df['time_group'] = l_ank_to_hip_df['frame'] / (fps)
    l_ank_to_hip_df['time_group'] = l_ank_to_hip_df['time_group'].apply(math.floor)
    
    # identify peaks and valleys in pixel Y position (approximate turn locations) 
    peaks, _ = sig.find_peaks(l_ank_to_hip_df['smooth_approx_height_Y_pix'], distance = fps)
    valleys, _ = sig.find_peaks(-l_ank_to_hip_df['smooth_approx_height_Y_pix'], distance = fps)
    peak_frames = l_ank_to_hip_df.iloc[peaks]['frame']
    valley_frames = l_ank_to_hip_df.iloc[valleys]['frame']
    peaks_valleys = np.concatenate((peak_frames, valley_frames))

    # plot pixel Y position with time groups in grey lines and peaks valleys as scatter plots 
    fig1, ax1 = plt.subplots(figsize=(5.75, 3))
#    sns.scatterplot(x = 'frame', y = 'approx_height_Y_pix', data = l_ank_to_hip_df, label = 'Y Dist L Hip to L Ank')
    plt.plot(l_ank_to_hip_df['frame'], l_ank_to_hip_df['smooth_approx_height_Y_pix'], color = 'black')
#    plt.scatter(peak_frames, l_ank_to_hip_df.iloc[peaks]['smooth_approx_height_Y_pix'], color='red', label='Local Maxima')
#    plt.scatter(valley_frames, l_ank_to_hip_df.iloc[valleys]['smooth_approx_height_Y_pix'], color='red', label='Local Minima')
    for time_group in l_ank_to_hip_df['time_group'].unique():
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

    for current_sec_group in l_ank_to_hip_df['time_group'].unique():
        current_l_ank_to_hip_df = l_ank_to_hip_df.loc[l_ank_to_hip_df['time_group'] == current_sec_group]

        current_start_frame = current_l_ank_to_hip_df['frame'].iloc[0]
        current_end_frame = current_l_ank_to_hip_df['frame'].iloc[-1]

        # if this current time_group group contains peak or valley, skip because likeley a turn 
        if current_l_ank_to_hip_df.loc[current_start_frame: current_end_frame, 'frame'].isin(peaks_valleys).any():
#            print('Skipped - contains peak or valley') 
            delta_pix_h_rel = np.nan
        else: 
            delta_pix_h_rel  = calc_pix_size_change(current_start_frame, current_end_frame, l_ank_to_hip_df, fps)
                
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



