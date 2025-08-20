#!/usr/bin/env python
# coding: utf-8

# Version 4 trying either mean of foot and ankle or mean of both?


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import scipy.signal as sig

################################################################
# functions to ID toe offs 
def find_gradient_max(time_series_series): 
    # find inflections points 
    original_data = time_series_series
    # remove NaN Values from original data to filter 
    original_data_no_na = original_data.dropna()
    
    # take second gradient 
    gradient = np.gradient(original_data_no_na)
    gradient_df = pd.Series(gradient)
    gradient_df.index = original_data_no_na.index

    # add missing data back into gradient  
    gradient_data_w_nan = original_data
    gradient_data_w_nan = gradient_data_w_nan.copy()
    non_nan_indices = original_data[original_data.notna()].index
    gradient_data_w_nan.loc[non_nan_indices] = gradient_df
    gradient_data_w_nan.name = 'gradient'

    # get peak gradient values in last 80% of frames  
    total_frames = gradient_data_w_nan.index[-1] - gradient_data_w_nan.index[0]
    first_frames = gradient_data_w_nan.index[0] + (total_frames * 0.20)
    last_frames = gradient_data_w_nan.index[-1] - (total_frames * 0.20)

    gradient_data_w_nan_last = gradient_data_w_nan.loc[(gradient_data_w_nan.index > first_frames) & (gradient_data_w_nan.index < last_frames)] 
    
    gradient_max_i = gradient_data_w_nan_last.idxmax()
    gradient_max = gradient_data_w_nan_last.max() 

    return(gradient_data_w_nan, gradient_max_i, gradient_max)

# inflection points between each peak and prior right base 
def save_RBase_peak_inflection(peaks_df, peaks_RBase_df, Y_position_df, fps):     
    # reset index for loop 
    peaks_df = peaks_df.reset_index(drop = True)
    peaks_RBase_df = peaks_RBase_df.reset_index(drop = True)

    # for each prominence, find inflection between left base and prominence
    event_frame_all = []
    event_frame_all_df = []
    for peak_i, peak_row in peaks_df.iterrows():
        
        peak_frame = peak_row['frame']
        # print(f"peak_frame: {peak_frame}") 
        
      #  print(f"peak_i: {peak_i}") 
        # half second before peak or to beginning of frame  
        if peak_i == 0: 
            if peak_frame > Y_position_df.index[0] + (fps * 0.50): 
                start_frame = peak_frame - (fps * 0.50) 
            else: 
                start_frame = Y_position_df.index[0]
            current_segment_df = Y_position_df.loc[(Y_position_df.index < peak_frame) & (Y_position_df.index > start_frame)]
        else:  
            # Previous right base 
            RBase_frame = peaks_RBase_df['frame'][peak_i-1]

            # current segment between prominence and previous right base 
            current_segment_df = Y_position_df.loc[(Y_position_df.index < peak_frame) & (Y_position_df.index > RBase_frame)] 

        # if less than X frames, skip this segment and move to next 
        if len(current_segment_df) < 3: 
            continue 
            
        # find inflection points  
     #   print('inflection points --------------') 
        grad_data, max_gradient_i, max_gradient = find_gradient_max(current_segment_df['Y_pose_negative_smooth']) 
        # plot 
        # plot current segment 
        fig_inflec, [ax1, ax2] = plt.subplots(nrows=2,  figsize=(2, 3)) 
        plt.suptitle('Toe Off') 
        ax1.plot(current_segment_df['Y_pose_negative_smooth'])
        
        ax2.plot(grad_data) 
        ax2.plot(max_gradient_i, max_gradient, 'o', color = 'red') 
        ax2.set_ylim([-.001, max(grad_data)+.001])

        plt.show() 
        plt.close() 
      
        # save all inflection frames 
        current_row_data = {'event_frame': max_gradient_i, 'peak_i': peak_i}
        event_frame_all.append(current_row_data) 
        
    # create dataframe 
    event_frame_all_df = pd.DataFrame(event_frame_all)
    
    return(event_frame_all_df)
    
########################################################################
# functions to find heel strikes
def count_consecutive_below(df, flag_col, event_indices):
    max_count = 0
    best_index = None

    df = df.reset_index(drop = False) 
    for idx in event_indices:
        count = 0
        df_backwards = df.loc[df['frame'] < idx]

        # move backwards from event position 
        for val in reversed(df_backwards[flag_col]):
            if val == 'below':
                count += 1
            else:
                break  # stop when it's not 'below'
    
        if count > max_count:
            max_count = count
            best_index = idx

    return best_index, max_count

def find_down_inflection_points(time_series_series, threshold): 
    # find inflections points 
    original_data = time_series_series
    # remove NaN Values from original data to filter 
    original_data_no_na = original_data.dropna()
    
    # take second gradient 
    gradient = np.gradient(original_data_no_na)
    gradient_df = pd.Series(gradient)
    gradient_df.index = original_data_no_na.index

    # add missing data back into gradient  
    gradient_data_w_nan = original_data
    gradient_data_w_nan = gradient_data_w_nan.copy()
    non_nan_indices = original_data[original_data.notna()].index
    gradient_data_w_nan.loc[non_nan_indices] = gradient_df
    gradient_data_w_nan.name = 'gradient'

    # upward crossings where velocity crosses below velocity threshold 
    inflection_point_indices = (gradient_data_w_nan.shift(1) < threshold) & (gradient_data_w_nan >= threshold)

    # filter series to only include values at inflection points 
    inflection_points_data = gradient_data_w_nan.loc[inflection_point_indices]

    # make df and add column - is velocity/gradient above or below threshold 
    gradient_data_w_nan = gradient_data_w_nan.to_frame()
    gradient_data_w_nan['above_below'] = 'above' 
    gradient_data_w_nan.loc[gradient_data_w_nan['gradient'] <= threshold, 'above_below'] = 'below'

    # get frames of inflection points 
    inflection_frames = inflection_points_data.index
    print(f"inflection_frames - 0.5: {inflection_frames - 0.5}") 
    
    return(gradient_data_w_nan, inflection_points_data, inflection_frames)


# find inflections points between peaks and next RBase 
def save_peak_Rbase_inflection(peaks_df, peaks_RBase_df, Y_position_df, threshold):     
    # reset index for loop 
    peaks_df = peaks_df.reset_index(drop = True)
    peaks_RBase_df = peaks_RBase_df.reset_index(drop = True)

    # for each prominence, find inflection between left base and prominence
    event_frames = []
    event_frame_all = []
    event_frame_all_df = []
    for peak_i, peak_row in peaks_df.iterrows():
        #print(f"peak_i: {peak_i}") 
        peak_frame = peak_row['frame'] 
        # get right base for that peak 
        RBase_frame = peaks_RBase_df['frame'][peak_i]

        # current segment between prominence and previous right base 
        current_segment_df = Y_position_df.loc[(Y_position_df.index < RBase_frame) & (Y_position_df.index > peak_frame)] 
        
        # if less than X frames, skip this segment and move to next 
        if len(current_segment_df) < 3: 
            continue 
            
        # find inflection points  
        #   print('inflection points --------------') 
        grad_2_data, inflec_data, event_frames = find_down_inflection_points(current_segment_df['Y_pose_negative_smooth'],
                                                                              threshold) 
        # Plot   
        # plot current segment 
        fig_inflec, [ax1, ax2] = plt.subplots(nrows=2,  figsize=(2, 3)) 
        plt.suptitle('Heel Strike') 
        ax1.plot(current_segment_df['Y_pose_negative_smooth'])
        
        ax2.plot(grad_2_data['gradient']) 
        ax2.plot(event_frames, inflec_data, 'o', color = 'black') 
        ax2.axhline(y=threshold, color='grey', linestyle='--', linewidth=0.8)
        ax2.set_ylim([min(grad_2_data['gradient'])-.001, 0.005])

        plt.show() 
        plt.close() 

        # total num frames between R base and next peak 
        total_frames = grad_2_data.index[-1] - grad_2_data.index[0]
        first_frames = grad_2_data.index[0] + (total_frames * 0.20)
        
        # if all gradient/velocity values are greater than threshold, use last frame 
            # use last frame = position of next right base 
        if len(inflec_data) == 0: 
            event_frame = grad_2_data.index[-1]

        # if there one inflection point where gradient/velocity crosses threshold, use that point 
        elif (len(inflec_data) == 1): # & (inflec_data.index[0] > first_frames):  # and inflection point in the first 3/4 of frames 
            event_frame = inflec_data.index[0]

        # if more than one inflectoin point, find one with longest consecutiv "below" values afterwards 
        # longest coneseucutive time below threshold 
        elif len(inflec_data) > 1:
            #print(f"*****event_frames before count consecutive fun: {event_frames}")
            event_frame, count = count_consecutive_below(grad_2_data, 'above_below', event_frames) 

        # if no conditions met, use last frame = position of right base 
        else:
            event_frame = grad_2_data.index[-1] 

        # save all inflection frames
        current_row_data = {'event_frame': event_frame, 'peak_i': peak_i}
        event_frame_all.append(current_row_data) 
         
    # create dataframe 
    event_frame_all_df = pd.DataFrame(event_frame_all)

    return(event_frame_all_df)



########################################################################################
## caalculate double and single support - main 
def id_calc_support_metrics(mp_df, fps, vid_in_path, dir_out_prefix, walk_num): 
    # create and save data frame as .csv 
    output_folder = os.path.join(dir_out_prefix, '005_gait_metrics', 'support_v3')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]
    
    # smooth position data -------------------------------
    # R heel
    mp_r_heel_df = mp_df.loc[mp_df['label'] == 'right_heel']
    mp_r_heel_df = mp_r_heel_df.copy()
    mp_r_heel_df['Y_smooth'] = mp_r_heel_df['Y_pose'].rolling(window=round(0.166* fps), min_periods=1).mean()
    mp_r_heel_df['Y_pose_negative_smooth'] = mp_r_heel_df['Y_pose_negative'].rolling(window=round(0.166 * fps), min_periods=1).mean()
    mp_r_heel_df.set_index('frame', inplace=True)
    
    # l heel 
    mp_l_heel_df = mp_df.loc[mp_df['label'] == 'left_heel']
    mp_l_heel_df = mp_l_heel_df.copy()
    mp_l_heel_df['Y_smooth'] = mp_l_heel_df['Y_pose'].rolling(window=round(0.166 * fps), min_periods=1).mean()
    mp_l_heel_df['Y_pose_negative_smooth'] = mp_l_heel_df['Y_pose_negative'].rolling(window=round(0.166 * fps), min_periods=1).mean()
    mp_l_heel_df.set_index('frame', inplace=True)
    
    # R foot (toe) 
    mp_r_foot_df = mp_df.loc[mp_df['label'] == 'right_foot_index']
    mp_r_foot_df = mp_r_foot_df.copy()
    mp_r_foot_df['Y_smooth'] = mp_r_foot_df['Y_pose'].rolling(window=round(0.166 * fps), min_periods=1).mean()
    mp_r_foot_df['Y_pose_negative_smooth'] = mp_r_foot_df['Y_pose_negative'].rolling(window=round(0.166 * fps), min_periods=1).mean()
    mp_r_foot_df.set_index('frame', inplace=True)
    
    # l foot (toe) 
    mp_l_foot_df = mp_df.loc[mp_df['label'] == 'left_foot_index']
    mp_l_foot_df = mp_l_foot_df.copy()
    mp_l_foot_df['Y_smooth'] = mp_l_foot_df['Y_pose'].rolling(window=round(0.166 * fps), min_periods=1).mean()
    mp_l_foot_df['Y_pose_negative_smooth'] = mp_l_foot_df['Y_pose_negative'].rolling(window=round(0.166 * fps), min_periods=1).mean()
    mp_l_foot_df.set_index('frame', inplace=True)

    # mean of right ankle and foot 
    mp_r_combined = mp_r_heel_df[['Y_pose_negative']].join(mp_r_foot_df[['Y_pose_negative']], how='inner',
                                                          lsuffix = '_heel', 
                                                          rsuffix = '_foot')
    mp_r_combined['mean'] = mp_r_combined.mean(axis=1)
    mp_r_combined['Y_pose_negative_smooth'] = mp_r_combined['mean'].rolling(window=round(0.166 * fps), min_periods=1).mean()

    # mean of left ankle and foot 
    mp_l_combined = mp_l_heel_df[['Y_pose_negative']].join(mp_l_foot_df[['Y_pose_negative']], how='inner', 
                                                          lsuffix = '_heel', 
                                                          rsuffix = '_foot')
    mp_l_combined['mean'] = mp_l_combined.mean(axis=1)
    mp_l_combined['Y_pose_negative_smooth'] = mp_l_combined['mean'].rolling(window=round(0.166 * fps), min_periods=1).mean()
    mp_l_combined.head() 
    
    # find peaks --------------------------------
    # right foot and ankle mean 
    r_combined_peaks_i, r_combined_prop = sig.find_peaks(mp_r_combined['Y_pose_negative_smooth'], prominence = (0.025, None))
    r_combined_peaks_df = pd.DataFrame(data = {'frame' : mp_r_combined.iloc[r_combined_peaks_i].index,
                                           'Y_pose_negative_smooth' :  mp_r_combined.iloc[r_combined_peaks_i]['Y_pose_negative_smooth']
                                           })
    r_combined_peaks_RBase = r_combined_prop['right_bases'] 
    r_combined_peaks_RBase_df = pd.DataFrame(data = {'frame' : mp_r_combined.iloc[r_combined_peaks_RBase].index,
                                                 'Y_pose_negative_smooth' :  mp_r_combined.iloc[r_combined_peaks_RBase]['Y_pose_negative_smooth']
                                           })

    r_combined_peaks_LBase = r_combined_prop['left_bases'] 
    r_combined_peaks_LBase_df = pd.DataFrame(data = {'frame' : mp_r_combined.iloc[r_combined_peaks_LBase].index,
                                                 'Y_pose_negative_smooth' :  mp_r_combined.iloc[r_combined_peaks_LBase]['Y_pose_negative_smooth']
                                           })

    # left foot and ankle mean 
    l_combined_peaks_i, l_combined_prop = sig.find_peaks(mp_l_combined['Y_pose_negative_smooth'], prominence = (0.025, None))
    l_combined_peaks_df = pd.DataFrame(data = {'frame' : mp_l_combined.iloc[l_combined_peaks_i].index,
                                           'Y_pose_negative_smooth' :  mp_l_combined.iloc[l_combined_peaks_i]['Y_pose_negative_smooth']
                                           })
    l_combined_peaks_RBase = l_combined_prop['right_bases'] 
    l_combined_peaks_RBase_df = pd.DataFrame(data = {'frame' : mp_l_combined.iloc[l_combined_peaks_RBase].index,
                                                 'Y_pose_negative_smooth' :  mp_l_combined.iloc[l_combined_peaks_RBase]['Y_pose_negative_smooth']
                                           })
    l_combined_peaks_LBase = l_combined_prop['left_bases'] 
    l_combined_peaks_LBase_df = pd.DataFrame(data = {'frame' : mp_l_combined.iloc[l_combined_peaks_LBase].index,
                                                 'Y_pose_negative_smooth' :  mp_l_combined.iloc[l_combined_peaks_LBase]['Y_pose_negative_smooth']
                                           })
    
    # -----------------------------------
    # save toe offs = inflection point between previous RBase and peak 
    # right 
    print('---------- right toe offs-------------')
    r_foot_toe_off_frames_df = save_RBase_peak_inflection(r_combined_peaks_df,
                                                          r_combined_peaks_RBase_df,
                                                          mp_r_combined, 
                                                         fps)
    r_foot_toe_off_frames_df['event'] = 'r_toe_off'
    
    print('---------- left toe offs-------------')
    l_foot_toe_off_frames_df = save_RBase_peak_inflection(l_combined_peaks_df,
                                                          l_combined_peaks_RBase_df,
                                                          mp_l_combined, 
                                                         fps)
    l_foot_toe_off_frames_df['event'] = 'l_toe_off' 

    # ---------------------------------
    # save heel strikes = inflection point between peak and next R base 
    # right 
    print('---------- right heel strikes-------------')
    r_heel_heel_strike_frames_df = save_peak_Rbase_inflection(peaks_df = r_combined_peaks_df,
                                                              peaks_RBase_df = r_combined_peaks_RBase_df, 
                                                              Y_position_df = mp_r_combined, 
                                                              threshold = -0.005)
    r_heel_heel_strike_frames_df['event'] = 'r_heel_strike' 

    # left 
    print('---------- left heel strikes-------------')
    l_heel_heel_strike_frames_df = save_peak_Rbase_inflection(peaks_df = l_combined_peaks_df,
                                                              peaks_RBase_df = l_combined_peaks_RBase_df, 
                                                              Y_position_df = mp_l_combined, 
                                                              threshold = -0.005)
    l_heel_heel_strike_frames_df['event'] = 'l_heel_strike' 
    
    
 #   print(f"r_foot_toe_off_frames:\n{r_foot_toe_off_frames_df}") 
 #   print(f"l_foot_toe_off_frames_df:\n{l_foot_toe_off_frames_df}")
#    print(f"r_heel_heel_strike_frames_df:\n{r_heel_heel_strike_frames_df}")
#    print(f"l_heel_heel_strike_frames_df: {l_heel_heel_strike_frames_df}")
    # PLOT ###################
    fig_test, [ax1, ax2, ax3] = plt.subplots(nrows=3,  figsize=(5.75, 5))  

                
    sns.lineplot(x = 'frame', y = 'Y_pose_negative_smooth',
                 data = mp_r_combined, 
                 color = 'gold',
                 ax = ax3) 
    sns.lineplot(x = 'frame', y = 'Y_pose_negative_smooth',
                 data = mp_r_combined, 
                 color = 'gold', label = 'mean right',
                 ax = ax1) 

    sns.lineplot(x = 'frame', y = 'Y_pose_negative_smooth',
                 data = mp_l_combined, 
                 color = 'purple',
                 ax = ax3) 
    sns.lineplot(x = 'frame', y = 'Y_pose_negative_smooth',
                 data = mp_l_combined, 
                 color = 'purple', label = 'mean left',
                 ax = ax2) 

    ax1.plot(r_combined_peaks_RBase_df['frame'], r_combined_peaks_RBase_df['Y_pose_negative_smooth'],
             "*", markersize = 5, color = 'gold', alpha = 0.5)
    ax2.plot(l_combined_peaks_RBase_df['frame'], l_combined_peaks_RBase_df['Y_pose_negative_smooth'],
             "*", markersize = 5, color = 'purple', alpha = 0.5)
    
    ax1.plot(r_combined_peaks_df['frame'], r_combined_peaks_df['Y_pose_negative_smooth'],
             "o", markersize = 5, color = 'gold', alpha = 0.5)
    ax2.plot(l_combined_peaks_df['frame'], l_combined_peaks_df['Y_pose_negative_smooth'],
             "o", markersize = 5, color = 'purple', alpha = 0.5)

    # add right foot toe offs on subplot 1 
    # right toe off from combined  
    if len(r_foot_toe_off_frames_df) > 0: 
        for r_to_i, r_toe_off_frame in enumerate(r_foot_toe_off_frames_df['event_frame']): 
            if r_to_i == 0: 
                ax3.axvline(r_toe_off_frame, color = 'gold', 
                            linestyle = '--', alpha = 0.5, label = 'R Toe Off')
                ax1.axvline(r_toe_off_frame, color = 'gold', 
                            linestyle = '--', alpha = 0.5)
            else: 
                ax3.axvline(r_toe_off_frame, color = 'gold', 
                            linestyle = '--', alpha = 0.5) 
                ax1.axvline(r_toe_off_frame, color = 'gold', 
                            linestyle = '--', alpha = 0.5)

    # left toe off 
    if len(l_foot_toe_off_frames_df) > 0: 
        for l_to_i, l_toe_off_frame in enumerate(l_foot_toe_off_frames_df['event_frame']): 
            if l_to_i == 0: 
                ax3.axvline(l_toe_off_frame, color = 'purple', 
                            linestyle = '--', alpha = 0.5, label = 'L Toe Off')
                ax2.axvline(l_toe_off_frame, color = 'purple', 
                            linestyle = '--', alpha = 0.5)
            else: 
                ax3.axvline(l_toe_off_frame, color = 'purple', 
                            linestyle = '--', alpha = 0.5)
                ax2.axvline(l_toe_off_frame, color = 'purple', 
                            linestyle = '--', alpha = 0.5)

    # right heel strikes  
    if len(r_heel_heel_strike_frames_df) > 0: 
        for r_hs_i, r_heel_strike_frame in enumerate(r_heel_heel_strike_frames_df['event_frame']): 
            if r_hs_i == 0: 
                ax3.axvline(r_heel_strike_frame, color = 'gold', 
                            linestyle = '-', alpha = 0.5, label = 'R Heel Strike')
                ax1.axvline(r_heel_strike_frame, color = 'gold', 
                            linestyle = '-', alpha = 0.5)
            else: 
                ax3.axvline(r_heel_strike_frame, color = 'gold', 
                            linestyle = '-', alpha = 0.5) 
                ax1.axvline(r_heel_strike_frame, color = 'gold', 
                            linestyle = '-', alpha = 0.5)

    # left heel strike 
    if len(l_heel_heel_strike_frames_df) > 0:
        for l_hs_i, l_heel_strike_frame in enumerate(l_heel_heel_strike_frames_df['event_frame']):
            if l_hs_i == 0:
                ax3.axvline(l_heel_strike_frame, color = 'purple',
                            linestyle = '-', alpha = 0.5, label = 'L Heel Strike')
                ax2.axvline(l_heel_strike_frame, color = 'purple',
                            linestyle = '-', alpha = 0.5)
            else:
                ax3.axvline(l_heel_strike_frame, color = 'purple', 
                            linestyle = '-', alpha = 0.5)
                ax2.axvline(l_heel_strike_frame, color = 'purple',
                            linestyle = '-', alpha = 0.5)
              
 #   ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize = 6)
 #   ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize = 6)
 #   ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize = 6)
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    ax3.set_ylabel("")

    plt.tight_layout()

    plot_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_' + walk_num + 'all_identified_gait_events.png')))
    plt.savefig(plot_path)
    plt.show() 
    plt.close() 

    # -------------------------------------------------------------------
    # combine all event dfs 
    all_events_df = pd.concat([r_foot_toe_off_frames_df,
                               l_foot_toe_off_frames_df,
                               r_heel_heel_strike_frames_df, 
                               l_heel_heel_strike_frames_df]) 

    all_gait_events = []

    # if any of these are empty, save empty dataframe 
    if (r_foot_toe_off_frames_df.empty) or (l_foot_toe_off_frames_df.empty) or (r_heel_heel_strike_frames_df.empty) or (l_heel_heel_strike_frames_df.empty):
        all_gait_events_df = pd.DataFrame(columns = ['first_toe_off_foot',	
                                                     'foot_1_heel_strike_a',
                                                     'foot_2_toe_off',
                                                     'foot_2_heel_strike',
                                                     'foot_1_toe_off',
                                                     'foot_1_heel_strike_b'], 
                                          index = [walk_num]) 
        
    # if there is data in all dataframes 
    else: 
        all_events_df = all_events_df.sort_values(by='event_frame')
        all_events_df = all_events_df.reset_index(drop = True) 
       # print(f"all_events_df:\n{all_events_df}") 
    
        # --------------------------------------------------------------------
        # blank gait events to populate with heel strikes and toe offs in order 
        for event_i, event_row in all_events_df.iterrows(): 

            length_df = len(all_events_df) 
            current_gait_events = []
        
            # RIGHT STRIDE 
            # event = right heel strike and at least four rows afterwards 
            if (event_row['event'] == 'r_heel_strike') & (event_i < (length_df - 4)): 
                first_toe_off_foot = 'right'
                heel_strike_1a = event_row['event_frame']
                
                # if next event = left toe off, save frame 
                if all_events_df.iloc[event_i + 1]['event'] == 'l_toe_off': 
                    toe_off_2 = all_events_df.iloc[event_i + 1]['event_frame'] 
                else: 
                    toe_off_2 = None 

                # if two events later is left heel strike, save frame 
                if all_events_df.iloc[event_i + 2]['event'] == 'l_heel_strike':
                    heel_strike_2 = all_events_df.iloc[event_i + 2]['event_frame']
                else: 
                    heel_strike_2 = None 

                # if three events later is right toe off, save frame 
                if all_events_df.iloc[event_i + 3]['event'] == 'r_toe_off':
                    toe_off_1 = all_events_df.iloc[event_i + 3]['event_frame']
                else: 
                    toe_off_1 = None

                # if four events later is right heel strike, save frame 
                if all_events_df.iloc[event_i + 4]['event'] == 'r_heel_strike': 
                    heel_strike_1b = all_events_df.iloc[event_i + 4]['event_frame']
                else: 
                    heel_strike_1b = None 
                
                # combine and save 
                current_gait_events = pd.DataFrame(data = {'first_toe_off_foot' : [first_toe_off_foot],
                                                           'foot_1_heel_strike_a' : [heel_strike_1a], 
                                                           'foot_2_toe_off' : [toe_off_2],
                                                           'foot_2_heel_strike' : [heel_strike_2], 
                                                           'foot_1_toe_off' : [toe_off_1], 
                                                           'foot_1_heel_strike_b' : [heel_strike_1b]
                                                          }) 
                all_gait_events.append(current_gait_events)

            # LEFT STRIDE 
            elif (event_row['event'] == 'l_heel_strike') & (event_i < (length_df - 4)):
                first_toe_off_foot = 'left'
                heel_strike_1a = event_row['event_frame']

                # if next event = left toe off, save frame 
                if all_events_df.iloc[event_i + 1]['event'] == 'r_toe_off': 
                    toe_off_2 = all_events_df.iloc[event_i + 1]['event_frame'] 
                else: 
                    toe_off_2 = None 

                # if two events later is left heel strike, save frame 
                if all_events_df.iloc[event_i + 2]['event'] == 'r_heel_strike':
                    heel_strike_2 = all_events_df.iloc[event_i + 2]['event_frame']
                else: 
                    heel_strike_2 = None 

                # if three events later is right toe off, save frame 
                if all_events_df.iloc[event_i + 3]['event'] == 'l_toe_off':
                    toe_off_1 = all_events_df.iloc[event_i + 3]['event_frame']
                else: 
                    toe_off_1 = None

                # if four events later is left heel strike, save fram 
                if all_events_df.iloc[event_i + 4]['event'] == 'l_heel_strike': 
                    heel_strike_1b = all_events_df.iloc[event_i + 4]['event_frame']
                else: 
                    heel_strike_1b = None 
                
                # combine and save 
                current_gait_events = pd.DataFrame(data = {'first_toe_off_foot' : [first_toe_off_foot],
                                                           'foot_1_heel_strike_a' : [heel_strike_1a], 
                                                           'foot_2_toe_off' : [toe_off_2],
                                                           'foot_2_heel_strike' : [heel_strike_2], 
                                                           'foot_1_toe_off' : [toe_off_1], 
                                                           'foot_1_heel_strike_b' : [heel_strike_1b]
                                                          })
                all_gait_events.append(current_gait_events)

        # if no strides identified with events in correct order, save empty data frame 
        if len(all_gait_events) == 0: 
            print('no strides identfied') 
            all_gait_events_df = pd.DataFrame(columns = ['first_toe_off_foot',
                                                         'foot_1_heel_strike_a',
                                                         'foot_2_toe_off',
                                                         'foot_2_heel_strike',
                                                         'foot_1_toe_off',
                                                         'foot_1_heel_strike_b'], 
                                              index = [walk_num]) 
        
        # if strides found, 
        # # concatenate all strides into single data frame and drop None 
        else: 
            all_gait_events_df = pd.concat(all_gait_events)
            all_gait_events_df = all_gait_events_df.reset_index(drop = True)
            all_gait_events_df = all_gait_events_df.dropna(axis = 0)



    #######################################################################
    # calculate metrics 
    # gait cycle time = first contact of one foot the the following first contact of the same foot 
    all_gait_events_df['gait_cycle_time_sec'] = (all_gait_events_df['foot_1_heel_strike_b'] - all_gait_events_df['foot_1_heel_strike_a']) / fps

    # stance time = time foot 1 is in contact with the ground 
    all_gait_events_df['stance_time_sec'] = (all_gait_events_df['foot_1_toe_off'] - all_gait_events_df['foot_1_heel_strike_a']) / fps 
    all_gait_events_df['stance_time_per'] = (all_gait_events_df['stance_time_sec'] / all_gait_events_df['gait_cycle_time_sec']) * 100

    # swing time - period of time foot 1 is not in contact with the ground 
    all_gait_events_df['swing_time_sec'] = all_gait_events_df['gait_cycle_time_sec'] - all_gait_events_df['stance_time_sec']
    all_gait_events_df['swing_time_per'] = (all_gait_events_df['swing_time_sec'] / all_gait_events_df['gait_cycle_time_sec']) * 100

    # single support time 
    # period of time when only the current foot is in contact with the ground 
    all_gait_events_df['singlesupport_time_sec'] = (all_gait_events_df['foot_2_heel_strike'] - all_gait_events_df['foot_2_toe_off']) / fps
    all_gait_events_df['singlesupport_per'] = (all_gait_events_df['singlesupport_time_sec'] / all_gait_events_df['gait_cycle_time_sec']) * 100
    
    # double support time 
    all_gait_events_df['ini_dsupport_sec'] = (all_gait_events_df['foot_2_toe_off'] - all_gait_events_df['foot_1_heel_strike_a']) / fps
    all_gait_events_df['term_dsupport_sec'] = (all_gait_events_df['foot_1_toe_off'] - all_gait_events_df['foot_2_heel_strike']) / fps
    all_gait_events_df['tot_dsupport_time_sec'] = all_gait_events_df['ini_dsupport_sec'] + all_gait_events_df['term_dsupport_sec'] 
    all_gait_events_df['tot_dsupport_per'] = (all_gait_events_df['tot_dsupport_time_sec'] / all_gait_events_df['gait_cycle_time_sec']) * 100

    # round values 
    temp_foot = all_gait_events_df['first_toe_off_foot'] 
    all_gait_events_df = all_gait_events_df.apply(pd.to_numeric, errors='coerce')
    all_gait_events_df = all_gait_events_df.round(2)
    all_gait_events_df['first_toe_off_foot'] = temp_foot 

    csv_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_' + walk_num + '_all_gait_events_df.csv')))
    all_gait_events_df.to_csv(csv_path) 

    return all_gait_events_df

# calculate stats per walk (mean, median, std)
def calc_support_stats(all_gait_events_df): 
    # median 
    median_df = pd.DataFrame(all_gait_events_df.median(numeric_only = True))
    median_df = median_df.reset_index() 
    median_df = median_df.rename(columns = {'index' : 'metric', 0 : 'median'}) 
    
    # mean 
    mean_df = pd.DataFrame(all_gait_events_df.mean(numeric_only = True))
    mean_df = mean_df.reset_index() 
    mean_df = mean_df.rename(columns = {'index' : 'metric', 0 : 'mean'})

    # standard devaition 
    std_df = pd.DataFrame(all_gait_events_df.std(numeric_only = True))
    std_df = std_df.reset_index()
    std_df = std_df.rename(columns = {'index' : 'metric', 0 : 'std'}) 

    # merge together 
    merged_stats_df = pd.merge(mean_df, median_df, on = 'metric', how = 'inner')
    merged_stats_df = pd.merge(merged_stats_df, std_df,  on = 'metric', how = 'inner')

    # make long and drop temporary index column 
    merged_stats_df['temp_idx'] = 0 
    merged_stats_df_long = merged_stats_df.pivot(index = 'temp_idx',
                                                    columns = 'metric')
    merged_stats_df_long.columns = [f'{col[1]}_{col[0]}' for col in merged_stats_df_long.columns]
    merged_stats_df_long = merged_stats_df_long.reset_index() 
    merged_stats_df_long.drop(['temp_idx'], axis=1, inplace=True)
    merged_stats_df_long = merged_stats_df_long.round(2)

    return merged_stats_df_long




