#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import scipy.signal as sig

################################################################
# functions to ID toe offs 
def count_consecutive_above(df, flag_col, event_indices):
    max_count = 0
    best_index = None

    for idx in event_indices:
        count = 0
        # Go row by row after the event index
        for val in df[flag_col].loc[df.index >= idx]:
            if val == 'above':
                count += 1
            else:
                break  # stop when it's not 'above'

        if count > max_count:
            max_count = count
            best_index = idx

    return best_index, max_count

def find_up_inflection_points(time_series_series, threshold): 
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

    # upward crossings where velocity crosses above velocity threshold 
    inflection_point_indices = (gradient_data_w_nan.shift(1) < threshold) & (gradient_data_w_nan >= threshold)

    # filter series to only include values at inflection points 
    inflection_points_data = gradient_data_w_nan.loc[inflection_point_indices]

    # make df and add column - is velocity/gradient above or below threshold 
    gradient_data_w_nan = gradient_data_w_nan.to_frame()
    gradient_data_w_nan['above_below'] = 'above' 
    gradient_data_w_nan.loc[gradient_data_w_nan['gradient'] <= threshold, 'above_below'] = 'below'

    # get frames of inflection points 
    inflection_frames = inflection_points_data.index
    
    return(gradient_data_w_nan, inflection_points_data, inflection_frames)

# inflection points between each peak and prior right base 
def save_RBase_peak_inflection(peaks_df, peaks_RBase_df, Y_position_df, threshold):     
    # reset index for loop 
    peaks_df = peaks_df.reset_index(drop = True)
    peaks_RBase_df = peaks_RBase_df.reset_index(drop = True)

    # for each prominence, find inflection between left base and prominence
    inflec_frames = []
    inflec_frame_all = []
    inflec_frame_all_df = []
    for peak_i, peak_row in peaks_df.iterrows():
      #  print(f"peak_i: {peak_i}") 
        if peak_i == 0: 
            continue # skip first peak 

        peak_frame = peak_row['frame'] 
       # print(f"peak_frame: {peak_frame}") 
        # Previous right base 
        RBase_frame = peaks_RBase_df['frame'][peak_i-1]
      #  print(f"LBase_frame: {RBase_frame}") 

        # current segment between prominence and previous right base 
        current_segment_df = Y_position_df.loc[(Y_position_df.index < peak_frame) & (Y_position_df.index > RBase_frame)] 

        # if less than X frames, skip this segment and move to next 
        if len(current_segment_df) < 3: 
            continue 
            
        # find inflection points  
     #   print('inflection points --------------') 
        grad_2_data, inflec_data, inflec_frames = find_up_inflection_points(current_segment_df['Y_pose_negative_smooth'],
                                                                         threshold) 
        # plot 
        # plot current segment 
        fig_inflec, [ax1, ax2] = plt.subplots(nrows=2,  figsize=(2, 3)) 
        plt.suptitle('Prev R Base to Peak') 
        ax1.plot(current_segment_df['Y_pose_negative_smooth'])
        
        ax2.plot(grad_2_data['gradient']) 
        ax2.plot(inflec_frames, inflec_data, 'o', color = 'black') 
        ax2.axhline(y=threshold, color='grey', linestyle='--', linewidth=0.8)
        ax2.set_ylim([-.001, max(grad_2_data['gradient'])+.001])

        plt.show() 
        plt.close() 
        # ------------------------
    #    print(f"len(inflec_data): {len(inflec_data)}")
    #    print(f"grad_2_data: {grad_2_data}")

        # total num frames between R base and next peak 
        total_frames = grad_2_data.index[-1] - grad_2_data.index[0]
        three_4ths_frames = grad_2_data.index[0] + (total_frames * 0.75)
    #    print(f"three_4ths_frames: {three_4ths_frames}")
        
        # if all gradient/velocity values are greater than threshold, use first frame 
            # use first frame = position of previous right base 
        if len(inflec_data) == 0: 
            inflec_frame = grad_2_data.index[0] 

        # if there one inflection point where gradient/velocity crosses threshold  
        # and inflection point in the first 3/4 of frames (not at very end near peak) 
        elif (len(inflec_data) == 1) & (inflec_data.index[0] < three_4ths_frames): 
            inflec_frame = inflec_data.index[0]

        # if more than one inflectoin point, find one with longest consecutiv "above" values afterwards  
        elif len(inflec_data) > 1:
            #print(f"*****inflec_frames before count consecutive fun: {inflec_frames}")
            inflec_frame, count = count_consecutive_above(grad_2_data, 'above_below', inflec_frames) 

        # if no conditions met, use first frame = position of previous right base 
        else:
            inflec_frame = grad_2_data.index[0] 
        
        # save all inflection frames 
        current_row_data = {'inflec_frame': inflec_frame, 'peak_i': peak_i}
        inflec_frame_all.append(current_row_data) 
        
    # create dataframe 
    inflec_frame_all_df = pd.DataFrame(inflec_frame_all)
    
    return(inflec_frame_all_df)
    
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
    
    return(gradient_data_w_nan, inflection_points_data, inflection_frames)


# find inflections points between peaks and next RBase 
def save_peak_Rbase_inflection(peaks_df, peaks_RBase_df, Y_position_df, threshold):     
    # reset index for loop 
    peaks_df = peaks_df.reset_index(drop = True)
    peaks_RBase_df = peaks_RBase_df.reset_index(drop = True)

    # for each prominence, find inflection between left base and prominence
    inflec_frames = []
    inflec_frame_all = []
    inflec_frame_all_df = []
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
        grad_2_data, inflec_data, inflec_frames = find_down_inflection_points(current_segment_df['Y_pose_negative_smooth'],
                                                                              threshold) 
        # Plot   
        # plot current segment 
        fig_inflec, [ax1, ax2] = plt.subplots(nrows=2,  figsize=(2, 3)) 
        plt.suptitle('Peak to Next R Base') 
        ax1.plot(current_segment_df['Y_pose_negative_smooth'])
        
        ax2.plot(grad_2_data['gradient']) 
        ax2.plot(inflec_frames, inflec_data, 'o', color = 'black') 
        ax2.axhline(y=threshold, color='grey', linestyle='--', linewidth=0.8)
     #   ax2.set_ylim([min(grad_2_data['gradient'])-.001, 0.001])

        plt.show() 
        plt.close() 

        # total num frames between R base and next peak 
        total_frames = grad_2_data.index[-1] - grad_2_data.index[0]
        one_4th_frames = grad_2_data.index[0] + (total_frames * 0.25)
        
        # if all gradient/velocity values are greater than threshold, use last frame 
            # use last frame = position of next right base 
        if len(inflec_data) == 0: 
            inflec_frame = grad_2_data.index[-1]

        # if there one inflection point where gradient/velocity crosses threshold  
        # and inflection point in the last 3/4 of frames 
        elif (len(inflec_data) == 1) & (inflec_data.index[0] > one_4th_frames): 
            inflec_frame = inflec_data.index[0]

        # if more than one inflectoin point, find one with longest consecutiv "below" values afterwards 
        # longest coneseucutive time below threshold 
        elif len(inflec_data) > 1:
            #print(f"*****inflec_frames before count consecutive fun: {inflec_frames}")
            inflec_frame, count = count_consecutive_below(grad_2_data, 'above_below', inflec_frames) 

        # if no conditions met, use last frame = position of right base 
        else:
            inflec_frame = grad_2_data.index[-1] 

        # save all inflection frames
        current_row_data = {'inflec_frame': inflec_frame, 'peak_i': peak_i}
        inflec_frame_all.append(current_row_data) 
         
    # create dataframe 
    inflec_frame_all_df = pd.DataFrame(inflec_frame_all)

    return(inflec_frame_all_df)


########################################################################################
## caalculate double and single support - main 
def id_calc_support_metrics(mp_df, fps, vid_in_path, dir_out_prefix, walk_num): 
    # create and save data frame as .csv 
    output_folder = os.path.join(dir_out_prefix, '005_gait_metrics', 'support_v2')
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


    # find peaks --------------------------------
    # right heel 
    r_heel_peaks_i, r_heel_prop = sig.find_peaks(mp_r_heel_df['Y_pose_negative_smooth'], prominence = (0.05, None))
    r_heel_peaks_df = pd.DataFrame(data = {'frame' : mp_r_heel_df.iloc[r_heel_peaks_i].index,
                                           'Y_pose_negative_smooth' :  mp_r_heel_df.iloc[r_heel_peaks_i]['Y_pose_negative_smooth']
                                           })
    r_heel_peaks_RBase = r_heel_prop['right_bases'] 
    r_heel_peaks_RBase_df = pd.DataFrame(data = {'frame' : mp_r_heel_df.iloc[r_heel_peaks_RBase].index,
                                                 'Y_pose_negative_smooth' :  mp_r_heel_df.iloc[r_heel_peaks_RBase]['Y_pose_negative_smooth']
                                           })
    r_heel_peaks_LBase = r_heel_prop['left_bases'] 
    r_heel_peaks_LBase_df = pd.DataFrame(data = {'frame' : mp_r_heel_df.iloc[r_heel_peaks_LBase].index,
                                                 'Y_pose_negative_smooth' :  mp_r_heel_df.iloc[r_heel_peaks_LBase]['Y_pose_negative_smooth']
                                           })
    
    # right foot 
    r_foot_peaks_i, r_foot_prop = sig.find_peaks(mp_r_foot_df['Y_pose_negative_smooth'], prominence = (0.03, None))
    r_foot_peaks_df = pd.DataFrame(data = {'frame' : mp_r_foot_df.iloc[r_foot_peaks_i].index,
                                           'Y_pose_negative_smooth' :  mp_r_foot_df.iloc[r_foot_peaks_i]['Y_pose_negative_smooth']
                                           })
    r_foot_peaks_RBase = r_foot_prop['right_bases'] 
    r_foot_peaks_RBase_df = pd.DataFrame(data = {'frame' : mp_r_foot_df.iloc[r_foot_peaks_RBase].index,
                                                 'Y_pose_negative_smooth' :  mp_r_foot_df.iloc[r_foot_peaks_RBase]['Y_pose_negative_smooth']
                                           })
    r_foot_peaks_LBase = r_foot_prop['left_bases'] 
    r_foot_peaks_LBase_df = pd.DataFrame(data = {'frame' : mp_r_foot_df.iloc[r_foot_peaks_LBase].index,
                                                 'Y_pose_negative_smooth' :  mp_r_foot_df.iloc[r_foot_peaks_LBase]['Y_pose_negative_smooth']
                                           })

    # left heel  
    l_heel_peaks_i, l_heel_prop = sig.find_peaks(mp_l_heel_df['Y_pose_negative_smooth'], prominence = (0.05, None))
    l_heel_peaks_df = pd.DataFrame(data = {'frame' : mp_l_heel_df.iloc[l_heel_peaks_i].index,
                                           'Y_pose_negative_smooth' :  mp_l_heel_df.iloc[l_heel_peaks_i]['Y_pose_negative_smooth']
                                           })
    l_heel_peaks_RBase = l_heel_prop['right_bases'] 
    l_heel_peaks_RBase_df = pd.DataFrame(data = {'frame' : mp_l_heel_df.iloc[l_heel_peaks_RBase].index,
                                                 'Y_pose_negative_smooth' :  mp_l_heel_df.iloc[l_heel_peaks_RBase]['Y_pose_negative_smooth']
                                           })
    l_heel_peaks_LBase = l_heel_prop['left_bases'] 
    l_heel_peaks_LBase_df = pd.DataFrame(data = {'frame' : mp_l_heel_df.iloc[l_heel_peaks_LBase].index,
                                                 'Y_pose_negative_smooth' :  mp_l_heel_df.iloc[l_heel_peaks_LBase]['Y_pose_negative_smooth']
                                           })

    # left foot 
    l_foot_peaks_i, l_foot_prop = sig.find_peaks(mp_l_foot_df['Y_pose_negative_smooth'], prominence = (0.03, None))
    l_foot_peaks_df = pd.DataFrame(data = {'frame' : mp_l_foot_df.iloc[l_foot_peaks_i].index,
                                           'Y_pose_negative_smooth' :  mp_l_foot_df.iloc[l_foot_peaks_i]['Y_pose_negative_smooth']
                                           })
    l_foot_peaks_RBase = l_foot_prop['right_bases'] 
    l_foot_peaks_RBase_df = pd.DataFrame(data = {'frame' : mp_l_foot_df.iloc[l_foot_peaks_RBase].index,
                                                 'Y_pose_negative_smooth' :  mp_l_foot_df.iloc[l_foot_peaks_RBase]['Y_pose_negative_smooth']
                                           })
    l_foot_peaks_LBase = l_foot_prop['left_bases'] 
    l_foot_peaks_LBase_df = pd.DataFrame(data = {'frame' : mp_l_foot_df.iloc[l_foot_peaks_LBase].index,
                                                 'Y_pose_negative_smooth' :  mp_l_foot_df.iloc[l_foot_peaks_LBase]['Y_pose_negative_smooth']
                                           })
    
    # -----------------------------------
    # save toe offs = inflection point between previous RBase and peak 
    # right 
    print('---------- right toe offs-------------')
    r_foot_toe_off_frames_df = save_RBase_peak_inflection(r_foot_peaks_df,
                                                          r_foot_peaks_RBase_df,
                                                          mp_r_foot_df,
                                                          threshold = 0.002)
    print('---------- left toe offs-------------')
    l_foot_toe_off_frames_df = save_RBase_peak_inflection(l_foot_peaks_df,
                                                          l_foot_peaks_RBase_df,
                                                          mp_l_foot_df,
                                                          threshold = 0.002)

    # ---------------------------------
    # save heel strikes = inflection point between peak and next R base 
    # right 
    print('---------- right heel strikes-------------')
    r_heel_heel_strike_frames_df = save_peak_Rbase_inflection(peaks_df = r_heel_peaks_df,
                                                              peaks_RBase_df = r_heel_peaks_RBase_df, 
                                                              Y_position_df = mp_r_heel_df, 
                                                              threshold = -0.01)
    

    # left 
    print('---------- left heel strikes-------------')
    l_heel_heel_strike_frames_df = save_peak_Rbase_inflection(peaks_df = l_heel_peaks_df,
                                                              peaks_RBase_df = l_heel_peaks_RBase_df, 
                                                              Y_position_df = mp_l_heel_df, 
                                                              threshold = -0.01)
    

    print(f"r_foot_toe_off_frames: {r_foot_toe_off_frames_df}") 
    print(f"l_foot_toe_off_frames_df: {l_foot_toe_off_frames_df}")
    print(f"r_heel_heel_strike_frames_df: {r_heel_heel_strike_frames_df}")
    print(f"l_heel_heel_strike_frames_df: {l_heel_heel_strike_frames_df}")
    # PLOT ###################
    fig_test, [ax1, ax2] = plt.subplots(nrows=2,  figsize=(5.75, 3))  

    
    # subplot 1 = right side ---------------------------
    # right foot 
    sns.lineplot(x = 'frame', y = 'Y_pose_negative_smooth', 
                     data = mp_r_foot_df, 
                     color = 'red', label = 'right foot Y',
                     ax = ax1) 
    
    # right heel 
    sns.lineplot(x = 'frame', y = 'Y_pose_negative_smooth', 
                     data = mp_r_heel_df, 
                     color = 'orange', label = 'right heel Y',
                     ax = ax1)  
    # right foot peaks and peak right bases 
    ax1.plot(r_foot_peaks_df['frame'], r_foot_peaks_df['Y_pose_negative_smooth'], 
             "o", markersize = 5, color = 'red', label = 'R Foot Peak')
    ax1.plot(r_foot_peaks_RBase_df['frame'], r_foot_peaks_RBase_df['Y_pose_negative_smooth'], 
             "*", markersize = 5, color = 'red', alpha = 0.5,  label = 'R Foot RBase')

    # right heel peaks and peak right bases 
    ax1.plot(r_heel_peaks_df['frame'], r_heel_peaks_df['Y_pose_negative_smooth'], 
             "o", markersize = 5, color = 'orange', label = 'R Heel Peak')
    ax1.plot(r_heel_peaks_RBase_df['frame'], r_heel_peaks_RBase_df['Y_pose_negative_smooth'], 
             "*", markersize = 5, color = 'orange', label = 'R Heel RBase')

    # add right foot toe offs on subplot 1 
    # if data frame with values saved - plot 
    if len(r_foot_toe_off_frames_df) > 0: 
        for r_to_i, r_toe_off_frame in enumerate(r_foot_toe_off_frames_df['inflec_frame']): 
            if r_to_i == 0: 
                ax1.axvline(r_toe_off_frame, color = 'red', 
                            linestyle = '--', alpha = 0.5, label = 'R Toe Off')
              #  ax2.axvline(r_toe_off_frame, color = 'red', 
                        #    linestyle = '--', alpha = 0.5, label = 'R Toe Off')
            else: 
                ax1.axvline(r_toe_off_frame, color = 'red', 
                            linestyle = '--', alpha = 0.5) 
               # ax2.axvline(r_toe_off_frame, color = 'red', 
                         #   linestyle = '--', alpha = 0.5) 

    # add right heel heel strikes on subplot 1 
    if len(r_heel_heel_strike_frames_df) > 0: 
        for r_hs_i, r_heel_strike_frame in enumerate(r_heel_heel_strike_frames_df['inflec_frame']): 
            if r_to_i == 0: 
                ax1.axvline(r_heel_strike_frame, color = 'orange', 
                            linestyle = '--', alpha = 0.5, label = 'R Heel Strike')
              #  ax2.axvline(r_heel_strike_frame, color = 'orange', 
                           # linestyle = '--', alpha = 0.5, label = 'R Heel Strike')
            else: 
                ax1.axvline(r_heel_strike_frame, color = 'orange', 
                            linestyle = '--', alpha = 0.5) 
              #  ax2.axvline(r_heel_strike_frame, color = 'orange', 
                           # linestyle = '--', alpha = 0.5) 

    ## subplot 2 - left ----------------------------
    # left foot 
    sns.lineplot(x = 'frame', y = 'Y_pose_negative_smooth',
                 data = mp_l_foot_df, 
                 color = 'green', label = 'left foot Y',
                 ax = ax2) 
    # left heel 
    sns.lineplot(x = 'frame', y = 'Y_pose_negative_smooth',
                 data = mp_l_heel_df,
                 color = 'blue', label = 'left heel Y',
                 ax = ax2)  

    # left foot peaks and peak right bases 
    ax2.plot(l_foot_peaks_df['frame'], l_foot_peaks_df['Y_pose_negative_smooth'], 
             "o", markersize = 5, color = 'green', label = 'L Foot Peak')
    ax2.plot(l_foot_peaks_RBase_df['frame'], l_foot_peaks_RBase_df['Y_pose_negative_smooth'], 
             "*", markersize = 5, color = 'green', alpha = 0.5,  label = 'L Foot RBase')

    # left heel peaks and peak right bases 
    ax2.plot(l_heel_peaks_df['frame'], l_heel_peaks_df['Y_pose_negative_smooth'], 
             "o", markersize = 5, color = 'blue', label = 'L Heel Peak')
    ax2.plot(l_heel_peaks_RBase_df['frame'], l_heel_peaks_RBase_df['Y_pose_negative_smooth'], 
             "*", markersize = 5, color = 'blue', label = 'L Heel RBase')

    # add left foot toe offs on subplot 2 
    # if data frame with values saved - plot 
    if len(l_foot_toe_off_frames_df) > 0: 
        for l_to_i, l_toe_off_frame in enumerate(l_foot_toe_off_frames_df['inflec_frame']): 
            if l_to_i == 0: 
                ax2.axvline(l_toe_off_frame, color = 'green', 
                            linestyle = '--', alpha = 0.5, label = 'L Toe Off')
            #    ax1.axvline(l_toe_off_frame, color = 'green', 
                          #  linestyle = '--', alpha = 0.5, label = 'L Toe Off')
            else: 
                ax2.axvline(l_toe_off_frame, color = 'green', 
                            linestyle = '--', alpha = 0.5)
               # ax1.axvline(l_toe_off_frame, color = 'green', 
                          #  linestyle = '--', alpha = 0.5)
                
    # add left foot heel strikes on subplot 2 
    if len(l_heel_heel_strike_frames_df) > 0:
        for l_hs_i, l_heel_strike_frame in enumerate(l_heel_heel_strike_frames_df['inflec_frame']):
            if l_to_i == 0:
                ax2.axvline(l_heel_strike_frame, color = 'blue',
                            linestyle = '--', alpha = 0.5, label = 'L Heel Strike')
              #  ax1.axvline(l_heel_strike_frame, color = 'blue',
                         #   linestyle = '--', alpha = 0.5, label = 'L Heel Strike')
            else:
                ax2.axvline(l_heel_strike_frame, color = 'blue', 
                            linestyle = '--', alpha = 0.5)
              #  ax1.axvline(l_heel_strike_frame, color = 'blue', 
                          #  linestyle = '--', alpha = 0.5)

    plt.legend(loc="center left",
               bbox_to_anchor=(1.05, 0.5)) 
    plt.show() 
    plt.close() 
    
    # -----------------------------
    # blank gait events to populate   
    all_gait_events = []

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




