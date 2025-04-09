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


# In[ ]:


def plot_events_per_stride(all_gait_events_df, mp_r_ank_df, mp_l_ank_df, output_folder, vid_in_path_no_ext, walk_num): 
    ## Plot each set of gait events data 
    for index, row in all_gait_events_df.iterrows():
        fig, axs = plt.subplots(figsize=(5.75, 3)) 
        # right ankle y position 
        sns.lineplot(x = 'frame', y = 'Y_pose_negative_smooth', 
                     data = mp_r_ank_df, 
                     color = 'darkorange', label = 'right ankle Y', alpha = 0.5, ax = axs) 
        # left ankle y position 
        sns.lineplot(x = 'frame', y = 'Y_pose_negative_smooth', 
                     data = mp_l_ank_df, 
                     color = 'darkblue', label = 'left ankle Y', alpha = 0.5, ax = axs) 
    
        # foot 1 toe off a
        axs.axvline(row['foot_1_toe_off_a'], color = 'grey', 
                       linestyle = '--', alpha = 0.5, label = 'foot_1_toe_off_a')
    
        # foot 1 heel strike a 
        axs.axvline(row['foot_1_heel_strike_a'], color = 'grey', 
                       linestyle = '-', alpha = 0.5, label = 'foot_1_heel_strike_a')
    
        # foot 2 toe off 
        axs.axvline(row['foot_2_toe_off'], color = 'black', 
                       linestyle = '--', alpha = 0.5, label = 'foot_2_toe_off')

        # foot 2 heel strike 
        axs.axvline(row['foot_2_heel_strike'], color = 'black', 
                    linestyle = '-', alpha = 0.5, label = 'foot_2_heel_strike')

        # foot 1 toe off b 
        axs.axvline(row['foot_1_toe_off_b'], color = 'grey', 
                    linestyle = '--', alpha = 0.5, label = 'foot_1_toe_off_b')

        # foot 1 heel strike 2 
        axs.axvline(row['foot_1_heel_strike_b'], color = 'grey', 
                    linestyle = '-', alpha = 0.5, label = 'foot_1_heel_strike_b')

        # title = foot 1 
 #       if row['first_toe_off_foot'] == 'left':
 #            fig.suptitle("Foot 1 = Left Foot") 
 #        elif row['first_toe_off_foot'] == 'right':
 #            fig.suptitle("Foot 1 = Right Foot")

        # axis labels 
        axs.set_ylim([-1, 0])
        axs.set_xlabel('Time (Frames)', fontsize = 11)
        axs.set_ylabel('Landmark Y Position (Pose)', fontsize = 11)
       # axs.set_xlim([row['foot_1_toe_off_a'] - 25, row['foot_1_heel_strike_b'] + 25]) 
        axs.tick_params(labelsize=10) 
        
        # legend 
        axs.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 10)
        events_legend = axs.get_legend()
        events_legend.texts[0].set_text("Right Ankle")
        events_legend.texts[1].set_text("Left Ankle")
        events_legend.texts[2].set_text("Foot 1: TO")
        events_legend.texts[3].set_text("Foot 1: HS")
        events_legend.texts[4].set_text("Foot 2: TO")
        events_legend.texts[5].set_text("Foot 2: HS")
        events_legend.texts[6].set_text("Foot 1: TO")
        events_legend.texts[7].set_text("Foot 1: HS")
        
        plt.tight_layout()
        plt.show()
        
        # save plot 
        outpath_plot = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_' + walk_num + # walk number 
                                                                     '_' + str(index) + # stride index 
                                                                     '_gait_events.png')))
        fig.savefig(outpath_plot, bbox_inches = 'tight')
        plt.close()
        plt.close(fig) 


def id_calc_support_metrics(mp_df, fps, vid_in_path, dir_out_prefix, walk_num): 
    # create and save data frame as .csv 
    output_folder = os.path.join(dir_out_prefix, '005_gait_metrics', 'support_v2')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]
    
    # smooth Y ankle and hip data  
    # r ankle 
    mp_r_ank_df = mp_df.loc[mp_df['label'] == 'right_ankle']
    mp_r_ank_df = mp_r_ank_df.copy()
    mp_r_ank_df['Y_smooth'] = mp_r_ank_df['Y_pose'].rolling(window=5, min_periods=1).mean()
    mp_r_ank_df['Y_pose_negative_smooth'] = mp_r_ank_df['Y_pose_negative'].rolling(window=5, min_periods=1).mean()
    mp_r_ank_df.set_index('frame', inplace=True)

    # l ankle 
    mp_l_ank_df = mp_df.loc[mp_df['label'] == 'left_ankle']
    mp_l_ank_df = mp_l_ank_df.copy()
    mp_l_ank_df['Y_smooth'] = mp_l_ank_df['Y_pose'].rolling(window=5, min_periods=1).mean()
    mp_l_ank_df['Y_pose_negative_smooth'] = mp_l_ank_df['Y_pose_negative'].rolling(window=5, min_periods=1).mean()
    mp_l_ank_df.set_index('frame', inplace=True) 

    # r hip 
    mp_r_hip_df = mp_df.loc[mp_df['label'] == 'right_hip']
    mp_r_hip_df = mp_r_hip_df.copy()
    mp_r_hip_df['Y_smooth'] = mp_r_hip_df['Y_pose'].rolling(window=5, min_periods=1).mean()
    mp_r_hip_df['Y_pose_negative_smooth'] = mp_r_hip_df['Y_pose_negative'].rolling(window=5, min_periods=1).mean()
    mp_r_hip_df.set_index('frame', inplace=True)

    # l hhip 
    mp_l_hip_df = mp_df.loc[mp_df['label'] == 'left_hip']
    mp_l_hip_df = mp_l_hip_df.copy()
    mp_l_hip_df['Y_smooth'] = mp_l_hip_df['Y_pose'].rolling(window=5, min_periods=1).mean()
    mp_l_hip_df['Y_pose_negative_smooth'] = mp_l_hip_df['Y_pose_negative'].rolling(window=5, min_periods=1).mean()
    mp_l_hip_df.set_index('frame', inplace=True)

    # calculate the difference between hip and ankle at each frame 
    r_hip_ank_diff = abs(mp_r_hip_df['Y_pose'] - mp_r_ank_df['Y_pose'])
    l_hip_ank_diff = abs(mp_l_hip_df['Y_pose'] - mp_l_ank_df['Y_pose'])

    diff_df = pd.DataFrame({'r_diff' : r_hip_ank_diff,
                            'l_diff' : l_hip_ank_diff}) 

    diff_df = diff_df.reset_index()

    # ----------------------------------------------------
    # find peaks in hip to ankle distance = ~heel strike 
    r_diff_peaks_i, _ = sig.find_peaks(diff_df['r_diff'], distance = 5)
    l_diff_peaks_i, _ = sig.find_peaks(diff_df['l_diff'], distance = 5)

    # make into data frame - selecting frame 
    r_diff_peaks_df = pd.DataFrame(data = {'frame' : diff_df.iloc[r_diff_peaks_i]['frame'],
                                           'peak_diff_val' : diff_df.iloc[r_diff_peaks_i]['r_diff']
                                           })

    l_diff_peaks_df = pd.DataFrame(data = {'frame' : diff_df.iloc[l_diff_peaks_i]['frame'],
                                           'peak_diff_val' : diff_df.iloc[l_diff_peaks_i]['l_diff']
                                           })


    # ------------------------------------
    # identify frames when right and left mp ankle values cross + .10 seconds ~toe off 
    #at  what frames right and left mp ankle values cross? 
    ankle_y_df = pd.DataFrame(data = {'r_ankle_neg_smooth_y' : mp_r_ank_df['Y_pose_negative_smooth'], 
                                      'l_ankle_neg_smooth_y' : mp_l_ank_df['Y_pose_negative_smooth']})

    # mean of 
    ankle_y_df['r_l_neg_y_mean'] =  ankle_y_df[['r_ankle_neg_smooth_y', 
                                                'l_ankle_neg_smooth_y']].mean(axis=1)

    # Create a new column to identify whether col1 is greater than col2
    ankle_y_df['r_greater'] = ankle_y_df['r_ankle_neg_smooth_y'] > ankle_y_df['l_ankle_neg_smooth_y']

    # Find the places where the value of r_greater changes
    ankle_y_df['change'] = ankle_y_df['r_greater'].ne(ankle_y_df['r_greater'].shift())

    # save df with only y crossing frames 
    ank_y_cross = ankle_y_df.loc[ankle_y_df['change'] == True]

    # reset index so frame is a colun 
    ank_y_cross = ank_y_cross.reset_index()
    ankle_y_df = ankle_y_df.reset_index() 

    #  convert to seconds 
    ank_y_cross['sec_diff'] = (ank_y_cross['frame'].diff())/fps
    # sec_diff = change in y position in next row 
    ank_y_cross['sec_diff'] = ank_y_cross['sec_diff'].shift(-1) 
    # y cross + tenth of a second 
        # why - heel starts to lift and cross slightly before true toe off (I think) 
    ank_y_cross['frame_tenth'] = ank_y_cross['frame'] + round(fps * .10)

    # separate into right and left dataframes 
    r_ank_y_cross = ank_y_cross.loc[ank_y_cross['r_greater'] == True]
    l_ank_y_cross = ank_y_cross.loc[ank_y_cross['r_greater'] == False]

    # ------------------------------------------------------------------------
    # save frames of each gait event for each stride (each row = stride) 
    # only calculate for rows with reasonable step time diff (between crosses) 
    ank_y_cross = ank_y_cross.loc[(ank_y_cross['sec_diff'] < 1) & (ank_y_cross['sec_diff'] > .1)]

    # blank gait events to populate 
    all_gait_events = [] 

    # iterate through each row of y cross df
    for index, row in ank_y_cross.iterrows(): 
    
        # if first event is right foot toe off 
        if row['r_greater'] == True: 
            first_toe_off_foot = 'right'
            to_df_1 = r_ank_y_cross
            hs_df_1 = r_diff_peaks_df
            to_df_2 = l_ank_y_cross
            hs_df_2 = l_diff_peaks_df

        # if first event is left toe off 
        elif row['r_greater'] == False: 
            first_toe_off_foot = 'left'
            to_df_1 = l_ank_y_cross
            hs_df_1 = l_diff_peaks_df
            to_df_2 = r_ank_y_cross
            hs_df_2 = r_diff_peaks_df

        # toe offs ------------------------
        # foot 1  toe off 1 = first y cross 
        toe_off_1a = row['frame_tenth']

        # foot 2 toe off 1 = next y cross 
        to_2_rows = to_df_2.loc[to_df_2['frame_tenth'] > toe_off_1a]
        if len(to_2_rows) > 0: 
            toe_off_2 = to_2_rows['frame_tenth'].iloc[0]
        else:
            toe_off_2 = None

        # foot 1 toe off # 2 
        to_1b_rows = to_df_1.loc[to_df_1['frame_tenth'] > toe_off_1a] 
        if len(to_1b_rows) > 0: 
            toe_off_1b = to_1b_rows['frame_tenth'].iloc[0]
        else: 
            toe_off_1b = None
            
        # -----------------------------------------
        
        # Heel strikes - between toe offs and at least 5 frames after previous toe off 

        # if none of the toe off values = none, calculate heel strikes 
        if (toe_off_1a is None) or (toe_off_2 is None) or (toe_off_1b is None) == False: 
            
            # foot 1 heel strike #1 
            hs_1a_rows = hs_df_1.loc[(hs_df_1['frame'] > toe_off_1a + 5) & (hs_df_1['frame'] <= toe_off_2)]
            if len(hs_1a_rows) > 0: 
                heel_strike_1a = hs_1a_rows['frame'].iloc[0]
            else:
                heel_strike_1a = None

            # foot 2 heel strike 
            hs_2_rows = hs_df_2.loc[(hs_df_2['frame'] > toe_off_2 + 5) & (hs_df_2['frame'] <= toe_off_1b)] 
            if len(hs_2_rows) > 0: 
                heel_strike_2 = hs_2_rows['frame'].iloc[0]
            else: 
                heel_strike_2 = None

            # foot 1 heel strike #2 
            hs_1b_rows = hs_df_1.loc[hs_df_1['frame'] > toe_off_1b + 5] 
            if len(hs_1b_rows) > 0:
                heel_strike_1b = hs_1b_rows['frame'].iloc[0]
            else: 
                heel_strike_1b = None

        # if any of the toe offs = none, all heel strikes = none 
        else: 
            heel_strike_1a = None 
            heel_strike_2 = None 
            heel_strike_1b = None 

        # -----------------------------
        
        # combine and save 
        current_gait_events = pd.DataFrame(data = {'y_cross_row_index' : [index],
                                                   'first_toe_off_foot' : [first_toe_off_foot],
                                                   'foot_1_toe_off_a' : [toe_off_1a], 
                                                   'foot_1_heel_strike_a' : [heel_strike_1a], 
                                                   'foot_2_toe_off' : [toe_off_2],
                                                   'foot_2_heel_strike' : [heel_strike_2], 
                                                   'foot_1_toe_off_b' : [toe_off_1b], 
                                                   'foot_1_heel_strike_b' : [heel_strike_1b]
                                                  }) 
#        print('gait events')
#        print(current_gait_events)

        all_gait_events.append(current_gait_events) 

    # if no strides identified, save empty data frame 
    if len(all_gait_events) == 0: 
        print('no strides identfied') 
        all_gait_events_df = pd.DataFrame(columns = ['y_cross_row_index',
                                                     'first_toe_off_foot',	
                                                     'foot_1_toe_off_a',	
                                                     'foot_1_heel_strike_a',
                                                     'foot_2_toe_off',
                                                     'foot_2_heel_strike',
                                                     'foot_1_toe_off_b',
                                                     'foot_1_heel_strike_b'], 
                                          index = [walk_num]) 
        
    # if strides found, 
    # # concatenate all strides into single data frame and drop None 
    else: 
        all_gait_events_df = pd.concat(all_gait_events)
        all_gait_events_df = all_gait_events_df.reset_index(drop = True)
        all_gait_events_df = all_gait_events_df.dropna()

    # Plot events per stide 
    plot_events_per_stride(all_gait_events_df, mp_r_ank_df, mp_l_ank_df, output_folder, vid_in_path_no_ext, walk_num)
    
    # ---------------------------------------------------------
    # calculate metrics 
    # frame diff columns 
    all_gait_events_df['frameDiff_to1a_hs1a'] = all_gait_events_df['foot_1_heel_strike_a'] - all_gait_events_df['foot_1_toe_off_a']
    all_gait_events_df['frameDiff_hs1a_to2'] = all_gait_events_df['foot_2_toe_off'] - all_gait_events_df['foot_1_heel_strike_a']
    all_gait_events_df['frameDiff_to2_hs2'] = all_gait_events_df['foot_2_heel_strike'] - all_gait_events_df['foot_2_toe_off']
    all_gait_events_df['frameDiff_hs2_to1b'] = all_gait_events_df['foot_1_toe_off_b'] - all_gait_events_df['foot_2_heel_strike']
    all_gait_events_df['frameDiff_to1b_hs1b'] = all_gait_events_df['foot_1_heel_strike_b'] - all_gait_events_df['foot_1_toe_off_b']

    # gait cycle time = first contact of one foot the the following first contact of the same foot 
    all_gait_events_df['gait_cycle_time_sec'] = (all_gait_events_df['foot_1_heel_strike_b'] - all_gait_events_df['foot_1_heel_strike_a']) / fps

    # stance time = time foot 1 is in contact with the ground 
    all_gait_events_df['stance_time_sec'] = (all_gait_events_df['foot_1_toe_off_b'] - all_gait_events_df['foot_1_heel_strike_a']) / fps 
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
    all_gait_events_df['term_dsupport_sec'] = (all_gait_events_df['foot_1_toe_off_b'] - all_gait_events_df['foot_2_heel_strike']) / fps
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




