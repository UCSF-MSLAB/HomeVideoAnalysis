#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# segment video into times when person is walking away from the camera, toward the camera, and turning 


# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sig 
import os 


# In[ ]:


# import functions from sandbox src 
#from frames_to_time import get_frames_per_second
#from sandbox_main_src_funs.filtering_functions.filter_single_axis import filter_landmark_single_axis


# In[ ]:


# manually input below if running one video at a time 
# long term goal - incorporate into pipeline 

# video file path 
#vid_in_path = r'..\..\tests\fixtures\all_videos\NW_HC_practice videos\NW_HC_gait_vertical_left.MOV' # vid_in_path set during process_dir() of run.py

# run.py outputs
#mp_all_filepath = r'..\..\temp\test_sandbox_pipeline_outputs\002_frames_to_time\NW_HC_gait_vertical_left_mediapipe_all_sec.csv'
#yolo_filepath = r'..\..\temp\test_sandbox_pipeline_outputs\002_frames_to_time\NW_HC_gait_vertical_left_yolo_sec.csv' 

#mp_all_df = pd.read_csv(mp_all_filepath, index_col = 0)
#yolo_df = pd.read_csv(yolo_filepath, index_col = 0)

# ground truth anotation of turn start and stop time 
# watch videos frame by frame: e on keyboard = move forward one frame 
#ground_truth_turn_frames_filepath = r'C:\Users\mmccu\Box\MM_Personal\5_Projects\BoveLab\3_Data_and_Code\poseEstimation_practice\data_example_videos\visual_annotation_ground_truth\vertical_turns_start_stop_frame.xlsx'
#ground_truth_turn_frames_df = pd.read_excel(ground_truth_turn_frames_filepath, sheet_name = 'Sheet1', engine='openpyxl')

#filter ground truth for this specific participant 
#ground_truth_turn_frames_df  = ground_truth_turn_frames_df.loc[ground_truth_turn_frames_df['filename'] == 'NW_HC_gait_vertical_left', :]
#print(ground_truth_turn_frames_df.head())

# outputs 
#output_parent_folder = r'..\..\temp\test_sandbox_pipeline_outputs'

# filtering vars 
#cutoff = 0.4  # Desired cutoff frequency of the filter in Hz
#order = 1  # Order of the filter (higher means sharper cutoff)


# In[ ]:


def filter_landmark_single_axis(df, landmark, axis_to_filter, video_fps, cutoff_hz, filter_order): 
    df_landmark = df.loc[df['label'] == landmark]
    df_landmark.index = df_landmark['frame'] # set index to frame

    # data = series, one landmark and one axis (column) 
    data = df_landmark[axis_to_filter]
       
    # Normalized cutoff frequency (cutoff frequency divided by the Nyquist frequency)
    nyquist = 0.5 * video_fps
    normal_cutoff = cutoff_hz / nyquist

    # Design a Butterworth low-pass filter
    b, a = sig.butter(filter_order, normal_cutoff, btype='low', analog=False)

    # filter data 
    filtered_data = sig.filtfilt(b, a, data)
    filtered_data = pd.Series(filtered_data)
    
    return ([data, filtered_data])


# In[ ]:


#inputs 
def segment_video_walks_turn(mp_all_df, yolo_df, fps, vid_in_path, output_parent_folder, 
                             cutoff, order, find_peaks_distance, find_peaks_prominence, flattening_point_atol, 
                             dist_turn_mid_to_flattening): 
    # -----------------------------------------------------------------------------
    # use hip z position to ID start, stop, and midpoint of turns in vertical videos 

    # filter right and left hip z pose data 
    [hip_r_mp_z, hip_r_mp_z_filt] = filter_landmark_single_axis(df = mp_all_df, 
                                                                landmark = 'right_hip',
                                                                axis_to_filter = 'Z_pose', 
                                                                video_fps = fps,
                                                                cutoff_hz = cutoff, 
                                                                filter_order = order)

    [hip_l_mp_z, hip_l_mp_z_filt] = filter_landmark_single_axis(df = mp_all_df,
                                                                landmark = 'left_hip', 
                                                                axis_to_filter = 'Z_pose', 
                                                                video_fps = fps, 
                                                                cutoff_hz = cutoff, 
                                                                filter_order = order)
    # frames for hip vars 
    hip_l_mp_z_frames = hip_l_mp_z.index
    hip_r_mp_z_frames = hip_r_mp_z.index

    # distance between l and r z and smooth
    hip_z_diff_mp_filt = hip_l_mp_z_filt - hip_r_mp_z_filt
    hip_z_diff_mp_filt = pd.Series(hip_z_diff_mp_filt).rolling(window=15, min_periods=1).mean()
    hip_z_diff_mp_filt.index = hip_l_mp_z_frames

    # find max and min of hip distance filtered 
    # max and min = frame of midpoint of turn 
    hip_z_diff_mp_filt_peak_frames, _ = sig.find_peaks(hip_z_diff_mp_filt, distance = find_peaks_distance, prominence = (find_peaks_prominence, None))
    hip_z_diff_mp_filt_peak_frames = hip_z_diff_mp_filt.index[hip_z_diff_mp_filt_peak_frames] # set to index, accounts for missing data where frame doesn't equal row index
    hip_z_diff_mp_filt_valley_frames, _ = sig.find_peaks(-hip_z_diff_mp_filt, distance = find_peaks_distance, prominence = (find_peaks_prominence, None))
    hip_z_diff_mp_filt_valley_frames = hip_z_diff_mp_filt.index[hip_z_diff_mp_filt_valley_frames]

    # merge together peaks and valleys of hip z diff df -> frames of each turn, ordered 
    hip_z_diff_mp_filt_turn_midpoints = np.concatenate((hip_z_diff_mp_filt_peak_frames, hip_z_diff_mp_filt_valley_frames), axis = None)
    hip_z_diff_mp_filt_turn_midpoints = np.sort(hip_z_diff_mp_filt_turn_midpoints)

    # rate of change of z hip distance 
    hip_z_diff_mp_filt_gradient = np.gradient(hip_z_diff_mp_filt)
    # make series and set index 
    hip_z_diff_mp_filt_gradient = pd.Series(hip_z_diff_mp_filt_gradient)
    hip_z_diff_mp_filt_gradient.index = hip_l_mp_z_frames

    # Identify where the slope is within absolute tolerance value (atol) away from zero 
    flattening_points = np.where(np.isclose(hip_z_diff_mp_filt_gradient, 0, atol=flattening_point_atol))[0]
    flattening_points = hip_z_diff_mp_filt_gradient.index[flattening_points]


    # Find first flattening point prior to turn midpoint
    turn_start_frames = np.array([], dtype='int16')
    for midpoint_i, current_midpoint in enumerate(hip_z_diff_mp_filt_turn_midpoints):
        # flattening points that are before current midpoint and at least 10 frames away from midpoint (exclude midpoint itself)
        before_peak_flattening_all = flattening_points[(flattening_points < current_midpoint) & (abs(current_midpoint - flattening_points) >= dist_turn_mid_to_flattening)]
        # select last element (closest to turn midpoint)
        before_peak_flattening_last = before_peak_flattening_all[-1]
        # save 
        turn_start_frames = np.append(turn_start_frames, before_peak_flattening_last)

    #Find first flattening point after hip midpoint 
    turn_stop_frames = np.array([], dtype='int16')
    for midpoint_i, current_midpoint in enumerate(hip_z_diff_mp_filt_turn_midpoints):
        # flattening points that are after current midpoint and at least 10 frames away from midpoint (exclude midpoint itself)
        after_peak_flattening_all = flattening_points[(flattening_points > current_midpoint) & (abs(current_midpoint - flattening_points) >= dist_turn_mid_to_flattening)]
        # select first element (closest to turn midpoint)
        after_peak_flattening_first = after_peak_flattening_all[0]
        # save 
        turn_stop_frames = np.append(turn_stop_frames, after_peak_flattening_first)

    # save all turn info as one df 
    turn_data = {'turn_num' : np.arange(0, len(hip_z_diff_mp_filt_turn_midpoints), step = 1),
                 'turn_start_frame' : turn_start_frames, 
                 'turn_midpoint' : hip_z_diff_mp_filt_turn_midpoints,
                 'turn_stop_frame' : turn_stop_frames,
                 'turn_time_frames' : turn_stop_frames - turn_start_frames, 
                 'turn_time_seconds' : (turn_stop_frames - turn_start_frames) / fps
                } 

    turn_df = pd.DataFrame(turn_data)
    # -----------------------------------------------------------------------------
    # Use distance between shoulders in pixels (yolo) to determine direction subject is moving 
    # shoulder width increasing = walking toward camera  
    #  shoulder width decreasing = walking away from camera
    # use start and stop of turns from hip z distance to ID walking times

    # create one df for r shoulder, one for l 
    shoulder_r_yolo_df = yolo_df.loc[(yolo_df['label'] == 'right_shoulder')]
    shoulder_r_yolo_df.index = shoulder_r_yolo_df['frame']

    shoulder_l_yolo_df = yolo_df.loc[(yolo_df['label'] == 'left_shoulder')]
    shoulder_l_yolo_df.index = shoulder_l_yolo_df['frame']

    # shoulder width 
    shoulder_width_yolo = abs(shoulder_r_yolo_df['X'] - shoulder_l_yolo_df['X'])
    shoulder_width_yolo_smooth = pd.Series(shoulder_width_yolo).rolling(window=15, min_periods=1).mean()

    # frames 
    frames = shoulder_r_yolo_df['frame']
    # walk start - start one second in to account for time for model to fit to person
        # start of entire video 
    first_walk_start_frame = frames[0] 
    # end of last walk 
    last_walk_end_frame = frames.iloc[-1]

    # create walk_df with start and stop of eaach walk, time per walk, and direction 
    walks_df = pd.DataFrame(index=range(len(turn_df) + 1), 
                            columns = ['walk_num', 
                                       'walk_start_frame', 
                                       'walk_end_frame', 
                                       'walk_time_frames', 
                                       'walk_time_turns', 
                                       'walk_direction'])

    number_of_walks = np.arange(0, len(walks_df), step = 1)

    for current_walk_num in number_of_walks: 
        # walk_num
        walks_df.iloc[current_walk_num, 0] = current_walk_num
    
        #walk_start_frame 
        # if walk 1 - start = first_walk_start_frame
        # all other walks = walk start = end of previous turn 
        if current_walk_num == 0:
            current_walk_start = first_walk_start_frame
        else:   
            turn_stop_frame = turn_df['turn_stop_frame'] 
            current_walk_start = turn_stop_frame[current_walk_num - 1]

        walks_df.iloc[current_walk_num, 1] = current_walk_start

        # walk end frame 
        # if current walk is the last walk, stop frame = last walk stop 
        if current_walk_num == max(number_of_walks): 
             current_walk_stop = last_walk_end_frame
        else:
            turn_start_frame = turn_df['turn_start_frame'] 
            current_walk_stop = turn_start_frame[current_walk_num]
        
        walks_df.iloc[current_walk_num, 2] = current_walk_stop

        # walk_time_frames 
        walks_df.iloc[current_walk_num, 3] = current_walk_stop - current_walk_start

        # walk_time_seconds 
        walks_df.iloc[current_walk_num, 4] = (current_walk_stop - current_walk_start) / fps

        # walk direction 
        # if shoulder width is bigger at walk stop than walk start, person is moving toward camera 
        if (shoulder_width_yolo_smooth[current_walk_stop] - shoulder_width_yolo_smooth[current_walk_start]) > 0: 
            walks_df.iloc[current_walk_num, 5] = 'toward'
        elif (shoulder_width_yolo_smooth[current_walk_stop] - shoulder_width_yolo_smooth[current_walk_start]) < 0:
            walks_df.iloc[current_walk_num, 5] = 'away'

    ## plots  -------------------------------------------
    # plot #1 - hip and shoulder positions 
    fig1, (ax1, ax2) = plt.subplots(2)
    fig1.suptitle(os.path.splitext(os.path.basename(vid_in_path))[0])

    # subplot 1 - mp z for each hip 
    ax1.plot(hip_r_mp_z_frames, hip_r_mp_z_filt, label = 'r_hip_z_filt', color = 'blue')
    ax1.plot(hip_l_mp_z_frames, hip_l_mp_z_filt, label = 'l_hip_z_filt', color = 'red')
    ax1.set_ylabel('MP Pose')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # subplot 2 - yolo x for each shoulder 
    ax2.plot(shoulder_r_yolo_df['frame'], shoulder_r_yolo_df['X'], label = 'r_shoulder_x', color = 'orange')
    ax2.plot(shoulder_r_yolo_df['frame'], shoulder_l_yolo_df['X'], label = 'l_shoulder_x', color = 'green')
    ax2.set_ylabel('Yolo Pixels')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig1.tight_layout()  # avoid plot overlap

    # plot 2 - hip and shoulder dist 
    # set plot with two subplots 
    fig2, (ax1, ax2) = plt.subplots(2)
    fig2.suptitle(os.path.splitext(os.path.basename(vid_in_path))[0])

    # subplot 1 - distance between right and left hip, use peaks and mins as turns 
    ax1.set_title('Turns')
    ax1.plot(hip_z_diff_mp_filt, label = 'l_hip_z_filt - r_hip_z_filt', color = 'black')
    ax1.vlines(x = turn_start_frames, ymin = -1, ymax = 1, color = 'green', alpha = 0.5, label = 'turn_start_calculated')
    ax1.vlines(x=turn_df['turn_midpoint'], ymin = -1, ymax = 1, color = 'yellow',  alpha = 0.5, label = 'turn_midpoint calculated')
    ax1.vlines(x = turn_stop_frames, ymin = -1, ymax = 1, color = 'red', alpha = 0.5,  label = 'turn_stop_calculated')
    ax1.set_ylabel('MP Pose')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # sublot 2 - yolo shoulder width 
    ax2.set_title('Walks')
    ax2.plot(shoulder_r_yolo_df['frame'], shoulder_width_yolo_smooth, label = "shoulder_width, abs, smooth", color = 'black') 
    ax2.vlines(x = walks_df['walk_start_frame'], ymin = 0, ymax = 1000, color = 'green', alpha = 0.5, label = 'walk_start_calculated')
    ax2.vlines(x = walks_df['walk_end_frame'], ymin = 0, ymax = 1000, color = 'red', alpha = 0.5,  label = 'walk_stop_calculated')
    ax2.vlines(x = first_walk_start_frame, ymin = 0, ymax = 1000,  color = 'green', linestyle = '--', label = 'first_walk_start_frame')
    ax2.vlines(x = last_walk_end_frame, ymin = 0, ymax = 1000, color = 'red', linestyle = '--', label = 'last_walk_end_frame')
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Yolo Pixels')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig2.tight_layout()  # avoid plot overlap

    # save outputs ------------  
    output_folder = os.path.join(output_parent_folder, '004_segment_towards_away_turn')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]

    # save plots 
    # plot 1 
    output_plot_1 = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_hip_z_mp_shoulder_x_yolo.png')))
    fig1.savefig(output_plot_1, bbox_inches = 'tight')

    # plot 2 
    output_plot_2 = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_turn_walk_start_stop.png')))
    fig2.savefig(output_plot_2, bbox_inches = 'tight')

    # save turn and walk data frames 
    turn_df_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_turn_start_stop_frames.csv')))
    turn_df.to_csv(turn_df_path)

    walk_df_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_walk_start_stop_frames.csv'))) 
    walks_df.to_csv(walk_df_path)

    # update mp_all_df and yolo_df with turns and 
    
    # outputs 
    return([turn_df, walks_df])






