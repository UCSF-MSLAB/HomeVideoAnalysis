
import pandas as pd 
import numpy as np
import os 
 
def save_all_pose_metrics(vid_in_path, valid_segments_all, stride_time_stats_all, cadence_mean_df, stride_time_stats_df, stride_width_stats_all, support_metrics_all, output_parent_folder): 

    # video name 
    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]

    # save info about linear walking segments - #, mean time, etc 
    total_segments = len(valid_segments_all)

    all_segment_duration = []
    for segment_i, current_segment in enumerate(valid_segments_all):
        start_sec = current_segment['time_seconds'].iloc[0]
        end_sec = current_segment['time_seconds'].iloc[-1]
        current_segment_duration = end_sec - start_sec
        all_segment_duration.append(current_segment_duration)

    all_segment_duration_mean = np.mean(all_segment_duration)
    all_segment_duration_median = np.median(all_segment_duration)

    walking_segment_info_df = pd.DataFrame(data = {'walking_segmets_n' : [total_segments],
                                                   'walking_segments_duration_mean': [round(all_segment_duration_mean, 2)],
                                                   'walking_segments_duration_median' : [round(all_segment_duration_median, 2)]}) 

    
    # test saving summary data frame 
    all_metrics_df = pd.DataFrame(data = {'video_id_date_name' : [vid_in_path_no_ext]})

    
    # merge stats into one data frame 
    all_metrics_df = pd.concat([all_metrics_df, walking_segment_info_df, stride_time_stats_all, cadence_mean_df, stride_width_stats_all, support_metrics_all],
                               axis = 1)
    
    
    all_metrics_df.columns = [col + '_pose' for col in all_metrics_df.columns]
    all_metrics_df = all_metrics_df.round(3)
    
    # save . csv with metrics 
    #output_folder = os.path.join(output_parent_folder, '005_gait_metrics')
    #if not os.path.exists(output_folder):
        #os.makedirs(output_folder)

    #all_metrics_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_pose_metrics.csv')))
    #all_metrics_df.to_csv(all_metrics_path)

    return(all_metrics_df)