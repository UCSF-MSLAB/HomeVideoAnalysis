
import pandas as pd 
import os 
 
def save_all_pose_metrics(id_date, vid_in_path, task, turn_df, cadence_df, stride_time_stats_df, stride_width_stats_df, support_metrics_df, output_parent_folder): 

    # video name 
    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]
    
    # test saving summary data frame 
        # can extract more from existing data frame, just for ACTRIMS 

    all_metrics_df = pd.DataFrame(data = {'id_date' : [id_date],
                                          'video_id_date_name' : [vid_in_path_no_ext], 
                                          'task' : [task],
                                          'turn_time_mean_sec' : [turn_df['turn_time_seconds'].mean(skipna = True)], 
                                          'turn_time_median_sec' : [turn_df['turn_time_seconds'].median(skipna = True)],
                                          'turn_time_sd' : [turn_df['turn_time_seconds'].std(skipna = True)],
                                          'turn_time_cv' :    [(turn_df['turn_time_seconds'].std(skipna = True) / turn_df['turn_time_seconds'].mean(skipna = True)) * 100],
                                          'cadence_step_per_min' : [cadence_df]
                                         })

    # pivot all stride time data to one row 
    stride_time_df_unstacked = stride_time_stats_df.unstack().to_frame().T 
    stride_time_df_unstacked.columns = [f'{col[0]}_{col[1]}' for col in stride_time_df_unstacked.columns] 
    stride_time_df_unstacked.columns = ['stride_time_' + col for col in stride_time_df_unstacked.columns] 
    stride_time_df_unstacked = stride_time_df_unstacked.astype(float)\
    
    # merge stats into one data frame 
    final_metrics_df = pd.concat([all_metrics_df, stride_time_df_unstacked, stride_width_stats_df, support_metrics_df], axis=1)
    final_metrics_df.columns = [col + '_pose' for col in final_metrics_df.columns]
    final_metrics_df = final_metrics_df.round(3)
    
    # save . csv with metrics 
    output_folder = os.path.join(output_parent_folder, '005_gait_metrics')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_metrics_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_pose_metrics.csv')))
    final_metrics_df.to_csv(all_metrics_path)

    return(final_metrics_df)