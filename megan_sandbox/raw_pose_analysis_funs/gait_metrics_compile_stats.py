
import pandas as pd 
import os 
 
def save_all_pose_metrics(id_date, vid_in_path, task, stride_time_stats_all, cadence_mean_df, stride_time_stats_df, stride_width_stats_all, support_metrics_all, output_parent_folder): 

    # video name 
    vid_in_path_no_ext = os.path.splitext(os.path.basename(vid_in_path))[0]
    
    # test saving summary data frame 
        # can extract more from existing data frame, just for ACTRIMS 

    all_metrics_df = pd.DataFrame(data = {'id_date' : [id_date],
                                          'video_id_date_name' : [vid_in_path_no_ext], 
                                          'task' : [task]})

    
    # merge stats into one data frame 
    all_metrics_df = pd.concat([all_metrics_df, stride_time_stats_all, cadence_mean_df, stride_width_stats_all, support_metrics_all], axis = 1)
    
    
    all_metrics_df.columns = [col + '_pose' for col in all_metrics_df.columns]
    all_metrics_df = all_metrics_df.round(3)
    
    # save . csv with metrics 
    #output_folder = os.path.join(output_parent_folder, '005_gait_metrics')
    #if not os.path.exists(output_folder):
        #os.makedirs(output_folder)

    #all_metrics_path = os.path.normpath(os.path.join(output_folder, (vid_in_path_no_ext + '_pose_metrics.csv')))
    #all_metrics_df.to_csv(all_metrics_path)

    return(all_metrics_df)