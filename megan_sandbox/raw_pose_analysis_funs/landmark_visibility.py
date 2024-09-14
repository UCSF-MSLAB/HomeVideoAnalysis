#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import seaborn as sns
import os 


# In[ ]:


# import mediapipe df from csv 
# yolo doesn't have visibility confidence score 
#mp_all_filepath = r'..\..\temp\test_sandbox_pipeline_outputs\002_frames_to_time\DS_HC_gait_vertical_left_mediapipe_all.csv'
#mp_all_df = pd.read_csv(mp_all_filepath, index_col = 0)

# path to video 
#vid_in_path = r'..\..\tests\fixtures\all_videos\DS_HC_practice videos\DS_HC_gait_vertical_left.mov'

# output folder
#output_parent_folder = r'..\temp\test_sandbox_pipeline_outputs'


# In[ ]:


# plot visibility mediapipe 
def mp_vis_all_labels_boxplot(mp_all_df, vid_in_path, output_parent_folder): 
    
    # save basename for plot title 
    vid_in_path_basename = os.path.basename(vid_in_path) 

    # plot 
    fig1, ax1 = plt.subplots()
    fig1.suptitle('All Labels Visibility: ' + vid_in_path_basename)
    ax1.xaxis.set_tick_params(rotation=90, labelsize=10)
    ax1.set_ylim([-.05, 1.05])

    # boxplot for each landmark label 
    for current_landmark_label in pd.unique(mp_all_df['label']):
        if current_landmark_label == 'no_labels_tracked' or pd.isna(current_landmark_label):
            print('label is na: skipped')
        else: 
            # filter to data frame that only includes one media pose landmark (nose, right foot, etc)
            current_label_df = mp_all_df.loc[(mp_all_df['label'] == current_landmark_label) | (mp_all_df['label'] == 'no_labels_tracked')]
            # reset all no_labels_tracked to current label to keep 0 vis scores in plot
            current_label_df.loc[:, 'label'] = current_landmark_label 
            ax1 = sns.boxplot(data = current_label_df, x='label', y='vis', color = 'grey')

    # save plot 
    output_folder = os.path.join(output_parent_folder, '003_landmark_visibility')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # create file name 
    input_file_no_ext_1 = os.path.splitext(os.path.basename(vid_in_path))[0]
    output_file_1 = os.path.normpath(os.path.join(output_folder, input_file_no_ext_1 +'_mp_visibility_boxplot.png'))

    # save figure 
    fig1.savefig(output_file_1, bbox_inches = 'tight')
    plt.close(fig1)
    plt.close()

    return(fig1)


# In[ ]:


# plot mediapipe visibility - line plot 
def mp_vis_lineplot(mp_all_df, vid_in_path, output_parent_folder): 
    # save basename for plot title 
    vid_in_path_basename = os.path.basename(vid_in_path)
    
    # change label to string for future filtering 
    mp_all_df['label'] = mp_all_df['label'].astype(str)

    # set labels to plot and filter data frame by label column
    labels_to_plot = ['left_foot_index', 'right_foot_index',
                      'left_heel', 'right_heel',
                      'left_ankle', 'right_ankle',
                      'left_knee', 'right_knee', 
                      'left_hip', 'right_hip'
                     ]

    # plot 
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.suptitle('Visibility by Frame: ' + vid_in_path_basename)
    
    # test new plot 
    #ax2 = sns.lineplot(data=mp_all_filt_df, x='frame', y='vis', hue='label', markers=True, dashes=False, estimator = None)
    for current_landmark_label in labels_to_plot:
        if current_landmark_label == 'nan' or pd.isna(current_landmark_label):
            print('label is na: skipped')
        else: 
            # filter to data frame that only includes one media pose landmark (nose, right foot, etc)
            current_label_df = mp_all_df.loc[(mp_all_df['label'] == current_landmark_label) | (mp_all_df['label'] == 'no_labels_tracked')]
            ax2 = sns.lineplot(data = current_label_df, x='frame', y='vis', label = current_landmark_label)
    
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_ylim([-.05, 1.05])
    
    # save .png
    output_folder = os.path.join(output_parent_folder, '003_landmark_visibility')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # create file name 
    input_file_no_ext_2 = os.path.splitext(os.path.basename(vid_in_path))[0]
    output_file_2 = os.path.normpath(os.path.join(output_folder, input_file_no_ext_2 +'_mp_visibility_lineplot.png'))

    # save figure 
    fig2.savefig(output_file_2, bbox_inches = 'tight')
    plt.close(fig2)
    plt.close()

    return([fig2])


# In[ ]:


# save mediapipe vis score stats as df 
def mp_save_vis_stats_by_label(mp_all_df, vid_in_path, output_parent_folder):
    # blank df to populate 
    vis_stats_df = pd.DataFrame(columns = ['label','mean_vis','median_vis','std_vis'])

    # save mean, median, and standard deviation of visibility score for each mediapipe landmark 
    for label in pd.unique(mp_all_df['label']):
        if label == 'no_labels_tracked' or pd.isna(label):
            print('skip: no_labels_tracked, included in each label')
        else: 
            # filter to data frame that only includes one media pose landmark (nose, right foot, etc)
            # label + no_labels tracked = all frames, including frames where no labels were tracked 
            current_label = mp_all_df.loc[(mp_all_df['label'] == label) | (mp_all_df['label'] == 'no_labels_tracked')]
            #mean, median, vis
            current_vis_stats_row = pd.DataFrame(data = {'label': [label],
                                                         'mean_vis': current_label['vis'].mean(),
                                                         'median_vis': current_label['vis'].median(),
                                                         'std_vis': current_label['vis'].std()}
                                                 )
            # concatanate
            vis_stats_df = pd.concat([vis_stats_df, current_vis_stats_row])
            # drop rows with all missing data 
            vis_stats_df = vis_stats_df.dropna(how='all')
            
    
    # save .csv 
    output_folder = os.path.join(output_parent_folder, '003_landmark_visibility')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

     # create file name 
    input_file_no_ext_3 = os.path.splitext(os.path.basename(vid_in_path))[0]
    output_file_3 = os.path.normpath(os.path.join(output_folder, input_file_no_ext_3 + '_mp_visibility_stats.csv'))

    # save csv 
    vis_stats_df.to_csv(output_file_3)


    return(vis_stats_df)
    

# yolo visibility ---------------------------------

def yolo_vis_lineplot(yolo_df, vid_in_path, output_parent_folder): 
    
    # add numberic value for landmark visibilty 
    yolo_df['landmark_visible_num'] = np.where((yolo_df['landmark_visible'] == 'yes'), 1, 0)

    # set labels to plot and filter data frame by label column
    labels_to_plot = ['left_ankle', 'right_ankle',
                      'left_knee', 'right_knee', 
                      'left_hip', 'right_hip', 
                      'right_shoulder', 'left_shoulder'
                     ]

    # one plot per label 
    for current_landmark_label in labels_to_plot: 
        current_yolo_df = yolo_df.loc[yolo_df['label'] == current_landmark_label]
    
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        fig3.suptitle(current_landmark_label)
        ax3.scatter(current_yolo_df['frame'], current_yolo_df['landmark_visible_num'], marker = '.')
        ax3.set_ylabel('Yolo Marker Present: 0 = No, 1 = Yes')
        ax3.set_ylim(-.05, 1.05)

        # save .png
        output_folder = os.path.join(output_parent_folder, '003_landmark_visibility')
    
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # create file name 
        input_file_no_ext_3 = os.path.splitext(os.path.basename(vid_in_path))[0]
        output_file_3 = os.path.normpath(os.path.join(output_folder, input_file_no_ext_3 +'_yolo_visibility_' + current_landmark_label + '.png'))

        # save figure 
        fig3.savefig(output_file_3, bbox_inches = 'tight')
        plt.close(fig3)
        plt.close()


    

    
    




