#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
import os 


# In[2]:


# import mediapipe df from csv 
# yolo doesn't have visibility confidence score 
#mp_all_filepath = r'..\temp\test_sandbox_pipeline_outputs\002_frames_to_time\DS_HC_gait_vertical_left_mediapipe_all.csv'
#mp_all_df = pd.read_csv(mp_all_filepath, index_col = 0)

# output folder
#output_parent_folder = r'..\temp\test_sandbox_pipeline_outputs'


# In[4]:


# plot visibility mediapipe 
def vis_all_labels_boxplot(mp_all_df, mp_all_filepath): 
    
    # save basename for plot title 
    mp_basename = os.path.basename(mp_all_filepath)

    # plot 
    plt.clf()
    fig1, ax = plt.subplots()
    ax = sns.boxplot(data=mp_all_df, x = 'label', y = 'vis')
    plt.xticks(rotation=90)
    plt.title('All Labels Visibility: ' + mp_basename)

    return(fig1)


# In[5]:


# save boxplot  
def save_visibility_boxplot(boxplot, mp_all_filepath, output_parent_folder): 
    output_folder = os.path.join(output_parent_folder, '003_landmark_visibility')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # create file name 
    output_file_1 = os.path.normpath(os.path.join(output_folder, 'mp_visibility_boxplot.png'))

    # save figure 
    boxplot.savefig(output_file_1, bbox_inches = 'tight')
    plt.close(boxplot)
    plt.close()
    


# In[6]:


# plot visibility - line plot 
def vis_lineplot(mp_all_df, mp_all_filepath): 
    # save basename for plot title 
    mp_basename = os.path.basename(mp_all_filepath)
    
    # change label to string for future filtering 
    mp_all_df['label'] = mp_all_df['label'].astype(str)

    # set labels to plot and filter data frame by label column
    labels_to_plot = ['left_foot_index', 'right_foot_index',
                      'left_heel', 'right_heel',
                      'left_ankle', 'right_ankle',
                      'left_knee', 'right_knee', 
                      'left_hip', 'right_hip']

    mp_all_filt_df = mp_all_df[mp_all_df['label'].str.contains('|'.join(labels_to_plot), case=False)]
    
    # plot 
    plt.figure(figsize=(10, 6))
    fig2, ax = plt.subplots()
    ax = sns.lineplot(data=mp_all_df, x='frame', y='vis', hue='label', markers=True, dashes=False, estimator = None)
    plt.legend(loc = 'right')
    plt.title('All Labels Visibility: ' + mp_basename)

    return(fig2)


# In[7]:


# save lineplot 
def save_visibility_lineplot(lineplot, mp_all_filepath, output_parent_folder): 
    output_folder = os.path.join(output_parent_folder, '003_landmark_visibility')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # create file name 
    output_file_2 = os.path.normpath(os.path.join(output_folder, 'mp_visibility_lineplot.png'))

    # save figure 
    lineplot.savefig(output_file_2, bbox_inches = 'tight')
    plt.close(lineplot)
    plt.close()


# In[8]:


# save vis score stats as df 
def save_vis_stats_by_label(mp_all_df):
    # blank df to populate 
    vis_stats_df = pd.DataFrame(columns = ['label','mean_vis','median_vis','std_vis'])

    # save mean, median, and standard deviation of visibility score for each mediapipe landmark 
    for label in pd.unique(mp_all_df['label']):
        if label == 'nan' or pd.isna(label):
            print('label is na: skipped')
        else: 
            # filter to data frame that only includes one mediapose landmark (nose, right foot, etc)
            current_label = mp_all_df.loc[mp_all_df['label'] == label]
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


    return(vis_stats_df)
    


# In[9]:


# save vis score stats as 
def save_vis_scores_csv(vis_stats_df, mp_all_filepath, output_parent_folder):
    output_folder = os.path.join(output_parent_folder, '003_landmark_visibility')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

     # create file name 
    output_file_3 = os.path.normpath(os.path.join(output_folder, 'mp_visibility_stats.csv'))

    # save csv 
    vis_stats_df.to_csv(output_file_3)


# In[10]:


## run functions 


# In[11]:


# boxplot 
#boxplot = vis_all_labels_boxplot(mp_all_df, mp_all_filepath)
#save_visibility_boxplot(boxplot, mp_all_filepath, output_parent_folder)


# In[12]:


# lineplot 
#lineplot = vis_lineplot(mp_all_df, mp_all_filepath)
#save_visibility_lineplot(lineplot, mp_all_filepath, output_parent_folder)


# In[13]:


# calculate and save vis score per label 
#vis_stats_df = save_vis_stats_by_label(mp_all_df)
#save_vis_scores_csv(vis_stats_df, mp_all_filepath, output_parent_folder)

