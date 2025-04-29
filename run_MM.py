import os
import sys
import cv2
import pandas as pd
from src.etl_funs import (extract_pose_data,
                          transform_pose_data,
                          load_pose_data)
# import warnings
import logging
logger = logging.getLogger(__name__)


# def fxn():
#     warnings.warn("deprecated", DeprecationWarning)


# with warnings.catch_warnings(action="ignore"):
#     fxn()

ALLOWED_VID_FORMATS = ["asf", "avi", "gif", "m4v",
                       "mkv", "mov", "mp4", "mpeg",
                       "mpg", "ts", "wmv", "webm"]

MODELS = ["yolo", "mediapipe", "mediapipe_world", "marigold"]


def get_frames_per_second(vid_in_path): 
    video = cv2.VideoCapture(vid_in_path) 
    fps = video.get(cv2.CAP_PROP_FPS)
    fps = round(fps)
    fps_df = pd.DataFrame(data = {'fps' : [fps]})
    
    return([fps,fps_df])

def process_dir(dir_in_path, dir_out_path, run_depth):

    for (dir_path, dir_names, file_names) in os.walk(dir_in_path):
        for file_name in file_names:
            name, ext = os.path.splitext(file_name)
            ext = ext.lower()[1:]
            # 10/8 - no TUG, DTW, QSLOS tasks from in-person video, just PWS and FWS from camera #1
            if (ext in ALLOWED_VID_FORMATS) & ('gait_vertical' in name) & ('DTW' not in name) & ('QSLOS' not in name) & ('TUG' not in name) & ('FW_2' not in name) & ('PWS_2' not in name):
                vid_in_path = os.path.join(dir_path, file_name) # full path to video with file name 
                print(f"vid_in_path: {vid_in_path}")
                vid_relpath= os.path.relpath(vid_in_path, dir_in_path) # relative path, with file name and extension
               # print('vid_relpath:' + vid_relpath)
                vid_subfolders = os.path.dirname(vid_relpath) # relative path with folders only, no file name
               # print('vid_subfolders:' + vid_subfolders)
                data_out_prefix = os.path.normpath(os.path.join(dir_out_path, vid_subfolders)) # create output folder that mirrors subfolder structure from dir_in_path
                print(f"data_out_prefix {data_out_prefix}")
                if not os.path.exists(data_out_prefix):
                    os.makedirs(data_out_prefix) # make directory if it doesn't exist already
                    print(f"Making directory: {data_out_prefix}")
                print(f"Processing: {file_name}")

                # make folder for raw pose estimation data 
                raw_data_out_folder = os.path.join(data_out_prefix, '000_raw_pose_data')
                print('raw_data_out_folder:' + raw_data_out_folder)
                
                # if raw pose directory does not exist yet, make folder 
                if not os.path.exists(raw_data_out_folder):
                    os.makedirs(raw_data_out_folder) # make directory if it doesn't exist already

                # get frames per second from video 
                [fps,fps_df] = get_frames_per_second(vid_in_path)
                print(os.path.normpath(os.path.join(raw_data_out_folder, name + '_fps.csv')))
                fps_df.to_csv(os.path.normpath(os.path.join(raw_data_out_folder, name + '_fps.csv')))

                # run pose estimation on video and save raw data as .csv file 
                model_results = extract_pose_data(vid_in_path, run_depth)
                for i, raw_data in enumerate(model_results):
                    try:
                        pose_data = transform_pose_data(raw_data)
                        load_pose_data(pose_data,
                                        os.path.normpath(os.path.join(raw_data_out_folder, name + f"_{MODELS[i]}")))
                    except Exception as e:
                        logger.info(e.args)

                              
                # analysis functions below 
                # model_results[1] = yolo_df
                # model_results[2] = mp_pose_df
                # model_results[3] = mp_world_df 
                # fps from step above
                    # Three above are inputs for first function inf raw_pose_analysis main 
                    # (as of 9/12/2024, merge_mp_pose_world) 
                    # Need to double check if any differences from python variable vs loading from .csv – index, missing data, structure



def main():

    logging.basicConfig(filename='hva.log', level=logging.INFO)
    logger.info("Started")

    args = sys.argv[1:]
    if len(args) < 2 or args[0] == "--help":
        # python3 -W ignore hva.py ./tmp/fixtures ./tmp/csv_output
        print("usage: python3 run.py <DEPTH: T/F> <DIR_IN_PATH> <DIR_OUT_PATH>")
        exit()

    if not args[0] in ["T", "F"]:
        print("Please specific if Depth Estimates should be made: T or F")
        sys.exit()

    run_depth = (args[0] == "T")
    dir_in_path = args[1]
    dir_out_path = args[2]

    print(f"Processing files in {dir_in_path}...")
    # process_folder(in_folder, out_folder)
    process_dir(dir_in_path, dir_out_path, run_depth)
    logger.info("Finished")


if __name__ == "__main__":
    main()
