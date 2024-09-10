import os
import re
import sys
import glob
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

MODELS = ["yolo", "mediapipe", "mediapipe_world"]


def process_dir(dir_in_path, dir_out_path):

    abs_path = os.path.abspath(dir_in_path)
    for file_name in glob.glob(abs_path + '/**/*.*',
                               recursive=True):
        name, ext = os.path.splitext(file_name[file_name.find(dir_in_path):])
        ext = ext.lower()[1:]
        if (ext in ALLOWED_VID_FORMATS):
            new_name = '_'.join(os.path.split(name))
            data_out_prefix = os.path.join(dir_out_path, new_name)
            print(f"Processing: {file_name}")
            model_results = extract_pose_data(file_name)
            for i, raw_data in enumerate(model_results):
                try:
                    pose_data = transform_pose_data(raw_data)
                    load_pose_data(pose_data,
                                   data_out_prefix + f"_{MODELS[i]}")
                    print(data_out_prefix)
                except Exception as e:
                    logger.info(e.args)


def main():

    logging.basicConfig(filename='hva.log', level=logging.INFO)
    logger.info("Started")

    args = sys.argv[1:]
    if len(args) < 2 or args[0] == "--help":
        # python3 -W ignore hva.py ./tmp/fixtures ./tmp/csv_output
        print("usage: python3 run.py <DIR_IN_PATH> <DIR_OUT_PATH>")
        exit()

    dir_in_path = args[0]
    dir_out_path = args[1]

    print(f"Processing files in {dir_in_path}...")
    # process_folder(in_folder, out_folder)
    process_dir(dir_in_path, dir_out_path)
    logger.info("Finished")


if __name__ == "__main__":
    main()
