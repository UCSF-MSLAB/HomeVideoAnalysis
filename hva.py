import os
import sys
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

MODELS = ["yolo", "mediapipe"]


def process_dir(dir_in_path, dir_out_path):

    for (dir_path, dir_names, file_names) in os.walk(dir_in_path):
        for file_name in file_names:
            name, ext = os.path.splitext(file_name)
            ext = ext.lower()[1:]
            if (ext in ALLOWED_VID_FORMATS):
                vid_in_path = os.path.join(dir_path, file_name)
                data_out_prefix = os.path.join(dir_out_path, name)
                print(f"Processing: {file_name}")

                model_results = extract_pose_data(vid_in_path)
                for i, raw_data in enumerate(model_results):
                    try:
                        pose_data = transform_pose_data(raw_data)
                        load_pose_data(pose_data,
                                       data_out_prefix + f"_{MODELS[i]}_")
                    except ValueError as err:
                        print(err.args)


def main():

    logging.basicConfig(filename='hva.log', level=logging.INFO)
    logger.info("Started")

    args = sys.argv[1:]
    if len(args) < 2 or args[0] == "--help":
        # python3 -W ignore hva.py ./tmp/fixtures ./tmp/csv_output
        print("usage: python3 hva.py <DIR_IN_PATH> <DIR_OUT_PATH>")
        exit()

    dir_in_path = args[0]
    dir_out_path = args[1]
    print(f"Processing files in {dir_in_path}...")
    # process_folder(in_folder, out_folder)
    process_dir(dir_in_path, dir_out_path)
    logger.info("Finished")


if __name__ == "__main__":
    main()
