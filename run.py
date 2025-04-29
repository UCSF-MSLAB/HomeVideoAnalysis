import os
# import re
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

MODELS = ["yolo", "mediapipe", "mediapipe_world", "marigold"]


# from https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def process_dir(dir_in_path, dir_out_path, run_depth):

    abs_path = os.path.abspath(dir_in_path)
    for file_name in glob.glob(abs_path + '/**/*.*', recursive=True):
        name, ext = os.path.splitext(os.path.basename(file_name))
        ext = ext.lower()[1:]
        if (ext in ALLOWED_VID_FORMATS):
            data_out_prefix = os.path.join(dir_out_path, name)
            print(f"Processing: {file_name}")
            model_results = extract_pose_data(file_name, run_depth)
            for i, raw_data in enumerate(model_results):
                try:
                    pose_data = transform_pose_data(raw_data)
                    load_pose_data(pose_data,
                                   data_out_prefix + f"_{MODELS[i]}")
                except Exception as e:
                    logger.info(e.args)


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
