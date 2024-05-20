import cv2
import mediapipe as mp
import sys
import pandas as pd
import os

poseDict = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index"
}

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def renameCols(col):
    lmNum = col.split('_')[0]
    lmVal = poseDict[int(lmNum)]
    return col.replace(lmNum, lmVal)

def process_video(inFile, outFile, exportVid=True):
    
    name = inFile.split('/')[-1].split('.')[0]
    data = pd.DataFrame()
    cap = cv2.VideoCapture(inFile)
    if exportVid:
        fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
        out = cv2.VideoWriter(f'./{outFile}_mpVid.mp4', fourcc, 20, (int(cap.get(3)),
                                                                     int(cap.get(4))))
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                #print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                tLndMrks = list(map(lambda lndMrk: (lndMrk.x, lndMrk.y,
                                                    lndMrk.z, lndMrk.visibility,
                                                    lndMrk.presence) if lndMrk else None,
                                    results.pose_landmarks.landmark))
            else:
                tLndMrks = None
                tData = None

            if tLndMrks and len(tLndMrks) == 33:
                tData = pd.DataFrame(tLndMrks, columns=['x', 'y', 'z', 'vis', 'pres']).reset_index().stack()
                tData.index = list(map(lambda idx: f"{idx[0]}_{idx[1]}", tData.index))
                tData = tData.to_frame().T
                data = pd.concat([data, tData], ignore_index=True)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if exportVid:
                out.write(image)
            if 0xFF == 27:
                break
    if exportVid:
        out.release()

    cap.release()
    data.columns = [renameCols(col) for col in data.columns]
    data.to_csv(f'./{outFile}_mpFrames.csv')

def process_folder(inFolderPath, outFolderPath):
    for (dirpath, dirnames, filenames) in os.walk(inFolderPath):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            ext = ext.lower()[1:]
            if (ext == "mov" or ext == "mp4"):
                 inPath = os.path.join(dirpath, filename)
                 outPath = os.path.join(outFolderPath, name)
                 print(f"Processing: {inPath}")
                 process_video(inPath, outPath)
                 
def main():
    args = sys.argv[1:]
    if len(args) < 2 or args[0] == "--help":
        print("usage: python mediapipe_pose.py <VIDEO_INPUT_FOLDER_PATH> <VIDEO_OUTPUT_FODLER_PATH>")
        exit()
    in_folder = args[0]
    out_folder = args[1]
    print(f"Starting Video Processing\nMediaPipe: {mp.__version__}")
    # process_folder(in_folder, out_folder)
    process_video(in_folder, out_folder)

if __name__ == "__main__":
    main()
