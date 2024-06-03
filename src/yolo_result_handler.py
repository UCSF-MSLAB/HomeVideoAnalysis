import pandas as pd
from src.keypoint_labels import keypoint_labels


class YoloResultHandler():

    LABELS = keypoint_labels.yolo_labels

    def __init__(self, i, yolo_results):

        self.frame = i
        self.keypoints = yolo_results[0].keypoints
        self.boxes = sorted(yolo_results[0].boxes.xyxy,
                            key=lambda x: x[0],
                            reverse=True)
        self.dfs = self.keypoints_to_dfs()

    def keypoints_to_dfs(self):

        indv_pose = []
        x_means = []
        for pose in self.keypoints:
            # return normalized coordinates
            pose_df = pd.DataFrame(pose.xy[0].numpy(),
                                   columns=['X', 'Y'])
            # pose_df['Y'] = pose_df['Y'] * (-1) + 1
            x_means.append(pose_df.mean()['X'])
            pose_df["label"] = pd.Series(self.LABELS.copy(), dtype='string')
            pose_df["frame"] = [self.frame]*len(self.LABELS)
            indv_pose.append(pose_df)

        # sort according to the x means, left-most first
        tups = sorted(zip(x_means, indv_pose), reverse=True)
        indv_pose = [t[1] for t in tups]

        return indv_pose
