import pandas as pd
from src.keypoint_labels import keypoint_labels


class YoloResultHandler():

    LABELS = keypoint_labels.yolo_labels
    CONF_THRESH = .85
    keypoints = None
    box = None
    dfs = None

    def __init__(self, i, yolo_results):

        self.frame = i
        candidates = zip(yolo_results[0].boxes.conf,
                         yolo_results[0].boxes.xyxy,
                         yolo_results[0].keypoints)
        candidates = sorted(candidates, key=lambda x: x[0].item(),
                            reverse=True)
        if candidates[0][0].item() >= self.CONF_THRESH:
            self.keypoints = candidates[0][2]
            self.box = candidates[0][1]
            self.dfs = self.keypoints_to_dfs()

    def keypoints_to_dfs(self):

        # return unnormalized coordinates
        pose_df = pd.DataFrame(self.keypoints.xy[0].numpy(),
                               columns=['X', 'Y'])
        # pose_df['Y'] = pose_df['Y'] * (-1) + 1
        pose_df["label"] = pd.Series(self.LABELS.copy(),
                                     dtype='string')
        pose_df["frame"] = [self.frame]*len(self.LABELS)
        return [pose_df]
