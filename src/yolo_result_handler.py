import pandas as pd


class YoloResultHandler():

    LABELS = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle"
    ]

    def __init__(self, yolo_res):

        self.keypoints = [frame.keypoints for frame in yolo_res]
        self.keypoints_to_df()

    def keypoints_to_df(self):

        individuals = []

        for i, kpt in enumerate(self.keypoints):
            indv_pose = []
            x_means = []
            for pose in kpt.xyn:
                pose_df = pd.DataFrame(pose.numpy(),
                                       columns=['X', 'Y'])
                pose_df['Y'] = pose_df['Y'] * (-1) + 1
                x_means.append(pose_df.mean()['X'])
                pose_df["label"] = self.LABELS.copy()
                pose_df["frame"] = [i]*len(self.LABELS)
                indv_pose.append(pose_df)

            # sort according to the x means, left-most first
            tups = sorted(zip(x_means, indv_pose), reverse=True)
            indv_pose = [t[1] for t in tups]

            individuals.append(indv_pose)

        individuals = zip(*individuals)
        return [pd.concat(indv) for indv in individuals]
