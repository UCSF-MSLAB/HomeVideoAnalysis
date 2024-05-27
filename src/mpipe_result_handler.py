import pandas as pd
from src.keypoint_labels import keypoint_labels


class MPipeResultHandler():

    LABELS = keypoint_labels.mpipe_labels

    def __init__(self, i, mpipe_results):

        self.frame = i
        self.df = self.landmarks_to_df(mpipe_results)

    def landmarks_to_df(self, mpipe_results):

        tLndMrks = list(map(lambda lndMrk: (lndMrk.x, lndMrk.y,
                                            lndMrk.z, lndMrk.visibility,
                                            lndMrk.presence) if lndMrk else
                            None,
                            mpipe_results.pose_landmarks.landmark))

        mpipe_df = pd.DataFrame(tLndMrks,
                                columns=['X', 'Y', 'Z', 'vis', 'pres'])
        mpipe_df['frame'] = [self.frame] * len(self.LABELS)
        mpipe_df['label'] = self.LABELS.copy()

        return mpipe_df
