import pandas as pd
from src.keypoint_labels import keypoint_labels


class MPipeResultHandler():

    LABELS = keypoint_labels.mpipe_labels
    EMPTY_DICT = {'X': [],
                  'Y': [],
                  'Z': [],
                  'vis': [],
                  'pres': [],
                  'frame': [],
                  'label': []}

    def __init__(self, i, mpipe_results):

        self.frame = i
        self.df = self.landmarks_to_df(mpipe_results)

    def landmarks_to_df(self, mpipe_results):

        if mpipe_results.pose_landmarks:
            tLndMrks = list(map(lambda lndMrk: (lndMrk.x, lndMrk.y,
                                                lndMrk.z, lndMrk.visibility,
                                                lndMrk.presence) if lndMrk else
                                None,
                                mpipe_results.pose_world_landmarks.landmark))
        else:
            return pd.DataFrame(self.EMPTY_DICT)

        mpipe_df = pd.DataFrame(tLndMrks,
                                columns=['X', 'Y', 'Z',
                                         'vis', 'pres'])
        # mpipe_df['Y'] = mpipe_df['Y'] * (-1) + 1
        mpipe_df['frame'] = [self.frame] * len(self.LABELS)
        mpipe_df['label'] = pd.Series(self.LABELS.copy(), dtype='string')

        return mpipe_df

    def denormalize(self, box, margin):
        self.df.X = self.df.X * (box[2] - box[0] + 2*margin) + box[0] - margin
        self.df.Y = self.df.Y * (box[3] - box[1] + 2*margin) + box[1] - margin
