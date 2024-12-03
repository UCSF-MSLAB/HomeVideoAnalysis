import pandas as pd
import math


class MarigoldResultHandler():

    def __init__(self, i, marigold_output):

        self.frame = i
        self.data = marigold_output['depth_np']

    def extract(self, mpipe_lndmrks):

        lndmrk_px_vals = []

        for i, row in mpipe_lndmrks.iterrows():
            px_x = row['X']
            px_y = row['Y']
            if px_x != 0 and px_y != 0 and \
               not math.isinf(px_x) and not math.isinf(px_y):
                depth_est = self.data[int(px_y), int(px_x)]
                lndmrk_px_vals.append({'frame': self.frame,
                                       'label': row['label'],
                                       'depth_est': depth_est})

        return pd.DataFrame(lndmrk_px_vals)
