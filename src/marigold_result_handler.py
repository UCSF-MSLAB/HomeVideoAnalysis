import pandas as pd


class MarigoldResultHandler():

    def __init__(self, i, marigold_output):

        self.frame = i
        self.data = marigold_output['depth_np']

    def extract(self, mpipe_lndmrks):

        lndmrk_px_vals = []

        for i, row in mpipe_lndmrks.iterrows():
            if row['X'] != 0 and row['Y'] != 0:
                depth_est = self.data[int(row.Y), int(row.X)]
                lndmrk_px_vals.append({'frame': self.i, 'label': row['label'],
                                       'depth_est': depth_est})

        return pd.DataFrame(lndmrk_px_vals)
