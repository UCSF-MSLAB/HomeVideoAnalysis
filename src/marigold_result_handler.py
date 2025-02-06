import pandas as pd
import math


class MarigoldResultHandler():

    def __init__(self, i, marigold_output):

        self.frame = i
        self.data = marigold_output
        self.EMPTY_DICT = pd.DataFrame({'frame': [i],
                                        'label': ['None'],
                                        'depth_est': [float('inf')]
                                        })
        self.EMPTY_DICT['label'] = pd.Series('None', dtype='string')

    def extract(self, mpipe_lndmrks):

        if self.data is None:
            return self.EMPTY_DICT

        lndmrk_px_vals = []

        for i, row in mpipe_lndmrks.iterrows():
            if row.X != 0 and row.Y != 0 and \
               not math.isinf(row.X) and not math.isinf(row.Y):
                # this is a quick fix.
                try:
                    depth_est = self.data['depth_np'][int(row.Y), int(row.X)]
                except Exception as e:
                    print(f"Error in frame {self.frame}:\n")
                    print(e)
                    continue
                lndmrk_px_vals.append({'frame': self.frame,
                                       'label': row['label'],
                                       'depth_est': depth_est})

        if len(lndmrk_px_vals) == 0:
            return self.EMPTY_DICT
        return pd.DataFrame(lndmrk_px_vals)
