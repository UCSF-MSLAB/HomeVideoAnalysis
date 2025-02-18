import pandas as pd
import numpy as np


class MarigoldResultHandler():

    def __init__(self, i, marigold_output):

        self.frame = i

        if marigold_output is None:
            self.latent = None
            self.data = None
        else:
            self.latent = marigold_output.latent
            prediction = marigold_output.prediction
            self.data_shape_x = prediction.shape[2]
            self.data_shape_y = prediction.shape[1]
            self.data = prediction.reshape(self.data_shape_y,
                                           self.data_shape_x)

        self.EMPTY_DICT = pd.DataFrame({'frame': [i],
                                        'label': ['None'],
                                        'depth_est': [float('inf')]
                                        })
        self.EMPTY_DICT['label'] = pd.Series('None', dtype='string')

    def get_latents(self):
        return self.latent

    def extract(self, mpipe_lndmrks):

        if self.data is None:
            return self.EMPTY_DICT

        lndmrk_px_vals = []

        for i, row in mpipe_lndmrks.iterrows():
            if np.isinf(row.Y) or np.isinf(row.X):
                continue
            else:
                y_ind = int(row.Y)
                x_ind = int(row.X)
            if (x_ind > 0 and x_ind < self.data_shape_x) and \
               (y_ind > 0 and y_ind < self.data_shape_y):
                try:
                    depth_est = self.data[y_ind, x_ind]
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
