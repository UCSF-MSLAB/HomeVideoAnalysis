import pandas as pd
import math


class MarigoldResultHandler():

    def __init__(self, i, marigold_output):

        self.frame = i
        self.data = marigold_output
        self.data_shape_x = len(self.data['depth_np'][0])
        self.data_shape_y = len(self.data['depth_np'])
        if marigold_output is not None:
            self.latent = marigold_output.latent
        else:
            self.latent = None
        self.EMPTY_DICT = pd.DataFrame({'frame': [i],
                                        'label': ['None'],
                                        'depth_est': [float('inf')]
                                        })
        self.EMPTY_DICT['label'] = pd.Series('None', dtype='string')

    def get_latents(self):
        return self.latents

    def extract(self, mpipe_lndmrks):

        if self.data is None:
            return self.EMPTY_DICT

        lndmrk_px_vals = []

        for i, row in mpipe_lndmrks.iterrows():
            y_ind = int(row.Y)
            x_ind = int(row.X)
            if (x_ind > 0 and x_ind < self.data_shape_x) and \
               (y_ind > 0 and y_ind < self.data_shape_y):
                try:
                    depth_est = self.data['depth_np'][y_ind, x_ind]
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
