# data_handler_2D.py

r''' Code contains class which handles the 2D data input.
'''

# libraries

import numpy as np

# internal modules

from .utils.utils import find_ind_val

class DataHandler2D:
    r''' Class takes the 2D data input and prepares it for ML procedures.

    Attribs:

    '''

    def __init__(self):
        self.batches = []
        self.prepared_data = []

    def load_batch_from_array(self, array):
        r''' Array should have a shape [i, x, y], where i runs through 
        different measurements, and x and y are values at x and y positions
        '''
        self.batches.append(array)

    def prepare_XYE_PCA(self, batch_ind=0):
        data2d = self.batches[batch_ind]
        a = np.reshape(data2d, (data2d.shape[0],
                 data2d.shape[1] * data2d.shape[2])).T

        self.prepared_data.append(np.array([a]))

    