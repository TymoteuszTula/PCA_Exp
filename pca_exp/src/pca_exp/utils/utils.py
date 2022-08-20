# utils.py

''' Code contains tools used by other part of the package '''

# libraries

import numpy as np

def find_ind_val(array, value):
    r''' Function finds array index of closest value.

    Args:
        array: list or numpy array, where we look for closest value.

        value: float of looked for value.
    '''
    ind = np.abs(array - value).argmin()
    return ind
