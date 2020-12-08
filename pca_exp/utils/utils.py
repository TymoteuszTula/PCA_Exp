# utils.py

''' Code contains tools used by other part of the package '''

# libraries

import numpy as np

def find_ind_val(array, value):
    ind = np.abs(array - value).argmin()
    return ind
