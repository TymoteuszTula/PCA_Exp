# data_handler.py

''' Code contains the class which handles the experimental data input.'''

# libraries

import numpy as np

class DataHandler:
    r''' Class which takes the experimental data and preprocess it if 
    neccessery before the ML procedure. '''

    def __init__(self):
        self.batches = []

    def load_batch(self, stsp, prenum='', ext='', loc='./', excep=[], name='',
                   indicators=[]):
        r''' Function that adds a batch of data to the class from a set of 
        files in one folder. The batch is stored as 3-D numpy array where
        each batch[:,:,n] matrix is a n-th measurements with columns 
        representing x and y variables and optionally standard deviation of y.
        Expected file names are of the form 'prenum' + 'N' + '.ext' with
        prenum being a string, N an integer numbering data and extension at
        the end.  

        Args:
            stsp: Tuple of two ints (start, stop) which indicate starting and 
            ending file number.
            prenum: String indicating the beginning of the filename, before
            numbering. Empty string in default
            ext: String defining type of extension at the end of the file 
            without '.'. Empty string in default.
            loc: String indicating location of the files. Set as the folder 
            with the python script in default.
            excep: List of integers that should be omitted when extracting data
            from textfiles. Set to an empty list as default.
            name: String. User can name a given batch of file (e.g. specific 
            material) so that it can be extracted later more intuitively.
            indicators: List of lists of floats (TODO: make indicators a 
            library)
        '''

        raise NotImplementedError