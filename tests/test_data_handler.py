# test_data_handler.py

import sys
sys.path.append('../pca_exp')

import numpy as np
import data_handler as dh

dataHandler = dh.DataHandler()

stsp_lng = (10, 27)
prenum_lng = 'R202'
ext_lng = '.DAT'
loc_lng = './ExData/'

stsp_ba = (56, 76)
prenum_ba = 'EMU585'
ext_ba = '.dat'
loc_ba = './BaFe2Se2O/'
delimiter_ba = ','
skiprows_ba = 2

dataHandler.load_batch(stsp=stsp_lng, prenum=prenum_lng, ext=ext_lng, 
                        loc=loc_lng)
dataHandler.load_batch(stsp=stsp_ba, prenum=prenum_ba, ext=ext_ba, loc=loc_ba,
                        delimiter=delimiter_ba, skiprows=skiprows_ba)



print(dataHandler.batches[0])




