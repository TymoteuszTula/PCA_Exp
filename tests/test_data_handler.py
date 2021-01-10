# test_data_handler.py

import sys
sys.path.append('./pca_exp')
sys.path.append('./pca_exp/generate_samples')

import numpy as np
import matplotlib.pyplot as plt
import data_handler as dh
from pca_machine import PCAMachine
from kubo_toyabe import generateKT

dataHandler = dh.DataHandler()

stsp_lng = (10, 27)
prenum_lng = 'R202'
ext_lng = '.DAT'
loc_lng = './tests/ExData/'

stsp_ba = (56, 76)
prenum_ba = 'EMU585'
ext_ba = '.dat'
loc_ba = './tests/BaFe2Se2O/'
delimiter_ba = ','
skiprows_ba = 2

dataHandler.load_batch(stsp=stsp_lng, prenum=prenum_lng, ext=ext_lng, 
                        loc=loc_lng)
dataHandler.load_batch(stsp=stsp_ba, prenum=prenum_ba, ext=ext_ba, loc=loc_ba,
                        delimiter=delimiter_ba, skiprows=skiprows_ba)



print(dataHandler.batches[0].shape)
print(dataHandler.batches[1].shape)

x_ind_a = (41, 780)
x_ind_b = (0, 760)

x_val_a = (0, 12)
x_val_b = (0, 12)

dataHandler.slice_batch(0, x_vals=x_val_a)
dataHandler.slice_batch(1, x_vals=x_val_b)

x_0 = np.linspace(1, 10, num=300)

dataHandler.bin_data(x_0, [0, 1])

dataHandler.prepare_XYE_PCA([0, 1])

pcaMachine = PCAMachine()

pcaMachine.perform_pca(dataHandler)

plt.figure(1)
plt.plot(dataHandler.prepared_data[0][1,:,0], pcaMachine.pc_curves[0][:,5])

plt.figure(2)
plt.plot(pcaMachine.pc_sing[0])

plt.show()






