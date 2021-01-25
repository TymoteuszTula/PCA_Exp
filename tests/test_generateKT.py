# test_generateKT.py

import sys
sys.path.append('./pca_exp/generate_samples')
sys.path.append('./pca_exp')

import numpy as np
import matplotlib.pyplot as plt
from pca_machine import PCAMachine
from kubo_toyabe import generateKT
from data_handler import DataHandler

t = np.linspace(0, 12, num=300)
a0 = 0.26
ab = 0
sig = (0.1, 0.5)
Lam = (0.1, 0.5)
er = 0.02 * (np.exp(0.1 * t) + 0.01)
no_samples = 2000

gKT, sig_param, Lam_param = generateKT(t, a0, ab, sig, Lam, er, no_samples)

dh = DataHandler()
dh.load_batch_from_array(gKT)
dh.filter_data()
dh.prepare_XYE_PCA()

pca_machine = PCAMachine(dh)
pca_machine.perform_pca()
pca_machine.show_pca_results_1(sig_param, Lam_param)



