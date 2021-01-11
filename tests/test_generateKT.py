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
sig = (0.1, 1)
Lam = (0.1, 2)
er = 0.02 * (np.exp(0.1 * t) + 0.01)
no_samples = 2000

gKT = generateKT(t, a0, ab, sig, Lam, er, no_samples)

dh = DataHandler()
dh.load_batch_from_array(gKT)
dh.filter_data()
dh.prepare_XYE_PCA()

pca_machine = PCAMachine()
pca_machine.perform_pca(dh)



