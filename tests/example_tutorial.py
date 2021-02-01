# example_tutorial.py

import sys
sys.path.append('./pca_exp')
sys.path.append('./pca_exp/generate_samples')

import numpy as np
import matplotlib.pyplot as plt
import data_handler as dh
from pca_machine import PCAMachine
from kubo_toyabe import generateKT

t = np.linspace(0, 12, num=300)
a0 = 0.26
ab = 0
sig = np.array([0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2, 0.25, 0.3])
Lam = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
T = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5] 
er = 0.002 * (np.exp(0.1 * t) + 0.01)

gKT, sig_param, Lam_param = generateKT(t, a0, ab, sig, Lam, er, no_samples=10)

plt.figure(1)
plt.plot(t, gKT[:,:,1])
plt.show()

