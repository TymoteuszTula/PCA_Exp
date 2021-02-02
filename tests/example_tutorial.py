# example_tutorial.py

import sys
sys.path.append('./pca_exp')
sys.path.append('./pca_exp/generate_samples')

import numpy as np
import matplotlib.pyplot as plt
from data_handler import DataHandler
from pca_machine import PCAMachine
from kubo_toyabe import generateKT

t = np.linspace(0, 12, num=300)
a0 = 0.26
ab = 0
sig = np.array([0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13])
Lam = np.array([0.07, 0.07, 0.07, 0.07, 0.07, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025])
T = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5] 
er = 0.002 * (np.exp(0.2 * t) + 0.001)

gKT, sig_param, Lam_param = generateKT(t, a0, ab, sig, Lam, er, no_samples=11)

plt.figure(1)
for i in range(len(T)):
    plt.plot(t, gKT[:,i,1], color=(1, i/11, 0, 1), label='T= ' + str(T[i]))

plt.title('Exp measurements')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()

dh = DataHandler()
dh.load_batch_from_array(gKT)
dh.filter_data()
dh.prepare_XYE_PCA()

pca_machine = PCAMachine(dh)
pca_machine.perform_pca()
pca_machine.show_pca_results_1(T, T)

for i in range(len(T)):
    np.savetxt('./exp_data_example/ede' + str(i) + '.txt',
               gKT[:,i,:])

