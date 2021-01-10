# pca_machine.py

''' Code contains the class which performs PCA and returns principal 
components, scores and other information
'''

# libraries
import numpy as np
import matplotlib.pyplot as plt

class PCAMachine:
    r''' Class which holds the functions and variables used in PCA of 
    experimental data
    '''

    def __init__(self):
        self.pc_scores = []
        self.pc_curves = []
        self.pc_av = []
        self.pc_sing = []
        self.pc_z = []

    def perform_pca(self, data_hand, prep_ind = 0):
        a = data_hand.prepared_data[prep_ind][0]
        av = np.sum(a, axis = 1)[np.newaxis].T / a.shape[1]
        z = a - av
        curves, sing, _ = np.linalg.svd(z)
        scores = np.dot(curves.T, z)

        self.pc_scores.append(scores)
        self.pc_curves.append(curves)
        self.pc_av.append(av)
        self.pc_sing.append(sing)
        self.pc_z.append(z)

    def show_pca_results(self):
        pass
        




    