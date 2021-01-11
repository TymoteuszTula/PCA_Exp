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

    def print_pca_representation(self):
        sing_total = np.sum(self.pc_sing[-1])
        sing_show = self.pc_sing[-1][:8] * 100 / sing_total
        
        print(str(int(sing_show[0])), '^')
        for i in range(10):
            print('   |', end='\t')
            for s_j in sing_show:
                if s_j / sing_show[0] < (10 - i) / 10:
                    print(' ', end='\t')
                else:
                    print('#', end='\t')
            print()
        print('    ---------------------------------------------' +
                    '-------------------->')


    def perform_pca(self, data_hand, prep_ind = 0):
        a = data_hand.prepared_data[prep_ind][0]
        av = np.sum(a, axis = 1)[np.newaxis].T / a.shape[1]
        z = a - av

        print('Performing PCA on prepared data')
        curves, sing, _ = np.linalg.svd(z)
        scores = np.dot(curves.T, z)

        print('Showing the percentage of covariance of most important PCs:')

        self.pc_scores.append(scores)
        self.pc_curves.append(curves)
        self.pc_av.append(av)
        self.pc_sing.append(sing)
        self.pc_z.append(z)

        # self.print_pca_representation()

    def show_pca_results(self):
        
        fig1 = plt.figure(1, figsize=[12, 8], dpi=100)
        

        




    