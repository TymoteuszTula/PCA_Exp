# pca_machine.py

''' Code contains the class which performs PCA and returns principal 
components, scores and other information.
'''

# libraries
import numpy as np
import matplotlib.pyplot as plt


class PCAMachine:
    r''' Class which holds the functions and variables used in PCA of 
    experimental data.

    Params:
        data_handler: pca_exp.data_handler class specifing the instance that
        holds processed data, used in PCA algorithm.

    Attribs:
        pc_scores: list of 2D numpy arrays that holds the PC scores of each
        PCA. The 2D array have indices [i, j], where i runs through PC 
        numbers and j runs through different measurements.

        pc_curves: list of 2D numpy arrays that holds the PC curves of each
        PCA. The 2D array have indices [i, j], where i runs through x values 
        and j runs through different PC numbers.

        pc_av: list of 1D numpy arrays, which hold the average of each PCA.

        pc_sing: list of 1D numy arrays, which hold singular values of each
        PCA.

        pc_z: list of 2D numpy array, which holds the initial measurement
        curves, with removed average.

        data_handler: pca_exp.data_handler class specifing the instance that
        holds processed data, used in PCA algorithm.
    '''

    def __init__(self, data_handler):
        self.pc_scores = []
        self.pc_curves = []
        self.pc_av = []
        self.pc_sing = []
        self.pc_z = []
        self.data_handler = data_handler

    def print_pca_representation(self):
        r''' Function that prints the scree plot in the console log of a last
        principal component analysis.
        '''
        sing_total = np.sum(self.pc_sing[-1])
        sing_show = self.pc_sing[-1][:8] * 100 / sing_total
        
        print(str(int(sing_show[0])) + '%', '^')
        for i in range(10):
            print('    |', end='\t')
            for s_j in sing_show:
                if s_j / sing_show[0] < (10 - i) / 10:
                    print(' ', end='\t')
                else:
                    print('#', end='\t')
            print()
        print('    ---------------------------------------------' +
                    '-------------------->')


    def perform_pca(self, prep_ind = 0):
        r''' Function that performs the principal component analysis on the 
        data specified by prep_ind. It save the results in attributes of the 
        class.

        Args:
            prep_ind: integer that specifies the data, on which PCA is 
            performed.
        '''
        data_hand = self.data_handler
        a = data_hand.prepared_data[prep_ind][0]
        av = np.sum(a, axis = 1)[np.newaxis].T / a.shape[1]
        z = a - av

        print('Performing PCA on prepared data')
        curves, sing, _ = np.linalg.svd(z)
        scores = np.dot(curves.T, z)  

        self.pc_scores.append(scores)
        self.pc_curves.append(curves)
        self.pc_av.append(av)
        self.pc_sing.append(sing)
        self.pc_z.append(z)

        print('Showing the percentage of covariance of most important PCs:')
        self.print_pca_representation()

    def show_pca_results_1(self, param1, param1_name='param1', res_idx=0,
                                 prep_idx=0):
        r''' Function that prints plots showing the result of PCA. This
        function shows scree plot, PC curves and 1st PC vs 2nd PC scores
        as well as up to 4th PC scores vs paramters specified in param1.

        Args:
            param1: list or 1D numpy.array of float values of a chosen 
            parameter

            param1_name: string of the name of the chosen parameter

            res_idx: integer, specifing the index of the results stored in this
            instance of the class.

            prep_idx: integer, specyfing the index of prepared data in 
            self.data_handler.
        '''

        x = self.data_handler.prepared_data[prep_idx][1][:,0]

        plt.style.use(['dark_background'])
        plt.rcParams['font.size'] = '12'
        plt.rcParams['lines.markersize'] = '5'


        # First figure
        fig1 = plt.figure(1, figsize=[12, 5], dpi=50)
        
        plt.subplot(121)
        plt.title('PC curves')
        plt.plot(x, self.pc_curves[res_idx][:,0], '-o', label='1st PC')
        plt.plot(x, self.pc_curves[res_idx][:,1], '-o', label='2nd PC')
        plt.plot(x, self.pc_curves[res_idx][:,2], '-o', label='3rd PC')
        plt.plot(x, self.pc_curves[res_idx][:,3], '-o', label='4th PC')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('PC vectors')
        plt.grid()

        plt.subplot(122)
        plt.title('Scree plot')
        sing_norm = 100 * self.pc_sing[res_idx] / np.sum(self.pc_sing[res_idx])
        plt.plot(np.arange(1, sing_norm.size+1), sing_norm, '-sr')
        plt.xlabel('PC no.')
        plt.ylabel('Covariance captured [%]')
        plt.grid()

        plt.tight_layout()
        
        # Second figure
        fig2 = plt.figure(2, figsize=[20, 5], dpi=50)
        plt.suptitle('Principal component scores vs ' + param1_name)
        
        plt.subplot(141)
        plt.title('1st PC scores')
        plt.plot(param1, self.pc_scores[res_idx][0,:], 'or')
        plt.xlabel(param1_name)
        plt.ylabel('PC scores vs ' + param1_name)
        plt.grid()

        plt.subplot(142)
        plt.title('2nd PC scores')
        plt.plot(param1, self.pc_scores[res_idx][1,:], 'ob')
        plt.xlabel(param1_name)
        plt.grid()

        plt.subplot(143)
        plt.title('3rd PC scores')
        plt.plot(param1, self.pc_scores[res_idx][2,:], 'og')
        plt.xlabel(param1_name)
        plt.grid()

        plt.subplot(144)
        plt.title('4th PC scores')
        plt.plot(param1, self.pc_scores[res_idx][3,:], 'oc')
        plt.xlabel(param1_name)
        plt.grid()

        plt.tight_layout()

        plt.show()

    def show_pca_results_2(self, param1, param2, res_idx=0, prep_idx=0):
        r''' Function that prints plots showing the result of PCA. This
        function shows scree plot, PC curves and 1st PC vs 2nd PC scores
        as well as up to 4th PC scores vs paramters specified in param1,
        param2.

        Args:
            param1: list or 1D numpy.array of float values of a first chosen 
            parameter

            param2: list or 1D numpy.array of float values of a second chosen 
            parameter

            res_idx: integer, specifing the index of the results stored in this
            instance of the class.

            prep_idx: integer, specyfing the index of prepared data in 
            self.data_handler.
        '''

        x = self.data_handler.prepared_data[prep_idx][1][:,0]

        plt.style.use(['dark_background'])
        plt.rcParams['font.size'] = '12'
        plt.rcParams['lines.markersize'] = '5'


        # First figure
        fig1 = plt.figure(1, figsize=[10, 6], dpi=50)
        
        plt.subplot(221)
        plt.title('PC curves')
        plt.plot(x, self.pc_curves[res_idx][:,0], '-o', label='1st PC')
        plt.plot(x, self.pc_curves[res_idx][:,1], '-o', label='2nd PC')
        plt.plot(x, self.pc_curves[res_idx][:,2], '-o', label='3rd PC')
        plt.plot(x, self.pc_curves[res_idx][:,3], '-o', label='4th PC')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('PC vectors')
        plt.grid()

        plt.subplot(222)
        plt.title('Scree plot')
        sing_norm = 100 * self.pc_sing[res_idx] / np.sum(self.pc_sing[res_idx])
        plt.plot(np.arange(1, sing_norm.size+1), sing_norm, '-sr')
        plt.xlabel('PC no.')
        plt.ylabel('Covariance captured [%]')
        plt.grid()

        plt.subplot(223)
        plt.title('PC1 vs PC2 (param1)')
        plt.scatter(self.pc_scores[res_idx][0,:], 
                    self.pc_scores[res_idx][1,:],
                    c=param1)
        cbar = plt.colorbar(pad = 0, fraction=0.08)
        plt.xlabel('PC1 score')
        plt.ylabel('PC2 score')
        plt.grid()

        plt.subplot(224)
        plt.title('PC1 vs PC2 (param1)')
        plt.scatter(self.pc_scores[res_idx][0,:], 
                    self.pc_scores[res_idx][1,:],
                    c=param2)
        cbar = plt.colorbar(pad = 0, fraction=0.08)
        plt.xlabel('PC1 score')
        plt.ylabel('PC2 score')
        plt.grid()

        plt.tight_layout()
        
        # Second figure
        fig2 = plt.figure(2, figsize=[20, 6], dpi=50)
        plt.suptitle('Principal component scores vs (param)')
        
        plt.subplot(241)
        plt.title('1st PC scores')
        plt.plot(param1, self.pc_scores[res_idx][0,:], 'or')
        plt.xlabel('param1')
        plt.ylabel('PC scores vs param1')
        plt.grid()

        plt.subplot(242)
        plt.title('2nd PC scores')
        plt.plot(param1, self.pc_scores[res_idx][1,:], 'ob')
        plt.xlabel('param1')
        plt.grid()

        plt.subplot(243)
        plt.title('3rd PC scores')
        plt.plot(param1, self.pc_scores[res_idx][2,:], 'og')
        plt.xlabel('param1')
        plt.grid()

        plt.subplot(244)
        plt.title('4th PC scores')
        plt.plot(param1, self.pc_scores[res_idx][3,:], 'oc')
        plt.xlabel('param1')
        plt.grid()

        plt.subplot(245)
        plt.plot(param2, self.pc_scores[res_idx][0,:], 'or')
        plt.ylabel('PC scores vs param2')
        plt.xlabel('param2')
        plt.grid()

        plt.subplot(246)
        plt.plot(param2, self.pc_scores[res_idx][1,:], 'ob')
        plt.xlabel('param2')
        plt.grid()

        plt.subplot(247)
        plt.plot(param2, self.pc_scores[res_idx][2,:], 'og')
        plt.xlabel('param2')
        plt.grid()

        plt.subplot(248)
        plt.plot(param2, self.pc_scores[res_idx][2,:], 'oc')
        plt.xlabel('param2')
        plt.grid()

        plt.tight_layout()

        plt.show()

    def turn_pc_into_2D(self, x_no, res_idx=0):
        r''' Function takes the results of PCA and turns it back to 2D data.
        '''

        self.pc_av[res_idx] = np.reshape(self.pc_av[res_idx], (x_no, 
                                        self.pc_ac[res_idx].shape[1] / x_no))
        self.pc_curves[res_idx] = np.reshape(self.pc_curves[res_idx], (x_no, 
                                    self.pc_curves[res_idx].shape[1] / x_no))

        
        

        




    