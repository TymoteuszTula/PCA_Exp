# data_handler.py

''' Code contains the class which handles the experimental data input.'''

# libraries

import numpy as np
from math import sqrt
import time

# internal modules

from .utils.utils import find_ind_val

class DataHandler:
    r''' Class which takes the experimental data and preprocess it if 
    neccessary, before the ML procedure.

    Attribs:
        batches: list of batches of data. Each batch is a 3d numpy array with
        indices [i, j, k], where i runs through x arguments of data, 
        j ∈ {x, y, error} and k runs through all measurements of the batch.

        batches_names: list of names for the batch (not yet implemented).

        prepared_data: list of preprocessed data that can be feed to 
        pca_machine module.
    '''

    def __init__(self):
        self.batches = []
        self.batches_names = []
        self.prepared_data = []
        self.batch_info = []

    def load_batch(self, stsp, prenum='', ext='', loc='./', excep=[], name=None,
                   indicators=[], delimiter=None, skiprows=0):
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
            numbering. Empty string in default.

            ext: String defining type of extension at the end of the file 
            without '.'. Empty string in default.

            loc: String indicating location of the files. Set as the folder 
            with the python script in default.

            excep: List of integers that should be omitted when extracting data
            from textfiles. Set to an empty list as default.

            name: String. User can name a given batch of file (e.g. specific 
            material) so that it can be extracted later more intuitively.
            indicators: List of lists of floats (TODO: make indicators work)
        '''

        no_meas = stsp[1] - stsp[0] - len(excep) + 1
        batch = []
    
        enum = np.delete(range(stsp[0], stsp[1] + 1), excep)

        for meas in enum:
            path = loc + prenum + str(meas) + ext
            temp_mat = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
            batch.append(temp_mat)

        batch = np.transpose(batch, (1, 0, 2))    

        self.batches.append(np.array(batch))
        if name == None:
            self.batches_names.append(str(len(self.batches_names)))
        else:
            self.batches_names.append(name)

    def load_batch_from_array(self, asymm, name=''):
        r''' Function that load batch from numpy array. The array should be of
        form (i, j, k), where i iterates over different x values, j
        iterates over different measurements and k can have three values 
        with k=0 representing two-dimensional array of x values,
        k=1 representing the y values and k=2 representing the y errors.

        Args:
            asymm: three-dimensional array of data

            name: name for the batch of data
        '''

        self.batches.append(asymm)
        self.batches_names.append(name)

    def prepare_XYE_PCA(self, batch_ind=[0], a_e=None, 
                        filter_data=True, preserve_batch_info=True):
        r''' Function that prepares the choosen data batches into matrix form,
        that is all of the y, x and error vectors are presented as matrices
        Y, X and E. (TODO: make batch_names work)

        Args:
            batch_ind: list of integers that specify which batches are 
            preprocessed together.

            batch_names: alternatively, names of batches that will be
            preprocessed together (not yet implemented)
        '''

        if filter_data:
            a, e, _, x, err = self.filter_data2(batch_ind=batch_ind, a=a_e)
        else:
            a = self.batches[batch_ind[0]][:,:,1]
            x = self.batches[batch_ind[0]][:,:,0]

        if preserve_batch_info:
            self.batch_info.append(
                [{"bidx": i,
                  "mlen": self.batches[i].shape[1]} for i in batch_ind])

        self.prepared_data.append(np.array([a, x, err]))

    def filter_data(self, batch_ind=[0], a=None, save_times=False):
        r''' Function that re-bin the data to equalise the error in each bin.

        Args:
            batch_ind: list of integers that specify which batches are 
            preprocessed together.
        '''

        # TODO write better code (this was taken from previous version of the
        # code).

        t = self.batches[batch_ind[0]][:,:,0]
        A = self.batches[batch_ind[0]][:,:,1]
        E = self.batches[batch_ind[0]][:,:,2]

        for batch_i in batch_ind[1:]:
            t = np.c_[t, self.batches[batch_ind[batch_i]][:,:,0]] 
            A = np.c_[A, self.batches[batch_ind[batch_i]][:,:,1]] 
            E = np.c_[E, self.batches[batch_ind[batch_i]][:,:,2]]

        xd = A.shape[0]
        yd = A.shape[1]

        # A1 = A[0,:][np.newaxis]
        # E1 = np.array([np.sqrt(np.sum(E[0,:] ** 2) / yd)])
        # Len1 = np.array([1])
        # t1 = t[0,:][np.newaxis]

        if a == None:
            a = np.sum(E[0,:] ** 2)

        A1 = []
        E1 = []
        Len1 = []
        t1 = []

        Etemp = 0
        Atemp = np.zeros((yd, )) # np.empty([0,yd])
        ttemp = np.zeros((yd, )) # np.empty([0,yd])
        no_of_bins = 0
        start_time = time.time()
        # Count times (only temporary)
        if save_times:
            times = []

        for ii in range(xd):
            
            #Etemp = np.append(Etemp, np.sum(E[ii,:] ** 2 / a))
            Etemp += np.sum(E[ii,:] ** 2) / a
            #Atemp = np.append(Atemp, A[ii,:][np.newaxis],axis=0)
            Atemp += A[ii,:]
            #ttemp = np.append(ttemp, t[ii,:][np.newaxis],axis=0)
            ttemp += t[ii,:]
            no_of_bins += 1

            #stop_time = time.time()
            print("Processed data: " + str(ii) + " / " + str(xd) + " # Number of new bins: " + str(len(Len1)))
            # if save_times:
            #     times.append(stop_time-start_time)
            #start_time = time.time()

            if Etemp == 0 or no_of_bins**2 / Etemp > 1 or ii == xd - 1:
                # #A1 = np.append(A1,(np.sum(Atemp,axis=0) / Atemp.shape[0])
                # #            [np.newaxis], axis=0)
                # A1 = np.append(A1, Atemp / no_of_bins, axis=0)
                # #E1 = np.append(E1,np.sqrt(np.sum( a * Etemp) / yd) /  len(Etemp))
                # E1 = np.append(E1, sqrt(a * Etemp) / yd / no_of_bins)
                # #t1 = np.append(t1,(np.sum(ttemp,axis=0) / ttemp.shape[0])
                # #            [np.newaxis], axis=0)
                # t1 = np.append(t1, ttemp / no_of_bins, axis=0)
                # #Len1 = np.append(Len1,Atemp.shape[0])
                # Len1 = np.append(Len1, no_of_bins)

                A1.append(Atemp / no_of_bins)
                E1.append(sqrt(a * Etemp) / yd / no_of_bins)
                t1.append(ttemp / no_of_bins)
                Len1.append(no_of_bins)

                Etemp = 0
                Atemp = np.zeros((yd, ))
                ttemp = np.zeros((yd, ))
                no_of_bins = 0

        A1 = A1[:-1]
        E1 = E1[:-1]
        t1 = t1[:-1]
        Len1 = Len1[:-1]

        A1 = np.array(A1)
        E1 = np.array(E1)
        t1 = np.array(t1)

        if save_times:
            return A1, E1, Len1, t1, times
        return A1, E1, Len1, t1 
    
    def filter_data2(self, batch_ind=[0], a=None, save_times=False):
        r''' Function that re-bin the data to equalise the error in each bin.

        Args:
            batch_ind: list of integers that specify which batches are 
            preprocessed together.
        '''

        # Code designed to work with error calculations

        t = self.batches[batch_ind[0]][:,:,0]
        A = self.batches[batch_ind[0]][:,:,1]
        E = self.batches[batch_ind[0]][:,:,2]

        for batch_i in batch_ind[1:]:
            t = np.c_[t, self.batches[batch_ind[batch_i]][:,:,0]] 
            A = np.c_[A, self.batches[batch_ind[batch_i]][:,:,1]] 
            E = np.c_[E, self.batches[batch_ind[batch_i]][:,:,2]]

        xd = A.shape[0]
        yd = A.shape[1]

        #A1 = A[0,:][np.newaxis]
        #E1 = np.array([np.sqrt(np.sum(E[0,:] ** 2) / yd)])
        #E_new = E[0,:].copy()[np.newaxis]
        #Len1 = np.array([1])
        #t1 = t[0,:][np.newaxis]
        A1 = np.empty([0, yd])
        E1 = np.empty([0, yd])
        E_new = np.empty([0, yd])
        Len1 = np.array([])
        t1 = np.empty([0, yd])

        if a == None:
            a = np.sum(E[0,:] ** 2)

        Etemp = np.array([])
        E_newtemp = np.empty([0,yd])
        Atemp = np.empty([0,yd])
        ttemp = np.empty([0,yd])

        for ii in range(0, xd):
            Etemp = np.append(Etemp, np.sum(E[ii,:] ** 2 / a))
            E_newtemp = np.append(E_newtemp, E[ii,:][np.newaxis],axis=0)
            Atemp = np.append(Atemp, A[ii,:][np.newaxis],axis=0)
            ttemp = np.append(ttemp, t[ii,:][np.newaxis],axis=0)
            
            
            if np.sum(1 / Etemp) > 1 or ii == xd - 1:
                A1 = np.append(A1,(np.sum(Atemp,axis=0) / Atemp.shape[0])
                            [np.newaxis], axis=0)
                E1 = np.append(E1,np.sqrt(np.sum( a * Etemp) / yd) /  len(Etemp))
                E_new = np.append(E_new, (np.sqrt(np.sum(E_newtemp**2, axis=0)) / len(Etemp))
                            [np.newaxis], axis=0)
                t1 = np.append(t1,(np.sum(ttemp,axis=0) / ttemp.shape[0])
                            [np.newaxis], axis=0)
                Len1 = np.append(Len1,Atemp.shape[0])

                Etemp = np.array([])
                E_newtemp = np.empty([0,yd])
                Atemp = np.empty([0,yd])
                ttemp = np.empty([0,yd])

        return A1, E1, Len1, t1, E_new

    def bin_data(self, x_0, batch_ind=[0], batch_names=[]):
        r''' Function that bin the data to common bins. Use it if your batches
        have different x sizes.

        Args:
            batch_ind: list of integers that specify which batches are 
            preprocessed together.

            batch_names: alternatively, names of batches that will be
            preprocessed together (not yet implemented)
        '''

        # TODO: write better code

        dx = x_0[1] - x_0[0]

        for batch_i in batch_ind:

            x_1 = self.batches[batch_i][:,0,0]

            batch_ph = np.zeros((len(x_0), self.batches[batch_i].shape[1], 3))

            h = 0
            for x_i in x_0:
                idx = (x_1 >= x_i - dx / 2) * (x_1 < x_i + dx / 2)
                no = self.batches[batch_i][idx,:,1].shape[0]
                batch_ph[h,:,1] = np.sum(self.batches[batch_i][idx,:,1],
                                            axis=0) / no
                batch_ph[h,:,2] = np.sqrt(np.sum(self.batches[batch_i][idx,:,2]
                             ** 2, axis=0)) / no
                h += 1

            batch_ph[:,:,0] = np.repeat(x_0[np.newaxis].T, 
                                        self.batches[batch_i].shape[1],
                                        axis = 1)

            self.batches[batch_i] = batch_ph
                        

    def slice_batch(self, batch_ind, x_inds=None, x_vals=None):
        r''' Function that cuts off the data points of a given batch. Can give
        index value or x cutoff value.

        Args:
            batch_ind: integer of the batch.

            x_inds: tuple of two integers (x_start, x_stop), which specify the
            cut off indices.

            x_vals: tuple of two floats (x_start_val, x_stop_val), which 
            specify cut off values.
        '''

        if x_vals == None and x_inds != None:
            self.batches[batch_ind] = (
                self.batches[batch_ind][x_inds[0]:x_inds[1],:,:])
        elif x_vals != None and x_inds == None:
            x_ind0 = find_ind_val(self.batches[batch_ind][:,0,0], x_vals[0])
            x_ind1 = find_ind_val(self.batches[batch_ind][:,0,0], x_vals[1])
            self.batches[batch_ind] = (
                self.batches[batch_ind][x_ind0:x_ind1,:,:])
        else:
            raise Exception("Either x_inds or x_vals needs to be specified!")

