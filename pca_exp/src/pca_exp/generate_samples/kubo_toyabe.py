# kubo_toyabe.py

''' Code contains function that generates the samples of Kubo Toyabe functions
with randomised noise.
'''

import numpy as np

def generateKT(t, a0, ab, sig=(0, 1), Lam=(0, 1), er=0.1, no_samples=100):
    r''' Function that generates the Kubo Toyabe curves with gaussian noise, 
    which can be dependent on the time variable t.
    
    Args:
        t: 1D numpy array that holds values of time windows.

        a0: float that specifies the initial asymmetry.

        ab: float that specifies the background asymmetry.

        sig: tuple of two floats (sig_min, sig_max) that specifies minimum 
        and maximum values of sigma parameter. Alternatively 1D numpy array
        that specifies values of sigma parameter directly.

        Lam: tuple of two floats (Lam_min, Lam_max) that specifies minimum 
        and maximum values of Lambda parameter. Alternatively 1D numpy array
        that specifies values of Lambda parameter directly. If both sig and Lam
        are 1D numpy arrays, then their size must match.

        er: float that specifies the uniform standard deviation over all 
        samples. Alternatively 1D numpy array of the same size as t, that 
        specify the error as a function of time.

        no_samples: number of measurements generated. Only needed if both sig
        and Lam are tuples.
    '''
    
    t = t[np.newaxis].T

    if type(sig) is np.ndarray and type(Lam) is np.ndarray:
        assert sig.shape == Lam.shape, ('Lam and sig arrays should' 
                                        + 'have same size.')
        assert sig.ndim == 1, 'Lam and sig have to be vectors.'
        sig_r = sig[np.newaxis]
        Lam_r = Lam[np.newaxis]
    elif type(sig) is np.ndarray and type(Lam) is tuple:
        assert sig.ndim == 1, 'sig has to be a vector.'
        assert len(Lam) == 2, 'Lam has to be a tuple of two numbers.'
        sig_r = sig[np.newaxis]
        no_samples = sig.size 
        Lam_r = (Lam[0] + Lam[1] * np.random.rand(no_samples))[np.newaxis]
    elif type(sig) is tuple and type(Lam) is np.ndarray:
        assert Lam.ndim == 1, 'Lam has to be a vector.'
        assert len(sig) == 2, 'sig has to be a tuple of two numbers.'
        Lam_r = Lam[np.newaxis]
        no_samples = Lam.size
        sig_r = (sig[0] + sig[1] * np.random.rand(no_samples))[np.newaxis]
    elif type(sig) is tuple and type(Lam) is tuple:
        assert len(Lam) == 2, 'Lam has to be a tuple of two numbers.'
        assert len(sig) == 2, 'sig has to be a tuple of two numbers.'
        Lam_r = (Lam[0] + Lam[1] * np.random.rand(no_samples))[np.newaxis]
        sig_r = (sig[0] + sig[1] * np.random.rand(no_samples))[np.newaxis]        
    else:
        raise ValueError('Lam and sig should be either numpy arrays or tuples'
                           + 'of two numbers.')
    
    a = (a0 * (1 / 3 + 2 / 3 * (1 - (sig_r * t) ** 2) * np.exp(- 
            (sig_r * t) ** 2 / 2)) * np.exp(-Lam_r * t) + ab)

    if type(er) == int:
        noise = er * np.random.randn(t.size, no_samples)
        e = er * np.ones((t.size, no_samples))
    elif type(er) == np.ndarray:
        assert er.ndim == 1, 'er has to be a vector.'
        assert er.size == t.size, 'er size has to be equal to t size.'
        noise = er[np.newaxis].T * np.random.randn(t.size, no_samples)
        e = np.repeat(er[np.newaxis].T, no_samples, axis=1)
    else:
        raise ValueError('er should be either integer or a vector.')

    return np.transpose(np.array([np.repeat(t, no_samples, axis=1), a + noise,
                    e]), axes=(1, 2, 0)), sig_r[0,:], Lam_r[0,:]

    
        
