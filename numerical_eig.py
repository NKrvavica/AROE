# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:44:37 2019

@author: PC-KRVAVICA
"""

import numpy as np
from scipy import linalg


def numerical_eig(A):
    ''' Computes real numerical eigenvalues `Lam` and eigenvectors `K`
    of matrix `A`::

        A = [[0, 1, 0, 0],
             [c1**2-u1**2, 2*u1, c1**2, 0],
             [0, 0, 0, 1],
             [r*c2**2, 0, c2**2 - u2**2, 2*u2]]

    If complex eigenvalues are computed, it corrects them by using real Jordan
    decomposition.

    Parameters
    ----------
    A: ndarray
        stacked array of flux matrix

    Returns
    -------
    Lam: ndarray
        stacked eigenvalue arrays
    K: ndarray or empty
        stacked 4x4 matrices whose columns are right eigenvectors (empty if
        eigevces is set to `False`)
    '''

    # Get eigenvalues and eigenvectors
    Lam, K = np.linalg.eig(A)

    # If complex eigenvalues are found apply real Jordan decomposition
    if np.iscomplex(Lam).any():
        idx = np.unique(np.argwhere(np.iscomplex(Lam))[:, 0])
        L_real, K_real = linalg.cdf2rdf(Lam[idx, :], K[idx, :])
        Lam = Lam.real
        for j in range(4):
            Lam[idx, j] = L_real[:, j, j]
        K = K.real
        K[idx, :, :] = K_real[:, :, :]

    return Lam, K
