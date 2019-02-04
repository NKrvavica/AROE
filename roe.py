# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:41:43 2019

@author: PC-KRVAVICA
"""

import numpy as np
import analytic_eig
import numerical_eig


def recompose_matrix(K, L):
    '''
    Recomposes a matrix from known eigenvalues `L` and eigevectors `K`.

    Instead of solving `Q = K@L@inv(K)` derived from equality `QK = KL`,
    the algorithm rewrites the equation as `K.T Q.T=(KL).T`,
    and uses `linalg.solve` to solve `Ax=B`, where `A=K.T`, `x=A.T`,
    and `B = (KL).T`. Finally `Q` is derived as `Q=x.T`.

    Parameters
    ----------
    K: stacked array
        matrices where columns are right eigenvectors
    L: stacked array
        matrices where diagonal values are eigenvalues

    Returns
    -------
    Q: stacked array
    '''
    A = K.transpose(0, 2, 1)
    B = (L[:, None, :] * K).transpose(0, 2, 1)
    return np.linalg.solve(A, B).transpose(0, 2, 1)


def harten_regularization(Lam, eps=1e-1):
    ''' Performs Harten regularization, which is needed if one of the
    eigenvelues is zero.

    Parameters
    ----------
    Lam: ndarray
        stacked eigenvalue arrays
    eps: float, optional
        Harten's parameter

    Returns
    ------_
    A_abs: ndarray
        stacked arrays of apsolute values of eigenvalues,
        eigenvalues are corrected by Hartens regularizations if one of them
        is equal to zero.
    '''
    A_abs = np.abs(Lam)
    A_abs += (0.5 * ((1 + np.sign(eps - A_abs))
              * ((Lam**2 + eps*eps) / (2*eps) - A_abs)))
    return A_abs


def comp_Q(u1, u2, h1, h2, r, g, A, eig_type='numerical', hyp_corr=True):
    ''' Returns the numerical viscosity matrix of a two-layer shallow water
    system. A analytical implementation of the Roe scheme, which belongs to
    the family of Riemann solvers.

    Parameters
    ----------
    u1: ndarray
        velocities of the upper layer
    u2: ndarray
        velocities of the lower layer
    h1: ndarray
        depths of the upper layer
    h2: ndarray
        depths of the lower layer
    r: flot or ndarray
        relative density `r = rho1/rho2`, where `rho1` and `rho2` are the
        respective densities of the upper and lower layer.
    g: float or ndarray, optional
        acceleration of gravity
    A: ndarray
        stacked flux Jacobian matrices of the two-layer shallow water system
    eig_type: string, optional
        type of eigenvalues ('numerical', 'analytical', 'approximated')
    hyp_corr: bool, optional
        if set to `True` hyperbolicity correction is performed

    Returns
    -------
    Q: ndarray
        stacked 4x4 numerical viscosity matrices
    P_plus: ndarray
        stacked 4x4 projection matrices of positive sign elements
    P_minus: ndarray
        stacked 4x4 projection matrices of negative sign elements
    Lam: ndarray
        stacked eigenvalue arrays
    F: ndarray
        array of correction friction (0 for hyperbolic system)
        '''
    # Get eigenvalues and eigenvectors
    if eig_type == 'numerical':
        Lam, K = numerical_eig.numerical_eig(A)
        F = []
    elif eig_type == 'analytical':
        Lam, K, F = analytic_eig.analytic_eig(u1, u2, h1, h2, r, g,
                                              hyp_corr=hyp_corr)
    elif eig_type == 'approximated':
        Lam, K, F = analytic_eig.analytic_eig(u1, u2, h1, h2, r, g,
                                              approx=True, hyp_corr=hyp_corr)
    else:
        raise ValueError('''wrong type of calculation, expected either '''
                         ''''numerical', 'analytical', or 'approximated' ''')

    # Perform Harten regularization
    A_abs = harten_regularization(Lam)

    # Recompose viscosity and projection matrices
    Q = recompose_matrix(K, A_abs)
    sign_A = 0.5 * np.sign(Lam)
    Pp = recompose_matrix(K, 0.5 + sign_A)
    Pm = recompose_matrix(K, 0.5 - sign_A)

    return Q, Pp, Pm, Lam, F
