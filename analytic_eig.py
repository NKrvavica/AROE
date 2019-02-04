# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:53:19 2019

@author: PC-KRVAVICA
"""

import numpy as np
import warnings


def analytic_eig(u1, u2, h1, h2, r, g,
                 eigvecs=True, approx=False, hyp_corr=True):
    ''' Computes the analytical closed-form solution to the eigenvalues `Lam`
    and eigenvectors `K` of matrix `A`::

        A = [[0, 1, 0, 0],
             [c1**2-u1**2, 2*u1, c1**2, 0],
             [0, 0, 0, 1],
             [r*c2**2, 0, c2**2 - u2**2, 2*u2]]

    If complex eigenvalues are predicted, it corrects the velocities by
    an optimal friction factor `F` so that the hyperbolic condition is
    restored.

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
    g: float or ndarray
        acceleration of gravity
    eigvecs: bool, optional
        if set to `True` the function computes and returns eigenvalues and
        eigevectors, otherwise it returns only eigevalues
    approx: bool, optional
        if set to `True` the function computes and returns the approximate
        eigenvalues (and eigevectors) insted of the closed-form solution
    hyp_corr: bool, optiona
        is set to `True` is performs the hyperboclity correction where needed.

    Returns
    -------
    K: ndarray or empty
        stacked 4x4 matrices whose columns are right eigenvectors (empty if
        eigevces is set to `False`)
    Lam: ndarray
        stacked eigenvalue arrays
    F: ndarray
        array of correction friction (0 for hyperbolic system)
    '''

    def corr_u(u1, u2, c1_2, c2_2, r, g, F):
        ''' Hyperbolicity correction.
        Modifies velocities `u1` and `u2` by the correction (friction)
        factor F'''
        u1 += F * np.sign(u2 - u1) / c1_2 * g
        u2 -= r * F * np.sign(u2 - u1) / c2_2 * g
        return u1, u2

    def char_poly_coeff(u1, u2, c1_2, c2_2, r, g):
        ''' Computes coefficients of a characteristic polynomial `p(x)`:
        p(x) = x^4 + a*x^3 + b*x^2 + c*x + d'''
        u_c_1 = u1*u1 - c1_2
        u_c_2 = u2*u2 - c2_2
        a = -2*(u1 + u2)
        b = u_c_1 + 4*u1*u2 + u_c_2
        c = - 2*u2*u_c_1 - 2*u1*u_c_2
        d = u_c_1*u_c_2 - r*c1_2*c2_2
        return a, b, c, d

    def get_discr(params, F=0):
        '''Computes the coefficients of the characteristic polynomial and
        finds the discriminant of the characteristic polynomial'''
        u1, u2, c1_2, c2_2, r, g = params
        if F != 0:
            u1, u2 = corr_u(*params, F)
        # Coefficients of the characteristic polynomial'''
        a, b, c, d = char_poly_coeff(u1, u2, c1_2, c2_2, r, g)
        d0 = b*b + 12*d - 3*a*c
        d1 = 27*a*a*d - 9*a*b*c + 2*b*b*b - 72*b*d + 27*c*c
        d0_3 = d0*d0*d0
        delta = d1*d1 - 4*d0_3
        return a, b, c, d, d0, d1, d0_3, delta

    def illinois(f, a, b, params, max_iterations=20, tol=1e-5):
        ''''Iterative Illions algorithm (adapted)'''
        *_, f_a = f(params, a)
        *_, f_b = f(params, b)
        for i in range(max_iterations):
            c = b - f_b * (a - b) / (f_a - f_b)
            *_, f_c = f(params, c)
            if f_b*f_c < 0:
                a = c
                f_a = f_c
            else:
                b = c
                f_b = f_c
                f_a = f_a / 2
            if np.abs(a-b) < tol:
                return np.max([a, b])
        else:
            warnings.warn('Not converging...')
        return np.max([a, b])

    def analytic_eigenvalues(a, b, c, d, d0, d1, d0_3, N):
        '''Computes analytical eigenvalues matrix by finding real roots of
        characteristic polynomial'''
        third = 1./3.
        A = -0.75*a*a + 2*b
        B = 0.25*a*a*a - a*b + 2*c
        fi = np.arccos(d1 * 0.5 / np.sqrt(d0_3))
        Z_2 = third * (2*np.sqrt(d0)*np.cos(third * fi) - A)
        Z = np.sqrt(Z_2)
        B_Z = B / Z
        sqrt1 = np.sqrt(-A - Z_2 + B_Z)
        sqrt2 = np.sqrt(-A - Z_2 - B_Z)
        a025 = - 0.25*a
        Lam = np.zeros((N, 4))
        Lam[:, 0] = a025 - 0.5*(Z + sqrt1)
        Lam[:, 1] = a025 - 0.5*(Z - sqrt1)
        Lam[:, 2] = a025 + 0.5*(Z - sqrt2)
        Lam[:, 3] = a025 + 0.5*(Z + sqrt2)
        return Lam

    def approximate_eigenvalues(u1, u2, c1_2, c2_2, r, g, N):
        ''' Computes the approximation of the eigenvalues for the two-layer
        system, based on the assumption that r=1, and u1=u2'''
        c12_2 = 1 / (c1_2 + c2_2)
        a = (u1 * c1_2 + u2 * c2_2) * c12_2
        b = np.sqrt(c1_2 + c2_2)
        c = (u1 * c2_2 + u2 * c1_2) * c12_2
        d = np.sqrt((1-r) * c1_2 * c2_2 * c12_2
                    * (1 - (u1 - u2)**2 / (1-r) * c12_2))
        Lam = np.zeros((N, 4))
        Lam[:, 0] = a - b
        Lam[:, 1] = c - d
        Lam[:, 2] = c + d
        Lam[:, 3] = a + b
        return Lam

    def analytic_eigenvectors(u1, c1_2, g, Lam, N):
        '''Computes analytical eigevector matrix from eigenvalues'''
        Id = np.ones((N, 4))
        k = 1 - (Lam - u1[:, np.newaxis])**2 / c1_2[:, np.newaxis]
        K = np.zeros((N, 4, 4))
        K[:, 0, :] = Id
        K[:, 1, :] = Lam
        K[:, 2, :] = -k
        K[:, 3, :] = -k * Lam
        return K

    # get coefficient and discriminant of the characteristic polynomial
    N = u1.size  # Sample size
    c1_2 = g * h1
    c2_2 = g * h2

    # Hyperbolicity correction
    F = np.zeros((N, 1))
    if hyp_corr or not approx:
        params = (u1, u2, c1_2, c2_2, r, g)
        (a, b, c, d, d0, d1, d0_3, delta) = get_discr(params)
        '''if any discriminant > 0, complex eigenvalues will be found,
        correct velocities to prevent hyperbolicity loss'''
        if (delta >= 0).any():
            C = (u1 - u2)**2 / ((1-r) * (c1_2 + c2_2))
            for idx in np.where(delta >= 0)[0]:
                a = 0
                b = (np.sqrt((1-r) * (c1_2[idx] + c2_2[idx]))
                     * (np.sqrt(C[idx]) - 1) / (1/c1_2[idx] + r/c2_2[idx]) / g)
                params_idx = (u1[idx], u2[idx], c1_2[idx], c2_2[idx], r, g)
                F[idx] = illinois(get_discr, a, b, params_idx)
                u1[idx], u2[idx] = corr_u(*params_idx, F[idx])
            (a, b, c, d, d0, d1, d0_3, delta) = get_discr(params)

    # Compute eigenvalues
    if approx:
        Lam = approximate_eigenvalues(u1, u2, c1_2, c2_2, r, g, N)
    else:
        Lam = analytic_eigenvalues(a, b, c, d, d0, d1, d0_3, N)

    # Compute eigenvectors
    if eigvecs:
        K = analytic_eigenvectors(u1, c1_2, g, Lam, N)
    else:
        K = []

    return Lam, K, F
