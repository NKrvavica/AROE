# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 17:29:40 2019

@author: Nino
"""

import roe
import timeit
import numpy as np


def get_flux_matrix(u1, u2, h1, h2, r, g=9.8):
    # matrix B (two-layer coupling)
    B = np.zeros([N, 4, 4])
    B[:, 1, 2] = -g * h1
    B[:, 3, 0] = -g * h2 * r

    # Jasobian matrix
    J = np.zeros([N, 4, 4])
    J[:, 0, 1] = 1
    J[:, 1, 0] = - u1*u1 + g * h1
    J[:, 1, 1] = 2*u1
    J[:, 2, 3] = 1
    J[:, 3, 2] = - u2*u2 + g * h2
    J[:, 3, 3] = 2*u2

    # flux matrix
    return J - B


def rmse(prediction, target):
    return np.sqrt(np.mean((prediction-target)**2))


# Generate random parameters
N = 10_000
g = 9.8
r = np.random.rand(N)*0.65 + 0.3
u1 = np.random.rand(N)*0.3
u2 = -np.random.rand(N)*0.3
h1 = np.random.rand(N)*1 + 1
h2 = np.random.rand(N)*1 + 1
A = get_flux_matrix(u1, u2, h1, h2, r, g)

runs = 10

print('\nCPU speed analysis:')

times = []
for i in range(runs):
    start = timeit.default_timer()
    (Q_NRoe, Pp_NRoe, Pm_NRoe,
     Lam_NRoe, F_NRoe) = roe.comp_Q(u1, u2, h1, h2, r, g, A,
                                    eig_type='numerical')
    stop = timeit.default_timer()
    time = stop - start
    times.append(time)
print('Numerical Roe: {:.2f} ms (average of {} runs)'
      .format(np.array(times).mean()*1000, runs))

times = []
for i in range(runs):
    start = timeit.default_timer()
    (Q_ARoe, Pp_ARoe, Pm_ARoe,
     Lam_ARoe, F_ARoe) = roe.comp_Q(u1, u2, h1, h2, r, g, A,
                                    eig_type='analytical', hyp_corr=True)
    stop = timeit.default_timer()
    time = stop - start
    times.append(time)
print('Analytical Roe: {:.2f} ms (average of {} runs)'
      .format(np.array(times).mean()*1000, runs))

times = []
for i in range(runs):
    start = timeit.default_timer()
    (Q_ERoe, Pp_ERoe, Pm_ERoe,
     Lam_ERoe, F_ERoe) = roe.comp_Q(u1, u2, h1, h2, r, g, A,
                                    eig_type='approximated', hyp_corr=True)
    stop = timeit.default_timer()
    time = stop - start
    times.append(time)
print('Approximated Roe: {:.2f} ms (average of {} runs)'
      .format(np.array(times).mean()*1000, runs))

print('\nError analysis:')
print('Analytical Roe RMSE: {:.2e}'.format(rmse(Q_ARoe, Q_NRoe)))
print('Approximated Roe RMSE: {:.2e}'.format(rmse(Q_ERoe, Q_NRoe)))
