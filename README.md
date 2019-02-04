# AROE

Analytical implementation of the Roe method for two-layer shallow water systems written in Python.

This is a numerical algorithm that supplements a recent article:

`Krvavica, N., Tuhtan, M., & Jelenić, G. (2018). Analytical implementation of Roe solver for two-layer shallow water equations with accurate treatment for loss of hyperbolicity. Advances in Water Resources, 122, 187-205. DOI: 10.1016/j.advwatres.2018.10.017`

A preprint of the article is available on [ArXiv](https://arxiv.org/abs/1810.11285).


# From the abstract...

A new implementation of the Roe scheme for solving two-layer shallow-water equations is presented. The proposed A-Roe scheme is based on the analytical solution to the characteristic quartic of the flux matrix, which is an efficient alternative to a numerical eigensolver. Additionally, an accurate method for maintaining the hyperbolic character of the governing system is proposed. The proposed A-Roe scheme is as accurate as the Roe scheme, but much faster.


# Requirements
 
Python 3+, Numpy, Scipy
 
 
# Usage

The main file is `roe.py`, which takes flow parameters (velocities, depth, relative density difference, and flux matrix) and computes the corresponding numerical viscosity and projection matrices crucial for the Roe scheme. The numerical vicosity matrix is based on the eigenstructure of the flux matrix. The eigenvalues and eigenvactors are computed either numerically (`numerical_eig.py` which calls `numpy.linalg.eig`), approximately (`analytical_eig.py` which uses approximate expressions for eigenvalues), or analytically (`analytical_eig.py` which uses closed-form solutions to the characteristic 4th order polynomials of the flux matrix).

For detail usage and performance see `test_Roe_implementation.py` which compares the efficiency (speed and accuracy) of all three implementation options.


# License
 
[MIT license](LICENSE)