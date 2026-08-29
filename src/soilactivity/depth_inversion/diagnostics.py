"""Resolution diagnostics for Fredholm depth inversion (analogous
to seismic tomography).

Provides resolution matrix (PSF), depth of investigation (DOI),
and SVD spectrum analysis.

References
----------
- Menke (2018) Geophysical Data Analysis.
- Backus & Gilbert (1968, 1970) resolving power.
- Zhdanov (2002) Geophysical Inverse Theory.
"""
from __future__ import division, print_function, absolute_import

import numpy as np


def _weighted(K, sigma):
    """Apply data weights w = 1/sigma to kernel."""
    K = np.asarray(K, float)
    if sigma is None:
        return K
    return K / np.asarray(sigma, float)[:, None]


def resolution_matrix(K, sigma=None, alpha=0.0, L=None):
    """Resolution matrix R = (A^T A + alpha * L^T L)^{-1} A^T A.

    R shows how a delta-function layer at depth j is recovered across
    all depths (point-spread function).  Diagonal dominance = good
    vertical resolution; off-diagonal spread = smearing.

    Parameters
    ----------
    K : array-like (m, n)
    sigma : array-like (m,) or None
    alpha : float
        Regularisation parameter.  alpha=0 gives unregularised R.
    L : array-like (p, n) or None

    Returns
    -------
    R : ndarray (n, n)
    """
    A = _weighted(K, sigma)
    n = A.shape[1]
    L = np.eye(n) if L is None else L
    M = A.T @ A + alpha * (L.T @ L)
    return np.linalg.solve(M, A.T @ A)


def depth_of_investigation(R, z, frac=0.5):
    """Depth of investigation z*: depth where cumulative |PSF_j| reaches frac.

    For each column j of the resolution matrix, the cumulative sum
    of |R[:, j]| gives the cumulative resolution power.  The depth
    at which this reaches *frac* (default 0.5) is the DOI for that
    measurement configuration.

    Parameters
    ----------
    R : ndarray (n, n)
        Resolution matrix.
    z : array-like (n,)
        Depth grid [m].
    frac : float
        Cumulative fraction threshold (0 to 1).

    Returns
    -------
    z_doi : np.ndarray (n,)
        Depth of investigation for each column.
    """
    z = np.asarray(z, float)
    out = np.empty(R.shape[1])
    for j in range(R.shape[1]):
        p = np.abs(R[:, j])
        s = p.sum()
        p = p / s if s > 0 else p
        cdf = np.cumsum(p)
        out[j] = np.interp(frac, cdf, z)
    return out


def singulars(K, sigma=None):
    """SVD spectrum of the depth-normalised weighted system.

    Returns
    -------
    sv : ndarray
        Singular values of A * S (unit-norm columns).
    scale : ndarray
        1 / ||K_i||  (the depth-scale factors).
    """
    A = _weighted(K, sigma)
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0
    sv = np.linalg.svd(A / scale, compute_uv=False)
    return sv, 1.0 / scale
