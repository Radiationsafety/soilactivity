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
from numpy.linalg import svd


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


# =====================================================================
# Extended geophysical diagnostics (2015-2025)
# =====================================================================

def model_covariance(K, sigma=None, alpha=1.0, L=None):
    """Linearised posterior model covariance (Tikhonov)."""
    A = _weighted(K, sigma)
    n = A.shape[1]
    L = np.eye(n) if L is None else L
    M = A.T @ A + alpha * (L.T @ L)
    return np.linalg.inv(M)


def spread_function(R, z):
    """Backus-Gilbert spread function."""
    z = np.asarray(z, float)
    n = len(z)
    dz = z[1] - z[0] if n > 1 else 1.0
    row_sum = np.sum(R, axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    Rn = R / row_sum
    spread = np.zeros(n)
    for i in range(n):
        spread[i] = 12.0 * np.sum(Rn[i] ** 2 * (z - z[i]) ** 2) * dz
    return spread


def checkerboard_test(K, sigma=None, alpha=1.0, L=None, z=None, block_size=2):
    """Checkerboard resolution test (seismic tomography)."""
    K = np.asarray(K, float)
    n = K.shape[1]
    m_true = np.zeros(n)
    for i in range(n):
        m_true[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
    d = K @ m_true
    from .solvers import _tikhonov_weighted
    A = _weighted(K, sigma)
    b = d.copy()
    if sigma is not None:
        b = b / np.asarray(sigma, float)
    m_rec = _tikhonov_weighted(A, b, alpha, L=L, nonneg=False)
    m_true_c = m_true - m_true.mean()
    m_rec_c = m_rec - m_rec.mean()
    denom = np.linalg.norm(m_true_c) * np.linalg.norm(m_rec_c)
    recovery = float(np.dot(m_true_c, m_rec_c) / denom) if denom > 0 else 0.0
    return {"model_true": m_true, "model_rec": m_rec, "recovery_ratio": recovery}


def data_resolution_matrix(K, sigma=None, alpha=0.0, L=None):
    """Data resolution matrix (hat matrix)."""
    A = _weighted(K, sigma)
    n = A.shape[1]
    L = np.eye(n) if L is None else L
    M = A.T @ A + alpha * (L.T @ L)
    return A @ np.linalg.solve(M, A.T)


def sensitivity_kernels(K, sigma=None, z=None):
    """Per-data-channel sensitivity kernels."""
    A = _weighted(K, sigma)
    m, n = A.shape
    kernels = np.empty_like(A)
    peak_depth = np.empty(m)
    integral = np.empty(m)
    for i in range(m):
        row = np.abs(A[i])
        peak = np.max(row)
        kernels[i] = row / peak if peak > 0 else row
        peak_depth[i] = (z[i] if z is not None else float(np.argmax(row)))
        integral[i] = float(np.sum(np.abs(A[i])))
    return {"kernels": kernels, "peak_depth": peak_depth, "integral": integral}


def information_content(K, sigma=None):
    """Effective degrees of freedom via trace."""
    A = _weighted(K, sigma)
    sv = svd(A, compute_uv=False)
    trace_N = float(np.sum(sv ** 2 / (sv ** 2 + 0.0)))
    rank_eff = float(np.sum(sv) / sv[0]) if sv[0] > 0 else 0.0
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    return {"trace_N": trace_N, "trace_R": trace_N, "cond": cond, "rank_eff": rank_eff}
