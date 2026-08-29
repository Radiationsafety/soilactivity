"""Regularised solvers for the discretised Fredholm equation of the first kind.

References
----------
- Tikhonov & Arsenin (1977).
- Hansen (1998) Rank-Deficient and Discrete Ill-Posed Problems.
- Engl, Hanke & Neubauer (1996) Regularization of Inverse Problems.
- Li & Oldenburg (1996, 1998) — depth weighting.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.linalg import svd
from scipy.optimize import lsq_linear


def diff_matrix(n):
    """First-order finite-difference matrix D_1 of shape (n-1, n).

    Used as the smoothness regularisation operator L.
    """
    D = np.zeros((n - 1, n))
    i = np.arange(n - 1)
    D[i, i] = -1.0
    D[i, i + 1] = 1.0
    return D


def depth_scale(K):
    """Column normalisation (Li & Oldenburg, 1996).

    Returns S = 1 / ||K_i|| so that transformed system A = K * S has
    unit-norm columns, counteracting the natural decay of kernel
    columns with depth.

    Parameters
    ----------
    K : array-like, shape (m, n)

    Returns
    -------
    S : np.ndarray, shape (n,)
    """
    S = np.linalg.norm(K, axis=0)
    S[S == 0] = 1.0
    return 1.0 / S


def _weighted(K, d, sigma):
    """Apply data weights w = 1/sigma (or w=1 if sigma is None)."""
    K = np.asarray(K, float)
    d = np.asarray(d, float)
    if sigma is None:
        return K.copy(), d.copy()
    w = 1.0 / np.asarray(sigma, float)
    return K * w[:, None], d * w


def _tikhonov_weighted(A, b, alpha, L=None, x0=None, nonneg=True):
    """Solve min ||Ax - b||^2 + alpha * ||L(x - x0)||^2  subject to x >= 0.

    Parameters
    ----------
    A : ndarray (m, n)
    b : ndarray (m,)
    alpha : float
    L : ndarray (p, n) or None
    x0 : ndarray (n,) or None
    nonneg : bool

    Returns
    -------
    x : ndarray (n,)
    """
    n = A.shape[1]
    L = np.eye(n) if L is None else L
    x0 = np.zeros(n) if x0 is None else np.asarray(x0, float)
    Aa = np.vstack([A, np.sqrt(alpha) * L])
    ba = np.concatenate([b, np.sqrt(alpha) * (L @ x0)])
    if nonneg:
        return lsq_linear(Aa, ba, bounds=(0.0, np.inf), tol=1e-12,
                         max_iter=500).x
    return np.linalg.lstsq(Aa, ba, rcond=None)[0]


def tikhonov(K, d, alpha, sigma=None, L=None, x0=None, nonneg=True):
    """Tikhonov regularisation with non-negativity constraint.

    Solves  min ||(Kx - d) / sigma||^2  +  alpha * ||L(x - x0)||^2,  x >= 0.

    Parameters
    ----------
    K : array-like (m, n)
    d : array-like (m,)
    alpha : float
        Regularisation parameter.
    sigma : array-like (m,) or None
        Data standard deviations.
    L : array-like (p, n) or None
        Regularisation operator.  Default: identity.
    x0 : array-like (n,) or None
        Reference model.  Default: zero.
    nonneg : bool
        Enforce non-negativity via NNLS.

    Returns
    -------
    x : np.ndarray (n,)
    """
    A, b = _weighted(K, d, sigma)
    return _tikhonov_weighted(A, b, alpha, L=L, x0=x0, nonneg=nonneg)


def tsvd(K, d, sigma=None, rel_cutoff=1e-3, nonneg=False):
    """Truncated SVD in depth-normalised column space.

    Parameters
    ----------
    K : array-like (m, n)
    d : array-like (m,)
    sigma : array-like (m,) or None
    rel_cutoff : float
        Keep singular values > sv[0] * rel_cutoff.
    nonneg : bool

    Returns
    -------
    x : np.ndarray (n,)
    info : dict with 'rank' and 'sv'.
    """
    A, b = _weighted(K, d, sigma)
    S = depth_scale(A)
    U, sv, Vt = svd(A * S, full_matrices=False)
    k = int(np.sum(sv > sv[0] * rel_cutoff))
    k = max(k, 1)
    y = Vt[:k].T @ ((U[:, :k].T @ b) / sv[:k])
    x = S * y
    if nonneg:
        x = np.maximum(x, 0.0)
    return x, {"rank": k, "sv": sv}


def landweber(K, d, sigma=None, x0=None, nonneg=True, chi2_target=None,
              max_iter=5000, relax=0.9):
    """Landweber iteration with semi-convergence stopping.

    Parameters
    ----------
    K, d, sigma : as in tikhonov()
    x0 : array-like or None
    chi2_target : float or None
        Stop when chi^2 <= chi2_target.  Default: len(d) (discrepancy).
    max_iter : int
    relax : float
        Relaxation factor omega < 2/sv_max^2.

    Returns
    -------
    x : np.ndarray (n,)
    info : dict
    """
    A, b = _weighted(K, d, sigma)
    sv = svd(A, compute_uv=False)
    omega = relax * (2.0 / sv[0] ** 2) if sv[0] > 0 else 1.0
    x = np.zeros(A.shape[1]) if x0 is None else np.asarray(x0, float).copy()
    m = len(b)
    if chi2_target is None:
        chi2_target = float(m)
    hist = np.empty(max_iter)
    for it in range(max_iter):
        r = b - A @ x
        x = x + omega * (A.T @ r)
        if nonneg:
            x = np.maximum(x, 0.0)
        hist[it] = float(r @ r)
        if hist[it] <= chi2_target:
            return x, {"iterations": it + 1, "chi2": hist[:it + 1],
                       "omega": omega}
    return x, {"iterations": max_iter, "chi2": hist, "omega": omega}


def cimmino(K, d, sigma=None, x0=None, nonneg=True, chi2_target=None,
            max_iter=20000, tol=1e-12):
    """Cimmino (row-action) method.

    Convenient for block-structured systems (lines x heights).

    Parameters
    ----------
    K, d, sigma : as in tikhonov()
    x0, nonneg, chi2_target, max_iter, tol : as in landweber()

    Returns
    -------
    x : np.ndarray (n,)
    info : dict
    """
    A, b = _weighted(K, d, sigma)
    nrm = np.einsum("ij,ij->i", A, A)
    nrm[nrm == 0] = 1.0
    x = np.zeros(A.shape[1]) if x0 is None else np.asarray(x0, float).copy()
    m = len(b)
    if chi2_target is None:
        chi2_target = float(m)
    it = 0
    for it in range(max_iter):
        r = b - A @ x
        x = x + ((r / nrm) @ A) / m
        if nonneg:
            x = np.maximum(x, 0.0)
        c2 = float(r @ r)
        if c2 <= chi2_target or c2 < tol:
            break
    return x, {"iterations": it + 1}
