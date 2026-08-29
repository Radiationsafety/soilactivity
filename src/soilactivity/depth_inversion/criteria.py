"""Regularisation parameter selection criteria: GCV, L-curve, discrepancy.

All operate in the weighted space (A = K/sigma, b = d/sigma).

References
----------
- Golub, Heath & Wahba (1979) GCV.
- Hansen (1992, 2006) L-curve.
- Morozov (1966) discrepancy principle.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.linalg import svd
from scipy.optimize import brentq


def chi2(K, d, x, sigma=None):
    """Chi-squared misfit: ||Kx - d||_W^2 where W = diag(1/sigma).

    Parameters
    ----------
    K : array-like (m, n)
    d : array-like (m,)
    x : array-like (n,)
    sigma : array-like (m,) or None

    Returns
    -------
    chi2 : float
    """
    r = np.asarray(K, float) @ np.asarray(x, float) - np.asarray(d, float)
    if sigma is not None:
        r = r / np.asarray(sigma, float)
    return float(r @ r)


def _lin_solve(A, b, alpha, L=None):
    """Solve the (unconstrained) Tikhonov system, return (x, M).
    M = A^T A + alpha * L^T L.
    """
    n = A.shape[1]
    L = np.eye(n) if L is None else L
    M = A.T @ A + alpha * (L.T @ L)
    x = np.linalg.solve(M, A.T @ b)
    return x, M


def _chi2_alpha(A, b, alpha, L):
    x, _ = _lin_solve(A, b, alpha, L)
    r = A @ x - b
    return float(r @ r)


def gcv_point(A, b, alpha, L=None, n_probe=24, seed=0):
    """GCV(alpha) = n * ||Ax - b||^2 / tr(I - H)^2.

    The trace is estimated via Hutchinson's stochastic trace estimator
    with n_probe random +/-1 probe vectors.

    Parameters
    ----------
    A : ndarray (m, n)  — weighted system matrix
    b : ndarray (m,)
    alpha : float
    L : ndarray (p, n) or None
    n_probe : int
    seed : int

    Returns
    -------
    gcv : float
    """
    x, M = _lin_solve(A, b, alpha, L)
    r = A @ x - b
    n = len(b)
    rng = np.random.default_rng(seed)
    V = rng.choice([-1.0, 1.0], size=(n, n_probe))
    HV = A @ np.linalg.solve(M, A.T @ V)
    tr = float(np.mean(np.sum(V * (V - HV), axis=0)))
    return float(n * (r @ r) / max(tr, 1e-12) ** 2)


def gcv_curve(A, b, alphas, L=None, **kw):
    """Evaluate GCV for a grid of alpha values.

    Returns
    -------
    gcv_vals : np.ndarray, shape (len(alphas),)
    """
    return np.array([gcv_point(A, b, a, L, **kw) for a in alphas])


def _menger_curvature(p1, p2, p3):
    """Menger curvature of three 2-D points (log-space)."""
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    a = np.hypot(x2 - x1, y2 - y1)
    b_ = np.hypot(x3 - x2, y3 - y2)
    c = np.hypot(x3 - x1, y3 - y1)
    s2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return 2.0 * s2 / max(a * b_ * c, 1e-300)


def lcurve_corner(A, b, alphas, L=None):
    """L-curve corner: maximum Menger curvature on (log ||Lx||, log ||r||).

    Parameters
    ----------
    A : ndarray (m, n)
    b : ndarray (m,)
    alphas : array-like
    L : ndarray (p, n) or None

    Returns
    -------
    alpha_corner : float
    kappa : ndarray, shape (len(alphas),)
        Curvature at each alpha (0 at endpoints).
    """
    alphas = np.asarray(alphas, float)
    n = A.shape[1]
    Lf = np.eye(n) if L is None else L
    reg = np.empty(len(alphas))
    res = np.empty(len(alphas))
    for j, a in enumerate(alphas):
        x, _ = _lin_solve(A, b, a, L)
        reg[j] = np.log(max(np.linalg.norm(Lf @ x), 1e-300))
        r = A @ x - b
        res[j] = np.log(max(float(r @ r), 1e-300))
    kappa = np.zeros(len(alphas))
    for j in range(1, len(alphas) - 1):
        kappa[j] = _menger_curvature(
            (reg[j - 1], res[j - 1]),
            (reg[j], res[j]),
            (reg[j + 1], res[j + 1]))
    return float(alphas[int(np.argmax(kappa))]), kappa


def choose_alpha_discrepancy(A, b, L=None, n_target=None,
                              lo=1e-12, hi=1e3):
    """Morozov discrepancy principle: find alpha such that chi^2(alpha) = n.

    Uses Brent's method on a log-spaced grid.

    Parameters
    ----------
    A : ndarray (m, n)
    b : ndarray (m,)
    L : ndarray or None
    n_target : float or None
        Target chi^2 (default: m = number of data).
    lo, hi : float
        Search bounds on alpha.

    Returns
    -------
    alpha : float
    """
    n = len(b) if n_target is None else n_target
    grid = np.geomspace(lo, hi, 64)
    vals = np.array([_chi2_alpha(A, b, a, L) for a in grid])
    idx = np.where(vals < n)[0]
    if idx.size == 0:
        return float(grid[-1])
    j = int(idx[-1])
    if j >= len(grid) - 1:
        return float(grid[-1])
    try:
        la = brentq(lambda t: _chi2_alpha(A, b, 10.0 ** t, L) - n,
                     np.log10(grid[j]), np.log10(grid[j + 1]), xtol=1e-3)
        return float(10.0 ** la)
    except ValueError:
        return float(grid[j])


def quasi_optimality(A, b, alphas, L=None):
    """Quasi-optimality criterion (Hochstenbach & Reichel, 2015).

    Selects alpha minimising noise-dominated SVD components.
    Does NOT require noise level estimate.
    """
    alphas = np.asarray(alphas, float)
    U, sv, Vt = svd(A, full_matrices=False)
    Ub = U.T @ b
    qo_vals = np.empty(len(alphas))
    for j, alpha in enumerate(alphas):
        mask = sv ** 2 < alpha
        if np.any(mask):
            qo_vals[j] = float(np.sqrt(np.sum((Ub[mask] / sv[mask]) ** 2)))
        else:
            qo_vals[j] = 0.0
    return float(alphas[int(np.argmin(qo_vals))]), qo_vals


def ncp_criterion(A, b, alphas, L=None):
    """Normalised Cumulative Periodogram (NCP) residual whiteness test.

    Selects alpha where residual is closest to white noise (KS test).
    """
    alphas = np.asarray(alphas, float)
    m = len(b)
    ncp_vals = np.empty(len(alphas))
    for j, alpha in enumerate(alphas):
        x, _ = _lin_solve(A, b, alpha, L)
        r = A @ x - b
        R = np.fft.rfft(r)
        I = np.abs(R) ** 2
        I[0] = 0.0
        I_sum = np.sum(I)
        if I_sum < 1e-30:
            ncp_vals[j] = 1.0
            continue
        cdf = np.cumsum(I) / I_sum
        j_arr = np.arange(len(cdf))
        ks = np.max(np.abs(cdf - j_arr / max(len(cdf) - 1, 1)))
        ncp_vals[j] = float(ks)
    return float(alphas[int(np.argmin(ncp_vals))]), ncp_vals


def snr_criterion(A, b, alphas, L=None):
    """Signal-to-Noise Ratio criterion.

    Selects alpha maximising estimated SNR of the solution.
    """
    alphas = np.asarray(alphas, float)
    m, n = A.shape
    U, sv, Vt = svd(A, full_matrices=False)
    Ub = U.T @ b
    snr_vals = np.empty(len(alphas))
    for j, alpha in enumerate(alphas):
        x_norm2 = float(np.sum((Ub * sv / (sv ** 2 + alpha)) ** 2))
        cov_tr = float(np.sum(1.0 / (sv ** 2 + alpha)))
        x_svd = Vt.T @ (Ub * sv / (sv ** 2 + alpha))
        r = A @ x_svd - b
        sigma2 = float(r @ r) / max(m - n, 1)
        snr_vals[j] = x_norm2 / max(sigma2 * cov_tr, 1e-30)
    return float(alphas[int(np.argmax(snr_vals))]), snr_vals


def gcv_weighted(A, b, alpha, L=None, sigma_data=None, n_probe=24, seed=0):
    """Weighted GCV for heteroscedastic noise (Poisson)."""
    if sigma_data is None:
        return gcv_point(A, b, alpha, L, n_probe, seed)
    w = 1.0 / (np.asarray(sigma_data, float) ** 2)
    Aw = A * np.sqrt(w)[:, None]
    bw = b * np.sqrt(w)
    x, M = _lin_solve(Aw, bw, alpha, L)
    r = Aw @ x - bw
    n = len(b)
    numer = float(r @ r)
    rng = np.random.default_rng(seed)
    V = rng.choice([-1.0, 1.0], size=(n, n_probe))
    HV = Aw @ np.linalg.solve(M, Aw.T @ V)
    tr = float(np.mean(np.sum(V * (V - HV), axis=0)))
    denom = max(tr, 1e-12) ** 2
    return float(n * numer / denom)


def gcv_weighted_curve(A, b, alphas, L=None, sigma_data=None, **kw):
    """Evaluate weighted GCV for a grid of alpha values."""
    return np.array([gcv_weighted(A, b, a, L, sigma_data=sigma_data, **kw)
                       for a in alphas])


def lcurve_corner_iter(hist_residual, hist_solution_norm):
    """L-curve corner for iterative methods (CGLS, Landweber)."""
    r = np.asarray(hist_residual, float)
    s = np.asarray(hist_solution_norm, float)
    eps = 1e-300
    lr = np.log(np.maximum(r, eps))
    ls = np.log(np.maximum(s, eps))
    n = len(lr)
    kappa = np.zeros(n)
    for j in range(1, n - 1):
        kappa[j] = _menger_curvature(
            (lr[j - 1], ls[j - 1]),
            (lr[j], ls[j]),
            (lr[j + 1], ls[j + 1]))
    return int(np.argmax(kappa)), kappa
