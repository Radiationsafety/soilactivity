"""Regularised solvers for the discretised Fredholm equation of the first kind.

Classical methods:
  - Tikhonov (NNLS via lsq_linear)
  - TSVD (truncated SVD with depth normalisation)
  - Landweber (steepest descent with semi-convergence)
  - Cimmino (row-action / simultaneous Kaczmarz)

Geophysical inversion methods (2015-2025):
  - CGLS (conjugate gradient least squares with early stopping)
  - Kaczmarz (sequential row-action with randomised ordering)
  - TV/ADMM (total variation via alternating direction method of multipliers)
  - FISTA (fast iterative shrinkage-thresholding for L1/sparse inversion)
  - IRLS-MLS (iteratively reweighted least squares, minimum-length support,
    Portniaguine & Zhdanov 1999 — focusing/compact inversion)

All solvers accept weighted system (K, d, sigma) and return non-negative
solutions by default.

References
----------
- Tikhonov & Arsenin (1977).
- Hansen (1998) Rank-Deficient and Discrete Ill-Posed Problems.
- Engl, Hanke & Neubauer (1996) Regularization of Inverse Problems.
- Li & Oldenburg (1996, 1998) — depth weighting.
- Chen et al. (2024) Adaptive CGLS, Pure Appl. Geophys. 181:203.
- Vatankhah et al. (2018) TV regularization 3-D gravity, GJI 213(1):695.
- Portniaguine & Zhdanov (1999) Focusing geophysical inversion, Geophysics 64(3):874.
- Beck & Teboulle (2009) FISTA, SIAM J. Imaging Sci. 2(1):183.
- Hasan et al. (2022, 2023) Regularised/Bayesian inversion borehole gamma, J. Env. Radioact.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.linalg import svd
from scipy.optimize import lsq_linear


# =====================================================================
# Utility functions
# =====================================================================

def diff_matrix(n, order=1):
    """Finite-difference matrix of given order.

    Parameters
    ----------
    n : int
        Number of depth cells.
    order : {1, 2}
        Order of the difference operator.

    Returns
    -------
    D : ndarray (n - order, n)
        First-order: D_1 (n-1, n).  Second-order: D_2 (n-2, n).
    """
    if order == 1:
        D = np.zeros((n - 1, n))
        i = np.arange(n - 1)
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    elif order == 2:
        D = np.zeros((n - 2, n))
        i = np.arange(n - 2)
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    else:
        raise ValueError("order must be 1 or 2, got {}".format(order))
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


# =====================================================================
# Classical solvers
# =====================================================================

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
    """Cimmino (row-action / simultaneous Kaczmarz) method.

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


# =====================================================================
# Geophysical solvers (2015-2025)
# =====================================================================

def cgls(K, d, sigma=None, x0=None, nonneg=True, chi2_target=None,
         max_iter=500, tol=1e-12):
    """Conjugate Gradient Least Squares with early stopping.

    Solves the normal equations A^T A x = A^T d iteratively.
    The iteration count k acts as the regularisation parameter:
    too few iterations = under-fitting, too many = noise amplification
    (semi-convergence).  Early stopping is the implicit regularisation.

    Based on Chen et al. (2024) "Adaptive CGLS", Pure Appl. Geophys. 181:203.

    Parameters
    ----------
    K : array-like (m, n)
    d : array-like (m,)
    sigma : array-like (m,) or None
        Data uncertainties.  Used for chi2_target computation.
    x0 : array-like (n,) or None
        Initial guess.  Default: zero.
    nonneg : bool
        Project solution onto x >= 0 at each iteration.
    chi2_target : float or None
        Stop when ||Ax - d||^2 <= chi2_target.
        Default: m (discrepancy principle for unit-weighted data).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on residual norm.

    Returns
    -------
    x : np.ndarray (n,)
    info : dict with 'iterations', 'chi2_history', 'residual_history'.
    """
    A, b = _weighted(K, d, sigma)
    n = A.shape[1]
    x = np.zeros(n) if x0 is None else np.asarray(x0, float).copy()
    if nonneg:
        x = np.maximum(x, 0.0)

    r = b - A @ x                    # residual
    s = A.T @ r                      # normal residual (A^T r)
    p = s.copy()                     # search direction
    sTs = float(s @ s)

    m = len(b)
    if chi2_target is None:
        chi2_target = float(m)

    res_hist = np.empty(max_iter)
    chi2_hist = np.empty(max_iter)

    for it in range(max_iter):
        q = A @ p                       # A @ search direction
        qTq = float(q @ q)
        if qTq < 1e-30:
            break
        alpha_cg = sTs / qTq
        x = x + alpha_cg * p
        if nonneg:
            x = np.maximum(x, 0.0)
        r = r - alpha_cg * q
        s_new = A.T @ r
        sTs_new = float(s_new @ s_new)
        beta = sTs_new / max(sTs, 1e-30)
        p = s_new + beta * p
        sTs = sTs_new

        res_hist[it] = float(r @ r)
        chi2_hist[it] = res_hist[it]

        if res_hist[it] <= chi2_target or res_hist[it] < tol:
            it += 1
            break

    return x, {
        "iterations": it,
        "chi2_history": chi2_hist[:it],
        "residual_history": res_hist[:it],
    }


def kaczmarz(K, d, sigma=None, x0=None, nonneg=True, chi2_target=None,
             max_iter=5000, seed=0, randomized=True):
    """Kaczmarz (ART — Algebraic Reconstruction Technique) method.

    Sequential row-action: projects onto each equation hyperplane
    x_{k+1} = x_k + (d_i - a_i^T x_k) / ||a_i||^2 * a_i.

    Randomised ordering (Strohmer & Vershynin, 2009) converges
    exponentially for row-norm-normalised systems.

    Parameters
    ----------
    K, d, sigma : as in tikhonov()
    x0, nonneg, chi2_target, max_iter : as in cgls()
    seed : int
        Random seed for row ordering.
    randomized : bool
        Use randomised cyclic order (True) or sequential (False).

    Returns
    -------
    x : np.ndarray (n,)
    info : dict with 'iterations', 'chi2_history'.
    """
    A, b = _weighted(K, d, sigma)
    m, n = A.shape
    x = np.zeros(n) if x0 is None else np.asarray(x0, float).copy()
    if nonneg:
        x = np.maximum(x, 0.0)

    row_norms_sq = np.einsum("ij,ij->i", A, A)
    row_norms_sq[row_norms_sq == 0] = 1.0

    if chi2_target is None:
        chi2_target = float(m)

    rng = np.random.default_rng(seed)
    chi2_hist = []

    for it in range(max_iter):
        if randomized:
            idx = rng.permutation(m)
        else:
            idx = np.arange(m)
        for i in idx:
            r_i = b[i] - A[i] @ x
            x = x + (r_i / row_norms_sq[i]) * A[i]
            if nonneg:
                x = np.maximum(x, 0.0)

        res = b - A @ x
        c2 = float(res @ res)
        chi2_hist.append(c2)
        if c2 <= chi2_target:
            it += 1
            break

    return x, {
        "iterations": it,
        "chi2_history": np.array(chi2_hist),
    }


def _soft_threshold(x, tau):
    """Soft-thresholding operator S_tau(x) = sign(x) * max(|x| - tau, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def fista(K, d, alpha, sigma=None, x0=None, nonneg=True,
          max_iter=2000, tol=1e-10):
    """FISTA: Fast Iterative Shrinkage-Thresholding Algorithm (L1 sparse).

    Solves  min ||Ax - b||^2 / 2  +  alpha * ||x||_1  subject to x >= 0.

    Uses Nesterov acceleration (Beck & Teboulle, 2009) for O(1/k^2)
    convergence vs O(1/k) for ISTA.

    The L1 penalty promotes sparsity — ideal for compact radionuclide
    profiles where activity is concentrated in a few thin layers.

    Parameters
    ----------
    K : array-like (m, n)
    d : array-like (m,)
    alpha : float
        L1 regularisation weight.
    sigma : array-like (m,) or None
    x0 : array-like (n,) or None
    nonneg : bool
        Enforce non-negativity (threshold at max(0, x - alpha/L)).
    max_iter : int
    tol : float
        Convergence tolerance on solution change.

    Returns
    -------
    x : np.ndarray (n,)
    info : dict with 'iterations', 'objective_history'.
    """
    A, b = _weighted(K, d, sigma)
    n = A.shape[1]
    L = float(np.linalg.svd(A, compute_uv=False)[0] ** 2)  # Lipschitz constant
    step = 1.0 / L
    tau = alpha * step

    x = np.zeros(n) if x0 is None else np.asarray(x0, float).copy()
    y = x.copy()
    t = 1.0

    obj_hist = []

    for it in range(max_iter):
        x_old = x.copy()
        grad = A.T @ (A @ y - b)           # gradient of data misfit at y
        x = y - step * grad

        # Proximal step: soft thresholding
        if nonneg:
            # Non-negative soft threshold: max(0, x - tau)
            x = np.maximum(x - tau, 0.0)
        else:
            x = _soft_threshold(x, tau)

        # Nesterov momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x + ((t - 1.0) / t_new) * (x - x_old)
        t = t_new

        # Objective: 0.5 * ||Ax - b||^2 + alpha * ||x||_1
        r = A @ x - b
        obj = 0.5 * float(r @ r) + alpha * float(np.sum(np.abs(x)))
        obj_hist.append(obj)

        if np.max(np.abs(x - x_old)) < tol:
            break

    return x, {
        "iterations": it + 1,
        "objective_history": np.array(obj_hist),
        "lipschitz": L,
    }


def tv_admm(K, d, alpha, sigma=None, x0=None, nonneg=True,
            rho=1.0, max_iter=500, tol=1e-8, anisotropic=True):
    """Total Variation (TV) regularisation via ADMM.

    Solves  min ||Ax - b||^2 / 2  +  alpha * TV(x)

    TV(x) = sum_i |D x_i|  (anisotropic, default)
    TV(x) = sum_i sqrt((D x_i)^2 + eps^2)  (isotropic)

    where D is the first-order finite-difference operator.

    TV preserves sharp layer boundaries — ideal for piecewise-constant
    radionuclide profiles with abrupt interfaces (e.g., contaminated
    layer boundaries).

    Split-Bregman / ADMM formulation:
      min ||Ax - b||^2 / 2 + alpha * ||z||_1   s.t. Dx = z
      m-update: (A^T A + rho D^T D) m = A^T b + rho D^T (z - u)
      z-update: z = S_{alpha/rho}(Dm + u)
      u-update: u = u + Dm - z

    Based on Vatankhah et al. (2018) GJI 213(1):695,
    Dong et al. (2026) J. Seismic Expl. 35(3).

    Parameters
    ----------
    K : array-like (m, n)
    d : array-like (m,)
    alpha : float
        TV regularisation weight.
    sigma : array-like (m,) or None
    x0 : array-like (n,) or None
    nonneg : bool
        Enforce x >= 0 (project after m-update).
    rho : float
        ADMM penalty parameter.  Default 1.0.
    max_iter : int
    tol : float
        Primal/dual residual tolerance.
    anisotropic : bool
        True: sum |Dx_i| (separable, no staircasing in 1D).
        False: sum sqrt((Dx_i)^2 + eps^2) (isotropic, smooth).

    Returns
    -------
    x : np.ndarray (n,)
    info : dict with 'iterations', 'primal_residual', 'objective_history'.
    """
    A, b = _weighted(K, d, sigma)
    m_dim = A.shape[1]
    D = diff_matrix(m_dim)               # (n-1, n)
    n_D = D.shape[0]

    x = np.zeros(m_dim) if x0 is None else np.asarray(x0, float).copy()
    z = D @ x
    u = np.zeros(n_D)

    eps_iso = 1e-6  # isotropic smoothing parameter

    # Pre-factor (A^T A + rho D^T D) — constant throughout iterations
    M = A.T @ A + rho * (D.T @ D)
    # Add small diagonal regularisation for numerical stability
    M += 1e-12 * np.eye(m_dim)
    M_chol = np.linalg.cholesky(M)

    obj_hist = []
    primal_res = []

    for it in range(max_iter):
        # m-update: solve (A^T A + rho D^T D) x = A^T b + rho D^T (z - u)
        rhs = A.T @ b + rho * D.T @ (z - u)
        x = np.linalg.solve(M_chol, rhs)  # using Cholesky
        # Back-substitution via cholesky: scipy.linalg.cho_solve would be more efficient
        # but np.linalg.solve with Cholesky factor works fine for small systems
        # Actually let's use proper cho_solve:
        from scipy.linalg import cho_solve
        x = cho_solve((M_chol, True), rhs)

        if nonneg:
            x = np.maximum(x, 0.0)

        Dx = D @ x

        # z-update: soft thresholding
        v = Dx + u
        if anisotropic:
            z = _soft_threshold(v, alpha / rho)
        else:
            # Isotropic TV: proximal of sqrt(v^2 + eps^2)
            norm_v = np.sqrt(v ** 2 + eps_iso ** 2)
            shrink = np.maximum(norm_v - alpha / rho, 0.0) / np.maximum(norm_v, 1e-30)
            z = v * shrink

        # u-update (dual variable)
        u = u + Dx - z

        # Compute residuals for convergence
        primal = float(np.linalg.norm(Dx - z))
        dual = float(rho * np.linalg.norm(D.T @ (z - (D @ x + u - u))))
        # Simplified dual: rho * ||D^T (z_new - z_old)||
        primal_res.append(primal)

        # Objective
        r = A @ x - b
        tv_val = float(np.sum(np.abs(Dx))) if anisotropic else float(np.sum(np.sqrt(Dx ** 2 + eps_iso ** 2)))
        obj = 0.5 * float(r @ r) + alpha * tv_val
        obj_hist.append(obj)

        if primal < tol and it > 5:
            break

    return x, {
        "iterations": it + 1,
        "primal_residual": np.array(primal_res),
        "objective_history": np.array(obj_hist),
    }


def focusing_irls(K, d, alpha, sigma=None, x0=None, nonneg=True,
                  mode="mgs", eps_focusing=1e-2, max_irls=30, tol=1e-6):
    """Focusing (compact) inversion via IRLS — Portniaguine & Zhdanov (1999).

    Minimises  ||Ax - b||^2 + alpha * S(x)  where S is a focusing
    stabilising functional that drives most model parameters to zero,
    producing compact, sharp-boundary solutions.

    Minimum Support (MS, mode='ms'):
        S_MS(x) = sum_i x_i^2 / (x_i^2 + e^2)
        Penalises non-zero values — produces sparse/compact solutions
        where only a few depth layers have activity.

    Minimum Gradient Support (MGS, mode='mgs', default):
        S_MGS(x) = sum_i (Dx_i)^2 / ((Dx_i)^2 + e^2)
        Penalises non-zero gradients — produces blocky/layered models
        with sharp interfaces and flat interiors.

    Solved via Iteratively Reweighted Least Squares (IRLS):
        At each iteration k, S(x) is approximated quadratically:
        S(x) ~ (x - x_ref)^T W(x_k)^{-1} (x - x_ref)
        where W_ii = x_i^2 + e^2 (MS) or (Dx_i)^2 + e^2 (MGS).
        Then solve the weighted Tikhonov problem:
        (A^T A + alpha W^{-1}) x = A^T b

    References:
        Portniaguine & Zhdanov (1999) Geophysics 64(3):874-887.
        Zhdanov (2002) Geophysical Inverse Theory, Elsevier.
        Zhang et al. (2012) GJI 189(1):296 (MGS for MT inversion).
        Xiang et al. (2017) Earth Planets Space 69:153 (MSG variant).

    Parameters
    ----------
    K : array-like (m, n)
    d : array-like (m,)
    alpha : float
        Focusing regularisation weight.
    sigma : array-like (m,) or None
    x0 : array-like (n,) or None
        Initial guess (hot-start).  Default: Tikhonov solution.
    nonneg : bool
    mode : {'ms', 'mgs'}
        'ms' = minimum support (sparse), 'mgs' = minimum gradient
        support (blocky/layered).  Default: 'mgs'.
    eps_focusing : float
        Focusing parameter e.  Controls sharpness of boundaries.
        Smaller e = sharper but potentially unstable.
        Typical: 0.01 * max(|x|).
    max_irls : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance on solution change.

    Returns
    -------
    x : np.ndarray (n,)
    info : dict with 'iterations', 'support_history', 'weight_history'.
    """
    A, b = _weighted(K, d, sigma)
    n = A.shape[1]
    D = diff_matrix(n) if mode == "mgs" else None

    # Hot-start: smooth Tikhonov solution
    if x0 is None:
        AtA = A.T @ A
        reg = np.eye(n)
        x = np.linalg.solve(AtA + alpha * reg, A.T @ b)
    else:
        x = np.asarray(x0, float).copy()

    if nonneg:
        x = np.maximum(x, 0.0)

    support_hist = []
    weight_hist = []

    for it in range(max_irls):
        x_old = x.copy()

        # Compute focusing weights W
        if mode == "mgs":
            gx = D @ x                     # gradients
            W_diag = gx ** 2 + eps_focusing ** 2   # (n-1,)
        else:  # 'ms'
            W_diag = x ** 2 + eps_focusing ** 2     # (n,)

        support_hist.append(float(np.sum(x ** 2 / (x ** 2 + eps_focusing ** 2))))
        weight_hist.append(W_diag.copy())

        # Solve: (A^T A + alpha * W^{-1}) x = A^T b
        # For MGS: the W operates on Dx, so the system is:
        #   (A^T A + alpha * D^T diag(1/W) D) x = A^T b
        # For MS: the W operates on x, so the system is:
        #   (A^T A + alpha * diag(1/W)) x = A^T b
        if mode == "mgs":
            W_inv = 1.0 / W_diag
            # D^T diag(W_inv) D
            DW = D.T @ (W_inv[:, None] * D)
            M = A.T @ A + alpha * DW
        else:
            W_inv = 1.0 / W_diag
            M = A.T @ A + alpha * np.diag(W_inv)

        # Small regularisation for numerical stability
        M += 1e-14 * np.eye(n)
        x = np.linalg.solve(M, A.T @ b)

        if nonneg:
            x = np.maximum(x, 0.0)

        # Adaptive eps: decrease focusing parameter to sharpen progressively
        # (Zhdanov 2002 recommends progressive focusing)
        eps_focusing = max(eps_focusing * 0.7, 1e-10)

        if np.max(np.abs(x - x_old)) < tol * max(np.max(np.abs(x)), 1e-30):
            break

    return x, {
        "iterations": it + 1,
        "support_history": np.array(support_hist),
        "mode": mode,
    }


# =====================================================================
# Advanced iterative solvers
# =====================================================================

def cgls_lcurve(K, d, sigma=None, x0=None, nonneg=True,
                max_iter=500, tol=1e-12):
    """CGLS with automatic L-curve corner stopping.

    Runs CGLS to max_iter, records residual and solution norm at
    each iteration, then selects the L-curve corner as the optimal
    stopping point (analogous to Tikhonov alpha selection).

    This combines the efficiency of iterative methods with the
    robustness of L-curve parameter selection.

    Parameters
    ----------
    K, d, sigma, x0, nonneg, max_iter, tol : as in cgls().

    Returns
    -------
    x : np.ndarray (n,)
    info : dict with 'iterations', 'corner_iter', 'chi2_history',
          'solution_norm_history'.
    """
    from .criteria import lcurve_corner_iter

    A, b = _weighted(K, d, sigma)
    n = A.shape[1]
    x = np.zeros(n) if x0 is None else np.asarray(x0, float).copy()
    if nonneg:
        x = np.maximum(x, 0.0)

    r = b - A @ x
    s = A.T @ r
    p = s.copy()
    sTs = float(s @ s)

    res_hist = []
    sol_norm_hist = []
    x_history = [x.copy()]

    for it in range(max_iter):
        q = A @ p
        qTq = float(q @ q)
        if qTq < 1e-30:
            break
        alpha_cg = sTs / qTq
        x = x + alpha_cg * p
        if nonneg:
            x = np.maximum(x, 0.0)
        r = r - alpha_cg * q
        s_new = A.T @ r
        sTs_new = float(s_new @ s_new)
        beta = sTs_new / max(sTs, 1e-30)
        p = s_new + beta * p
        sTs = sTs_new

        res_val = float(r @ r)
        res_hist.append(res_val)
        sol_norm_hist.append(float(np.linalg.norm(x)))
        x_history.append(x.copy())

        if res_val < tol:
            break

    # Find L-curve corner
    if len(res_hist) > 2:
        corner, kappa = lcurve_corner_iter(
            np.array(res_hist), np.array(sol_norm_hist))
        corner = min(corner, len(x_history) - 1)
        x = x_history[corner]
    else:
        corner = len(res_hist)

    return x, {
        "iterations": corner,
        "corner_iter": corner,
        "chi2_history": np.array(res_hist),
        "solution_norm_history": np.array(sol_norm_hist),
    }


def crossval_alpha(K, d, sigma, n_folds=5, alphas=None, L=None,
                   nonneg=True, seed=0):
    """K-fold cross-validation for Tikhonov alpha selection.

    Splits data into n_folds, solves on n_folds-1, evaluates
    misfit on the held-out fold.  Selects alpha minimising
    the mean held-out misfit.

    This is a noise-level-free alternative to GCV and discrepancy.

    Parameters
    ----------
    K : array-like (m, n)
    d : array-like (m,)
    sigma : array-like (m,)
    n_folds : int
    alphas : array-like or None
        If None, auto-computed from SVD.
    L : array-like or None
        Regularisation operator.
    nonneg : bool
    seed : int

    Returns
    -------
    alpha_best : float
    cv_scores : ndarray (len(alphas),)
    """
    A, b = _weighted(K, d, sigma)
    m = len(b)
    n = A.shape[1]
    L = np.eye(n) if L is None else L

    if alphas is None:
        lmax = float(np.linalg.svd(A, compute_uv=False)[0] ** 2)
        alphas = np.geomspace(lmax * 1e-10, lmax * 1e-1, 32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(m)
    folds = np.array_split(idx, n_folds)

    cv_err = np.zeros(len(alphas))

    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        A_tr, b_tr = A[train_idx], b[train_idx]
        A_te, b_te = A[test_idx], b[test_idx]

        for j, alpha in enumerate(alphas):
            if nonneg:
                x = _tikhonov_weighted(A_tr, b_tr, alpha, L=L, nonneg=True)
            else:
                M = A_tr.T @ A_tr + alpha * (L.T @ L)
                x = np.linalg.solve(M, A_tr.T @ b_tr)
            cv_err[j] += float(np.sum((A_te @ x - b_te) ** 2))

    cv_err /= m
    return float(alphas[np.argmin(cv_err)]), cv_err


def depth_weighted_tikhonov(K, d, alpha, z, sigma=None, L=None, x0=None,
                              nonneg=True, weight_method='combined', **kw):
    """Tikhonov with depth-dependent weighting of the model.

    Applies depth weighting w(z) so that the regularisation is:
        ||W^{1/2} L (x - x0)||^2
    This counteracts depth-dependent sensitivity decay.

    Parameters
    ----------
    K, d, alpha, sigma, L, x0, nonneg : as in tikhonov().
    z : array-like (n,)
        Depth grid [m].
    weight_method : str
        Passed to compose_weighting in geophysics module.
    **kw : extra keyword arguments for compose_weighting.

    Returns
    -------
    x : ndarray (n,)
    """
    from .geophysics import compose_weighting

    A, b = _weighted(K, d, sigma)
    w = compose_weighting(A, z, method=weight_method, **kw)
    Lw = np.diag(np.sqrt(w)) @ (L if L is not None else np.eye(len(z)))
    return _tikhonov_weighted(A, b, alpha, L=Lw, x0=x0, nonneg=nonneg)

