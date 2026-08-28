"""Diagnostics for the Fredholm SLAE quality assessment.

Provides condition number estimation and error bounds for the
inverse problem of SAD reconstruction from ADER.

Refs: Chizhov et al (2023) J. Radiol. Prot. 43 041505, eq.(10-12).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "slae_condition_number",
    "slae_error_bound",
    "slae_finer_error_estimate",
]


def slae_condition_number(
    F: np.ndarray,
    norm: str = "spectral",
) -> float:
    """Compute condition number of matrix F.

    Parameters
    ----------
    F : np.ndarray (N, N)
        Fredholm matrix.
    norm : str
        'spectral' (2-norm), '1-norm', or 'inf-norm'.

    Returns
    -------
    float
        cond(F).
    """
    if norm == "spectral":
        # Use SVD for stability
        s = np.linalg.svd(F, compute_uv=False)
        return float(s[0] / s[-1]) if s[-1] > 0 else float('inf')
    elif norm == "1-norm":
        return float(np.linalg.cond(F, 1))
    elif norm == "inf-norm":
        return float(np.linalg.cond(F, np.inf))
    else:
        raise ValueError(f"Unknown norm: {norm}")


def slae_error_bound(
    F: np.ndarray,
    delta_P: float,
    delta_F: float = 0.0,
    norm: str = "spectral",
) -> float:
    """Upper bound on relative error of SLAE solution.

    ||dA|| / ||A|| <= cond(F) * (||dP||/||P|| + ||dF||/||F||)

    Parameters
    ----------
    F : np.ndarray
        Fredholm matrix.
    delta_P : float
        Relative error in right-hand side ||dP||/||P||.
    delta_F : float
        Relative error in matrix ||dF||/||F||. Default 0.
    norm : str
        Norm type for condition number.

    Returns
    -------
    float
        Upper bound on relative error epsilon_A.
    """
    cond_F = slae_condition_number(F, norm)
    return cond_F * (delta_P + delta_F)


def slae_finer_error_estimate(
    F: np.ndarray,
    A: np.ndarray,
    P: np.ndarray,
    alpha: float,
) -> float:
    """Finer error estimate nu(P) (Tikhonov-regularised solution).

    nu(P) = ||(alpha*E + F^T F)^{-1}|| * ||F^T P|| / ||A||

    1 <= nu <= cond(F). This is a tighter bound than cond(F).

    Parameters
    ----------
    F : np.ndarray (N, N)
    P : np.ndarray (N,)
    A : np.ndarray (N,)
        Reconstructed solution.
    alpha : float
        Regularisation parameter.

    Returns
    -------
    float
        nu(P) estimate.
    """
    N = F.shape[0]
    FtF = F.T @ F
    reg = alpha * np.eye(N) + FtF
    FtP = F.T @ P

    inv_reg_norm = float(1.0 / np.linalg.norm(reg, 2))  # approx
    FtP_norm = float(np.linalg.norm(FtP, 2))
    A_norm = float(np.linalg.norm(A, 2))

    if A_norm < 1e-30:
        return float('inf')

    return inv_reg_norm * FtP_norm / A_norm
