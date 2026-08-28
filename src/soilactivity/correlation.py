from __future__ import annotations

import numpy as np

__all__ = [
    "information_correlation_coefficient",
    "entropy",
]


def _histogram_counts(x: np.ndarray, bins: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram counts and probabilities."""
    counts, edges = np.histogram(x, bins=bins, density=False)
    total = counts.sum()
    if total == 0:
        return np.array([]), np.array([])
    probs = counts.astype(np.float64) / total
    return probs, edges


def entropy(x: np.ndarray, bins: int = 30, base: float = 2.0) -> float:
    """Shannon entropy of a 1D sample via histogram binning.

    H(X) = -sum(p_i * log_b(p_i))

    Parameters
    ----------
    x : np.ndarray
        Data values.
    bins : int
        Number of histogram bins.
    base : float
        Logarithm base (2 for bits, e for nats).

    Returns
    -------
    float
        Shannon entropy.
    """
    p, _ = _histogram_counts(x, bins)
    if len(p) == 0:
        return 0.0
    p = p[p > 0]
    return float(-np.sum(p * np.log(p) / np.log(base)))


def _joint_entropy(X: np.ndarray, Y: np.ndarray, bins: int) -> float:
    """Joint entropy H(X,Y) via 2D histogram."""
    H2d, _, _ = np.histogram2d(X, Y, bins=bins)
    p2d = H2d.ravel().astype(np.float64)
    total = p2d.sum()
    if total == 0:
        return 0.0
    p2d = p2d / total
    p2d = p2d[p2d > 0]
    return float(-np.sum(p2d * np.log(p2d)))


def information_correlation_coefficient(
    X: np.ndarray,
    Y: np.ndarray,
    bins: int = 30,
) -> float:
    """Information correlation coefficient R(X, Y) after Linfoot (1957).

    R(X,Y) = sqrt(1 - exp(-2 * I(X;Y) / H(X,Y)))

    where I(X;Y) = H(X) + H(Y) - H(X,Y) is the mutual information
    estimated from 2D histogram binning.

    This coefficient can be used for arbitrarily distributed random
    variables (not only Gaussian), unlike Pearson's r.

    R ranges from 0 (no correlation) to 1 (perfect dependence).

    Parameters
    ----------
    X, Y : np.ndarray
        Paired measurements (same length).
    bins : int
        Number of bins per axis for 2D histogram.

    Returns
    -------
    float
        Information correlation coefficient R in [0, 1].

    References
    ----------
    1. Linfoot E (1957) An informational measure of correlation.
       Information and Control 1 85-89.
    2. Cover T M, Thomas J A (2012) Elements of Information Theory.
    3. Chizhov et al (2019) J. Radiol. Prot. 39 354-372, eq.(18).
    """
    X = np.asarray(X, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    assert len(X) == len(Y), "X and Y must have same length"

    H_X = entropy(X, bins)
    H_Y = entropy(Y, bins)
    H_XY = _joint_entropy(X, Y, bins)

    if H_XY < 1e-15:
        return 0.0

    # Normalised mutual information
    I_XY = H_X + H_Y - H_XY
    I_XY = max(I_XY, 0.0)  # numerical guard
    T = I_XY / H_XY  # normalised, in [0, 0.5]

    R = np.sqrt(1.0 - np.exp(-2.0 * T))
    return float(np.clip(R, 0.0, 1.0))
