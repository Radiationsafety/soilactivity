"""Lorenz curves for comparing compactness of SAD and ADER distributions.

The Lorenz curve is a graphical representation of the cumulative
distribution function.  In the context of SAD/ADER analysis, it is used
to assess how compact (non-uniform) a spatial distribution is.

A 45-degree line represents a perfectly uniform distribution.
A curve that bows towards the lower-right corner indicates a
non-uniform (compact) distribution.

When the Lorenz curve for SAD is more compact than for ADER, solving
the Fredholm equation provides useful additional information about
the spatial distribution of radiation sources.

Refs: Chizhov et al (2023) J. Radiol. Prot. 43 041505, Section 2.4.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "lorenz_curve",
    "lorenz_gini_coefficient",
    "lorenz_compactness_ratio",
]


def lorenz_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Lorenz curve for a 1D array of values.

    Parameters
    ----------
    values : np.ndarray
        Flattened raster values (SAD or ADER per cell).

    Returns
    -------
    (x, y) : tuple of np.ndarray
        Cumulative fraction of cells (x) and cumulative fraction
        of total value (y).  Both in [0, 1].  Length = len(values) + 1
        (includes origin (0, 0)).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = np.maximum(v, 0.0)  # non-negative

    # Sort ascending
    v_sorted = np.sort(v)
    cumsum = np.cumsum(v_sorted)
    total = cumsum[-1] if cumsum[-1] > 0 else 1.0

    n = len(v_sorted)
    # x: fraction of cells, y: fraction of total value
    x = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    y = np.concatenate([[0.0], cumsum / total])

    return x, y


def lorenz_gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient from values (alternative to full curve).

    Gini = 1 - 2 * integral(Lorenz_curve)  (trapezoidal rule).
    Gini = 0 for perfectly uniform, approaches 1 for extreme inequality.

    Parameters
    ----------
    values : np.ndarray
        Flattened raster values.

    Returns
    -------
    float
        Gini coefficient in [0, 1).
    """
    x, y = lorenz_curve(values)
    # Trapezoidal integral of Lorenz curve
    area = np.trapz(y, x)
    gini = 1.0 - 2.0 * area
    return float(max(gini, 0.0))


def lorenz_compactness_ratio(
    sad: np.ndarray,
    ader: np.ndarray,
) -> float:
    """Ratio of Gini coefficients Gini(SAD) / Gini(ADER).

    If > 1, SAD is more compact than ADER, meaning the Fredholm
    equation solution adds useful spatial information.

    If < 1, ADER is already more compact; the inverse problem may
    not significantly improve localisation.

    Parameters
    ----------
    sad : np.ndarray
        SAD raster (flattened or 2D).
    ader : np.ndarray
        ADER raster (same shape).

    Returns
    -------
    float
        Gini(SAD) / Gini(ADER). Returns 0.0 if Gini(ADER) = 0.
    """
    g_sad = lorenz_gini_coefficient(sad)
    g_ader = lorenz_gini_coefficient(ader)
    if g_ader < 1e-15:
        return 0.0
    return float(g_sad / g_ader)
