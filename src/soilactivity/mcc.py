"""Method of Conversion Coefficients (MCC) for ADER -> SAD.

The simplest approach to estimate surface activity density (SAD) from
ambient dose equivalent rate (ADER): divide ADER by a conversion
coefficient that relates the two quantities for a given radionuclide.

    SAD = ADER / C_mcc

where C_mcc depends on radionuclide, detection height L, and the
dose-rate quantity used.

The MCC method is adequate only for LOW-GRADIENT contamination patterns.
For heterogeneous (complex) distributions use the Fredholm equation
solver (soilactivity.fredholm) instead.

References
----------
1. Chizhov K et al (2019) J. Radiol. Prot. 39 354-372.
2. Saito K et al (2012) Radiat. Environ. Biophys. 51 411-423.
3. Ninkovic M, Adrovic F (2012) Air Kerma Rate Constants for Nuclides
   Important to Gamma Ray Dosimetry. DOI: 10.5772/39170.
"""
from __future__ import annotations

import numpy as np

from .radionuclides import KERMA_CONSTANTS, get_normalizing_factor

__all__ = [
    "mcc_ader_to_sad",
    "mcc_sad_to_ader",
    "mcc_coefficient",
    "mcc_total_activity",
]


def mcc_coefficient(
    kerma_constant: float,
    height_m: float = 1.0,
    cell_area_m2: float = 1.0,
    dose_quantity: str = "H_star_10",
    radionuclide: str = "Cs-137",
) -> float:
    """Compute the MCC conversion coefficient C_mcc.

    For an infinite uniform plane source at height L, the kerma rate at
    height L above the surface is related to SAD by an integral that
    yields a height-dependent coefficient.  For a single cell of area
    S_cell the relationship simplifies to:

        ADER ~ W * K_gamma * A * S_cell / (L^2)  (approximate, near field)

    For the standard MCC (infinite-plane model):

        C_mcc = W * K_gamma * pi  [for L >> contamination extent]

    In practice, the coefficient is often determined empirically or via
    the specific air kerma rate (SAKR) for a given geometry.

    This function returns the coefficient as:

        C_mcc = W * K_gamma * pi

    which corresponds to the theoretical infinite-plane approximation.

    Parameters
    ----------
    kerma_constant : float
        Kerma constant K_gamma [aGy m^2 s^-1 Bq^-1].
    height_m : float
        Detection height above surface [m].  Used for documentation;
        the standard MCC coefficient is height-independent in the
        infinite-plane limit.
    cell_area_m2 : float
        Area of one raster cell [m^2]. Default 1.0.
    dose_quantity : str
        'D_air', 'K_air', 'H_star_10', or 'X'.
    radionuclide : str
        Radionuclide name for W lookup.

    Returns
    -------
    float
        C_mcc in units of [P_unit / Bq], where P_unit depends on
        dose_quantity (e.g. Sv/s for H_star_10).
    """
    W = get_normalizing_factor(dose_quantity, radionuclide, kerma_constant)
    return W * kerma_constant * np.pi


def mcc_ader_to_sad(
    ader: np.ndarray,
    kerma_constant: float,
    dose_quantity: str = "H_star_10",
    radionuclide: str = "Cs-137",
    cell_area_m2: float = 1.0,
) -> np.ndarray:
    """Convert ADER raster to SAD using the Method of Conversion Coefficients.

    SAD_i = ADER_i / C_mcc

    Parameters
    ----------
    ader : np.ndarray
        2D array of ADER values at each raster cell.
    kerma_constant : float
        Kerma constant K_gamma [aGy m^2 s^-1 Bq^-1].
    dose_quantity : str
        Dose-rate quantity of the ADER values.
    radionuclide : str
        Radionuclide name for W lookup.
    cell_area_m2 : float
        Area of one raster cell [m^2].

    Returns
    -------
    np.ndarray
        2D array of SAD values [Bq] per cell.

    Notes
    -----
    MCC is adequate ONLY for low-gradient SAD distributions.
    For heterogeneous contamination, use the Fredholm equation solver.
    """
    C = mcc_coefficient(kerma_constant, cell_area_m2=cell_area_m2,
                        dose_quantity=dose_quantity, radionuclide=radionuclide)
    sad = ader / C
    # SAD cannot be negative
    return np.maximum(sad, 0.0)


def mcc_sad_to_ader(
    sad: np.ndarray,
    kerma_constant: float,
    dose_quantity: str = "H_star_10",
    radionuclide: str = "Cs-137",
    cell_area_m2: float = 1.0,
) -> np.ndarray:
    """Convert SAD raster to ADER (forward problem) using MCC.

    ADER_i = C_mcc * SAD_i

    Parameters
    ----------
    sad : np.ndarray
        2D array of SAD values [Bq] per cell.
    kerma_constant : float
        Kerma constant K_gamma.
    dose_quantity : str
        Dose-rate quantity to compute.
    radionuclide : str
        Radionuclide name.
    cell_area_m2 : float
        Area of one raster cell [m^2].

    Returns
    -------
    np.ndarray
        2D array of ADER values.
    """
    C = mcc_coefficient(kerma_constant, cell_area_m2=cell_area_m2,
                        dose_quantity=dose_quantity, radionuclide=radionuclide)
    return C * sad


def mcc_total_activity(
    ader: np.ndarray,
    kerma_constant: float,
    cell_area_m2: float = 1.0,
    dose_quantity: str = "H_star_10",
    radionuclide: str = "Cs-137",
) -> float:
    """Estimate total surface activity from ADER map using MCC.

    A_total = sum(ADER_i) / C_mcc

    Parameters
    ----------
    ader : np.ndarray
        2D array of ADER values.
    kerma_constant : float
        Kerma constant K_gamma.
    cell_area_m2 : float
        Area of one raster cell [m^2].
    dose_quantity : str
        Dose-rate quantity.
    radionuclide : str
        Radionuclide name.

    Returns
    -------
    float
        Total surface activity [Bq].
    """
    C = mcc_coefficient(kerma_constant, cell_area_m2=cell_area_m2,
                        dose_quantity=dose_quantity, radionuclide=radionuclide)
    return float(np.sum(ader) / C)
