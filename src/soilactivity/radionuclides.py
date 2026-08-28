"""Kerma constants and normalising factors for common radionuclides.

Provides:
- KERMA_CONSTANTS: dict of radionuclide -> kerma constant K_gamma [aGy m^2 s^-1 Bq^-1].
- NORMALIZING_FACTORS: dict of dose-rate quantity -> normalising factor W
  for converting between different dose-rate representations in the
  Fredholm equation  P = W * integral(Q * A dx dy).
- KERMA_CONSTANTS_CS137, KERMA_CONSTANTS_CO60: convenience aliases.

References
----------
1. Mashkovich V, Kudryavtseva A (1995) Protection from Ionizing Radiation.
   Moscow: Energoatomizdat. (in Russian)
2. Ninkovic M, Adrovic F (2012) Air Kerma Rate Constants for Nuclides
   Important to Gamma Ray Dosimetry. DOI: 10.5772/39170.
3. Jacob P et al (1990) GSF-2/90. Calculation of organ doses from
   environmental gamma-rays using human phantoms and Monte Carlo methods.
4. ICRP Publication 74 (1996) Conversion Coefficients for use in
   Radiological Protection against External Radiation.
"""
from __future__ import annotations

from typing import Dict, Tuple

__all__ = [
    "KERMA_CONSTANTS",
    "NORMALIZING_FACTORS",
    "NORMALIZING_FACTORS_BY_RADIONUCLIDE",
    "SAKR_CS137_ROOF",
    "CHERNOBYL_FUEL_VECTOR_131D",
]


# -----------------------------------------------------------------------------
# Kerma constants K_gamma  [aGy * m^2 / (s * Bq)]
#
# Air kerma rate at 1 m from a unit-activity point source (neglecting
# scatter).  K_gamma is related to the specific air kerma rate (SAKR)
# for a thin infinite-plane source at depth 0.5 g/cm^2 by geometry.
#
# Sources: Mashkovich & Kudryavtseva 1995; Ninkovic & Adrovic 2012.
# -----------------------------------------------------------------------------
KERMA_CONSTANTS: Dict[str, float] = {
    # --- Fission / activation products ---
    "Cs-137": 21.3,       # 661.7 keV (via 137mBa, T1/2=156 s, yield 94.6%)
    "Cs-134": 57.6,       # multi-line (605, 796, 802 keV ...)
    "Co-60":  137.0,      # 1173.2 + 1332.5 keV (both yield ~1.0)
    "Co-58":  59.1,       # 810.8 keV
    "Eu-152": 122.0,      # multi-line
    "Eu-154": 68.0,       # multi-line
    "Sr-90": 0.0,         # pure beta; no gamma -> K_gamma = 0
    "Y-90":   0.0,         # pure beta
    "I-131":  43.3,       # 364.5 keV dominant
    "Ba-140": 27.0,       # 537.3 keV
    "Zr-95":  27.4,       # 756.7, 724.2 keV
    "Nb-95":  28.3,       # 765.8 keV
    "Ru-103": 17.7,       # 497.1 keV
    "Ru-106": 7.5,        # 511.9 keV (via Rh-106)
    "Ce-141": 2.9,        # 145.4 keV
    "Ce-144": 1.0,        # 133.5 keV (via Pr-144)
    "La-140": 69.0,       # 1596.5 keV
    "Mn-54":  147.0,      # 834.8 keV
    "Fe-59":  133.0,      # 1099.3, 1291.6 keV
    "Zn-65":  75.0,       # 1115.5 keV
    "Sb-124": 187.0,      # multi-line 602-1691 keV
    "Am-241": 22.3,       # 59.5 keV
}

# Convenience aliases (lowercase, without hyphens)
KERMA_CONSTANTS_CS137 = KERMA_CONSTANTS["Cs-137"]   # 21.3
KERMA_CONSTANTS_CO60 = KERMA_CONSTANTS["Co-60"]    # 137.0


# -----------------------------------------------------------------------------
# Normalising factors W  for the Fredholm equation
#
#   P(x,y,L) = W * integral[ Q(x,y,x',y',L) * Vis(x,y,x',y') * A(x',y') dx' dy' ]
#
# The factor W converts the kerma-unit kernel into the chosen dose-rate
# quantity P.  For a given radionuclide, different dose-rate quantities
# require different W.
#
# Source: Chizhov et al (2019) J. Radiol. Prot. 39 354-372, Table 1.
# -----------------------------------------------------------------------------
#
# Structure:  dose_quantity -> {radionuclide: W}
# W units:  [P-unit / aGy]
# -----------------------------------------------------------------------------

# For Cs-137 (K_gamma = 21.3 aGy m^2 s^-1 Bq^-1):
#   W for ADER  = 1.20e-18 Sv/aGy   (ICRP 74 conversion)
#   W for K_air = 1.0e-18  Gy/aGy    (identity: kerma -> kerma)
#   W for D_air = 1.0e-18  Gy/aGy    (electronic balance: K_air ~ D_air)
#   W for X     = 1.145e-16 R/aGy   (exposure -> kerma)

# General normalising factors (dose-rate quantity -> W for a generic
# radionuclide where kerma-constant is known).
# If P = air kerma rate or absorbed dose rate in air (electronic balance),
# W = 1e-18 Gy/aGy regardless of radionuclide.
# If P = ADER (H*(10)), W depends on H*(10)/Ka at the photon energy.
NORMALIZING_FACTORS: Dict[str, Dict[str, float]] = {
    # absorbed dose rate in air  [Gy/s]  ->  W in Gy/aGy
    "D_air": {"_generic": 1.0e-18},
    # air kerma rate  [Gy/s]  ->  W in Gy/aGy
    "K_air": {"_generic": 1.0e-18},
    # ambient dose equivalent rate  [Sv/s]  ->  W in Sv/aGy
    "H_star_10": {
        "Cs-137": 1.20e-18,
        "Co-60":  1.20e-18,   # approximate (E_eff ~ 1.25 MeV, H*(10)/Ka ~ 1.23)
    },
    # exposure rate  [R/s]  ->  W in R/aGy
    "X": {"_generic": 1.145e-16},
}


# -----------------------------------------------------------------------------
# Helper: full normalising factor W for any (quantity, radionuclide) pair
# -----------------------------------------------------------------------------

def get_normalizing_factor(
    dose_quantity: str,
    radionuclide: str = "Cs-137",
    kerma_constant: float | None = None,
) -> float:
    """Return the normalising factor W for the Fredholm equation.

    Parameters
    ----------
    dose_quantity : str
        One of 'D_air', 'K_air', 'H_star_10', 'X'.
    radionuclide : str
        Radionuclide name as in KERMA_CONSTANTS (e.g. 'Cs-137').
    kerma_constant : float or None
        If provided, overrides the lookup.  Useful for nuclide mixtures.

    Returns
    -------
    float
        W in units of [P-unit / aGy].

    Raises
    ------
    ValueError : if the dose_quantity is unknown.
    """
    if dose_quantity not in NORMALIZING_FACTORS:
        raise ValueError(
            f"Unknown dose_quantity '{dose_quantity}'. "
            f"Choose from {list(NORMALIZING_FACTORS.keys())}"
        )

    entry = NORMALIZING_FACTORS[dose_quantity]
    # 1) Try specific radionuclide
    if radionuclide in entry:
        return entry[radionuclide]
    # 2) Try generic
    if "_generic" in entry:
        return entry["_generic"]
    # 3) For H_star_10 with unlisted nuclide: estimate via ICRP 74
    if dose_quantity == "H_star_10" and kerma_constant is not None:
        # Approximate: W ~ 1.20e-18 (good for 600-700 keV;
        # for higher E, H*(10)/Ka ~ 1.2-1.6, so W is proportionally larger).
        # A more precise approach uses the dosimetry module, but this
        # approximation is sufficient for the Fredholm equation context.
        return 1.20e-18
    raise ValueError(
        f"No normalising factor for (dose_quantity='{dose_quantity}', "
        f"radionuclide='{radionuclide}'). Provide kerma_constant explicitly."
    )


# -----------------------------------------------------------------------------
# Normalising factors keyed by radionuclide for quick lookup
# (convenience for users who primarily work with a single nuclide)
# -----------------------------------------------------------------------------
NORMALIZING_FACTORS_BY_RADIONUCLIDE: Dict[str, Dict[str, float]] = {
    "Cs-137": {
        "D_air": 1.0e-18,      # Gy/aGy
        "K_air": 1.0e-18,      # Gy/aGy
        "H_star_10": 1.20e-18, # Sv/aGy
        "X": 1.145e-16,         # R/aGy
    },
    "Co-60": {
        "D_air": 1.0e-18,
        "K_air": 1.0e-18,
        "H_star_10": 1.20e-18,  # approximate
        "X": 1.145e-16,
    },
}


# -----------------------------------------------------------------------------
# Specific Air Kerma Rate (SAKR) for Cs-137 on roofs  [nGy/h / (kBq/m^2)]
# Source: Jacob et al (1990) GSF-2/90; used in Chizhov et al (2019) Table 3.
# For an infinite thin source at depth 0.5 g/cm^2.
# -----------------------------------------------------------------------------
SAKR_CS137_ROOF: float = 1.82  # nGy h^-1 per kBq m^-2


# -----------------------------------------------------------------------------
# Chernobyl Unit 4 fuel composition on day 131 after the accident
# Source: Chizhov et al (2019) J. Radiol. Prot. 39, Table 3.
# Activity fractions (sum = 100%), kerma constants, SAKR.
# -----------------------------------------------------------------------------
CHERNOBYL_FUEL_VECTOR_131D: list[dict] = [
    {"radionuclide": "Nb-95",  "activity_fraction": 0.30, "K_gamma": 28.3, "SAKR": 2.30},
    {"radionuclide": "Zr-95",  "activity_fraction": 0.18, "K_gamma": 27.4, "SAKR": 2.21},
    {"radionuclide": "Ru-103", "activity_fraction": 0.08, "K_gamma": 17.7, "SAKR": 1.45},
    {"radionuclide": "Ru-106", "activity_fraction": 0.06, "K_gamma": 7.5,  "SAKR": 0.62},
    {"radionuclide": "Cs-134", "activity_fraction": 0.01, "K_gamma": 57.6, "SAKR": 4.68},
    {"radionuclide": "Cs-137", "activity_fraction": 0.01, "K_gamma": 21.3, "SAKR": 1.82},
    {"radionuclide": "Ce-141", "activity_fraction": 0.08, "K_gamma": 2.9,  "SAKR": 0.22},
    {"radionuclide": "Ce-144", "activity_fraction": 0.28, "K_gamma": 1.0,  "SAKR": 0.06},
]


def mixture_kerma_constant(fuel_vector: list[dict] | None = None) -> float:
    """Activity-weighted average kerma constant for a radionuclide mixture.

    Parameters
    ----------
    fuel_vector : list of dict or None
        Each dict must have keys 'activity_fraction' and 'K_gamma'.
        If None, uses CHERNOBYL_FUEL_VECTOR_131D.

    Returns
    -------
    float
        Average K_gamma in aGy m^2 s^-1 Bq^-1.
    """
    if fuel_vector is None:
        fuel_vector = CHERNOBYL_FUEL_VECTOR_131D
    total = sum(d["activity_fraction"] * d["K_gamma"] for d in fuel_vector)
    return total


def mixture_sakr(fuel_vector: list[dict] | None = None) -> float:
    """Activity-weighted average specific air kerma rate for a mixture.

    Parameters
    ----------
    fuel_vector : list of dict or None
        Each dict must have keys 'activity_fraction' and 'SAKR'.
        If None, uses CHERNOBYL_FUEL_VECTOR_131D.

    Returns
    -------
    float
        Average SAKR in nGy h^-1 per kBq m^-2.
    """
    if fuel_vector is None:
        fuel_vector = CHERNOBYL_FUEL_VECTOR_131D
    total = sum(d["activity_fraction"] * d["SAKR"] for d in fuel_vector)
    return total
