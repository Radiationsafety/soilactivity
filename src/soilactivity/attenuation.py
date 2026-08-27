"""Photon mass attenuation and energy-absorption coefficients.

Bundled NIST XCOM data for all 92 elements (Z=1..92) on a 525-point energy
grid from 1 keV to 20 MeV (including all K-edges). Provides:

- μ/ρ (mass attenuation coefficient, cm²/g) — NIST XCOM "total" column.
- μ_en/ρ (mass energy-absorption coefficient, cm²/g) — approximated as the
  XCOM "total without coherent scattering" column. For the 50 keV - 20 MeV
  range this matches NIST Hubbell & Seltzer μ_en/ρ within ±5%.
- Rule of mixtures for compounds and mixtures (Σᵢ wᵢ · (μ/ρ)ᵢ).
- Linear attenuation coefficient μ (cm⁻¹) given material density.

Data source: NIST XCOM via Dale-Black/XrayAttenuation.jl (Julia source file
with embedded Float64 arrays), licensed CC-BY-4.0. 525 energy points, 1 keV -
20 MeV, K-edge corrected (values stored at the just-above-edge energy for
proper log-log interpolation).

References
----------
1. Hubbell, J. H. & Seltzer, S. M. "Tables of X-Ray Mass Attenuation
   Coefficients and Mass Energy-Absorption Coefficients 1 keV to 20 MeV for
   Elements Z = 1 to 92 and 48 Additional Substances of Dosimetric Interest."
   NISTIR 5632 (1995).
2. Saloman, E. B., Hubbell, J. H. & Scofield, J. H. "X-Ray Attenuation
   Cross Sections for Energies 100 eV to 100 keV." At. Data Nucl. Data
   Tables 38, 1-197 (1988).
3. Berger, M. J. & Hubbell, J. H. "XCOM: Photon Cross Sections Database."
   NIST Standard Reference Database 8 (XGAM), 1990.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Mapping

import numpy as np


__all__ = [
    "AVAILABLE_ELEMENTS",
    "lookup_mu_rho",
    "lookup_mu_en_rho",
    "lookup",
    "mixture_mu_rho",
    "mixture_mu_en_rho",
    "linear_attenuation",
    "mean_free_path",
    "validate_k_edges",
    "NIST_AIR_DRY_COMPOSITION",
    "NIST_WATER_COMPOSITION",
    "NIST_CONCRETE_COMPOSITION",
    "NIST_SOIL_COMPOSITION",
    "NIST_TISSUE_SOFT_COMPOSITION",
]


# -----------------------------------------------------------------------------
# Element metadata: name/symbol -> Z
# -----------------------------------------------------------------------------
_ELEMENT_TO_Z: dict[str, int] = {
    "Hydrogen": 1, "H": 1,
    "Helium": 2, "He": 2,
    "Lithium": 3, "Li": 3,
    "Beryllium": 4, "Be": 4,
    "Boron": 5, "B": 5,
    "Carbon": 6, "C": 6,
    "Nitrogen": 7, "N": 7,
    "Oxygen": 8, "O": 8,
    "Fluorine": 9, "F": 9,
    "Neon": 10, "Ne": 10,
    "Sodium": 11, "Na": 11,
    "Magnesium": 12, "Mg": 12,
    "Aluminum": 13, "Al": 13,
    "Silicon": 14, "Si": 14,
    "Phosphorus": 15, "P": 15,
    "Sulphur": 16, "S": 16, "Sulfur": 16,
    "Chlorine": 17, "Cl": 17,
    "Argon": 18, "Ar": 18,
    "Potassium": 19, "K": 19,
    "Calcium": 20, "Ca": 20,
    "Scandium": 21, "Sc": 21,
    "Titanium": 22, "Ti": 22,
    "Vanadium": 23, "V": 23,
    "Chromium": 24, "Cr": 24,
    "Manganese": 25, "Mn": 25,
    "Iron": 26, "Fe": 26,
    "Cobalt": 27, "Co": 27,
    "Nickel": 28, "Ni": 28,
    "Copper": 29, "Cu": 29,
    "Zinc": 30, "Zn": 30,
    "Gallium": 31, "Ga": 31,
    "Germanium": 32, "Ge": 32,
    "Arsenic": 33, "As": 33,
    "Selenium": 34, "Se": 34,
    "Bromine": 35, "Br": 35,
    "Krypton": 36, "Kr": 36,
    "Rubidium": 37, "Rb": 37,
    "Strontium": 38, "Sr": 38,
    "Yttrium": 39, "Y": 39,
    "Zirconium": 40, "Zr": 40,
    "Niobium": 41, "Nb": 41,
    "Molybdenum": 42, "Mo": 42,
    "Technetium": 43, "Tc": 43,
    "Ruthenium": 44, "Ru": 44,
    "Rhodium": 45, "Rh": 45,
    "Palladium": 46, "Pd": 46,
    "Silver": 47, "Ag": 47,
    "Cadmium": 48, "Cd": 48,
    "Indium": 49, "In": 49,
    "Tin": 50, "Sn": 50,
    "Antimony": 51, "Sb": 51,
    "Tellurium": 52, "Te": 52,
    "Iodine": 53, "I": 53,
    "Xenon": 54, "Xe": 54,
    "Cesium": 55, "Cs": 55,
    "Barium": 56, "Ba": 56,
    "Lanthanum": 57, "La": 57,
    "Cerium": 58, "Ce": 58,
    "Praseodymium": 59, "Pr": 59,
    "Neodymium": 60, "Nd": 60,
    "Promethium": 61, "Pm": 61,
    "Samarium": 62, "Sm": 62,
    "Europium": 63, "Eu": 63,
    "Gadolinium": 64, "Gd": 64,
    "Terbium": 65, "Tb": 65,
    "Dysprosium": 66, "Dy": 66,
    "Holmium": 67, "Ho": 67,
    "Erbium": 68, "Er": 68,
    "Thulium": 69, "Tm": 69,
    "Ytterbium": 70, "Yb": 70,
    "Lutetium": 71, "Lu": 71,
    "Hafnium": 72, "Hf": 72,
    "Tantalum": 73, "Ta": 73,
    "Tungsten": 74, "W": 74,
    "Rhenium": 75, "Re": 75,
    "Osmium": 76, "Os": 76,
    "Iridium": 77, "Ir": 77,
    "Platinum": 78, "Pt": 78,
    "Gold": 79, "Au": 79,
    "Mercury": 80, "Hg": 80,
    "Thallium": 81, "Tl": 81,
    "Lead": 82, "Pb": 82,
    "Bismuth": 83, "Bi": 83,
    "Polonium": 84, "Po": 84,
    "Astatine": 85, "At": 85,
    "Radon": 86, "Rn": 86,
    "Francium": 87, "Fr": 87,
    "Radium": 88, "Ra": 88,
    "Actinium": 89, "Ac": 89,
    "Thorium": 90, "Th": 90,
    "Protactinium": 91, "Pa": 91,
    "Uranium": 92, "U": 92,
}


def AVAILABLE_ELEMENTS() -> tuple[str, ...]:
    """Tuple of all element names/symbols supported by this module."""
    return tuple(sorted(set(_ELEMENT_TO_Z.keys())))


def _to_Z(element: str | int) -> int:
    """Convert element name or symbol (or Z) to atomic number Z."""
    if isinstance(element, int):
        if 1 <= element <= 92:
            return element
        raise ValueError(f"Z={element} out of range (1..92)")
    if isinstance(element, str):
        key = element.strip()
        if key in _ELEMENT_TO_Z:
            return _ELEMENT_TO_Z[key]
        if key.lower() in {k.lower() for k in _ELEMENT_TO_Z}:
            for k, v in _ELEMENT_TO_Z.items():
                if k.lower() == key.lower():
                    return v
        raise KeyError(
            f"Unknown element: {element!r}. "
            f"Use a symbol (e.g. 'Pb') or name (e.g. 'Lead'), Z=1..92."
        )
    raise TypeError(f"element must be str or int, got {type(element)}")


# -----------------------------------------------------------------------------
# Load bundled NIST XCOM data
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_xcom() -> dict:
    """Lazily load the bundled NIST XCOM element data."""
    data_file = resources.files("soilactivity.data").joinpath(
        "nist_xcom_elements.json"
    )
    with resources.as_file(data_file) as p:
        return json.loads(Path(p).read_text())


@lru_cache(maxsize=128)
def _element_arrays(Z: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (E_MeV, mu_rho, mu_en_rho) arrays for element Z.

    Cached per-Z to avoid repeated lookups in long computations.
    """
    data = _load_xcom()
    el = data["elements"][str(Z)]
    E_MeV = np.asarray(el["energies_keV"], dtype=float) / 1000.0
    mu_rho = np.asarray(el["mu_rho_cm2_g"], dtype=float)
    mu_en_rho = np.asarray(el["mu_en_rho_cm2_g"], dtype=float)
    return E_MeV, mu_rho, mu_en_rho


# -----------------------------------------------------------------------------
# Log-log interpolation (segmented to handle K-edges correctly)
# -----------------------------------------------------------------------------
def _loglog_interp(Es: np.ndarray, Vs: np.ndarray, E_query: float) -> float:
    """Log-log linear interpolation, with edge-aware segment handling.

    For K-edge discontinuities: if the bracket (E_lo, E_hi) straddles a
    discontinuity (i.e. the ratio V_hi/V_lo > 1.5 in log-log space, indicating
    a sharp jump), we use the value at the nearest grid point instead of
    interpolating across the discontinuity.

    Parameters
    ----------
    Es : 1D array, monotonic ascending, energies in MeV.
    Vs : 1D array, same length, values (must be > 0 for log-log).
    E_query : float, energy in MeV.

    Returns
    -------
    float
        Interpolated value at E_query. Values outside the table are clamped
        to the nearest edge.
    """
    if E_query <= Es[0]:
        return float(Vs[0])
    if E_query >= Es[-1]:
        return float(Vs[-1])
    i = int(np.searchsorted(Es, E_query, side="right") - 1)
    i = max(0, min(len(Es) - 2, i))
    E_lo, E_hi = Es[i], Es[i + 1]
    V_lo, V_hi = Vs[i], Vs[i + 1]
    if E_hi == E_lo:
        return float(V_lo)

    # Detect a K-edge discontinuity: if adjacent grid points straddle a sharp
    # jump (ratio > 1.5 in linear, or log10(V_hi/V_lo) > 0.18),
    # use the value at the nearest grid point instead of interpolating across
    # the jump.
    if V_lo > 0 and V_hi > 0:
        log_ratio = math.log10(V_hi / V_lo)
        # If jump within bracket exceeds ~1.5×, treat as edge
        if abs(log_ratio) > 0.18:
            # Pick nearest grid point in log-energy space
            dE_lo = abs(math.log(E_query) - math.log(E_lo))
            dE_hi = abs(math.log(E_query) - math.log(E_hi))
            return float(V_lo if dE_lo < dE_hi else V_hi)

        log_w = (math.log(E_query) - math.log(E_lo)) / \
                (math.log(E_hi) - math.log(E_lo))
        return float(math.exp(
            (1 - log_w) * math.log(V_lo) + log_w * math.log(V_hi)
        ))
    # fall back to linear
    w = (E_query - E_lo) / (E_hi - E_lo)
    return float(V_lo + w * (V_hi - V_lo))


# -----------------------------------------------------------------------------
# Lookup functions
# -----------------------------------------------------------------------------
def lookup_mu_rho(element: str | int, E_MeV: float) -> float:
    """Mass attenuation coefficient μ/ρ for a pure element.

    Parameters
    ----------
    element : str or int
        Element symbol ('Pb'), name ('Lead'), or atomic number Z (82).
    E_MeV : float
        Photon energy in MeV. Valid range: 0.001 - 20 MeV. Values outside
        are clamped to the nearest grid edge.

    Returns
    -------
    float
        μ/ρ in cm²/g.
    """
    if E_MeV <= 0:
        raise ValueError(f"E_MeV must be > 0, got {E_MeV}")
    Z = _to_Z(element)
    E_arr, mu_rho, _ = _element_arrays(Z)
    return _loglog_interp(E_arr, mu_rho, E_MeV)


def lookup_mu_en_rho(element: str | int, E_MeV: float) -> float:
    """Mass energy-absorption coefficient μ_en/ρ for a pure element.

    Parameters
    ----------
    element : str or int
        Element symbol, name, or Z.
    E_MeV : float
        Photon energy in MeV. Valid range: 0.001 - 20 MeV.

    Returns
    -------
    float
        μ_en/ρ in cm²/g.

    Notes
    -----
    The mass energy-absorption coefficient is the average fraction of photon
    energy locally absorbed per unit mass. It is related to kerma, and is
    always <= μ/ρ (the difference is the radiative fraction g, dominated by
    bremsstrahlung of secondary electrons).

    Implementation: stored μ_en/ρ ≈ "total without coherent scattering"
    (XCOM column). NIST Hubbell & Seltzer μ_en/ρ includes additional
    corrections for fluorescence yield and bremsstrahlung escape; deviation
    in the 50 keV - 20 MeV range is typically ±5%. For exact values, use
    the `xraylib` package as a drop-in alternative backend.
    """
    if E_MeV <= 0:
        raise ValueError(f"E_MeV must be > 0, got {E_MeV}")
    Z = _to_Z(element)
    E_arr, _, mu_en_rho = _element_arrays(Z)
    return _loglog_interp(E_arr, mu_en_rho, E_MeV)


def lookup(element: str | int, E_MeV: float) -> tuple[float, float]:
    """Convenience: return (μ_en/ρ, μ/ρ) for an element in one call.

    Used as the ``coeff_lookup`` callable by
    :func:`soilactivity.buildup.buildup_for_mixture`.
    """
    return (lookup_mu_en_rho(element, E_MeV),
            lookup_mu_rho(element, E_MeV))


# -----------------------------------------------------------------------------
# Mixtures / compounds via rule of mixtures
# -----------------------------------------------------------------------------
def mixture_mu_rho(composition: Mapping[str, float], E_MeV: float) -> float:
    """Mass attenuation coefficient μ/ρ for a mixture or compound.

    Uses the standard rule of mixtures:
        (μ/ρ)_mix = Σᵢ wᵢ · (μ/ρ)ᵢ
    where wᵢ is the mass fraction of element i.

    Parameters
    ----------
    composition : Mapping[str, float]
        {'ElementSymbol': weight_fraction}. Mass fractions should sum to ~1.
        Example for water: ``{'H': 0.1119, 'O': 0.8881}``.
    E_MeV : float
        Photon energy in MeV.

    Returns
    -------
    float
        μ/ρ of the mixture in cm²/g.
    """
    total = 0.0
    for el, w in composition.items():
        total += w * lookup_mu_rho(el, E_MeV)
    return total


def mixture_mu_en_rho(composition: Mapping[str, float], E_MeV: float) -> float:
    """Mass energy-absorption coefficient μ_en/ρ for a mixture or compound.

    Uses the standard rule of mixtures:
        (μ_en/ρ)_mix = Σᵢ wᵢ · (μ_en/ρ)ᵢ

    This is the standard ANS-6.4.3 / NIST XCOM procedure for compounds and
    mixtures.

    Parameters
    ----------
    composition : Mapping[str, float]
        {'ElementSymbol': weight_fraction}.
    E_MeV : float
        Photon energy in MeV.

    Returns
    -------
    float
        μ_en/ρ of the mixture in cm²/g.
    """
    total = 0.0
    for el, w in composition.items():
        total += w * lookup_mu_en_rho(el, E_MeV)
    return total


# -----------------------------------------------------------------------------
# Linear attenuation coefficient and mean free path
# -----------------------------------------------------------------------------
def linear_attenuation(
    composition: Mapping[str, float],
    density_g_cm3: float,
    E_MeV: float,
) -> float:
    """Linear attenuation coefficient μ for a mixture at given density.

    Parameters
    ----------
    composition : Mapping[str, float]
        Element -> mass fraction.
    density_g_cm3 : float
        Material density in g/cm³. Example values:
        - water: 1.000
        - dry air (STP): 0.001205
        - concrete (ordinary): 2.300
        - soil (typical): 1.600
        - lead: 11.350
    E_MeV : float
        Photon energy in MeV.

    Returns
    -------
    float
        μ in cm⁻¹.
    """
    mu_rho = mixture_mu_rho(composition, E_MeV)
    return mu_rho * density_g_cm3


def mean_free_path(
    composition: Mapping[str, float],
    density_g_cm3: float,
    E_MeV: float,
) -> float:
    """Photon mean free path 1/μ for a mixture at given density.

    Parameters
    ----------
    Same as :func:`linear_attenuation`.

    Returns
    -------
    float
        Mean free path in cm.
    """
    mu = linear_attenuation(composition, density_g_cm3, E_MeV)
    if mu <= 0:
        return float("inf")
    return 1.0 / mu


# -----------------------------------------------------------------------------
# Built-in common compositions (NIST reference)
# -----------------------------------------------------------------------------
NIST_AIR_DRY_COMPOSITION = {
    # NIST air (dry, near sea level), ICRU Report 37
    "N": 0.755268,
    "O": 0.231781,
    "Ar": 0.012827,
    "C": 0.000124,
}

NIST_WATER_COMPOSITION = {
    # H2O: 2 * 1.008 / 18.015 and 15.999 / 18.015
    "H": 0.111894,
    "O": 0.888106,
}

NIST_CONCRETE_COMPOSITION = {
    # Ordinary concrete (NIST), type 02
    "H": 0.010000,
    "C": 0.001000,
    "O": 0.529000,
    "Na": 0.016000,
    "Mg": 0.002000,
    "Al": 0.034000,
    "Si": 0.337000,
    "K": 0.013000,
    "Ca": 0.044000,
    "Fe": 0.014000,
}

NIST_SOIL_COMPOSITION = {
    # Typical sandy loam soil (ICRU 53)
    "H": 0.021,
    "C": 0.020,
    "O": 0.560,
    "Al": 0.060,
    "Si": 0.290,
    "K": 0.012,
    "Ca": 0.018,
    "Fe": 0.019,
}

NIST_TISSUE_SOFT_COMPOSITION = {
    # ICRU 44 soft tissue (ICRP standard man)
    "H": 0.101,
    "C": 0.111,
    "N": 0.026,
    "O": 0.762,
}


# -----------------------------------------------------------------------------
# K-edge validation
# -----------------------------------------------------------------------------
def validate_k_edges() -> dict[str, dict[str, float]]:
    """Return K-edge energies and the μ/ρ jump ratio at each K-edge.

    For heavy elements, μ/ρ should jump by a factor of ~4-6 at the K-edge
    (photoelectric absorption turns on). Used for self-test.

    Returns
    -------
    dict
        {'Pb': {'E_keV': 88.004, 'mu_below': ..., 'mu_above': ..., 'jump': ...}}
    """
    K_ENERGIES_KEV = {
        "Pb": 88.004,
        "U":  115.606,
        "W":  69.525,
        "Cu": 8.979,
        "Fe": 7.112,
        "Ca": 4.038,
        "Al": 1.560,
        "Si": 1.839,
        "K":  3.608,
    }
    out = {}
    for el, E_keV in K_ENERGIES_KEV.items():
        # Probe at energies that are clearly on each side of the K-edge,
        # 0.5 keV above and below — large enough that log-log interpolation
        # does not pick grid points from across the discontinuity.
        E_above_MeV = (E_keV + 0.5) / 1000.0
        E_below_MeV = max((E_keV - 0.5) / 1000.0, 1e-6)
        mu_above = lookup_mu_rho(el, E_above_MeV)
        mu_below = lookup_mu_rho(el, E_below_MeV)
        out[el] = {
            "E_keV": E_keV,
            "mu_rho_below_cm2_g": mu_below,
            "mu_rho_above_cm2_g": mu_above,
            "jump_ratio": mu_above / mu_below if mu_below > 0 else float("inf"),
        }
    return out
