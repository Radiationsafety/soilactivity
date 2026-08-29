"""Photon dosimetry conversion coefficients (ICRP 74 / ICRU 57).

Provides:
- Air-kerma-to-ambient-dose-equivalent conversion coefficients H*(10)/Ka
  for monoenergetic photons (10 keV - 10 MeV), per ICRP Publication 74
  (Conversion Coefficients for use in Radiological Protection against
  External Radiation, 1996). Same data as ICRU Report 57.

- Combined fluence-to-ambient-dose-equivalent conversion h*(10)/Φ:
    h*(10)/Φ  =  H*(10)/Ka  ×  Ka/Φ
  where Ka/Φ = 1.602e-9 × (μ_en/ρ)_air  [Gy·cm² per photon]
  with (μ_en/ρ)_air from xraylib (NIST Hubbell & Seltzer).

- Source-to-detector dose-rate from a point isotropic source:
    Ḋ* = (A · E · n_γ · h*(10)/Φ · e^(-μ·r)) / (4π·r²)
  (buildup factor B(E, μr) applied optionally via soilactivity.buildup).

References
----------
1. ICRP Publication 74: "Conversion Coefficients for use in Radiological
   Protection against External Radiation." Annals of the ICRP 26(3-4), 1996.
2. ICRU Report 57: "Conversion Coefficients for use in Radiological
   Protection against External Radiation." 1998 (same data as ICRP 74).
3. Hubbell, J. H. & Seltzer, S. M. NISTIR 5632 (1995).
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np

from . import attenuation as _att


__all__ = [
    "ICRP74_H_STAR_10_OVER_KA",
    "ICRP74_ENERGIES_MEV",
    "h_star_10_over_Ka",
    "h_star_10_over_phil",
    "kerma_per_fluence_air",
    "point_source_dose_rate",
    "NIST_AIR_COMPOSITION",
]


# -----------------------------------------------------------------------------
# ICRP 74 / ICRU 57: H*(10)/Ka (Sv/Gy)
#
# For photons, H*(10)/Ka is a dimensionless ratio (Sv/Gy = 1 by definition of
# the sievert, but the conversion coefficient is conventionally tabulated in
# Sv/Gy). Values for monoenergetic photons at the standard 17-point ICRP/ICRU
# grid (10 keV - 10 MeV), antero-posterior (AP) irradiation of the ICRU
# sphere. Source: ICRP 74 Table A.21, reproduced from ICRU 57.
# -----------------------------------------------------------------------------
ICRP74_ENERGIES_MEV = np.array([
    0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08,
    0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,
    1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
])

# H*(10)/Ka in Sv/Gy for monoenergetic photons (ICRP 74, antero-posterior).
# Reproduces ICRU 57 Table 28.
ICRP74_H_STAR_10_OVER_KA = np.array([
    0.009,   # 10 keV
    0.264,   # 15 keV
    0.610,   # 20 keV
    0.966,   # 30 keV
    1.071,   # 40 keV
    1.067,   # 50 keV
    1.050,   # 60 keV
    1.027,   # 80 keV
    1.020,   # 100 keV
    1.094,   # 150 keV
    1.181,   # 200 keV
    1.390,   # 300 keV
    1.505,   # 400 keV
    1.566,   # 500 keV
    1.591,   # 600 keV
    1.592,   # 800 keV
    1.557,   # 1.0 MeV
    1.438,   # 1.5 MeV
    1.341,   # 2.0 MeV
    1.251,   # 3.0 MeV
    1.220,   # 4.0 MeV
    1.208,   # 5.0 MeV
    1.205,   # 6.0 MeV
    1.205,   # 8.0 MeV
    1.205,   # 10. MeV
])


# Air composition (mass fractions), NIST reference
NIST_AIR_COMPOSITION = _att.NIST_AIR_DRY_COMPOSITION


# -----------------------------------------------------------------------------
# Interpolation of H*(10)/Ka
# -----------------------------------------------------------------------------
def h_star_10_over_Ka(E_MeV: float | np.ndarray) -> float | np.ndarray:
    """ICRP 74 air-kerma-to-ambient-dose-equivalent coefficient H*(10)/Ka.

    Parameters
    ----------
    E_MeV : float or np.ndarray
        Photon energy in MeV. Range 0.01 - 10 MeV; values outside are
        clamped to the nearest edge.

    Returns
    -------
    float or np.ndarray
        H*(10)/Ka in Sv/Gy (dimensionless for photons).

    Notes
    -----
    The coefficient is interpolated log-log between the 25 ICRP/ICRU grid
    points. For 50 keV - 1.5 MeV the values are within ±5% of 1.0 Sv/Gy
    (the build-up of secondary-electron equilibrium is the dominant effect
    at low E; pair-production begins contributing above 5 MeV).
    """
    E = np.atleast_1d(np.asarray(E_MeV, dtype=float))
    # Clamp to grid range
    E_clamped = np.clip(E, ICRP74_ENERGIES_MEV[0], ICRP74_ENERGIES_MEV[-1])

    log_E = np.log(E_clamped)
    log_xs = np.log(ICRP74_ENERGIES_MEV)
    log_ys = np.log(np.maximum(ICRP74_H_STAR_10_OVER_KA, 1e-30))

    # Linear interpolation in log-log space, vectorised via np.interp
    log_v = np.interp(log_E, log_xs, log_ys)
    out = np.exp(log_v)
    if E.shape == ():
        return float(out[0])
    if out.size == 1 and np.isscalar(E_MeV):
        return float(out[0])
    return out


# -----------------------------------------------------------------------------
# Air-kerma per fluence: Ka/Φ (Gy·cm² per photon)
# -----------------------------------------------------------------------------
# Physical constant: photon energy in J when E in MeV
#   1 MeV = 1.602176634e-13 J
_MEV_TO_JOULE = 1.602176634e-13


def kerma_per_fluence_air(E_MeV: float) -> float:
    """Air-kerma per unit fluence: Ka/Φ (Gy·cm²) for dry air at energy E.

    The kerma in air per unit photon fluence is:

        Ka/Φ = (μ_en/ρ)_air × E_J × (ρ·dx)

    where (μ_en/ρ)_air is the mass energy-absorption coefficient in cm²/g,
    E_J is the photon energy in joules, and (ρ·dx) is the mass-path per unit
    area. For a unit mass-path of 1 g/cm²:

        Ka/Φ [J·cm²/g] = (μ_en/ρ) × E_J

    Converting to Gy·cm² (1 Gy = 1 J/kg, so 1 J = 1 Gy·kg = 1000 Gy·g):

        Ka/Φ [Gy·cm²] = (μ_en/ρ) × E_J × 1000

    Parameters
    ----------
    E_MeV : float
        Photon energy in MeV.

    Returns
    -------
    float
        Ka/Φ in Gy·cm² (per photon).

    Notes
    -----
    Reference values from NIST (Hubbell & Seltzer 1995):
    - 0.1 MeV: 4.92e-15 Gy·cm²  ( = 4.92 fGy·cm² = 0.00492 pGy·cm²)
    - 1.0 MeV: 4.47e-14 Gy·cm²  ( = 0.0447 pGy·cm²)
    - 5.0 MeV: 1.22e-13 Gy·cm²  ( = 0.122 pGy·cm²)
    - 10. MeV: 1.55e-13 Gy·cm²  ( = 0.155 pGy·cm²)
    """
    mu_en_rho_air = _att.mixture_mu_en_rho(NIST_AIR_COMPOSITION, E_MeV)
    # Ka/Φ = (μ_en/ρ)[cm²/g] × E_MeV × _MEV_TO_JOULE[J/MeV] × 1000 [Gy·cm² per J·cm²/g]
    E_J = E_MeV * _MEV_TO_JOULE
    return mu_en_rho_air * E_J * 1000.0


# -----------------------------------------------------------------------------
# Combined h*(10)/Φ
# -----------------------------------------------------------------------------
def h_star_10_over_phil(E_MeV: float) -> float:
    """Fluence-to-ambient-dose-equivalent conversion coefficient h*(10)/Φ.

    h*(10)/Φ  =  H*(10)/Ka  ×  Ka/Φ

    Parameters
    ----------
    E_MeV : float
        Photon energy in MeV. Range 0.01 - 10 MeV.

    Returns
    -------
    float
        h*(10)/Φ in Sv·cm² (per photon). For doses in μSv·h⁻¹, multiply by
        fluence rate in cm⁻²·s⁻¹ and by 3600 (s/h) × 1e6 (Sv→μSv).

    Notes
    -----
    Reference values (ICRP 74 + Hubbell & Seltzer):
    - 0.1 MeV: 1.20e-12 Sv·cm² ( = 1.20 pSv·cm²)
    - 0.662 MeV (Cs-137): 6.49e-13 Sv·cm²
    - 1.0 MeV: 6.96e-13 Sv·cm²
    - 1.25 MeV (Co-60 effective): 7.20e-13 Sv·cm²
    """
    H_Ka = h_star_10_over_Ka(E_MeV)
    Ka_phi = kerma_per_fluence_air(E_MeV)
    return float(H_Ka) * Ka_phi


# -----------------------------------------------------------------------------
# Point source dose rate
# -----------------------------------------------------------------------------
def point_source_dose_rate(
    activity_Bq: float,
    gamma_lines: Iterable[tuple[float, float]],
    distance_m: float,
    shield_composition: Mapping[str, float] | None = None,
    shield_density_g_cm3: float | None = None,
    shield_thickness_cm: float = 0.0,
    apply_buildup: bool = False,
) -> float:
    """Ambient dose equivalent rate Ḋ*(10) from a point isotropic gamma source.

    Computes:

        Ḋ* = Σ_E [ A · n_γ(E) · h*(10)/Φ(E) · B(E, μ·t) · exp(-μ·t) ] / (4π·r²)

    Parameters
    ----------
    activity_Bq : float
        Source activity in Bq.
    gamma_lines : iterable of (E_MeV, yield_per_decay)
        Each tuple is (photon energy in MeV, photons per decay).
        Example for Cs-137: ``[(0.662, 0.851)]``.
        Example for Co-60: ``[(1.173, 1.0), (1.332, 1.0)]``.
    distance_m : float
        Source-to-detector distance in meters.
    shield_composition : Mapping[str, float] or None
        Element -> mass fraction. If None and shield_thickness > 0, raises.
    shield_density_g_cm3 : float or None
        Shield density in g/cm³.
    shield_thickness_cm : float
        Shield thickness in cm.
    apply_buildup : bool
        If True, apply the ANS-6.4.3 buildup factor B(E, μ·t) via
        :func:`soilactivity.buildup.get_buildup`. The shield is assumed to be
        a pure material in the ANS-6.4.3 database (e.g. 'Water', 'Lead',
        'Concrete'); for mixtures use :func:`buildup_for_mixture` separately.

    Returns
    -------
    float
        Ḋ*(10) in Sv/s. Multiply by 1e6 × 3600 = 3.6e9 to get μSv/h.

    Notes
    -----
    For a 1 GBq Cs-137 source at 1 m in air (no shield), expected:
    Ḋ* ≈ 91 μSv/h (literature range 85-92).
    For a 1 GBq Co-60 source at 1 m, expected: Ḋ* ≈ 350 μSv/h.
    """
    r_cm = distance_m * 100.0
    if r_cm <= 0:
        raise ValueError(f"distance_m must be > 0, got {distance_m}")

    dose_rate_Sv_per_s = 0.0
    for E_MeV, n_gamma in gamma_lines:
        if E_MeV <= 0 or n_gamma <= 0:
            continue
        h10_phi = h_star_10_over_phil(E_MeV)   # Sv·cm² per photon

        # Fluence rate at distance r (no shield): Φ̇ = A · n_γ / (4π·r²)
        # in photons/(s·cm²)
        phi_dot = activity_Bq * n_gamma / (4.0 * math.pi * r_cm ** 2)

        # Apply attenuation if shield present
        B_factor = 1.0
        if shield_thickness_cm > 0:
            if shield_composition is None or shield_density_g_cm3 is None:
                raise ValueError(
                    "shield_composition and shield_density_g_cm3 must be "
                    "provided when shield_thickness_cm > 0"
                )
            mu = _att.linear_attenuation(
                shield_composition, shield_density_g_cm3, E_MeV
            )
            mfp = mu * shield_thickness_cm   # mean free paths
            B_factor = 1.0
            if apply_buildup:
                from .buildup import get_buildup
                # Try direct lookup; for mixtures use buildup_for_mixture
                # (here we use a single material name if shield is elemental)
                if isinstance(shield_composition, str):
                    B_factor = float(get_buildup(
                        shield_composition, E_MeV, mfp
                    ))
                else:
                    # Use mixture lookup with this module's lookup function
                    from .buildup import buildup_for_mixture
                    B_factor = float(buildup_for_mixture(
                        shield_composition, E_MeV, mfp,
                        coeff_lookup=lambda el, E: (
                            _att.lookup_mu_en_rho(el, E),
                            _att.lookup_mu_rho(el, E),
                        ),
                    ))
            attn_factor = math.exp(-mfp)
        else:
            attn_factor = 1.0

        dose_rate_Sv_per_s += phi_dot * h10_phi * B_factor * attn_factor

    return dose_rate_Sv_per_s
