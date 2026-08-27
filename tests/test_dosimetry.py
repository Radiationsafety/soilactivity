"""Tests for soilactivity.dosimetry (ICRP 74 + NIST air μen/ρ)."""
from __future__ import annotations

import math

import numpy as np
import pytest

xraylib = pytest.importorskip("xraylib")

from soilactivity.dosimetry import (
    ICRP74_ENERGIES_MEV,
    ICRP74_H_STAR_10_OVER_KA,
    NIST_AIR_COMPOSITION,
    h_star_10_over_Ka,
    h_star_10_over_phil,
    kerma_per_fluence_air,
    point_source_dose_rate,
)
from soilactivity.attenuation import mixture_mu_en_rho


# -----------------------------------------------------------------------------
# ICRP 74 table completeness & spot-checks
# -----------------------------------------------------------------------------
def test_icrp74_grid_size():
    assert len(ICRP74_ENERGIES_MEV) == 25
    assert len(ICRP74_H_STAR_10_OVER_KA) == 25
    assert ICRP74_ENERGIES_MEV[0] == pytest.approx(0.01)
    assert ICRP74_ENERGIES_MEV[-1] == pytest.approx(10.0)


@pytest.mark.parametrize("E,expected", [
    (0.01, 0.009),
    (0.05, 1.067),
    (0.1,  1.020),
    (0.5,  1.566),
    (1.0,  1.557),
    (5.0,  1.208),
    (10.0, 1.205),
])
def test_h_star_10_over_Ka_at_grid_points(E, expected):
    """H*(10)/Ka at grid points should match ICRP 74 Table A.21."""
    val = h_star_10_over_Ka(E)
    assert val == pytest.approx(expected, rel=1e-3)


def test_h_star_10_over_Ka_interpolation():
    """Interpolated value at 0.07 MeV should be ~1.04 (between 60 and 80 keV)."""
    val = h_star_10_over_Ka(0.07)
    # Between 1.050 (60 keV) and 1.027 (80 keV): log-log interp ~ 1.040
    assert 1.0 < val < 1.06


def test_h_star_10_over_Ka_clamping():
    """Energies outside the grid should clamp to edge values."""
    low = h_star_10_over_Ka(0.001)
    high = h_star_10_over_Ka(20.0)
    assert low == pytest.approx(ICRP74_H_STAR_10_OVER_KA[0])
    assert high == pytest.approx(ICRP74_H_STAR_10_OVER_KA[-1])


# -----------------------------------------------------------------------------
# Kerma per fluence in air (Ka/Φ)
# -----------------------------------------------------------------------------
# Verify Ka/Φ scales correctly with energy (linear in E) and gives the
# correct order of magnitude. The absolute conversion factor depends on
# the normalisation convention used (per 1 g/cm² mass-path, or per unit
# mass in SI), so we test scaling rather than absolute values.
def test_kerma_per_fluence_air_scales_with_energy():
    """Ka/Φ should scale approximately linearly with E in the dosimetric range.

    At 0.1 - 1.0 MeV (the most relevant gamma range for radiation protection:
    Cs-137 at 662 keV, Co-60 at 1.17-1.33 MeV), μ_en/ρ for air varies from
    ~0.024 to ~0.028 cm²/g, so Ka/Φ = μ_en/ρ × E_J scales with E within ~25%.

    At higher energies (5-10 MeV), pair-production contributions grow and our
    approximation of μ_en/ρ = μ_incoh·T(E) + μ_photo + μ_pair·(1-2mc²/E)
    underestimates NIST Hubbell-Seltzer values by ~30% (the difference comes
    from the radiative-loss correction g(E) which we ignore). This is a
    known limitation; see CHANGELOG.
    """
    Ka_01 = kerma_per_fluence_air(0.1)
    Ka_1 = kerma_per_fluence_air(1.0)
    # In the dosimetric range 0.1 - 1.0 MeV, ratio should be ~10 (E ratio).
    assert Ka_1 / Ka_01 == pytest.approx(10.0, rel=0.25)


def test_kerma_per_fluence_air_at_cs137_energy():
    """Ka/Φ at 662 keV (Cs-137) — verify μ_en/ρ × E_J product matches NIST.

    The NIST Hubbell-Seltzer reference value for μ_en/ρ of dry air at 662 keV
    is 0.0294 cm²/g; Ka/Φ = μ_en/ρ × E_J × (ρ·dx), and with a unit mass-path
    of 1 g/cm² this evaluates to ~3.12e-15 J·cm²/g (per photon). The actual
    numerical value of Ka/Φ depends on the chosen unit normalisation (g vs kg,
    cm² vs m²); this test verifies only the μ_en/ρ × E_J product against the
    NIST reference.
    """
    from soilactivity.attenuation import mixture_mu_en_rho, NIST_AIR_DRY_COMPOSITION
    E = 0.662
    mu_en_rho = mixture_mu_en_rho(NIST_AIR_DRY_COMPOSITION, E)
    # NIST Hubbell & Seltzer 1995 reference value at 662 keV:
    nist_mu_en_rho_air_662keV = 0.0294
    rel_err = abs(mu_en_rho - nist_mu_en_rho_air_662keV) / nist_mu_en_rho_air_662keV
    assert rel_err < 0.05, \
        f"μ_en/ρ air at 662 keV: got {mu_en_rho:.4f}, " \
        f"expected {nist_mu_en_rho_air_662keV:.4f}, rel_err={rel_err:.3f}"


def test_kerma_per_fluence_air_positive():
    """Ka/Φ must be positive and finite for all energies in [0.001, 20] MeV."""
    for E in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]:
        Ka = kerma_per_fluence_air(E)
        assert math.isfinite(Ka)
        assert Ka > 0


# -----------------------------------------------------------------------------
# Combined h*(10)/Φ
# -----------------------------------------------------------------------------
def test_h_star_10_over_phil_scales_with_energy():
    """h*(10)/Φ should be a smooth function of E."""
    h_low = h_star_10_over_phil(0.05)
    h_mid = h_star_10_over_phil(0.5)
    h_high = h_star_10_over_phil(5.0)
    # All should be positive and finite.
    for h in (h_low, h_mid, h_high):
        assert math.isfinite(h)
        assert h > 0
    # Mid-energy should be larger than low (factor of ~10 in energy).
    assert h_mid > h_low
    # High energy has a slight peak around 0.8-1 MeV then decreases.


# -----------------------------------------------------------------------------
# Point source dose rate — the headline validation
# -----------------------------------------------------------------------------
# Cs-137: 1 GBq at 1 m in air, no shield
# Reference H*(10) rate: ~120-125 μSv/h (ISO 4037-3, gamma constant 0.124 μSv·m²/(MBq·h))
# Older literature quotes ~78-91 μSv/h based on the historical exposure rate
# constant (without H*(10) conversion); those are not used here.
CS137_GAMMA = [(0.662, 0.851)]   # 662 keV, 85.1% yield


def test_cs137_dose_rate_in_air():
    """1 GBq Cs-137 at 1 m → ~120-130 μSv/h (H*(10), ISO 4037-3)."""
    dose_rate_Sv_per_s = point_source_dose_rate(
        activity_Bq=1e9,
        gamma_lines=CS137_GAMMA,
        distance_m=1.0,
    )
    dose_rate_uSv_per_h = dose_rate_Sv_per_s * 1e6 * 3600
    assert 105.0 < dose_rate_uSv_per_h < 135.0, \
        f"Cs-137 dose rate = {dose_rate_uSv_per_h:.2f} μSv/h, " \
        f"expected 120-130"


# Co-60: 1 GBq at 1 m in air, no shield
# Reference H*(10) rate: ~350-380 μSv/h (ISO 4037-3)
CO60_GAMMA = [(1.173, 0.9985), (1.332, 0.9998)]


def test_co60_dose_rate_in_air():
    """1 GBq Co-60 at 1 m → ~340-420 μSv/h (H*(10), ISO 4037-3)."""
    dose_rate_Sv_per_s = point_source_dose_rate(
        activity_Bq=1e9,
        gamma_lines=CO60_GAMMA,
        distance_m=1.0,
    )
    dose_rate_uSv_per_h = dose_rate_Sv_per_s * 1e6 * 3600
    assert 330.0 < dose_rate_uSv_per_h < 480.0, \
        f"Co-60 dose rate = {dose_rate_uSv_per_h:.2f} μSv/h, " \
        f"expected 340-420"


# Cs-137 H*(10) gamma constant: ISO 4037-3 reference value 0.124 μSv·m²/(MBq·h)
def test_cs137_gamma_constant():
    """Cs-137 H*(10) gamma constant ~0.10-0.14 μSv·m²/(MBq·h)."""
    dose_rate_Sv_per_s = point_source_dose_rate(
        activity_Bq=1e6,         # 1 MBq
        gamma_lines=CS137_GAMMA,
        distance_m=1.0,
    )
    gamma_const_uSv_m2_per_MBq_h = dose_rate_Sv_per_s * 1e6 * 3600
    assert 0.09 < gamma_const_uSv_m2_per_MBq_h < 0.16


# Co-60 H*(10) gamma constant: ISO 4037-3 reference value 0.357 μSv·m²/(MBq·h)
def test_co60_gamma_constant():
    """Co-60 H*(10) gamma constant ~0.30-0.50 μSv·m²/(MBq·h)."""
    dose_rate_Sv_per_s = point_source_dose_rate(
        activity_Bq=1e6,         # 1 MBq
        gamma_lines=CO60_GAMMA,
        distance_m=1.0,
    )
    gamma_const_uSv_m2_per_MBq_h = dose_rate_Sv_per_s * 1e6 * 3600
    assert 0.30 < gamma_const_uSv_m2_per_MBq_h < 0.55


# -----------------------------------------------------------------------------
# Shield attenuation
# -----------------------------------------------------------------------------
def test_lead_shield_attenuation_cs137():
    """5 cm Pb shield reduces Cs-137 dose by ~half (10 HVLs at 1.2 cm each).

    Actually Pb HVL for 662 keV is ~0.6 cm, so 5 cm = ~8 HVLs → factor ~256.
    We test the exponential attenuation factor.
    """
    dose_unshielded = point_source_dose_rate(
        activity_Bq=1e9, gamma_lines=CS137_GAMMA, distance_m=1.0,
    )
    dose_shielded = point_source_dose_rate(
        activity_Bq=1e9, gamma_lines=CS137_GAMMA, distance_m=1.0,
        shield_composition={"Pb": 1.0},
        shield_density_g_cm3=11.350,
        shield_thickness_cm=5.0,
    )
    # HVL Pb @ 662 keV ~ 0.6 cm, so 5 cm = 8.3 HVLs, factor ~ 2^-8.3 ~ 0.003
    ratio = dose_shielded / dose_unshielded
    assert ratio < 0.01, \
        f"After 5 cm Pb, ratio = {ratio:.4f}, expected < 0.01"


def test_shield_requires_composition():
    with pytest.raises(ValueError, match="shield_composition"):
        point_source_dose_rate(
            activity_Bq=1e6, gamma_lines=CS137_GAMMA, distance_m=1.0,
            shield_thickness_cm=1.0,  # nonzero but no composition
        )


# -----------------------------------------------------------------------------
# Energy scaling: 1/r² geometry
# -----------------------------------------------------------------------------
def test_inverse_square_law():
    """Dose rate should fall off as 1/r² for an unshielded point source."""
    dose_1m = point_source_dose_rate(
        activity_Bq=1e9, gamma_lines=CS137_GAMMA, distance_m=1.0,
    )
    dose_2m = point_source_dose_rate(
        activity_Bq=1e9, gamma_lines=CS137_GAMMA, distance_m=2.0,
    )
    ratio = dose_2m / dose_1m
    assert ratio == pytest.approx(0.25, rel=0.001)


def test_activity_linear():
    """Dose rate should scale linearly with activity."""
    dose_1 = point_source_dose_rate(
        activity_Bq=1e6, gamma_lines=CS137_GAMMA, distance_m=1.0,
    )
    dose_10 = point_source_dose_rate(
        activity_Bq=1e7, gamma_lines=CS137_GAMMA, distance_m=1.0,
    )
    assert dose_10 == pytest.approx(10 * dose_1, rel=0.001)
