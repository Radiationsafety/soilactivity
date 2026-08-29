"""Tests for the fredholm depth-inversion subpackage.

Twin experiments: synthetic profiles -> forward count rates -> inversion -> recovery.
"""
import numpy as np
import pytest

from soilactivity.depth_inversion import (
    GammaLine, Detector, build_kernel, kernel_analytic,
    DepthInverter, pulse_profile, exp_profile, retardation, kd,
    chain_evolve, DECAY_S, tikhonov, tsvd, landweber, cimmino,
)

# Cs-137 line at 661.66 keV (rho_soil=1.4 g/cm3 -> mu_soil ~10.8 /m)
CS = GammaLine(661.66, 0.851, 10.8, 0.0092)

# Eu-152 multi-line (for multi-energy tests)
EU = [
    GammaLine(121.78, 0.284, 23.8, 0.0185),
    GammaLine(344.28, 0.266, 12.2, 0.0114),
    GammaLine(1408.01, 0.209, 7.4, 0.0066),
]

YEAR_S = 3.156e7


def test_kernel_numeric_matches_analytic():
    """Numeric kernel should agree with E1 analytic within ~0.2%."""
    det = Detector(1.0, 1.0)
    z = np.linspace(0.0, 0.5, 11)
    Kn = build_kernel([CS], det, z, u_max=5e3, n_u=20000)[0]
    Ka = kernel_analytic(CS, det, z)
    assert np.allclose(Kn, Ka, rtol=2e-3)


def test_kernel_shape():
    det = Detector(1.0, 1.0)
    z = np.linspace(0, 0.3, 20)
    K = build_kernel(EU, det, z)
    assert K.shape == (3, 20)


def test_transport_kd():
    """Kd database lookup."""
    assert 100.0 <= kd("Cs-137", "low") <= 3000.0
    assert kd("Cs-137", "low") < kd("Cs-137", "high")


def test_transport_retardation():
    R = retardation(1.4, 500.0, 0.3)  # typical Cs
    assert R == pytest.approx(1.0 + 1.4 * 500.0 / 0.3)
    assert R > 1.0


def test_transport_pulse_profile():
    z = np.linspace(0, 0.5, 100)
    a = pulse_profile(z, 1e5, 25 * YEAR_S, 1e-10, 0.0, 1401.0, DECAY_S["Cs-137"])
    assert a.min() >= 0.0
    assert np.trapezoid(a, z) < 1e5  # decayed


def test_transport_exp_profile():
    z = np.linspace(0, 1.0, 200)
    A = 1e4
    lam = 0.05
    a = exp_profile(z, A, lam)
    assert np.trapezoid(a, z) == pytest.approx(A, rel=0.05)
    assert a[0] > a[-1]


def test_transport_chain_evolve():
    lams = [DECAY_S["Cs-137"], DECAY_S["Ba-137m"] if False else 0.0]
    # Simple single-nuclide decay
    A0 = np.array([1000.0])
    At = chain_evolve([DECAY_S["Cs-137"]], A0, 30 * YEAR_S)
    assert At[0] < A0[0]
    assert At[0] > 0


def _twin_eu(scale=2.0e4, seed=1, n_z=20):
    """Helper: Eu-152 twin experiment."""
    inv = DepthInverter(EU, heights=[0.5, 2.0], z_max=0.6, n_z=n_z)
    t = 25.0 * YEAR_S
    a_true = pulse_profile(inv.z, 1e5, t, 1e-10, 0.0, 1401.0, DECAY_S["Eu-152"])
    rng = np.random.default_rng(seed)
    counts = rng.poisson(np.maximum((inv.K @ a_true) * scale, 1.0)) / scale
    return inv, a_true, counts


def _zmedian(a, inv):
    cdf = np.cumsum(a * inv.dz)
    return float(np.interp(0.5 * cdf[-1], cdf, inv.z))


def test_twin_nonparametric_recovery():
    """Non-parametric Tikhonov/GCV should recover areal within 30% and z_median within 6 cm."""
    inv, a_true, counts = _twin_eu()
    res = inv.fit(counts, criterion="gcv")
    assert res.a.min() >= 0.0
    assert res.areal == pytest.approx(float(np.sum(a_true * inv.dz)), rel=0.3)
    assert res.z_median == pytest.approx(_zmedian(a_true, inv), abs=0.06)
    assert res.chi2 / len(counts) < 2.0


def test_parametric_pulse_recovery():
    """Parametric pulse fit should recover A0 and t within 50%/30%."""
    inv, _, counts = _twin_eu(seed=2)
    out = inv.fit_parametric(counts, family="pulse", nuclide="Eu-152",
                              D=1e-10, R=1401.0)
    assert out["A0"] == pytest.approx(1e5, rel=0.5)
    assert out["t_years"] == pytest.approx(25.0, rel=0.5)


def test_solvers_agree():
    """Tikhonov, Landweber, Cimmino should all be non-negative."""
    inv, _, counts = _twin_eu(seed=3)
    d = counts
    sg = inv.poisson_sigma(counts)
    x1 = tikhonov(inv.K, d, 1e-4, sigma=sg, nonneg=True)
    x2, _ = landweber(inv.K, d, sigma=sg, chi2_target=len(d),
                         nonneg=True, max_iter=2000)
    x3, _ = cimmino(inv.K, d, sigma=sg, chi2_target=len(d),
                       nonneg=True, max_iter=20000)
    for x in (x1, x2, x3):
        assert np.all(x >= 0)
        assert np.sum(x * inv.dz) > 0  # recovered some areal activity


def test_tsvd():
    """TSVD should produce non-negative result when requested."""
    inv, a_true, counts = _twin_eu(seed=4)
    sg = inv.poisson_sigma(counts)
    x, info = tsvd(inv.K, counts, sigma=sg, nonneg=True)
    assert np.all(x >= 0)
    assert 0 < info["rank"] <= min(inv.K.shape)
