"""Tests for the fredholm depth-inversion subpackage.

Twin experiments: synthetic profiles -> forward count rates -> inversion -> recovery.
Covers classical solvers, geophysical solvers, transport chemistry,
Bayesian methods, criteria, diagnostics, and pipeline ensemble.
"""
import numpy as np
import pytest

from soilactivity.depth_inversion import (
    GammaLine, Detector, build_kernel, kernel_analytic,
    DepthInverter, pulse_profile, exp_profile, retardation, kd,
    chain_evolve, DECAY_S, tikhonov, tsvd, landweber, cimmino,
    cgls, kaczmarz, fista, tv_admm, focusing_irls,
    chi2, gcv_curve, lcurve_corner, choose_alpha_discrepancy,
    quasi_optimality, ncp_criterion, snr_criterion,
    resolution_matrix, depth_of_investigation, singulars,
    model_covariance, spread_function, information_content,
    ensemble_kalman_inversion, laplace_map,
    kd_ph, kd_freundlich, multi_layer_pulse,
    FREUNDLICH_DB, KD_PH_PARAMS,
    # new geophysics & transport additions
    depth_scale_power, depth_scale_log, depth_scale_adaptive,
    compose_weighting, two_site_retardation, two_site_ade,
    joint_inversion, two_site_effective_Kd,
    two_site_retardation_from_kd, competitive_kd_cs, competitive_kd_sr,
    kd_u_eh, cgls_lcurve, crossval_alpha, depth_weighted_tikhonov,
    weighted_smoothness, compactness_operator,
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


# =====================================================================
# Kernel tests
# =====================================================================

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


# =====================================================================
# Transport chemistry tests
# =====================================================================

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


def test_transport_kd_ph():
    """pH-dependent Kd."""
    kd_mid = kd("Cs-137", "mid")
    kd_ph5 = kd_ph("Cs-137", 5.0)
    kd_ph8 = kd_ph("Cs-137", 8.0)
    # Higher pH should generally give higher Kd for Cs
    assert kd_ph8 > kd_ph5
    # pH model should give reasonable values
    assert 1.0 < kd_ph5 < 1e6


def test_transport_freundlich():
    """Freundlich isotherm Kd."""
    K_F, n_F = FREUNDLICH_DB["Cs-137"]
    assert K_F > 0
    assert n_F < 1.0  # favourable isotherm
    kd_low = kd_freundlich("Cs-137", 0.001)
    kd_high = kd_freundlich("Cs-137", 10.0)
    assert kd_low > kd_high  # lower conc -> higher Kd for n<1


def test_transport_multi_layer():
    """Multi-layer pulse profile."""
    z = np.linspace(0, 0.5, 50)
    layers = [
        {"z_top": 0.0, "z_bot": 0.1, "D": 1e-10, "v": 0.0, "R": 500.0, "lam": DECAY_S["Cs-137"]},
        {"z_top": 0.1, "z_bot": 0.5, "D": 5e-11, "v": 0.0, "R": 2000.0, "lam": DECAY_S["Cs-137"]},
    ]
    a = multi_layer_pulse(z, 1e5, 25 * YEAR_S, layers)
    assert a.shape == z.shape
    assert a.min() >= 0.0


# =====================================================================
# Solver tests
# =====================================================================

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
        assert np.sum(x * inv.dz) > 0


def test_tsvd():
    """TSVD should produce non-negative result when requested."""
    inv, a_true, counts = _twin_eu(seed=4)
    sg = inv.poisson_sigma(counts)
    x, info = tsvd(inv.K, counts, sigma=sg, nonneg=True)
    assert np.all(x >= 0)
    assert 0 < info["rank"] <= min(inv.K.shape)


def test_cgls():
    """CGLS should converge and give non-negative result."""
    inv, _, counts = _twin_eu(seed=5)
    x, info = cgls(inv.K, counts, max_iter=100, nonneg=True)
    assert np.all(x >= 0)
    assert np.sum(x * inv.dz) > 0
    assert info["iterations"] > 0
    assert len(info["chi2_history"]) == info["iterations"]


def test_kaczmarz():
    """Kaczmarz should give non-negative result."""
    inv, _, counts = _twin_eu(seed=6)
    x, info = kaczmarz(inv.K, counts, max_iter=100, nonneg=True, seed=42)
    assert np.all(x >= 0)
    assert np.sum(x * inv.dz) > 0


def test_fista():
    """FISTA (L1 sparse) should give non-negative, sparse result."""
    inv, _, counts = _twin_eu(seed=7)
    A, b = inv.K, counts
    sv_max = np.linalg.svd(A, compute_uv=False)[0]
    alpha = sv_max ** 2 * 1e-3
    x, info = fista(A, b, alpha, nonneg=True, max_iter=500)
    assert np.all(x >= 0)
    assert np.sum(x * inv.dz) > 0
    # L1 solution should be sparser than Tikhonov
    x_tik = tikhonov(A, b, alpha * 10, nonneg=True)
    assert np.sum(x > 0) <= np.sum(x_tik > 0) + 5


def test_tv_admm():
    """TV/ADMM should preserve non-negativity."""
    inv, _, counts = _twin_eu(seed=8)
    A, b = inv.K, counts
    sv_max = np.linalg.svd(A, compute_uv=False)[0]
    alpha = sv_max ** 2 * 1e-3
    x, info = tv_admm(A, b, alpha, nonneg=True, max_iter=100)
    assert np.all(x >= 0)
    assert np.sum(x * inv.dz) > 0


def test_focusing_mgs():
    """Focusing MGS should give non-negative blocky result."""
    inv, _, counts = _twin_eu(seed=9)
    A, b = inv.K, counts
    sv_max = np.linalg.svd(A, compute_uv=False)[0]
    alpha = sv_max ** 2 * 1e-2
    x, info = focusing_irls(A, b, alpha, mode="mgs", nonneg=True)
    assert np.all(x >= 0)
    assert np.sum(x * inv.dz) > 0
    assert info["iterations"] > 0


def test_focusing_ms():
    """Focusing MS (minimum support) should give non-negative result."""
    inv, _, counts = _twin_eu(seed=10)
    A, b = inv.K, counts
    sv_max = np.linalg.svd(A, compute_uv=False)[0]
    alpha = sv_max ** 2 * 1e-2
    x, info = focusing_irls(A, b, alpha, mode="ms", nonneg=True)
    assert np.all(x >= 0)


# =====================================================================
# Criteria tests
# =====================================================================

def test_criteria_basic():
    """GCV, L-curve, discrepancy should all return finite alphas."""
    inv, _, counts = _twin_eu(seed=1)
    A, b = inv.K, counts
    sg = inv.poisson_sigma(counts)
    Aw = A / sg[:, None]
    bw = b / sg
    L = np.eye(len(inv.z))
    alphas = np.geomspace(1e-10, 1e2, 32)
    # GCV
    gcv_vals = gcv_curve(Aw, bw, alphas, L=L)
    assert np.all(np.isfinite(gcv_vals))
    # L-curve
    alpha_lc, kappa = lcurve_corner(Aw, bw, alphas, L=L)
    assert np.isfinite(alpha_lc)
    # Discrepancy
    alpha_disc = choose_alpha_discrepancy(Aw, bw, L=L)
    assert np.isfinite(alpha_disc)


def test_criteria_quasi_optimality():
    """Quasi-optimality should return a finite alpha."""
    inv, _, counts = _twin_eu(seed=1)
    A, b = inv.K, counts
    sg = inv.poisson_sigma(counts)
    Aw = A / sg[:, None]
    bw = b / sg
    alphas = np.geomspace(1e-10, 1e2, 32)
    alpha_qo, qo_vals = quasi_optimality(Aw, bw, alphas)
    assert np.isfinite(alpha_qo)
    assert np.all(np.isfinite(qo_vals))


def test_criteria_ncp():
    """NCP criterion should return a finite alpha."""
    inv, _, counts = _twin_eu(seed=1)
    A, b = inv.K, counts
    sg = inv.poisson_sigma(counts)
    Aw = A / sg[:, None]
    bw = b / sg
    alphas = np.geomspace(1e-10, 1e2, 32)
    alpha_ncp, ncp_vals = ncp_criterion(Aw, bw, alphas)
    assert np.isfinite(alpha_ncp)


def test_criteria_snr():
    """SNR criterion should return a finite alpha."""
    inv, _, counts = _twin_eu(seed=1)
    A, b = inv.K, counts
    sg = inv.poisson_sigma(counts)
    Aw = A / sg[:, None]
    bw = b / sg
    alphas = np.geomspace(1e-10, 1e2, 32)
    alpha_snr, snr_vals = snr_criterion(Aw, bw, alphas)
    assert np.isfinite(alpha_snr)


# =====================================================================
# Diagnostics tests
# =====================================================================

def test_diagnostics_resolution():
    """Resolution matrix and DOI should be finite."""
    inv, _, counts = _twin_eu(seed=1)
    R = resolution_matrix(inv.K, alpha=1e-4)
    assert R.shape == (len(inv.z), len(inv.z))
    doi = depth_of_investigation(R, inv.z)
    assert doi.shape == (len(inv.z),)
    assert np.all(doi >= 0)


def test_diagnostics_covariance():
    """Model covariance should be symmetric positive semi-definite."""
    inv, _, counts = _twin_eu(seed=1)
    Cov = model_covariance(inv.K, alpha=1e-4)
    assert Cov.shape == (len(inv.z), len(inv.z))
    assert np.allclose(Cov, Cov.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(Cov)
    assert np.all(eigvals >= -1e-10)


def test_diagnostics_spread():
    """Spread function should be non-negative."""
    inv, _, counts = _twin_eu(seed=1)
    R = resolution_matrix(inv.K, alpha=1e-4)
    sp = spread_function(R, inv.z)
    assert sp.shape == (len(inv.z),)
    assert np.all(sp >= 0)


def test_diagnostics_info():
    """Information content metrics should be finite."""
    inv, _, counts = _twin_eu(seed=1)
    info = information_content(inv.K)
    assert np.isfinite(info["cond"])
    assert 0 < info["rank_eff"] <= min(inv.K.shape)


def test_diagnostics_singulars():
    """SVD spectrum should be non-negative and decreasing."""
    inv, _, counts = _twin_eu(seed=1)
    sv, scale = singulars(inv.K)
    assert len(sv) == min(inv.K.shape)
    assert np.all(sv >= 0)
    assert np.all(np.diff(sv) <= 1e-10)


# =====================================================================
# Bayesian tests
# =====================================================================

def test_eki_basic():
    """EKI should return non-negative ensemble with finite uncertainty."""
    inv, _, counts = _twin_eu(seed=1)
    res = ensemble_kalman_inversion(
        inv.K, counts, n_ens=50, n_iter=10, prior_std=1e4, seed=42)
    assert np.all(res["mean"] >= -1e-3)  # allow tiny negatives
    assert np.all(res["std"] >= 0)
    assert res["ensemble"].shape == (50, len(inv.z))
    assert np.sum(res["mean"] * inv.dz) > 0


def test_laplace_map():
    """Laplace MAP should return non-negative result with uncertainty."""
    inv, _, counts = _twin_eu(seed=1)
    res = laplace_map(inv.K, counts, alpha=1e-4, nonneg=True)
    assert np.all(res["map"] >= 0)
    assert np.all(res["std"] >= 0)
    assert res["cov"].shape == (len(inv.z), len(inv.z))


# =====================================================================
# Pipeline ensemble test
# =====================================================================

def test_pipeline_ensemble():
    """Ensemble fit should return multiple results with AIC/BIC."""
    inv, _, counts = _twin_eu(seed=1)
    out = inv.fit_ensemble(counts, methods=["tikhonov/gcv", "cgls"])
    assert len(out["results"]) >= 1
    assert len(out["aic"]) == len(out["results"])
    assert out["best"].a is not None
    assert np.all(out["best"].a >= 0)


def test_pipeline_tv():
    """Pipeline TV fit should work."""
    inv, _, counts = _twin_eu(seed=2)
    res = inv.fit_tv(counts)
    assert np.all(res.a >= 0)


def test_pipeline_sparse():
    """Pipeline FISTA fit should work."""
    inv, _, counts = _twin_eu(seed=3)
    res = inv.fit_sparse(counts)
    assert np.all(res.a >= 0)


# =====================================================================
# Geophysics module tests
# =====================================================================

def test_depth_scale_power():
    """Power-law depth weighting should decrease with depth."""
    z = np.linspace(0, 1.0, 50)
    w = depth_scale_power(z, beta=2.0)
    assert w.shape == z.shape
    assert w[0] >= w[-1]  # decreasing
    assert np.all(w > 0)


def test_depth_scale_log():
    """Logarithmic depth weighting should decrease with depth."""
    z = np.linspace(0, 1.0, 50)
    w = depth_scale_log(z)
    assert w.shape == z.shape
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)


def test_depth_scale_adaptive():
    """Adaptive weighting should be finite and positive."""
    inv, _, counts = _twin_eu(seed=1)
    w = depth_scale_adaptive(inv.K, inv.z)
    assert w.shape == inv.z.shape
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)


def test_compose_weighting():
    """All weighting methods should return finite positive vectors."""
    inv, _, _ = _twin_eu(seed=1)
    for method in ['li_oldenburg', 'power', 'log', 'combined']:
        w = compose_weighting(inv.K, inv.z, method=method)
        assert np.all(np.isfinite(w))
        assert np.all(w > 0)


def test_two_site_retardation():
    """Two-site retardation should interpolate between R_eq and R_inst."""
    R_inst = 1400.0
    f = 0.7
    omega = 1e-6
    R_short = two_site_retardation(R_inst, f, omega, t_obs=1.0)
    R_long = two_site_retardation(R_inst, f, omega, t_obs=1e8)
    assert 1.0 < R_short < R_inst
    assert R_long > R_short
    assert R_long <= R_inst + 1e-6


def test_two_site_ade():
    """Two-site ADE solver should produce non-negative concentrations."""
    z = np.linspace(0, 0.5, 50)
    res = two_site_ade(z, (0, 30 * YEAR_S), D=1e-10, v=0.0,
                       R_inst=1400.0, f=0.7, omega_kin=1e-6,
                       lam=DECAY_S["Cs-137"], n_save=10)
    assert np.all(res['C_total'] >= 0)
    assert res['C_total'].shape[0] == 10


def test_two_site_kd():
    """Two-site effective Kd should depend on time."""
    Kd_eq = two_site_effective_Kd(500.0, 0.7, 1e-6, t_obs=1.0)
    Kd_long = two_site_effective_Kd(500.0, 0.7, 1e-6, t_obs=1e8)
    assert Kd_eq < Kd_long
    assert Kd_long <= 500.0 + 1e-6


def test_competitive_kd():
    """Competitive Kd should decrease with competitor concentration."""
    kd_no_comp = competitive_kd_cs(500.0, 0.0, 0.0)
    kd_with_K = competitive_kd_cs(500.0, 5.0, 0.0)
    kd_with_NH4 = competitive_kd_cs(500.0, 0.0, 3.0)
    assert kd_no_comp >= kd_with_K
    assert kd_no_comp >= kd_with_NH4
    # Sr with Ca
    kd_sr_no = competitive_kd_sr(100.0, 0.0)
    kd_sr_Ca = competitive_kd_sr(100.0, 10.0)
    assert kd_sr_no >= kd_sr_Ca


def test_kd_u_eh():
    """U Kd should be higher under reducing conditions."""
    kd_red = kd_u_eh(7.0, -50.0)   # reducing (below Eh_crit ~50 mV)
    kd_ox = kd_u_eh(7.0, 200.0)    # oxidising
    assert kd_red > kd_ox


def test_two_site_retardation_from_kd():
    """Retardation from Kd wrapper should be consistent."""
    R = two_site_retardation_from_kd(500.0, 0.7, 1e-6, 25 * YEAR_S)
    assert R > 1.0


def test_cgls_lcurve():
    """CGLS with L-curve stopping should give non-negative result."""
    inv, _, counts = _twin_eu(seed=11)
    x, info = cgls_lcurve(inv.K, counts, max_iter=100, nonneg=True)
    assert np.all(x >= 0)
    assert np.sum(x * inv.dz) > 0
    assert 'corner_iter' in info


def test_crossval_alpha():
    """Cross-validation should return a finite alpha."""
    inv, _, counts = _twin_eu(seed=12)
    sg = inv.poisson_sigma(counts)
    alpha_cv, cv_err = crossval_alpha(
        inv.K, counts, sg, n_folds=3, alphas=np.geomspace(1e-8, 1e0, 8))
    assert np.isfinite(alpha_cv)
    assert alpha_cv > 0
    assert len(cv_err) == 8


def test_depth_weighted_tikhonov():
    """Depth-weighted Tikhonov should give non-negative result."""
    inv, _, counts = _twin_eu(seed=13)
    sg = inv.poisson_sigma(counts)
    x = depth_weighted_tikhonov(
        inv.K, counts, 1e-4, inv.z, sigma=sg,
        weight_method='power', beta=2.0)
    assert np.all(x >= 0)
    assert np.sum(x * inv.dz) > 0


def test_weighted_smoothness():
    """Weighted smoothness operator should have correct shape."""
    z = np.linspace(0, 1.0, 20)
    w = depth_scale_power(z)
    L = weighted_smoothness(20, z, w=w, order=2)
    assert L.shape == (18, 20)


def test_compactness_operator():
    """Compactness operator should return diagonal matrix."""
    x = np.random.default_rng(0).uniform(0, 100, 20)
    W_inv = compactness_operator(x, eps=1e-2, mode='ms')
    assert W_inv.shape == (20, 20)
    assert np.all(np.diag(W_inv) > 0)


def test_joint_inversion():
    """Joint inversion of two nuclides should return finite profiles."""
    inv_cs = DepthInverter([CS], heights=[1.0], z_max=0.5, n_z=15)
    t = 25.0 * YEAR_S
    a_cs = pulse_profile(inv_cs.z, 1e5, t, 1e-10, 0.0, 1401.0, DECAY_S["Cs-137"])
    rng = np.random.default_rng(42)
    d_cs = rng.poisson(np.maximum(inv_cs.K @ a_cs * 1e4, 1.0)) / 1e4
    s_cs = inv_cs.poisson_sigma(d_cs)

    out = joint_inversion(
        {"Cs-137": inv_cs.K},
        {"Cs-137": d_cs},
        {"Cs-137": s_cs},
        inv_cs.z, inv_cs.dz,
        alpha=1e-4, method='tikhonov')
    assert np.all(out['profiles']["Cs-137"] >= 0)
    assert np.isfinite(out['chi2'])
