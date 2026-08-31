"""Fredholm depth inversion subpackage.

Reconstruct vertical volumetric activity profiles a(z) from in-situ
gamma-spectrometry measurements using the Fredholm equation of
the first kind, with geophysics-inspired regularisation and
transport-chemistry-informed parametric priors.

Classical solvers: Tikhonov, TSVD, Landweber, Cimmino.
Geophysical solvers: CGLS, CGLS+L-curve, Kaczmarz, FISTA (L1),
  TV/ADMM, IRLS focusing (MGS/MS), depth-weighted Tikhonov,
  K-fold cross-validation alpha.
Parameter selection: GCV, L-curve, discrepancy, quasi-optimality,
  NCP, SNR, weighted GCV.
Diagnostics: resolution matrix, DOI, spread function, checkerboard,
  sensitivity kernels, model covariance, data resolution, info content.
Transport: ADE with Kd, pH-dependent sorption, Freundlich isotherm,
  multi-layer soil, Crank-Nicolson solver, Bateman chains,
  two-site kinetic sorption, competitive ion exchange, Eh-dependent U.
Bayesian: Ensemble Kalman Inversion, Laplace MAP, GP prior.
Kernels: point, lateral, collimated, multi-layer.
Geophysics: depth weighting (Li-Oldenburg, power-law, logarithmic,
  adaptive), joint multi-nuclide inversion, compactness operators.
Pipeline: DepthInverter with AIC/BIC ensemble model selection.

Modules
-------
kernels      -- Forward Fredholm kernel K(E, z) and buildup factors.
transport    -- ADE migration, Kd database, Bateman chains, kinetic sorption.
solvers      -- Tikhonov, TSVD, Landweber, Cimmino, CGLS, Kaczmarz,
               FISTA, TV/ADMM, IRLS focusing, CGLS+L-curve, crossval.
criteria     -- GCV, L-curve, discrepancy, quasi-optimality, NCP, SNR.
diagnostics  -- Resolution matrix, DOI, SVD, covariance, checkerboard.
bayesian     -- Ensemble Kalman Inversion, Laplace MAP.
geophysics   -- Depth weighting, joint inversion, two-site sorption.
pipeline     -- DepthInverter end-to-end pipeline.
"""
from .kernels import (
    GammaLine, Detector, build_kernel, kernel_analytic, buildup_taylor,
    kernel_lateral, kernel_collimated, kernel_multilayer,
)
from .transport import (
    KD_DB, DECAY_S, retardation, kd,
    millington_quirk_d_gas, pulse_profile, exp_profile, chain_evolve,
    kd_ph, kd_freundlich, retardation_freundlich,
    multi_layer_pulse, effective_properties, ade_solve,
    FREUNDLICH_DB, KD_PH_PARAMS,
    two_site_effective_Kd, two_site_retardation_from_kd,
    competitive_kd_cs, competitive_kd_sr, kd_u_eh,
)
from .solvers import (
    tikhonov, tsvd, landweber, cimmino,
    cgls, kaczmarz, fista, tv_admm, focusing_irls,
    diff_matrix, depth_scale,
    cgls_lcurve, crossval_alpha, depth_weighted_tikhonov,
)
from .criteria import (
    chi2, gcv_point, gcv_curve, lcurve_corner,
    choose_alpha_discrepancy,
    quasi_optimality, ncp_criterion, snr_criterion,
    gcv_weighted, gcv_weighted_curve, lcurve_corner_iter,
)
from .diagnostics import (
    resolution_matrix, depth_of_investigation, singulars,
    model_covariance, spread_function, checkerboard_test,
    data_resolution_matrix, sensitivity_kernels, information_content,
)
from .bayesian import (
    ensemble_kalman_inversion, laplace_map, gp_prior_covariance,
)
from .geophysics import (
    depth_scale_power, depth_scale_log, depth_scale_adaptive,
    compose_weighting,
    joint_kernel, joint_inversion, joint_coupling_matrix,
    two_site_retardation, two_site_ade,
    weighted_smoothness, compactness_operator,
)
from .pipeline import DepthInverter, DepthInversion

__all__ = [
    # kernels
    "GammaLine", "Detector", "build_kernel", "kernel_analytic", "buildup_taylor",
    "kernel_lateral", "kernel_collimated", "kernel_multilayer",
    # transport
    "KD_DB", "DECAY_S", "retardation", "kd",
    "millington_quirk_d_gas", "pulse_profile", "exp_profile", "chain_evolve",
    "kd_ph", "kd_freundlich", "retardation_freundlich",
    "multi_layer_pulse", "effective_properties", "ade_solve",
    "FREUNDLICH_DB", "KD_PH_PARAMS",
    "two_site_effective_Kd", "two_site_retardation_from_kd",
    "competitive_kd_cs", "competitive_kd_sr", "kd_u_eh",
    # solvers
    "tikhonov", "tsvd", "landweber", "cimmino",
    "cgls", "kaczmarz", "fista", "tv_admm", "focusing_irls",
    "diff_matrix", "depth_scale",
    "cgls_lcurve", "crossval_alpha", "depth_weighted_tikhonov",
    # criteria
    "chi2", "gcv_point", "gcv_curve", "lcurve_corner",
    "choose_alpha_discrepancy",
    "quasi_optimality", "ncp_criterion", "snr_criterion",
    "gcv_weighted", "gcv_weighted_curve", "lcurve_corner_iter",
    # diagnostics
    "resolution_matrix", "depth_of_investigation", "singulars",
    "model_covariance", "spread_function", "checkerboard_test",
    "data_resolution_matrix", "sensitivity_kernels", "information_content",
    # bayesian
    "ensemble_kalman_inversion", "laplace_map", "gp_prior_covariance",
    # geophysics
    "depth_scale_power", "depth_scale_log", "depth_scale_adaptive",
    "compose_weighting",
    "joint_kernel", "joint_inversion", "joint_coupling_matrix",
    "two_site_retardation", "two_site_ade",
    "weighted_smoothness", "compactness_operator",
    # pipeline
    "DepthInverter", "DepthInversion",
]
