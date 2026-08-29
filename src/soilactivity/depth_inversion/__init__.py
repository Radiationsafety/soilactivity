"""Fredholm depth inversion subpackage.

Reconstruct vertical volumetric activity profiles a(z) from in-situ
gamma-spectrometry measurements using the Fredholm equation of
the first kind, with geophysics-inspired regularisation and
transport-chemistry-informed parametric priors.

Classical solvers: Tikhonov, TSVD, Landweber, Cimmino.
Geophysical solvers: CGLS, Kaczmarz, FISTA (L1), TV/ADMM,
  IRLS focusing (MGS/MS).
Parameter selection: GCV, L-curve, discrepancy, quasi-optimality,
  NCP, SNR, weighted GCV.
Diagnostics: resolution matrix, DOI, spread function, checkerboard,
  sensitivity kernels, model covariance.
Transport: ADE with Kd, pH-dependent sorption, Freundlich isotherm,
  multi-layer soil, Crank-Nicolson numerical solver, Bateman chains.
Bayesian: Ensemble Kalman Inversion, Laplace MAP, GP prior.
Kernels: point, lateral, collimated, multi-layer.
Pipeline: DepthInverter with AIC/BIC ensemble model selection.

Modules
-------
kernels      -- Forward Fredholm kernel K(E, z) and buildup factors.
transport    -- ADE migration, Kd database, Bateman chains.
solvers      -- Tikhonov, TSVD, Landweber, Cimmino, CGLS, Kaczmarz,
               FISTA, TV/ADMM, IRLS focusing.
criteria     -- GCV, L-curve, discrepancy, quasi-optimality, NCP, SNR.
diagnostics  -- Resolution matrix, DOI, SVD, covariance, checkerboard.
bayesian     -- Ensemble Kalman Inversion, Laplace MAP.
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
)
from .solvers import (
    tikhonov, tsvd, landweber, cimmino,
    cgls, kaczmarz, fista, tv_admm, focusing_irls,
    diff_matrix, depth_scale,
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
    # solvers
    "tikhonov", "tsvd", "landweber", "cimmino",
    "cgls", "kaczmarz", "fista", "tv_admm", "focusing_irls",
    "diff_matrix", "depth_scale",
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
    # pipeline
    "DepthInverter", "DepthInversion",
]
