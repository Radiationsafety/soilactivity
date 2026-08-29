"""Fredholm depth inversion subpackage.

Reconstruct vertical volumetric activity profiles a(z) from in-situ
gamma-spectrometry measurements using the Fredholm equation of
the first kind, with geophysics-inspired regularisation and
transport-chemistry-informed parametric priors.

Modules
-------
kernels    — Forward Fredholm kernel K(E, z) and buildup factors.
transport  — ADE migration, Kd database, Bateman chains.
solvers    — Tikhonov, TSVD, Landweber, Cimmino.
criteria   — GCV, L-curve, discrepancy principle.
diagnostics — Resolution matrix, depth of investigation, SVD.
pipeline   — DepthInverter end-to-end pipeline.
"""
from .kernels import GammaLine, Detector, build_kernel, kernel_analytic, buildup_taylor
from .transport import (
    KD_DB, DECAY_S, retardation, kd,
    millington_quirk_d_gas, pulse_profile, exp_profile, chain_evolve,
)
from .solvers import (
    tikhonov, tsvd, landweber, cimmino,
    diff_matrix, depth_scale,
)
from .criteria import (
    chi2, gcv_point, gcv_curve, lcurve_corner,
    choose_alpha_discrepancy,
)
from .diagnostics import resolution_matrix, depth_of_investigation, singulars
from .pipeline import DepthInverter, DepthInversion

__all__ = [
    # kernels
    "GammaLine", "Detector", "build_kernel", "kernel_analytic", "buildup_taylor",
    # transport
    "KD_DB", "DECAY_S", "retardation", "kd",
    "millington_quirk_d_gas", "pulse_profile", "exp_profile", "chain_evolve",
    # solvers
    "tikhonov", "tsvd", "landweber", "cimmino", "diff_matrix", "depth_scale",
    # criteria
    "chi2", "gcv_point", "gcv_curve", "lcurve_corner",
    "choose_alpha_discrepancy",
    # diagnostics
    "resolution_matrix", "depth_of_investigation", "singulars",
    # pipeline
    "DepthInverter", "DepthInversion",
]
