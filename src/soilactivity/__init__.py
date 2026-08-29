"""SoilActivity: radionuclide activity reconstruction and dosimetry.

Submodules:
- core: 3D volumetric unfolding (Unfolder)
- fredholm: 2D Fredholm equation solver for SAD from ADER
- reconstructor: high-level SadReconstructor API
- visibility: barrier geometry (buildings) for Fredholm equation
- radionuclides: kerma constants, normalising factors, fuel vectors
- mcc: Method of Conversion Coefficients (ADER -> SAD)
- lorenz: Lorenz curve compactness analysis
- correlation: information correlation coefficient (Linfoot)
- diagnostics: SLAE condition numbers and error bounds
- dosimetry: ICRP 74 conversion coefficients
- attenuation: NIST XCOM mass attenuation coefficients
- buildup: ANS-6.4.3 exposure buildup factors
"""

from .core import Unfolder, UnfoldingResult
from .buildup import (
    get_buildup,
    gp_buildup_water,
    buildup_for_mixture,
    AVAILABLE_MATERIALS,
    ANS_ENERGIES,
    ANS_DEPTHS,
)
from .attenuation import (
    lookup_mu_rho,
    lookup_mu_en_rho,
    lookup,
    mixture_mu_rho,
    mixture_mu_en_rho,
    linear_attenuation,
    mean_free_path,
    NIST_AIR_DRY_COMPOSITION,
    NIST_WATER_COMPOSITION,
    NIST_CONCRETE_COMPOSITION,
    NIST_SOIL_COMPOSITION,
)
from .dosimetry import (
    h_star_10_over_Ka,
    h_star_10_over_phil,
    kerma_per_fluence_air,
    point_source_dose_rate,
    ICRP74_ENERGIES_MEV,
)
from .reconstructor import SadReconstructor, SadResult
from .radionuclides import (
    KERMA_CONSTANTS,
    get_normalizing_factor,
    mixture_kerma_constant,
    mixture_sakr,
    CHERNOBYL_FUEL_VECTOR_131D,
)
from .mcc import mcc_ader_to_sad, mcc_sad_to_ader, mcc_coefficient, mcc_total_activity
from .lorenz import lorenz_curve, lorenz_gini_coefficient, lorenz_compactness_ratio
from .correlation import information_correlation_coefficient, entropy
from .diagnostics import slae_condition_number, slae_error_bound, slae_finer_error_estimate
from .visibility import compute_visibility_matrix, visibility_radius_mask
from .spatial_interpolation import (
    Interpolator2D,
    InterpolationAutoSelector,
    SparseResultInterpolator,
    MeasurementSensitivityAnalyzer,
    idw_interpolate,
    barnes_interpolate,
    cressman_interpolate,
    AVAILABLE_METHODS,
)
from .fredholm import (
    build_fredholm_matrix,
    build_fredholm_matrix_no_vis,
    solve_fredholm_tikhonov,
    solve_fredholm_tikhonov_nn,
    raster_coords,
    raster_to_vector,
    vector_to_raster,
)

__version__ = "0.5.0"
__all__ = [
    "Unfolder",
    "UnfoldingResult",
    # Buildup factors (ANS-6.4.3)
    "get_buildup",
    "gp_buildup_water",
    "buildup_for_mixture",
    "AVAILABLE_MATERIALS",
    "ANS_ENERGIES",
    "ANS_DEPTHS",
    # Attenuation (NIST XCOM via xraylib)
    "lookup_mu_rho",
    "lookup_mu_en_rho",
    "lookup",
    "mixture_mu_rho",
    "mixture_mu_en_rho",
    "linear_attenuation",
    "mean_free_path",
    "NIST_AIR_DRY_COMPOSITION",
    "NIST_WATER_COMPOSITION",
    "NIST_CONCRETE_COMPOSITION",
    "NIST_SOIL_COMPOSITION",
    # Dosimetry (ICRP 74)
    "h_star_10_over_Ka",
    "h_star_10_over_phil",
    "kerma_per_fluence_air",
    "point_source_dose_rate",
    "ICRP74_ENERGIES_MEV",
    # 2D Fredholm SAD reconstruction (Chizhov et al 2019-2024)
    "SadReconstructor",
    "SadResult",
    "KERMA_CONSTANTS",
    "get_normalizing_factor",
    "mixture_kerma_constant",
    "mixture_sakr",
    "CHERNOBYL_FUEL_VECTOR_131D",
    "mcc_ader_to_sad",
    "mcc_sad_to_ader",
    "mcc_coefficient",
    "mcc_total_activity",
    "lorenz_curve",
    "lorenz_gini_coefficient",
    "lorenz_compactness_ratio",
    "information_correlation_coefficient",
    "entropy",
    "slae_condition_number",
    "slae_error_bound",
    "slae_finer_error_estimate",
    "compute_visibility_matrix",
    "visibility_radius_mask",
    "build_fredholm_matrix",
    "build_fredholm_matrix_no_vis",
    "solve_fredholm_tikhonov",
    "solve_fredholm_tikhonov_nn",
    "raster_coords",
    "raster_to_vector",
    "vector_to_raster",
    # Spatial interpolation
    "Interpolator2D",
    "InterpolationAutoSelector",
    "SparseResultInterpolator",
    "MeasurementSensitivityAnalyzer",
    "idw_interpolate",
    "barnes_interpolate",
    "cressman_interpolate",
    "AVAILABLE_METHODS",
]
