"""SoilActivity: 3D reconstruction of radionuclide volumetric activity."""

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

__version__ = "0.3.0"
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
]
