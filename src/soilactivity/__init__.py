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

__version__ = "0.2.0"
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
]
