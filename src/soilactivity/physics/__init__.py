"""Physics subpackage: photon transport, dosimetry, buildup factors.

Modules
-------
buildup     : ANSI/ANS-6.4.3 exposure buildup factors B(E, x).
"""
from .buildup import (
    BUILDUP_DATA,
    GP_WATER,
    ANS_ENERGIES,
    ANS_DEPTHS,
    AVAILABLE_MATERIALS,
    get_buildup,
    gp_buildup_water,
    buildup_for_mixture,
)

__all__ = [
    "BUILDUP_DATA",
    "GP_WATER",
    "ANS_ENERGIES",
    "ANS_DEPTHS",
    "AVAILABLE_MATERIALS",
    "get_buildup",
    "gp_buildup_water",
    "buildup_for_mixture",
]
