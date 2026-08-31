"""Photon exposure buildup factors B(E, x) per ANSI/ANS-6.4.3-1991.

Provides:
- Lookup of tabulated B(E, x) for 26 materials (elements + water/air/concrete).
- 2D log-log interpolation for arbitrary (E, x).
- GP (Geometric Progression, Harima) computation for water (faster, ~1-3% accuracy).
- Rule of mixtures via equivalent atomic number Zeq for compounds/mixtures.

Data source: Trubey 1988 (ORNL/RSIC-49), ANSI/ANS-6.4.3-1991 standard reference
data. Energy range 0.015-15 MeV; penetration depths 0.5-40 mfp.

References
----------
1. Trubey, D. K. "New Gamma-Ray Buildup Factor Data for Point Kernel
   Calculations: ANS-6.4.3 Standard Reference Data." ORNL/RSIC-49 (1988).
2. Harima, Y. et al. "Validity of the Geometric-Progression Formula in
   Approximating Gamma-Ray Buildup Factors." Nucl. Sci. Eng. 94, 24-35 (1986).
3. Olarinoye, I. O. "EXABCal: A program for calculating photon exposure and
   energy absorption buildup factors." MethodsX 6, 1755-1763 (2019).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "BUILDUP_DATA",
    "GP_WATER",
    "get_buildup",
    "gp_buildup_water",
    "buildup_for_mixture",
    "ANS_ENERGIES",
    "ANS_DEPTHS",
    "AVAILABLE_MATERIALS",
]

# Standard ANS-6.4.3 energy grid (MeV), 25 log-spaced points 0.015 - 15 MeV
ANS_ENERGIES = np.array([
    0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1,
    0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,
    1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0,
])

# Standard ANS-6.4.3 penetration depths in mfp (16 points, 0.5 - 40)
ANS_DEPTHS = np.array([
    0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0,
])


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_buildup_data() -> dict:
    """Lazily load the ANS-6.4.3 EBF tables bundled with the package."""
    data_file = resources.files("soilactivity.data").joinpath(
        "buildup_factors_ans643.json"
    )
    with resources.as_file(data_file) as p:
        return json.loads(Path(p).read_text())


@lru_cache(maxsize=1)
def _load_gp_water() -> dict:
    """Lazily load the water GP fitting coefficients (Harima)."""
    data_file = resources.files("soilactivity.data").joinpath(
        "gp_coefficients_water_ans643.json"
    )
    with resources.as_file(data_file) as p:
        return json.loads(Path(p).read_text())


# Module-level lazy proxies
class _LazyBuildupData:
    def __getattr__(self, name):
        return getattr(_load_buildup_data(), name)

    def __getitem__(self, key):
        return _load_buildup_data()[key]

    def __contains__(self, key):
        return key in _load_buildup_data()

    def keys(self):
        return _load_buildup_data().keys()


class _LazyGPWater:
    def __getattr__(self, name):
        return getattr(_load_gp_water(), name)

    def __getitem__(self, key):
        return _load_gp_water()[key]

    def keys(self):
        return _load_gp_water().keys()


BUILDUP_DATA = _LazyBuildupData()
GP_WATER = _LazyGPWater()


def AVAILABLE_MATERIALS() -> tuple[str, ...]:
    """Tuple of material names with ANS-6.4.3 buildup-factor tables."""
    return tuple(sorted(BUILDUP_DATA["materials"].keys()))


# -----------------------------------------------------------------------------
# Core lookup: B(E, x) for a single material
# -----------------------------------------------------------------------------
def _loglog_interp_2d(
    xs: np.ndarray, ys: np.ndarray, Z: np.ndarray,
    x: float, y: float,
) -> float:
    """Bilinear interpolation in log-log space.

    Parameters
    ----------
    xs : 1D array, monotonic ascending
        Row coordinates (depths in mfp).
    ys : 1D array, monotonic ascending
        Column coordinates (energies in MeV).
    Z : 2D array, shape (len(xs), len(ys))
        Tabulated values (all must be > 0 for log-log).
    x, y : scalars
        Target point.
    """
    x = max(xs[0], min(xs[-1], x))
    y = max(ys[0], min(ys[-1], y))

    log_x = math.log(x)
    log_y = math.log(y)
    log_xs = np.log(xs)
    log_ys = np.log(ys)

    i = int(np.searchsorted(log_xs, log_x, side="right") - 1)
    i = max(0, min(len(xs) - 2, i))
    j = int(np.searchsorted(log_ys, log_y, side="right") - 1)
    j = max(0, min(len(ys) - 2, j))

    wx = (log_x - log_xs[i]) / (log_xs[i + 1] - log_xs[i])
    wy = (log_y - log_ys[j]) / (log_ys[j + 1] - log_ys[j])

    z_ll = Z[i,     j]
    z_lh = Z[i,     j + 1]
    z_hl = Z[i + 1, j]
    z_hh = Z[i + 1, j + 1]

    if min(z_ll, z_lh, z_hl, z_hh) > 0:
        ll = math.log(z_ll); lh = math.log(z_lh)
        hl = math.log(z_hl); hh = math.log(z_hh)
        log_v = (ll * (1 - wx) * (1 - wy) +
                 lh * (1 - wx) * wy +
                 hl * wx * (1 - wy) +
                 hh * wx * wy)
        return math.exp(log_v)
    v = (z_ll * (1 - wx) * (1 - wy) +
         z_lh * (1 - wx) * wy +
         z_hl * wx * (1 - wy) +
         z_hh * wx * wy)
    return max(v, 0.0)


def get_buildup(
    material: str,
    E_MeV: float | Sequence[float] | np.ndarray,
    x_mfp: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Exposure buildup factor B(E, x) for a pure ANS-6.4.3 material.

    Parameters
    ----------
    material : str
        One of :func:`AVAILABLE_MATERIALS` (e.g. 'Water', 'Concrete', 'Lead').
    E_MeV : float or array-like
        Photon energy in MeV. Valid range 0.015 - 15 MeV; values outside
        are clamped to the nearest grid edge.
    x_mfp : float or array-like
        Penetration depth in mean free paths. Valid range 0.5 - 40 mfp.

    Returns
    -------
    float or np.ndarray
        Buildup factor B(E, x). Shape follows broadcasting of inputs.

    Raises
    ------
    ValueError
        If material is not in the ANS-6.4.3 database.
    """
    data = _load_buildup_data()
    mats = data["materials"]
    if material not in mats:
        raise ValueError(
            f"Unknown material '{material}'. "
            f"Available: {sorted(mats.keys())}"
        )
    mat = mats[material]
    Es = np.asarray(mat["energies_MeV"], dtype=float)
    xs = np.asarray(mat["depths_mfp"], dtype=float)
    B = np.asarray(mat["B"], dtype=float)

    # Energies in the source are descending; sort ascending
    order = np.argsort(Es)
    Es = Es[order]
    B = B[:, order]

    E_arr = np.atleast_1d(np.asarray(E_MeV, dtype=float))
    x_arr = np.atleast_1d(np.asarray(x_mfp, dtype=float))
    E_b, x_b = np.broadcast_arrays(E_arr, x_arr)

    flat_E = E_b.ravel()
    flat_x = x_b.ravel()
    flat_out = np.empty(flat_E.size, dtype=float)
    for k in range(flat_E.size):
        flat_out[k] = _loglog_interp_2d(xs, Es, B, flat_x[k], flat_E[k])
    out = flat_out.reshape(E_b.shape)

    if out.size == 1 and np.isscalar(E_MeV) and np.isscalar(x_mfp):
        return float(out[0])
    return out


# -----------------------------------------------------------------------------
# GP formula for water (faster than table lookup, ~1-3% accuracy)
# -----------------------------------------------------------------------------
def gp_buildup_water(
    E_MeV: float,
    x_mfp: float,
    response: str = "air",
) -> float:
    """B(E, x) for water using the Geometric Progression (GP) formula.

    Uses the Harima GP coefficients tabulated by ANS-6.4.3. Validated against
    the tabulated values to within 0.5-3 % across the full grid.

    Parameters
    ----------
    E_MeV : float
        Photon energy in MeV. Range 0.015 - 15 MeV (clamped otherwise).
    x_mfp : float
        Penetration depth in mean free paths. Range 0 - 40 mfp.
    response : {'air', 'water'}, default 'air'
        'air'   -> exposure buildup factor (air-kerma response)
        'water' -> energy absorption buildup factor (water-kerma response)

    Returns
    -------
    float
        Buildup factor B(E, x).

    Notes
    -----
    The GP fitting formula (Harima et al. 1986) is::

        B(E, x) = 1 + (b - 1) * (K**x - 1) / (K - 1)   for K != 1
        B(E, x) = 1 + (b - 1) * x                       for K == 1

        K(E, x) = c * x**a + d * T(x, Xk)
        T(x, Xk) = (tanh(x/Xk - 2) - tanh(-2)) / (1 - tanh(-2))

    where b, c, a, Xk, d are energy-dependent fitting coefficients.
    """
    if response not in ("air", "water"):
        raise ValueError(f"response must be 'air' or 'water', got {response!r}")
    gp = _load_gp_water()
    key = "air_kerma_response" if response == "air" else "water_kerma_response"
    rows = gp[key]
    if not rows:
        raise RuntimeError(f"No GP coefficients for response={response!r}")

    E_clamped = max(rows[0]["E_MeV"], min(rows[-1]["E_MeV"], E_MeV))

    Es = np.array([r["E_MeV"] for r in rows])
    i = int(np.searchsorted(Es, E_clamped, side="right") - 1)
    i = max(0, min(len(rows) - 2, i))
    r_lo = rows[i]
    r_hi = rows[i + 1]
    if r_hi["E_MeV"] == r_lo["E_MeV"]:
        b, c, a, Xk, d = r_lo["b"], r_lo["c"], r_lo["a"], r_lo["Xk"], r_lo["d"]
    else:
        w = (math.log(E_clamped) - math.log(r_lo["E_MeV"])) / \
            (math.log(r_hi["E_MeV"]) - math.log(r_lo["E_MeV"]))
        b  = r_lo["b"]  + (r_hi["b"]  - r_lo["b"])  * w
        c  = r_lo["c"]  + (r_hi["c"]  - r_lo["c"])  * w
        a  = r_lo["a"]  + (r_hi["a"]  - r_lo["a"])  * w
        Xk = r_lo["Xk"] + (r_hi["Xk"] - r_lo["Xk"]) * w
        d  = r_lo["d"]  + (r_hi["d"]  - r_lo["d"])  * w

    if x_mfp <= 0:
        return 1.0
    if x_mfp > 40.0:
        x_mfp = 40.0

    tanh_term = (math.tanh(x_mfp / Xk - 2) - math.tanh(-2)) / (1 - math.tanh(-2))
    K = c * x_mfp ** a + d * tanh_term

    if abs(K - 1.0) < 1e-10:
        B = 1.0 + (b - 1.0) * x_mfp
    else:
        B = 1.0 + (b - 1.0) * (K ** x_mfp - 1.0) / (K - 1.0)
    return max(B, 1.0)


# -----------------------------------------------------------------------------
# Rule of mixtures via equivalent atomic number Zeq
# -----------------------------------------------------------------------------
# Atomic numbers for ANS elements
_Z_OF = {
    "Beryllium": 4, "Boron": 5, "Carbon": 6, "Nitrogen": 7, "Oxygen": 8,
    "Sodium": 11, "Magnesium": 12, "Aluminum": 13, "Silicon": 14,
    "Phosphorus": 15, "Sulphur": 16, "Argon": 18, "Potassium": 19,
    "Calcium": 20, "Iron": 26, "Copper": 29, "Molybdenum": 42,
    "Tin": 50, "Lanthanum": 57, "Gadolinium": 64,
    "Tungsten": 74, "Lead": 82, "Uranium": 92,
}

# Common short symbols -> ANS material names
_ELEMENT_ALIASES = {
    "H": "Hydrogen", "Be": "Beryllium", "B": "Boron", "C": "Carbon", "N": "Nitrogen",
    "O": "Oxygen", "Na": "Sodium", "Mg": "Magnesium", "Al": "Aluminum",
    "Si": "Silicon", "P": "Phosphorus", "S": "Sulphur", "Ar": "Argon",
    "K": "Potassium", "Ca": "Calcium", "Fe": "Iron", "Cu": "Copper",
    "Mo": "Molybdenum", "Sn": "Tin", "La": "Lanthanum", "Gd": "Gadolinium",
    "W": "Tungsten", "Pb": "Lead", "U": "Uranium",
}


def _canonical_element_name(name: str) -> str:
    """Map element symbol ('Pb', 'Si', 'O') to ANS material name."""
    s = name.strip()
    for k, v in _ELEMENT_ALIASES.items():
        if s.lower() == k.lower() or s.lower() == v.lower():
            return v
    raise KeyError(f"Unknown element symbol/name: {name!r}")


def _equivalent_atomic_number(
    E_MeV: float,
    composition: dict[str, float],
    coeff_lookup: Callable[[str, float], tuple[float | None, float | None]],
) -> float:
    """Compute the equivalent atomic number Zeq for a mixture at energy E.

    Zeq is defined by matching the mixture's R = (mu_en/rho)/(mu/rho) to the
    elemental R(E, Z) curve (linear interpolation in Z).

    Parameters
    ----------
    E_MeV : float
        Photon energy.
    composition : dict[str, float]
        {'ElementSymbol': weight_fraction}. Sums should be ~1.
    coeff_lookup : callable
        Function (element_name, E_MeV) -> (mu_en_over_rho, mu_over_rho)
        in cm^2/g.

    Returns
    -------
    float
        Equivalent atomic number Zeq (real-valued).
    """
    Zs: list[int] = []
    Rs: list[float] = []
    for el in _Z_OF:
        mu_en, mu = coeff_lookup(el, E_MeV)
        if mu_en is None or mu is None or mu <= 0:
            continue
        Zs.append(_Z_OF[el])
        Rs.append(mu_en / mu)

    if not Zs:
        raise RuntimeError("No NIST data available for any ANS element at this E")

    num = 0.0
    den = 0.0
    for el, w in composition.items():
        el_canonical = _canonical_element_name(el)
        mu_en, mu = coeff_lookup(el_canonical, E_MeV)
        if mu_en is None or mu is None:
            raise ValueError(
                f"No NIST coefficients for element {el!r} "
                f"(resolved as {el_canonical!r})"
            )
        num += w * mu_en
        den += w * mu
    if den <= 0:
        raise ValueError("Mixture has zero/negative total mu/rho")
    R_mix = num / den

    Zs_arr = np.array(Zs, dtype=float)
    Rs_arr = np.array(Rs, dtype=float)
    if R_mix <= Rs_arr.min():
        return float(Zs_arr[int(np.argmin(Rs_arr))])
    if R_mix >= Rs_arr.max():
        return float(Zs_arr[int(np.argmax(Rs_arr))])
    order = np.argsort(Zs_arr)
    Zs_sorted = Zs_arr[order]
    Rs_sorted = Rs_arr[order]
    k = int(np.searchsorted(Rs_sorted, R_mix, side="right") - 1)
    k = max(0, min(len(Zs_sorted) - 2, k))
    R_lo = Rs_sorted[k]; R_hi = Rs_sorted[k + 1]
    Z_lo = Zs_sorted[k]; Z_hi = Zs_sorted[k + 1]
    if R_hi == R_lo:
        return float(Z_lo)
    w = (R_mix - R_lo) / (R_hi - R_lo)
    return float(Z_lo + w * (Z_hi - Z_lo))


def buildup_for_mixture(
    composition: dict[str, float],
    E_MeV: float,
    x_mfp: float,
    coeff_lookup: Callable[[str, float], tuple[float | None, float | None]] | None = None,
) -> float:
    """Buildup factor for a mixture/compound via the Zeq method.

    Parameters
    ----------
    composition : dict[str, float]
        {'ElementSymbol': weight_fraction}. Example for water:
        ``{'H': 0.1119, 'O': 0.8881}``. For typical concrete:
        ``{'H':0.01,'O':0.529,'Mg':0.002,'Al':0.034,'Si':0.337,'Ca':0.044,'Fe':0.013}``.
    E_MeV : float
        Photon energy in MeV.
    x_mfp : float
        Penetration depth in mean free paths.
    coeff_lookup : callable, optional
        Function ``(element_name, E_MeV) -> (mu_en_over_rho, mu_over_rho)``.
        If None, falls back to ``soilactivity.physics.attenuation.lookup``.

    Returns
    -------
    float
        Buildup factor B(E, x) for the mixture.

    Notes
    -----
    The Zeq method is the standard ANS-6.4.3 procedure for compounds and
    mixtures. For layered/heterogeneous shields use Broder or Kalos formulas
    instead (not yet implemented here).
    """
    if coeff_lookup is None:
        try:
            from . import attenuation as _att  # noqa: WPS433
        except ImportError as e:
            raise ImportError(
                "buildup_for_mixture() requires either an explicit "
                "coeff_lookup callable or the soilactivity.physics.attenuation "
                "module. Provide coeff_lookup=(el,E)->(mu_en,mu) to bypass."
            ) from e
        coeff_lookup = lambda el, E: _att.lookup(el, E)  # noqa: E731

    Zeq = _equivalent_atomic_number(E_MeV, composition, coeff_lookup)
    data = _load_buildup_data()
    pairs: list[tuple[int, str]] = [
        (_Z_OF[n], n) for n in data["materials"] if n in _Z_OF
    ]
    pairs.sort()
    if Zeq <= pairs[0][0]:
        return float(get_buildup(pairs[0][1], E_MeV, x_mfp))
    if Zeq >= pairs[-1][0]:
        return float(get_buildup(pairs[-1][1], E_MeV, x_mfp))
    for k in range(len(pairs) - 1):
        Z_lo, n_lo = pairs[k]
        Z_hi, n_hi = pairs[k + 1]
        if Z_lo <= Zeq <= Z_hi:
            w = (Zeq - Z_lo) / (Z_hi - Z_lo)
            B_lo = float(get_buildup(n_lo, E_MeV, x_mfp))
            B_hi = float(get_buildup(n_hi, E_MeV, x_mfp))
            return float(B_lo + w * (B_hi - B_lo))
    raise RuntimeError("Unreachable: Zeq not in any ANS bracket")
