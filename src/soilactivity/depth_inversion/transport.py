"""Transport chemistry: ADE with sorption (Kd), decay, Bateman chains.

R_j * dC_j/dt = d/dz(D_j * dC_j/dz) - v_j * dC_j/dz - lambda_j * R_j * C_j
                   + lambda_{j-1} * b_{j-1} * R_{j-1} * C_{j-1}
R = 1 + rho_b * Kd / theta.

Kd depends on pH/Eh, mineralogy, competitors (Cs<->K+/NH4+,
Sr<->Ca2+, U<->carbonates/Eh, Pu<->colloids).

References
----------
- IAEA TRS-472 (2010) — radiological assessment.
- Sheppard & Thibault (1990) — Kd compilation.
- van Genuchten (1981) — analytical solutions for ADE.
- Millington & Quirk (1961) — gas diffusivity in porous media.
- Bateman (1910) — radioactive decay chains.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.linalg import expm


# Kd ranges, ml/g (IAEA TRS-472, 2010; Sheppard & Thibault, 1990)
KD_DB = {
    "Cs-137": (100.0, 3000.0),
    "Sr-90": (10.0, 500.0),
    "Co-60": (50.0, 1000.0),
    "U-238": (1.0, 5000.0),
    "Pu-239": (100.0, 5.0e4),
    "Am-241": (100.0, 1.0e4),
    "Ra-226": (10.0, 1000.0),
    "Pb-210": (100.0, 5000.0),
}

# Decay constants, 1/s
DECAY_S = {
    "Cs-137": 7.31e-10,
    "Sr-90": 7.27e-10,
    "Co-60": 4.16e-9,
    "Eu-152": 1.62e-9,
    "U-238": 4.92e-18,
    "Ra-226": 1.37e-11,
    "Rn-222": 2.10e-6,
    "Pb-210": 9.93e-10,
    "Am-241": 1.60e-11,
    "Pu-239": 9.05e-13,
}


def retardation(rho_b, kd_ml_g, theta):
    """Retardation factor R = 1 + rho_b * Kd / theta.

    Parameters
    ----------
    rho_b : float
        Bulk density [g/cm^3].
    kd_ml_g : float
        Distribution coefficient [ml/g].
    theta : float
        Volumetric water content (dimensionless, 0–1).

    Returns
    -------
    R : float
    """
    return 1.0 + rho_b * kd_ml_g / theta


def kd(nuclide, scenario="mid"):
    """Kd from database for a given uncertainty scenario.

    Parameters
    ----------
    nuclide : str
        Nuclide key (e.g. 'Cs-137').
    scenario : {'low', 'mid', 'high'}
        Uncertainty scenario.  'mid' uses geometric mean.

    Returns
    -------
    kd_val : float
        Distribution coefficient [ml/g].
    """
    lo, hi = KD_DB.get(nuclide, (10.0, 1000.0))
    return {"low": lo, "mid": float(np.sqrt(lo * hi)), "high": hi}[scenario]


def millington_quirk_d_gas(D0, theta, theta_a):
    """Effective gas diffusivity (Rn-222) in porous media [m^2/s].

    D_e = D0 * theta_a^(10/3) / theta^2   (Millington & Quirk, 1961).

    Parameters
    ----------
    D0 : float
        Free-air molecular diffusion coefficient [m^2/s].
    theta : float
        Total porosity.
    theta_a : float
        Air-filled porosity.

    Returns
    -------
    D_e : float
    """
    return D0 * theta_a ** (10.0 / 3.0) / max(theta, 1e-6) ** 2


def pulse_profile(z, A0, t, D, v, R, lam):
    """a(z, t) [Bq/m^3]: instantaneous surface deposition A0 [Bq/m^2] at t=0.

    Gaussian spreading with retardation and reflection at z=0:
    a(z,t) = A0 * exp(-lam*t) / sqrt(4*pi*D_s*t) * [
        exp(-(z - v_s*t)^2 / (4*D_s*t)) +
        exp(-(z + v_s*t)^2 / (4*D_s*t))]
    where D_s = D/R, v_s = v/R.

    Parameters
    ----------
    z : array-like
        Depth grid [m] (>= 0).
    A0 : float
        Initial surface deposition [Bq/m^2].
    t : float
        Time since deposition [s].
    D : float
        Effective diffusion coefficient [m^2/s].
    v : float
        Effective advection velocity [m/s] (positive = downward).
    R : float
        Retardation factor.
    lam : float
        Radioactive decay constant [1/s].

    Returns
    -------
    a : np.ndarray, shape (len(z),)
    """
    z = np.asarray(z, float)
    if t <= 0.0:
        return np.zeros_like(z)
    Ds = D / R
    vs = v / R
    s2 = 4.0 * Ds * t
    g = (np.exp(-((z - vs * t) ** 2) / s2)
         + np.exp(-((z + vs * t) ** 2) / s2)) / np.sqrt(np.pi * s2)
    return A0 * np.exp(-lam * t) * g


def exp_profile(z, A, lam_relax_m):
    """Diffusion-limited profile: a(z) = (A / lam) * exp(-z / lam).

    Normalised so that integral_0^inf a(z) dz = A [Bq/m^2].

    Parameters
    ----------
    z : array-like
        Depth grid [m].
    A : float
        Areal activity [Bq/m^2].
    lam_relax_m : float
        Relaxation length [m].

    Returns
    -------
    a : np.ndarray
    """
    lam = max(float(lam_relax_m), 1e-6)
    return A / lam * np.exp(-np.asarray(z, float) / lam)


def chain_evolve(lams, A0, t, branch=None):
    """Bateman equations for a decay chain.

    A(t) = expm(M * t) @ A0,  where M[i,i] = -lambda_i,
    M[i, i-1] = lambda_{i-1} * b_{i-1}.

    Parameters
    ----------
    lams : array-like
        Decay constants [1/s], ordered parent -> daughter -> ...
    A0 : array-like
        Initial activities [Bq].
    t : float
        Time [s].
    branch : array-like, optional
        Branching ratios (length len(lams)-1).  Default 1.0.

    Returns
    -------
    A : np.ndarray, shape (len(lams),)
    """
    n = len(lams)
    M = np.zeros((n, n))
    idx = np.arange(n)
    M[idx, idx] = -np.asarray(lams, float)
    for i in range(1, n):
        b = branch[i - 1] if branch is not None else 1.0
        M[i, i - 1] = lams[i - 1] * b
    return expm(M * t) @ np.asarray(A0, float)
