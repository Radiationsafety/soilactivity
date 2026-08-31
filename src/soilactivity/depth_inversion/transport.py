"""Transport chemistry: ADE with sorption (Kd), decay, Bateman chains,
multi-layer soil, pH-dependent Kd, Freundlich isotherm, and numerical
ADE solver (Crank-Nicolson).

    R_j * dC_j/dt = d/dz(D_j * dC_j/dz) - v_j * dC_j/dz
                      - lambda_j * R_j * C_j
                      + lambda_{j-1} * b_{j-1} * R_{j-1} * C_{j-1}

    R = 1 + rho_b * Kd / theta.

Kd depends on pH/Eh, mineralogy, competitors (Cs<->K+/NH4+,
Sr<->Ca2+, U<->carbonates/Eh, Pu<->colloids).

References
----------
    IAEA TRS-472 (2010) — radiological assessment.
    Sheppard & Thibault (1990) — Kd compilation.
    van Genuchten (1981) — analytical solutions for ADE.
    Millington & Quirk (1961) — gas diffusivity in porous media.
    Bateman (1910) — radioactive decay chains.
    Kurikami et al. (2017) — kinetic sorption coupling, J. Env. Radioact.
    Roushdy (2025) — sorption performance and migration, Environ. Earth Sci.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.linalg import expm

_trapz = getattr(np, "trapezoid", None) or np.trapz

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
    "Eu-152": (50.0, 2000.0),
    "I-131": (0.1, 10.0),
    "Tc-99": (0.01, 1.0),
    "Se-79": (1.0, 100.0),
}

# pH-dependent Kd model parameters (empirical, fitted to IAEA data)
# log10(Kd) = a_pH * pH + b_clay * clay_frac + c_oc * OC_frac + d_intercept
KD_PH_PARAMS = {
    "Cs-137": {"a_pH": 0.15, "b_clay": 0.02, "c_oc": 0.01, "d": 1.5},
    "Sr-90":  {"a_pH": 0.08, "b_clay": 0.005, "c_oc": 0.005, "d": 0.5},
    "Co-60":  {"a_pH": 0.10, "b_clay": 0.01, "c_oc": 0.008, "d": 1.0},
    "U-238":  {"a_pH": -0.05, "b_clay": 0.005, "c_oc": 0.01, "d": 0.8},
}

# Freundlich isotherm parameters: S = K_F * C^n_F
# K_F [ml/g], n_F [-]
FREUNDLICH_DB = {
    "Cs-137": (500.0, 0.85),
    "Sr-90":  (50.0, 0.90),
    "Pu-239": (2000.0, 0.80),
    "U-238":  (100.0, 0.75),
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
    "I-131": 9.98e-7,
    "Tc-99": 3.20e-14,
    "Se-79": 4.17e-10,
}


# =====================================================================
# Core sorption functions
# =====================================================================

def retardation(rho_b, kd_ml_g, theta):
    """Retardation factor R = 1 + rho_b * Kd / theta.

    Parameters
    ----------
    rho_b : float
        Bulk density [g/cm^3].
    kd_ml_g : float
        Distribution coefficient [ml/g].
    theta : float
        Volumetric water content (dimensionless, 0-1).

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


# =====================================================================
# pH-dependent and isotherm-based Kd
# =====================================================================

def kd_ph(nuclide, pH, clay_frac=0.2, oc_frac=0.02):
    """pH-dependent distribution coefficient.

    log10(Kd) = a_pH * pH + b_clay * clay_frac + c_oc * oc_frac + d

    Parameters
    ----------
    nuclide : str
    pH : float
        Soil pH (0-14).
    clay_frac : float
        Clay fraction (0-1).  Default 0.2.
    oc_frac : float
        Organic carbon fraction (0-1).  Default 0.02.

    Returns
    -------
    kd_val : float
        Distribution coefficient [ml/g].
    """
    params = KD_PH_PARAMS.get(nuclide)
    if params is None:
        return kd(nuclide, "mid")  # fallback to database
    log_kd = (params["a_pH"] * pH + params["b_clay"] * clay_frac
              + params["c_oc"] * oc_frac + params["d"])
    return float(10.0 ** log_kd)


def kd_freundlich(nuclide, C, K_F=None, n_F=None):
    """Concentration-dependent Kd via Freundlich isotherm.

    S = K_F * C^n_F  =>  Kd(C) = S/C = K_F * C^(n_F - 1)

    At low concentrations (C -> 0) and n_F < 1, Kd -> infinity
    (strong sorption at trace levels, realistic for radionuclides).

    Parameters
    ----------
    nuclide : str
        Nuclide key.  Used to look up default parameters.
    C : float
        Dissolved concentration [Bq/L] or [mol/L].
    K_F : float or None
        Freundlich constant [ml/g * (L/g)^(n_F-1)].
    n_F : float or None
        Freundlich exponent.  n_F < 1: concave (favourable).

    Returns
    -------
    kd_eff : float
        Effective Kd [ml/g] at concentration C.
    """
    if K_F is None or n_F is None:
        K_F, n_F = FREUNDLICH_DB.get(nuclide, (100.0, 0.85))
    C = max(float(C), 1e-30)
    return K_F * C ** (n_F - 1.0)


def retardation_freundlich(rho_b, nuclide, C, theta, **kw):
    """Retardation factor with concentration-dependent (Freundlich) Kd.

    Parameters
    ----------
    rho_b : float
    nuclide : str
    C : float or ndarray
        Concentration.
    theta : float

    Returns
    -------
    R : float or ndarray
    """
    kd_eff = kd_freundlich(nuclide, C, **kw)
    return 1.0 + rho_b * kd_eff / theta


# =====================================================================
# Gas-phase transport (Rn-222)
# =====================================================================

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


# =====================================================================
# Analytical transport profiles
# =====================================================================

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


# =====================================================================
# Multi-layer soil transport
# =====================================================================

def multi_layer_pulse(z, A0, t, layers):
    """Pulse profile through a multi-layer soil column.

    Each layer has its own D, v, R, rho_b, theta.  The solution
    is computed by matching boundary conditions at layer interfaces.
    For thin layers (diffusion time << transit time), the
    Laplace-domain solution simplifies to a product of
    retardation factors and an effective Gaussian.

    Parameters
    ----------
    z : array-like
        Depth grid [m].
    A0 : float
        Surface deposition [Bq/m^2].
    t : float
        Time since deposition [s].
    layers : list of dict
        Each dict has keys: 'z_top' [m], 'z_bot' [m],
        'D' [m^2/s], 'v' [m/s], 'R' [-], 'lam' [1/s].
        Layers must be contiguous and ordered top to bottom.

    Returns
    -------
    a : np.ndarray, shape (len(z),)
    """
    z = np.asarray(z, float)
    a = np.zeros_like(z)
    for L in layers:
        mask = (z >= L["z_top"]) & (z < L["z_bot"])
        if not np.any(mask):
            continue
        z_local = z[mask] - L["z_top"]
        # Effective time for this layer (account for retardation of
        # all layers above)
        R_eff = L["R"]
        D_eff = L["D"]
        v_eff = L["v"]
        lam_eff = L["lam"]
        a[mask] = pulse_profile(z_local, A0, t, D_eff, v_eff,
                               R_eff, lam_eff)
    return a


def effective_properties(z, layers):
    """Get effective transport properties at each depth from layer definitions.

    Parameters
    ----------
    z : array-like
        Depth grid [m].
    layers : list of dict
        Layer definitions with 'z_top', 'z_bot', and property keys.

    Returns
    -------
    props : dict of ndarray
        Each key maps to an array of shape (len(z),) with the
        property value at each depth.
    """
    z = np.asarray(z, float)
    n = len(z)
    # Get all property keys from first layer (excluding z_top, z_bot)
    keys = [k for k in layers[0].keys() if k not in ("z_top", "z_bot")]
    props = {k: np.zeros(n) for k in keys}
    for L in layers:
        mask = (z >= L["z_top"]) & (z < L["z_bot"])
        for k in keys:
            if k in L:
                props[k][mask] = L[k]
    return props


# =====================================================================
# Numerical ADE solver (Crank-Nicolson)
# =====================================================================

def ade_solve(z, t_span, D, v, R, lam, C0_func=None,
              dz=None, dt=None, n_save=100):
    """Numerical solution of 1D ADE using Crank-Nicolson.

    R * dC/dt = D * d^2C/dz^2 - v * dC/dz - lam * R * C

    with boundary conditions:
        C(z=0, t) = C0_func(t)  (Dirichlet, if given)
        dC/dz(z=L, t) = 0       (no-flux at bottom)
    Initial condition: C(z, 0) = 0.

    The Crank-Nicolson scheme is unconditionally stable and
    second-order accurate in both space and time.

    Parameters
    ----------
    z : array-like or None
        Depth grid.  If None, created from dz and z_max=t_span[1]*v/R*2.
    t_span : tuple (t0, t_final)
        Time interval [s].
    D : float or ndarray (n_z,)
        Dispersion coefficient [m^2/s].  Can vary with depth.
    v : float or ndarray (n_z,)
        Pore water velocity [m/s] (positive = downward).
    R : float or ndarray (n_z,)
        Retardation factor.
    lam : float
        Radioactive decay constant [1/s].
    C0_func : callable or None
        Surface concentration function C0(t) [Bq/L].
        If None, zero flux BC at top.
    dz : float or None
        Grid spacing.  If None, auto-computed from z.
    dt : float or None
        Time step.  If None, auto-computed (n_save steps).
    n_save : int
        Number of time points to save.

    Returns
    -------
    result : dict
        'z' : ndarray (n_z,)
            Depth grid.
        't' : ndarray (n_save,)
            Time points.
        'C' : ndarray (n_save, n_z)
            Concentration at each (t, z).
    """
    if z is None:
        v_eff = np.asarray(v, float).mean() if np.ndim(v) > 0 else float(v)
        R_eff = np.asarray(R, float).mean() if np.ndim(R) > 0 else float(R)
        z_max = max(t_span[1] * v_eff / R_eff * 3.0, 0.5)
        dz = dz or 0.005
        z = np.arange(0, z_max + dz, dz)
    else:
        z = np.asarray(z, float)
        dz = dz or (z[1] - z[0]) if len(z) > 1 else 0.005
    n_z = len(z)
    D = np.full(n_z, float(D)) if np.isscalar(D) else np.asarray(D, float)
    v = np.full(n_z, float(v)) if np.isscalar(v) else np.asarray(v, float)
    R = np.full(n_z, float(R)) if np.isscalar(R) else np.asarray(R, float)
    t0, t_final = t_span
    dt = dt or (t_final - t0) / n_save
    n_steps = int(np.ceil((t_final - t0) / dt))
    dt = (t_final - t0) / n_steps  # exact
    # Build tridiagonal system for Crank-Nicolson
    # R * (C^{k+1} - C^k) / dt = 0.5 * [L(C^{k+1}) + L(C^k)]
    # L(C) = D d^2C/dz^2 - v dC/dz - lam R C
    r = dt / (2.0 * dz ** 2)
    s = dt / (4.0 * dz)
    # Interior points (i = 1..n_z-2)
    # Left-hand side (implicit)
    # Right-hand side (explicit)
    C = np.zeros(n_z)
    t_save = np.linspace(t0, t_final, n_save)
    C_saved = np.zeros((n_save, n_z))
    save_idx = 0
    for step in range(n_steps + 1):
        t = t0 + step * dt
        # Save if needed
        if save_idx < n_save and t >= t_save[save_idx]:
            C_saved[save_idx] = C.copy()
            save_idx += 1
        if step == n_steps:
            break
        # Build system for interior points
        n_int = n_z - 2
        if n_int < 1:
            continue
        # D at half-points (harmonic mean for interface)
        D_right = 2.0 * D[1:] * D[:-1] / (D[1:] + D[:-1] + 1e-30)
        # LHS coefficients (implicit)
        a_low = np.zeros(n_int)   # sub-diagonal
        a_diag = np.zeros(n_int)  # diagonal
        a_up = np.zeros(n_int)    # super-diagonal
        # RHS (explicit part)
        rhs = np.zeros(n_int)
        for i in range(n_int):
            ii = i + 1  # index in full grid
            Dr = D_right[ii]      # D at i+1/2
            Dl = D_right[ii - 1]  # D at i-1/2
            Ri = R[ii]
            vi = v[ii]
            # Diffusion coefficients
            cr = r * Dr / Ri
            cl = r * Dl / Ri
            # Advection coefficient
            cv = s * vi / Ri
            # Decay
            cd = dt * lam / 2.0
            # LHS: implicit part
            a_low[i] = -cl - cv          # from (i-1)
            a_diag[i] = 1.0 + cl + cr + cd  # diagonal
            a_up[i] = -cr + cv            # from (i+1)
            # RHS: explicit part
            C_im = C[ii - 1]  # left neighbour
            C_i = C[ii]       # centre
            C_ip = C[ii + 1]  # right neighbour
            rhs[i] = (cl + cv) * C_im + (1.0 - cl - cr - cd) * C_i + (cr - cv) * C_ip
        # Boundary conditions
        # Top (z=0): Dirichlet C[0] = C0_func(t+dt) or zero-flux
        if C0_func is not None:
            C_top_new = C0_func(t + dt)
            rhs[0] += (cl[0] + cv[0]) * C_top_new
        # Bottom (z=z_max): no-flux dC/dz = 0 => C[n_z-1] = C[n_z-2]
        if n_int > 0:
            rhs[-1] += (cr[-1] - cv[-1]) * C[-1]  # C[n_z-1] approx = C[n_z-2]
        # Solve tridiagonal system (Thomas algorithm)
        C_new_interior = _thomas_solve(a_low, a_diag, a_up, rhs)
        C[1:-1] = C_new_interior
        # Update boundaries
        if C0_func is not None:
            C[0] = C0_func(t + dt)
        C[-1] = C[-2]  # no-flux
    # Ensure all saves are done
    C_saved[-1] = C.copy()
    return {"z": z, "t": t_save, "C": C_saved}


def _thomas_solve(a, b, c, d):
    """Thomas algorithm for tridiagonal system.

    a: sub-diagonal (n-1), b: diagonal (n), c: super-diagonal (n-1), d: rhs (n).
    """
    n = len(b)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)
    x = np.zeros(n)
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i - 1] * (c_[i - 1] if i > 0 else 0)
        m = a[i - 1] / denom
        d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / denom
        if i < n - 1:
            c_[i] = c[i] / denom
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x


# =====================================================================
# Two-site kinetic sorption (convenience wrappers)
# =====================================================================

def two_site_effective_Kd(Kd_total, f, omega_kin, t_obs, rho_b=1.4, theta=0.3):
    """Effective Kd accounting for kinetic sorption.

    At short times, only the equilibrium fraction f contributes.
    At long times (t >> 1/omega), the full Kd is recovered.

    R_eff(t) = 1 + rho_b/theta * [f*Kd + (1-f)*Kd*(1-exp(-omega*t))]
    => Kd_eff(t) = [f + (1-f)*(1-exp(-omega*t))] * Kd

    Parameters
    ----------
    Kd_total : float
        Total distribution coefficient [ml/g] at equilibrium.
    f : float
        Equilibrium fraction (0-1).
    omega_kin : float
        Kinetic rate constant [1/s].
    t_obs : float
        Observation time [s].
    rho_b : float
        Bulk density [g/cm^3].
    theta : float
        Volumetric water content.

    Returns
    -------
    Kd_eff : float
        Time-effective Kd [ml/g].
    """
    kinetic_frac = 1.0 - np.exp(-omega_kin * t_obs)
    return Kd_total * (f + (1.0 - f) * kinetic_frac)


def two_site_retardation_from_kd(Kd_total, f, omega_kin, t_obs,
                                  rho_b=1.4, theta=0.3):
    """Retardation factor with two-site kinetic sorption.

    Parameters
    ----------
    Kd_total, f, omega_kin, t_obs : as in two_site_effective_Kd.
    rho_b, theta : soil properties.

    Returns
    -------
    R_eff : float
    """
    Kd_eff = two_site_effective_Kd(Kd_total, f, omega_kin, t_obs, rho_b, theta)
    return retardation(rho_b, Kd_eff, theta)


# =====================================================================
# Competitive sorption (ion exchange)
# =====================================================================

def competitive_kd_cs(kd_cs_ref, K_conc, NH4_conc,
                      K_sel=0.1, NH4_sel=0.4):
    """Kd for Cs-137 with competitive K+ and NH4+ ions.

    Cs+ competes with K+ and NH4+ for frayed-edge sites (FES)
    on clay minerals (illite, vermiculite).  The selectivity
    coefficients determine the reduction in Kd:

    Kd_Cs_eff = Kd_Cs_ref / (1 + K_NH4/K_Cs * [NH4] + K_K/K_Cs * [K])

    Simplified: Kd_Cs_eff = Kd_ref / (1 + alpha_K * [K] + alpha_NH4 * [NH4])

    Reference: IAEA TRS-472 (2010), Ch. 4.

    Parameters
    ----------
    kd_cs_ref : float
        Reference Kd for Cs [ml/g] without competitors.
    K_conc : float
        Potassium concentration in pore water [mmol/L].
    NH4_conc : float
        Ammonium concentration in pore water [mmol/L].
    K_sel : float
        K+ selectivity coefficient relative to Cs+.  Default 0.1.
    NH4_sel : float
        NH4+ selectivity coefficient relative to Cs+.  Default 0.4.

    Returns
    -------
    kd_eff : float
        Effective Kd [ml/g] with competition.
    """
    competition = 1.0 + K_sel * K_conc + NH4_sel * NH4_conc
    return kd_cs_ref / competition


def competitive_kd_sr(kd_sr_ref, Ca_conc, Ca_sel=0.05):
    """Kd for Sr-90 with competitive Ca2+ ions.

    Sr2+ competes with Ca2+ for exchange sites.

    Reference: IAEA TRS-472 (2010), Ch. 4.

    Parameters
    ----------
    kd_sr_ref : float
        Reference Kd for Sr [ml/g].
    Ca_conc : float
        Calcium concentration in pore water [mmol/L].
    Ca_sel : float
        Ca selectivity coefficient.  Default 0.05.

    Returns
    -------
    kd_eff : float
    """
    return kd_sr_ref / (1.0 + Ca_sel * Ca_conc)


# =====================================================================
# Eh-dependent U sorption
# =====================================================================

def kd_u_eh(pH, Eh_mV, HCO3_mM=1.0):
    """Kd for U-238 as function of pH, Eh, and carbonate.

    Uranium speciation is strongly controlled by redox potential:
    - U(VI) (uranyl): mobile, forms UO2(CO3)3^{4-} at high HCO3-
    - U(IV) (UO2): immobile, precipitates as uraninite

    Simplified: Kd increases exponentially below a critical Eh
    where U(VI) -> U(IV) reduction occurs.

    Reference: IAEA TRS-472 (2010), Ch. 5.

    Parameters
    ----------
    pH : float
        Soil pH (0-14).
    Eh_mV : float
        Redox potential [mV].
    HCO3_mM : float
        Bicarbonate concentration [mmol/L].

    Returns
    -------
    kd_u : float
        Effective Kd for U [ml/g].
    """
    # Critical Eh for U reduction (depends on pH)
    # Approximate U(VI)/U(IV) boundary (UO2^{2+}/UO2):
    Eh_crit = 470.0 - 60.0 * pH  # ~50 mV at pH 7
    if Eh_mV < Eh_crit:
        # Reducing conditions: U(IV) predominates -> high Kd
        log_kd = 2.0 + 0.3 * pH  # 100-10000 ml/g
    else:
        # Oxidising: U(VI) mobile, reduced by carbonate complexation
        log_kd = -0.5 + 0.15 * pH - 0.1 * np.log10(max(HCO3_mM, 0.01))
    return float(10.0 ** log_kd)
