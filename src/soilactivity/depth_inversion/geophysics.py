"""Geophysical inversion utilities: depth weighting, joint inversion,
two-site kinetic sorption, and advanced regularisation operators.

Adapted from gravity/magnetic/EM geophysics (Li & Oldenburg 1996, 1998;
Zhdanov 2002; Portniaguine & Zhdanov 1999) to radionuclide depth
profiling via Fredholm inversion of in-situ gamma spectra.

Functions
--------
Depth weighting:
    depth_scale_li_oldenburg  -- standard 1/||K_i|| (column norm)
    depth_scale_power         -- z^beta power-law weighting
    depth_scale_log           -- logarithmic depth weighting
    depth_scale_adaptive      -- data-driven adaptive weighting
    compose_weighting         -- combine kernel + depth + custom weights

Joint multi-nuclide inversion:
    joint_kernel              -- build stacked kernel for multiple nuclides
    joint_inversion           -- simultaneous inversion with shared geology

Two-site kinetic sorption:
    two_site_retardation      -- effective retardation with kinetic exchange
    two_site_ade              -- coupled ADE for dissolved + sorbed phases

Regularisation operators:
    weighted_smoothness       -- depth-weighted 2nd-order difference
    compactness_operator      -- minimum-gradient-support reweighting matrix

References
----------
- Li & Oldenburg (1996) Geophysics 61(2):394-408.
- Li & Oldenburg (1998) Geophysics 63(1):270-279.
- Zhdanov (2002) Geophysical Inverse Theory and Optimization Problems.
- Portniaguine & Zhdanov (1999) Geophysics 64(3):874-887.
- van Genuchten & Wagenet (1989) Soil Sci. Soc. Am. J. 53:1305-1312.
- Selim et al. (1976) Soil Sci. Soc. Am. J. 40:560-566.
- Sperry et al. (2018) Adv. Water Resour. 111:49-67.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.linalg import svd


# =====================================================================
# Depth weighting functions
# =====================================================================

def depth_scale_li_oldenburg(K):
    """Standard column-norm depth weighting (Li & Oldenburg, 1996).

    S_i = 1 / ||K_i|| so that A' = K * diag(S) has unit-norm columns.
    Counteracts the natural decay of kernel columns with depth.

    Parameters
    ----------
    K : array-like (m, n)

    Returns
    -------
    S : ndarray (n,)
    """
    K = np.asarray(K, float)
    S = np.linalg.norm(K, axis=0)
    S[S == 0] = 1.0
    return 1.0 / S


def depth_scale_power(z, beta=2.0, z_ref=1.0):
    """Power-law depth weighting w(z) = (z/z_ref + 1)^{-beta/2}.

    Originally developed for 3-D gravity inversion (Li & Oldenburg,
    1998) where kernel amplitude decays as 1/z^3.  Adapted to 1-D
    depth profiling where the kernel (E1) decays exponentially.

    The exponent beta controls the weighting strength:
      beta=0: no weighting
      beta=2: moderate (default, suitable for gravity)
      beta=3: strong (suitable for magnetic dipole)

    For gamma-spectrometry kernels, beta in [1, 3] is typical.

    Parameters
    ----------
    z : array-like (n,)
        Depth grid [m] (positive downwards).
    beta : float
        Power-law exponent.  Default 2.0.
    z_ref : float
        Reference depth [m].  Default 1.0.

    Returns
    -------
    w : ndarray (n,)
    """
    z = np.asarray(z, float)
    return (z / z_ref + 1.0) ** (-beta / 2.0)


def depth_scale_log(z, z0=0.01):
    """Logarithmic depth weighting w(z) = 1 / log(1 + z/z0).

    Provides milder depth weighting than power-law, useful when
    the kernel already includes some depth normalisation.

    Parameters
    ----------
    z : array-like (n,)
        Depth grid [m].
    z0 : float
        Scale depth [m].  Default 0.01 m (1 cm).

    Returns
    -------
    w : ndarray (n,)
    """
    z = np.asarray(z, float)
    val = np.log(1.0 + z / max(z0, 1e-6))
    val[val == 0] = 1e-10
    return 1.0 / val


def depth_scale_adaptive(K, z, alpha=1e-4):
    """Data-driven adaptive depth weighting.

    Combines kernel column norm (Li-Oldenburg) with power-law depth
    weighting and a smoothness-based correction.  The adaptive
    component down-weights depths where the kernel has poor
    sensitivity (small singular values).

    w_adaptive = w_kernel * w_power * (1 + alpha * w_smoothness)

    where w_smoothness penalises oscillatory sensitivity patterns.

    Parameters
    ----------
    K : array-like (m, n)
    z : array-like (n,)
    alpha : float
        Smoothness correction strength.

    Returns
    -------
    w : ndarray (n,)
    """
    K = np.asarray(K, float)
    z = np.asarray(z, float)
    w_kern = depth_scale_li_oldenburg(K)
    w_pow = depth_scale_power(z, beta=2.0)
    # Smoothness: penalise rows where sensitivity oscillates
    col_norms = np.linalg.norm(K, axis=0)
    col_norms /= max(col_norms.max(), 1e-30)
    # Derivative of normalised column norms
    dcn = np.abs(np.diff(col_norms))
    smoothness = np.zeros_like(col_norms)
    smoothness[1:] = dcn
    smoothness[0] = dcn[0] if len(dcn) > 0 else 0.0
    w_adapt = w_kern * w_pow * (1.0 + alpha / (smoothness + 1e-10))
    return w_adapt / max(w_adapt.max(), 1e-30)


def compose_weighting(K, z, method='li_oldenburg', **kw):
    """Compute composite depth weighting vector.

    Parameters
    ----------
    K : array-like (m, n)
    z : array-like (n,)
    method : {'li_oldenburg', 'power', 'log', 'adaptive', 'combined'}
        Weighting method.  'combined' = li_oldenburg * power.
    **kw : extra parameters passed to the chosen method.

    Returns
    -------
    w : ndarray (n,)
    """
    if method == 'li_oldenburg':
        return depth_scale_li_oldenburg(K)
    elif method == 'power':
        return depth_scale_power(z, **kw)
    elif method == 'log':
        return depth_scale_log(z, **kw)
    elif method == 'adaptive':
        return depth_scale_adaptive(K, z, **kw)
    elif method == 'combined':
        return depth_scale_li_oldenburg(K) * depth_scale_power(z, **kw)
    else:
        raise ValueError("Unknown weighting method: {}".format(method))


# =====================================================================
# Joint multi-nuclide inversion
# =====================================================================

def joint_kernel(kernels_dict, z, dz=None):
    """Build stacked kernel matrix for joint multi-nuclide inversion.

    Different nuclides may have been measured with different detector
    configurations (heights, efficiencies).  This function stacks all
    kernels into a single system:

        [K_nuclide1]         [a_nuclide1(z)]
        [K_nuclide2] * dz  = [a_nuclide2(z)]
        [    ...   ]         [     ...      ]

    Each nuclide has its own activity profile a_j(z).  If the
    nuclides share a common contamination history (e.g., global
    fallout), a coupling constraint can be added externally.

    Parameters
    ----------
    kernels_dict : dict
        Keys: nuclide names (str).  Values: kernel matrices (m_j, n_z).
        Each kernel may have different number of rows (measurements)
        but must share the same depth grid (n_z columns).
    z : array-like (n_z,)
        Shared depth grid [m].
    dz : array-like (n_z,) or None
        Depth cell widths.  If None, computed from z.

    Returns
    -------
    K_joint : ndarray (M, N)
        Block-diagonal stacked kernel, where M = sum(m_j),
        N = n_nuclides * n_z.
    nuclide_order : list of str
        Order of nuclides in the block-diagonal.
    """
    z = np.asarray(z, float)
    if dz is None:
        dz = np.diff(z, prepend=0.0)
        dz[0] = dz[1] if len(dz) > 1 else z[0] if z[0] > 0 else 0.01
    else:
        dz = np.asarray(dz, float)

    nuclide_order = list(kernels_dict.keys())
    n_z = len(z)
    blocks = []
    for name in nuclide_order:
        Kj = np.asarray(kernels_dict[name], float)
        assert Kj.shape[1] == n_z, (
            "Kernel for {} has {} depth columns, expected {}".format(
                name, Kj.shape[1], n_z))
        blocks.append(Kj * dz[None, :])
    return np.block([[np.zeros_like(b) if i != j else b
                      for j in range(len(blocks))]
                     for i, b in enumerate(blocks)]), nuclide_order


def joint_coupling_matrix(n_nuclides, n_z, coupling='proportional'):
    """Build coupling constraint matrix for joint inversion.

    Enforces a relationship between nuclide depth profiles.

    Parameters
    ----------
    n_nuclides : int
        Number of nuclides.
    n_z : int
        Number of depth cells.
    coupling : {'proportional', 'same_shape', 'none'}
        'proportional': a_j(z) = c_j * s(z)  (same shape, different amplitudes)
        'same_shape': a_j(z) = s(z)  (identical profiles, useful for decay chains)
        'none': identity (no coupling, independent inversion)

    Returns
    -------
    C : ndarray (n_z, n_nuclides * n_z) or None
        Coupling matrix such that s = C @ a_joint.
        For 'none', returns None (no coupling).
    """
    N = n_nuclides * n_z
    if coupling == 'none':
        return None
    elif coupling == 'same_shape':
        # s = mean of all nuclide profiles
        C = np.zeros((n_z, N))
        for j in range(n_nuclides):
            C[:, j * n_z:(j + 1) * n_z] = np.eye(n_z) / n_nuclides
        return C
    elif coupling == 'proportional':
        # Reference profile is the first nuclide
        C = np.zeros((n_z, N))
        C[:, :n_z] = np.eye(n_z)
        return C
    else:
        raise ValueError("Unknown coupling: {}".format(coupling))


def joint_inversion(kernels_dict, data_dict, sigma_dict, z, dz=None,
                    coupling=None, alpha=1e-4, method='tikhonov',
                    nonneg=True, **solver_kw):
    """Joint multi-nuclide inversion with optional coupling.

    Parameters
    ----------
    kernels_dict : dict
        {nuclide_name: K_j (m_j, n_z)}
    data_dict : dict
        {nuclide_name: d_j (m_j,)}
    sigma_dict : dict
        {nuclide_name: sigma_j (m_j,)}
    z : array-like (n_z,)
    dz : array-like or None
    coupling : {'proportional', 'same_shape', 'none'} or None
    alpha : float
    method : {'tikhonov', 'cgls', 'fista'}
    nonneg : bool
    **solver_kw : extra solver parameters.

    Returns
    -------
    dict with keys:
        'profiles' : dict {nuclide_name: a_j(z)}
        'chi2' : float
        'info' : dict
    """
    from .solvers import tikhonov, cgls, fista, _weighted, diff_matrix

    z = np.asarray(z, float)
    if dz is None:
        dz = np.diff(z, prepend=0.0)
        dz[0] = dz[1] if len(dz) > 1 else 0.01
    dz = np.asarray(dz, float)

    # Build joint system
    K_joint, nuclide_order = joint_kernel(kernels_dict, z, dz)
    d_joint = np.concatenate([np.asarray(data_dict[n], float) for n in nuclide_order])
    sigma_joint = np.concatenate([np.asarray(sigma_dict[n], float) for n in nuclide_order])

    A, b = _weighted(K_joint, d_joint, sigma_joint)
    n_z = len(z)
    N = len(nuclide_order) * n_z

    if coupling is not None and coupling != 'none':
        C = joint_coupling_matrix(len(nuclide_order), n_z, coupling)
        if C is not None:
            # Reduce to coupled system: K_reduced = K_joint @ C.T
            # For 'same_shape': this gives n_z unknowns
            K_red = K_joint @ C.T
            A_red, b_red = _weighted(K_red, d_joint, sigma_joint)
            if method == 'tikhonov':
                L = diff_matrix(n_z) if n_z > 2 else np.eye(n_z)
                x_red = tikhonov(A_red, b_red, alpha, L=L, nonneg=nonneg)
            elif method == 'cgls':
                x_red, info = cgls(A_red, b_red, sigma=None, nonneg=nonneg, **solver_kw)
            else:
                x_red = tikhonov(A_red, b_red, alpha, nonneg=nonneg)
            # Expand back
            x_joint = C.T @ x_red
        else:
            x_joint, info = _solve_joint(A, b, method, alpha, nonneg, N, n_z, **solver_kw)
    else:
        x_joint, info = _solve_joint(A, b, method, alpha, nonneg, N, n_z, **solver_kw)

    # Split per nuclide
    profiles = {}
    for j, name in enumerate(nuclide_order):
        profiles[name] = x_joint[j * n_z:(j + 1) * n_z]

    r = K_joint @ x_joint - d_joint
    chi2 = float(np.sum((r / sigma_joint) ** 2))

    return {'profiles': profiles, 'chi2': chi2, 'info': info,
            'nuclide_order': nuclide_order}


def _solve_joint(A, b, method, alpha, nonneg, N, n_z, **kw):
    """Solve the joint system."""
    from .solvers import tikhonov, cgls, fista, diff_matrix
    L = diff_matrix(n_z) if n_z > 2 else np.eye(n_z)
    # Block-diagonal regularisation (same L for each nuclide block)
    n_blocks = N // n_z
    L_big = np.zeros((L.shape[0] * n_blocks, N))
    for j in range(n_blocks):
        L_big[j * L.shape[0]:(j + 1) * L.shape[0],
             j * n_z:(j + 1) * n_z] = L
    if method == 'tikhonov':
        x = tikhonov(A, b, alpha, L=L_big, nonneg=nonneg)
        return x, {}
    elif method == 'cgls':
        x, info = cgls(A, b, sigma=None, nonneg=nonneg, **kw)
        return x, info
    elif method == 'fista':
        x, info = fista(A, b, alpha, nonneg=nonneg, **kw)
        return x, info
    else:
        x = tikhonov(A, b, alpha, L=L_big, nonneg=nonneg)
        return x, {}


# =====================================================================
# Two-site kinetic sorption model
# =====================================================================

def two_site_retardation(R_inst, f, omega_kin, t_obs):
    """Effective retardation factor for two-site kinetic sorption.

    The two-site model (van Genuchten & Wagenet, 1989) partitions
    sorption sites into:
      - Type 1 (fraction f): instantaneous (equilibrium) sorption
      - Type 2 (fraction 1-f): kinetic (first-order) sorption

    The kinetic rate equation is:
        dS2/dt = omega * (Kd * C - S2)

    The effective retardation depends on observation time:
        R_eff(t) = 1 + rho_b/theta * [f*Kd + (1-f)*Kd*(1 - exp(-omega*t))]

    At short times (t << 1/omega), R_eff -> 1 + rho_b*f*Kd/theta (only
    equilibrium sites contribute).  At long times (t >> 1/omega),
    R_eff -> 1 + rho_b*Kd/theta (all sites equilibrated = linear Kd).

    Parameters
    ----------
    R_inst : float
        Instantaneous retardation (all sites at equilibrium).
        R_inst = 1 + rho_b * Kd / theta.
    f : float
        Fraction of equilibrium (Type 1) sites (0 to 1).
    omega_kin : float
        First-order kinetic rate constant [1/s].
        Typical: 1e-7 to 1e-4 /s.
    t_obs : float
        Observation/exposure time [s].

    Returns
    -------
    R_eff : float
        Effective retardation factor at time t_obs.
    """
    # R_inst = 1 + rho_b * Kd / theta
    # R_eq = 1 + rho_b * f * Kd / theta = 1 + f * (R_inst - 1)
    R_eq = 1.0 + f * (R_inst - 1.0)
    # Kinetic contribution
    kinetic_frac = 1.0 - np.exp(-omega_kin * t_obs)
    # R_eff = R_eq + (1-f) * (R_inst - 1) * kinetic_frac
    #       = 1 + f*(R-1) + (1-f)*(R-1)*(1 - exp(-omega*t))
    #       = 1 + (R-1) * [f + (1-f)*(1-exp(-omega*t))]
    R_eff = 1.0 + (R_inst - 1.0) * (f + (1.0 - f) * kinetic_frac)
    return R_eff


def two_site_ade(z, t_span, D, v, R_inst, f, omega_kin, lam,
                 theta=0.3, rho_b=1.4, Kd_total=None,
                 dz=None, n_save=100):
    """Two-site kinetic sorption ADE solver.

    Coupled system for dissolved (C) and kinetically-sorbed (S2):

        theta * dC/dt + rho_b * dS2/dt = theta * D * d2C/dz2
                                  - theta * v * dC/dz - lambda * theta * C
                                  - lambda * rho_b * S2

        dS2/dt = omega * (Kd2 * C - S2)

    where Kd2 = (1-f) * Kd_total is the kinetic-site distribution
    coefficient and Kd1 = f * Kd_total is the equilibrium contribution.

    Solved with operator splitting: Crank-Nicolson for transport,
    analytical integration for kinetic sorption.

    Reference: van Genuchten & Wagenet (1989) Soil Sci. Soc. Am. J. 53.

    Parameters
    ----------
    z : array-like or None
        Depth grid [m].  If None, auto-computed.
    t_span : tuple (t0, t_final)
        Time interval [s].
    D : float
        Dispersion coefficient [m^2/s].
    v : float
        Pore water velocity [m/s] (positive downward).
    R_inst : float
        Instantaneous retardation (all sites at equilibrium).
    f : float
        Fraction of equilibrium sites (0-1).
    omega_kin : float
        Kinetic rate constant [1/s].
    lam : float
        Radioactive decay constant [1/s].
    theta : float
        Volumetric water content.  Default 0.3.
    rho_b : float
        Bulk density [g/cm^3].  Default 1.4.
    Kd_total : float or None
        Total Kd [ml/g].  If None, derived from R_inst.
    dz : float or None
        Grid spacing.
    n_save : int
        Number of time snapshots.

    Returns
    -------
    result : dict
        'z', 't', 'C' (dissolved), 'S2' (kinetically sorbed),
        'C_total' (total = C + rho_b/theta * S1 + rho_b/theta * S2).
    """
    from .transport import _thomas_solve

    if z is None:
        v_eff = v
        R_mean = two_site_retardation(R_inst, f, omega_kin,
                                        0.5 * (t_span[0] + t_span[1]))
        z_max = max(t_span[1] * v_eff / R_mean * 3.0, 0.5)
        dz = dz or 0.005
        z = np.arange(0, z_max + dz, dz)
    else:
        z = np.asarray(z, float)
        dz = dz or (z[1] - z[0]) if len(z) > 1 else 0.005

    if Kd_total is None:
        Kd_total = (R_inst - 1.0) * theta / rho_b

    Kd_eq = f * Kd_total        # equilibrium sites
    Kd_kin = (1.0 - f) * Kd_total  # kinetic sites

    n_z = len(z)
    t0, t_final = t_span
    dt = (t_final - t0) / n_save
    n_steps = int(np.ceil((t_final - t0) / dt))
    dt = (t_final - t0) / n_steps

    # State variables
    C = np.zeros(n_z)    # dissolved concentration
    S2 = np.zeros(n_z)   # kinetically sorbed concentration [Bq/g]

    # Effective retardation for transport step
    R_eq = 1.0 + rho_b * Kd_eq / theta  # equilibrium retardation

    t_save = np.linspace(t0, t_final, n_save)
    C_saved = np.zeros((n_save, n_z))
    S2_saved = np.zeros((n_save, n_z))
    save_idx = 0

    for step in range(n_steps + 1):
        t = t_now = t0 + step * dt

        # Save if needed
        if save_idx < n_save and t >= t_save[save_idx]:
            C_saved[save_idx] = C.copy()
            S2_saved[save_idx] = S2.copy()
            save_idx += 1

        if step == n_steps:
            break

        # --- Operator splitting ---

        # Step 1: Transport (Crank-Nicolson with R_eq retardation)
        n_int = n_z - 2
        if n_int < 1:
            continue
        r = dt / (2.0 * dz ** 2)
        s = dt / (4.0 * dz)

        a_low = np.zeros(n_int)
        a_diag = np.zeros(n_int)
        a_up = np.zeros(n_int)
        rhs = np.zeros(n_int)

        cr_val = r * D / R_eq
        cl_val = r * D / R_eq
        cv_val = s * v / R_eq
        cd_val = dt * lam / 2.0

        for i in range(n_int):
            ii = i + 1

            a_low[i] = -cl_val - cv_val
            a_diag[i] = 1.0 + cl_val + cr_val + cd_val
            a_up[i] = -cr_val + cv_val

            C_im = C[ii - 1]
            C_i = C[ii]
            C_ip = C[ii + 1]
            rhs[i] = ((cl_val + cv_val) * C_im
                      + (1.0 - cl_val - cr_val - cd_val) * C_i
                      + (cr_val - cv_val) * C_ip)

        # Bottom BC: no flux
        if n_int > 0:
            rhs[-1] += (cr_val - cv_val) * C[-1]

        C_new = _thomas_solve(a_low, a_diag, a_up, rhs)
        C[1:-1] = C_new
        C[-1] = C[-2]

        # Step 2: Kinetic sorption (analytical integration over dt)
        # dS2/dt = omega * (Kd_kin * C - S2)
        # Solution: S2(t+dt) = Kd_kin*C + (S2(t) - Kd_kin*C) * exp(-omega*dt)
        S2_eq = Kd_kin * C  # equilibrium value for kinetic sites
        decay_factor = np.exp(-omega_kin * dt)
        S2 = S2_eq + (S2 - S2_eq) * decay_factor

        # Radioactive decay on both phases
        decay_factor_total = np.exp(-lam * dt)
        C *= decay_factor_total
        S2 *= decay_factor_total

    # Ensure all saves done
    C_saved[-1] = C.copy()
    S2_saved[-1] = S2.copy()

    # Total concentration in soil [Bq/m^3 of bulk soil]
    # C_total = theta * C + rho_b * Kd_eq * C + rho_b * S2
    C_total_saved = theta * C_saved + rho_b * Kd_eq * C_saved + rho_b * S2_saved

    return {
        'z': z, 't': t_save,
        'C': C_saved,           # dissolved [Bq/L]
        'S2': S2_saved,          # kinetically sorbed [Bq/g]
        'C_total': C_total_saved,  # total [Bq/m^3 bulk]
    }


# =====================================================================
# Advanced regularisation operators
# =====================================================================

def weighted_smoothness(n, z, w=None, order=2):
    """Depth-weighted smoothness (difference) operator.

    L_w = diag(w) @ D  where D is the finite-difference matrix.
    Deeper layers with smaller w get less penalty for roughness,
    allowing sharper features at depth where data have less control.

    Parameters
    ----------
    n : int
        Number of depth cells.
    z : array-like (n,)
        Depth grid [m].
    w : array-like (n,) or None
        Weight vector.  If None, unit weights.
    order : {1, 2}
        Difference order.

    Returns
    -------
    L : ndarray
        Weighted difference operator.
    """
    from .solvers import diff_matrix
    D = diff_matrix(n, order=order)
    if w is None:
        return D
    w = np.asarray(w, float)
    # Weight rows: use average of adjacent w values
    n_rows = D.shape[0]
    w_row = np.zeros(n_rows)
    if order == 1:
        for i in range(n_rows):
            w_row[i] = 0.5 * (w[i] + w[i + 1])
    elif order == 2:
        for i in range(n_rows):
            w_row[i] = 0.5 * (w[i] + w[i + 2])
    return w_row[:, None] * D


def compactness_operator(x, eps=1e-2, mode='mgs'):
    """Compute IRLS reweighting matrix for compact inversion.

    W_ii = x_i^2 + eps^2  (MS) or (Dx_i)^2 + eps^2 (MGS).
    Used as W^{-1} in the regularisation term.

    Parameters
    ----------
    x : ndarray (n,)
        Current model estimate.
    eps : float
        Focusing parameter.
    mode : {'ms', 'mgs'}

    Returns
    -------
    W_inv : ndarray (n, n) or (n-1, n-1)
        Diagonal matrix of inverse weights.
    """
    from .solvers import diff_matrix
    if mode == 'mgs':
        n = len(x)
        D = diff_matrix(n)
        gx = D @ x
        w = gx ** 2 + eps ** 2
        return np.diag(1.0 / w)
    else:
        w = x ** 2 + eps ** 2
        return np.diag(1.0 / w)
