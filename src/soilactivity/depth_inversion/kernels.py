"""Forward Fredholm kernel K(E, z) for in-situ gamma depth inversion.

C_i = integral k_i(z) a(z) dz  — count rate in line *i* for volumetric
activity a(z) [Bq/m^3].

k_i(z) = (y_i * eps / 2) * int_0^{pi/2} tan(theta) * g_i(theta) * B(tau) * exp(-tau) dtheta,
tau = (mu_air * h + mu_soil * z) / cos(theta).

Without buildup and for constant angular efficiency (g=const) the integral
reduces to the analytic E1 form:

    k_i(z) = (y_i * eps / 2) * E1(mu_air * h + mu_soil * z).

Extended kernel models:

- kernel_lateral(): Laterally-extended source (finite field of view).
- kernel_collimated(): Collimated/shielded detector with field of view.
- kernel_multilayer(): Multi-layer soil with per-layer attenuation.

References
----------
- Beck & de Planque (1968) HASL-234.
- Zombori et al. (1992) IAEA-314.
- Tyler (2008) J. Environ. Radioact. 99, 143–161.
- Hasan et al. (2022) J. Environ. Radioact. 251:106877.
"""from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.special import exp1

_trapz = getattr(np, "trapezoid", None) or np.trapz


class GammaLine:
    """Immutable descriptor for a single gamma-emission line.

    Parameters
    ----------
    energy_kev : float
        Photon energy [keV].
    yield_ : float
        Photons per disintegration in this line.
    mu_soil : float
        Linear attenuation coefficient in soil [1/m].
    mu_air : float
        Linear attenuation coefficient in air [1/m].
    """
    __slots__ = ('energy_kev', 'yield_', 'mu_soil', 'mu_air')

    def __init__(self, energy_kev, yield_, mu_soil, mu_air):
        self.energy_kev = float(energy_kev)
        self.yield_ = float(yield_)
        self.mu_soil = float(mu_soil)
        self.mu_air = float(mu_air)

    def __repr__(self):
        return ("GammaLine(E={:.2f} keV, y={:.4f}, "
                "mu_soil={:.2f}, mu_air={:.5f})".format(
                    self.energy_kev, self.yield_, self.mu_soil, self.mu_air))


class Detector:
    """Immutable detector geometry descriptor.

    Parameters
    ----------
    height_m : float
        Height of detector centre above ground [m].
    eff_4pi : float, optional
        Full-sphere (4*pi) detection efficiency.  Default 1.0.
    """
    __slots__ = ('height_m', 'eff_4pi')

    def __init__(self, height_m, eff_4pi=1.0):
        self.height_m = float(height_m)
        self.eff_4pi = float(eff_4pi)

    def __repr__(self):
        return "Detector(h={:.2f} m, eff={:.4f})".format(
            self.height_m, self.eff_4pi)


def build_kernel(lines, detector, z, buildup=None, angular_eff=None,
                 u_max=2.0e3, n_u=4096):
    """Build the Fredholm kernel matrix K: (n_lines, n_z).

    Each element K[i, j] gives the count rate [1/s] per unit volumetric
    activity [Bq/m^3] per unit depth [m] for line *i* at depth z[j].

    The integration variable is u = 1/cos(theta), ranging from 1 to *u_max*.

    Parameters
    ----------
    lines : list of GammaLine
        Gamma-emission line descriptors.
    detector : Detector
        Detector descriptor.
    z : array-like
        Depth grid [m] (positive downwards).
    buildup : callable, optional
        Buildup factor B(tau).  If None, B=1.
    angular_eff : callable, optional
        Angular efficiency g(energy_keV, u_array).  If None, g=1.
    u_max : float, optional
        Upper integration limit for u = 1/cos(theta).  Default 2000.
    n_u : int, optional
        Number of integration points (log-spaced).  Default 4096.

    Returns
    -------
    K : np.ndarray, shape (n_lines, len(z))
    """
    z = np.asarray(z, dtype=float)
    u = np.geomspace(1.0, u_max, n_u)[:, None]          # (n_u, 1)
    K = np.empty((len(lines), z.size))
    for i, ln in enumerate(lines):
        tau = (ln.mu_air * detector.height_m + ln.mu_soil * z)[None, :] * u
        g = np.ones_like(tau) if angular_eff is None else angular_eff(ln.energy_kev, u)
        B = np.ones_like(tau) if buildup is None else buildup(tau)
        K[i] = 0.5 * ln.yield_ * detector.eff_4pi * _trapz(
            B * np.exp(-tau) * g / u, u.ravel(), axis=0)
    return K


def kernel_analytic(line, detector, z):
    """Analytic kernel (no buildup, isotropic detector): 0.5 * y * eps * E1(mu_a*h + mu_s*z).

    Parameters
    ----------
    line : GammaLine
    detector : Detector
    z : array-like
        Depth grid [m].

    Returns
    -------
    k : np.ndarray, shape (len(z),)
    """
    C = line.mu_air * detector.height_m + line.mu_soil * np.asarray(z, float)
    return 0.5 * line.yield_ * detector.eff_4pi * exp1(C)


def buildup_taylor(tau, a=1.0, b=8.0):
    """Taylor buildup factor B(tau) = 1 + a * tau * exp(-tau / b).

    Parameters
    ----------
    tau : array-like
        Optical thickness.
    a : float, optional
        Amplitude parameter.  Default 1.0.
    b : float, optional
        Shape parameter.  Default 8.0.

    Returns
    -------
    B : array-like
        Same shape as *tau*.
    """
    return 1.0 + a * tau * np.exp(-tau / b)


# =====================================================================
# Extended kernel models
# =====================================================================

def kernel_lateral(line, detector, z, r_fov=50.0, buildup=None,
                   u_max=2.0e3, n_u=2048, n_r=64):
    """Kernel for laterally-extended source with finite field of view.

    Accounts for gamma photons arriving from off-axis points within
    a circular field of view of radius r_fov.  At each depth z, the
    contribution is integrated over the lateral distance r from 0
    to r_fov:

    k(z) = int_0^{r_fov} k_point(r, z) * 2*pi*r dr

    where k_point includes the geometric spreading from the off-axis
    point to the detector.  This is important for in-situ measurements
    over large contaminated areas (aerial surveys, wide-area ground
    surveys).

    Reference: Tyler (2008) J. Environ. Radioact. 99, 143–161.

    Parameters
    ----------
    line : GammaLine
    detector : Detector
    z : array-like
        Depth grid [m].
    r_fov : float
        Field-of-view radius [m].  Default 50.0.
    buildup : callable or None
    u_max, n_u : integration parameters for angular integral.
    n_r : int
        Number of radial integration points.

    Returns
    -------
    k : np.ndarray, shape (len(z),)
    """
    z = np.asarray(z, float)
    h = detector.height_m
    r = np.linspace(0.0, r_fov, n_r)
    # Lateral distance increases the path length through soil and air
    # Slant range: L = sqrt((h + z)^2 + r^2)
    # Extra soil path: delta_soil = sqrt(z^2 + r^2) - z  (approx for r << z)
    # For a point at (r, z), the path length through soil is:
    #   path_soil = sqrt(z^2 + r^2)  (from point to surface)
    #   path_air  = sqrt(h^2 + r^2)  (from surface to detector)
    # Total optical depth: mu_soil * sqrt(z^2 + r^2) + mu_air * sqrt(h^2 + r^2)
    k = np.zeros_like(z)
    for j, zj in enumerate(z):
        path_soil = line.mu_soil * np.sqrt(zj ** 2 + r ** 2)
        path_air = line.mu_air * np.sqrt(h ** 2 + r ** 2)
        tau = path_soil + path_air
        # Geometric factor: solid angle subtended by detector
        # For a point source at (r, z), solid angle ~ cos(theta) / L^2
        L = np.sqrt((h + zj) ** 2 + r ** 2)
        geom = h / (L ** 2 + 1e-30)  # cos(theta) / L^2 * h approximation
        B = np.ones_like(tau) if buildup is None else buildup(tau)
        integrand = line.yield_ * detector.eff_4pi * B * np.exp(-tau) * geom * 2.0 * np.pi * r
        k[j] = _trapz(integrand, r)
    return k


def kernel_collimated(line, detector, z, collimator_angle_deg=30.0,
                      buildup=None, u_max=2.0e3, n_u=4096):
    """Kernel for a collimated/shielded detector.

    The collimator restricts the field of view to a cone with
    half-angle theta_c.  Only photons arriving from directions
    within this cone contribute to the signal.

    k(z) = (y * eps / 2) * int_1^{u_max_coll} tan(theta) * B(tau) * exp(-tau) * g(theta) / u du

    where u_max_coll = 1/cos(theta_c), and g(theta) = 1 for
    theta < theta_c, 0 otherwise.

    Parameters
    ----------
    line : GammaLine
    detector : Detector
    z : array-like
        Depth grid [m].
    collimator_angle_deg : float
        Collimator half-angle [degrees].  Default 30.
    buildup : callable or None
    u_max : float
        Maximum u for integration (should be > 1/cos(theta_c)).
    n_u : int

    Returns
    -------
    k : np.ndarray, shape (len(z),)
    """
    z = np.asarray(z, float)
    u_max_c = 1.0 / np.cos(np.radians(collimator_angle_deg))
    u_max_eff = min(u_max, u_max_c * 10)  # a bit beyond collimator edge
    u = np.geomspace(1.0, u_max_eff, n_u)
    k = np.zeros_like(z)
    for j, zj in enumerate(z):
        tau = (line.mu_air * detector.height_m + line.mu_soil * zj) * u
        B = np.ones_like(tau) if buildup is None else buildup(tau)
        # Apply collimator window: g = 1 for u < u_max_c, 0 otherwise
        # Use smooth cutoff to avoid numerical artifacts
        window = 0.5 * (1.0 - np.tanh(20.0 * (u - u_max_c)))
        integrand = B * np.exp(-tau) * window / u
        k[j] = 0.5 * line.yield_ * detector.eff_4pi * _trapz(integrand, u)
    return k


def kernel_multilayer(lines, detector, z, layer_tops, mu_soil_layers,
                       buildup=None, angular_eff=None,
                       u_max=2.0e3, n_u=4096):
    """Kernel for multi-layer soil with per-layer attenuation.

    Each layer has its own mu_soil (different density, composition).
    The total optical depth is computed by summing contributions
    from all layers above the point of interest.

    This is important for real soils where surface organic layers
    (low density, low mu) overlie mineral soil (high density, high mu),
    or where buried contaminated layers exist beneath clean backfill.

    Parameters
    ----------
    lines : list of GammaLine
    detector : Detector
    z : array-like
        Depth grid [m].
    layer_tops : array-like
        Top depths of each layer [m].  First element must be 0.0.
    mu_soil_layers : array-like
        Linear attenuation coefficient for each layer [1/m].
        Length must equal len(layer_tops).
    buildup, angular_eff, u_max, n_u : as in build_kernel.

    Returns
    -------
    K : np.ndarray, shape (n_lines, len(z))
    """
    z = np.asarray(z, float)
    layer_tops = np.asarray(layer_tops, float)
    mu_layers = np.asarray(mu_soil_layers, float)
    n_layers = len(layer_tops)
    u = np.geomspace(1.0, u_max, n_u)[:, None]
    K = np.empty((len(lines), z.size))
    # For each depth z_j, compute mu_soil(z_j) from layer assignment
    mu_soil_z = np.empty_like(z)
    for j, zj in enumerate(z):
        # Find which layer zj belongs to
        idx = np.searchsorted(layer_tops, zj, side='right') - 1
        idx = max(0, min(idx, n_layers - 1))
        mu_soil_z[j] = mu_layers[idx]
    for i, ln in enumerate(lines):
        # Use effective mu_soil at each depth
        tau = (ln.mu_air * detector.height_m + mu_soil_z * z)[None, :] * u
        g = np.ones_like(tau) if angular_eff is None else angular_eff(ln.energy_kev, u)
        B = np.ones_like(tau) if buildup is None else buildup(tau)
        K[i] = 0.5 * ln.yield_ * detector.eff_4pi * _trapz(
            B * np.exp(-tau) * g / u, u.ravel(), axis=0)
    return K
