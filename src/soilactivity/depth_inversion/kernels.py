"""Forward Fredholm kernel K(E, z) for in-situ gamma depth inversion.

C_i = integral k_i(z) a(z) dz  — count rate in line *i* for volumetric
activity a(z) [Bq/m^3].

k_i(z) = (y_i * eps / 2) * int_0^{pi/2} tan(theta) * g_i(theta) * B(tau) * exp(-tau) dtheta,
tau = (mu_air * h + mu_soil * z) / cos(theta).

Without buildup and for constant angular efficiency (g=const) the integral
reduces to the analytic E1 form:

    k_i(z) = (y_i * eps / 2) * E1(mu_air * h + mu_soil * z).

References
----------
- Beck & de Planque (1968) HASL-234.
- Zombori et al. (1992) IAEA-314.
- Tyler (2008) J. Environ. Radioact. 99, 143–161.
"""
from __future__ import division, print_function, absolute_import

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
    lines.
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
