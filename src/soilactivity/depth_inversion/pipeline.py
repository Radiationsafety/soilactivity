"""DepthInverter: end-to-end pipeline from peak count rates to a(z) + 1-sigma.

Supports both non-parametric (Tikhonov/NNLS with GCV/L-curve/discrepancy)
and parametric (transport-chemistry-informed profile) inversion.

References
----------
- Beck & de Planque (1968) HASL-234.
- Zombori et al. (1992) IAEA-314.
- Tyler (2008) J. Environ. Radioact. 99, 143-161.
- IAEA TRS-472 (2010) — Kd database.
"""
from __future__ import division, print_function, absolute_import

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .kernels import Detector, build_kernel
from .solvers import _tikhonov_weighted, _weighted, diff_matrix
from .criteria import gcv_curve, lcurve_corner, choose_alpha_discrepancy
from .transport import pulse_profile, exp_profile, DECAY_S


@dataclass
class DepthInversion:
    """Container for depth-inversion results.

    Attributes
    ----------
    z : np.ndarray (n_z,)
        Depth grid centres [m].
    dz : np.ndarray (n_z,)
        Depth cell widths [m].
    a : np.ndarray (n_z,)
        Reconstructed volumetric activity [Bq/m^3].
    a_std : np.ndarray (n_z,)
        Linearised 1-sigma uncertainty [Bq/m^3].
    areal : float
        Areal activity [Bq/m^2] = sum(a * dz).
    areal_std : float
        Uncertainty of areal activity.
    z_median : float
        Median depth [m] (depth below which 50% of activity resides).
    chi2 : float
        Weighted chi-squared of the fit.
    alpha : float
        Regularisation parameter used.
    method : str
        Description of method (e.g. 'tikhonov/gcv').
    info : dict
        Extra diagnostic information.
    """
    z: np.ndarray
    dz: np.ndarray
    a: np.ndarray
    a_std: np.ndarray
    areal: float
    areal_std: float
    z_median: float
    chi2: float
    alpha: float
    method: str
    info: dict = field(default_factory=dict)

    def report(self):
        return ("areal A={:.3e}+/-{:.1e} Bq/m2, "
                "z_median={:.1f} cm, chi2={:.1f}, "
                "alpha={:.2e}, method={}").format(
                    self.areal, self.areal_std,
                    self.z_median * 100, self.chi2,
                    self.alpha, self.method)


class DepthInverter:
    """End-to-end Fredholm depth inversion.

    Constructs the kernel K from gamma lines and detector heights,
    then provides non-parametric and parametric inversion methods.

    Parameters
    ----------
    lines : list of GammaLine
        Gamma-emission line descriptors.
    heights : float or list of float
        Detector height(s) above ground [m].
    z_max : float
        Maximum depth of the grid [m].  Default 1.0.
    n_z : int
        Number of depth cells.  Default 50.
    eff : float
        Full-sphere detection efficiency.  Default 1.0.
    buildup : callable or None
        Buildup factor function.  Default None (no buildup).
    angular_eff : callable or None
        Angular efficiency function.  Default None (isotropic).
    """

    def __init__(self, lines, heights=1.0, z_max=1.0, n_z=50, eff=1.0,
                 buildup=None, angular_eff=None):
        self.heights = [heights] if np.isscalar(heights) else list(heights)
        edges = np.linspace(0.0, z_max, n_z + 1)
        self.dz = np.diff(edges)
        self.z = 0.5 * (edges[:-1] + edges[1:])
        self.lines = list(lines)
        K = np.vstack([
            build_kernel(self.lines, Detector(h, eff), self.z,
                        buildup=buildup, angular_eff=angular_eff)
            for h in self.heights
        ])
        self.K = K * self.dz[None, :]

    @staticmethod
    def poisson_sigma(counts):
        """Poisson standard deviations (floor at 1)."""
        return np.sqrt(np.maximum(np.asarray(counts, float), 1.0))

    def _grid_alphas(self, A):
        lmax = np.linalg.svd(A, compute_uv=False)[0] ** 2
        return np.geomspace(lmax * 1e-10, lmax * 1e-1, 48)

    # ------ non-parametric inversion ------
    def fit(self, counts, sigma=None, alpha=None, criterion="gcv",
            smoothness=True, nonneg=True):
        """Non-parametric Tikhonov inversion with automatic alpha selection.

        Parameters
        ----------
        counts : array-like (m,)
            Measured peak count rates [1/s].
        sigma : array-like (m,) or None
            Data uncertainties.  Default: Poisson.
        alpha : float or None
            Fixed regularisation parameter.  If None, selected by *criterion*.
        criterion : {'gcv', 'lcurve', 'discrepancy'}
        smoothness : bool
            Use first-difference operator L (True) or identity (False).
        nonneg : bool
            Enforce non-negativity.

        Returns
        -------
        result : DepthInversion
        """
        counts = np.asarray(counts, float)
        if sigma is None:
            sigma = self.poisson_sigma(counts)
        A, b = _weighted(self.K, counts, sigma)
        L = diff_matrix(len(self.z)) if smoothness else np.eye(len(self.z))
        if alpha is None:
            alphas = self._grid_alphas(A)
            if criterion == "gcv":
                alpha = float(alphas[int(np.argmin(
                    gcv_curve(A, b, alphas, L=L)))])
            elif criterion == "lcurve":
                alpha, _ = lcurve_corner(A, b, alphas, L=L)
            elif criterion == "discrepancy":
                alpha = choose_alpha_discrepancy(A, b, L=L)
            else:
                raise ValueError("Unknown criterion: {}".format(criterion))
        x = _tikhonov_weighted(A, b, alpha, L=L, nonneg=nonneg)
        cov = np.linalg.inv(A.T @ A + alpha * (L.T @ L))
        a_std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        areal = float(np.sum(x * self.dz))
        areal_std = float(np.sqrt(max(self.dz @ cov @ self.dz, 0.0)))
        cdf = np.cumsum(np.clip(x, 0.0, None) * self.dz)
        z_med = (float(np.interp(0.5 * areal, cdf, self.z))
                 if areal > 0 else float(self.z[0]))
        r = A @ x - b
        return DepthInversion(
            self.z, self.dz, x, a_std, areal, areal_std,
            z_med, float(r @ r), alpha,
            "tikhonov/{}".format(criterion),
            {"L": "diff" if smoothness else "I"})

    # ------ parametric (transport-informed) inversion ------
    def fit_parametric(self, counts, sigma=None, family="pulse",
                       nuclide="Cs-137", D=1e-10, v=0.0, R=200.0,
                       p0=None):
        """Parametric inversion using a transport-chemistry profile.

        Fits (A0, t_eff) for family='pulse' or (A, lambda_relax)
        for family='exp'.  For a single line (e.g. Cs-137) this is
        strongly preferred over non-parametric inversion.

        Parameters
        ----------
        counts : array-like (m,)
        sigma : array-like (m,) or None
        family : {'pulse', 'exp'}
        nuclide : str
            Nuclide key (for decay constant).
        D : float
            Effective diffusion coefficient [m^2/s].
        v : float
            Advection velocity [m/s].
        R : float
            Retardation factor.
        p0 : list or None
            Initial parameter guesses (log-space).

        Returns
        -------
        out : dict
            Keys: 'params', 'profile', 'chi2', and family-specific keys.
        """
        counts = np.asarray(counts, float)
        if sigma is None:
            sigma = self.poisson_sigma(counts)
        lam = DECAY_S.get(nuclide, 0.0)

        if family == "pulse":
            def prof(p):
                return pulse_profile(self.z, np.exp(p[0]), np.exp(p[1]),
                                     D, v, R, lam)
            p0 = p0 or [np.log(1e5), np.log(30 * 3.156e7)]
            lo = [np.log(1e1), np.log(1e6)]
            hi = [np.log(1e9), np.log(200 * 3.156e7)]
        elif family == "exp":
            def prof(p):
                return exp_profile(self.z, np.exp(p[0]), np.exp(p[1]))
            p0 = p0 or [np.log(1e5), np.log(0.03)]
            lo = [np.log(1e1), np.log(1e-4)]
            hi = [np.log(1e9), np.log(2.0)]
        else:
            raise ValueError("Unknown family: {}".format(family))

        sol = least_squares(
            lambda p: (self.K @ prof(p) - counts) / sigma,
            p0, bounds=(lo, hi), x_scale="jac")

        out = {
            "params": sol.x,
            "profile": prof(sol.x),
            "chi2": float(sol.fun @ sol.fun),
        }
        if family == "pulse":
            out["A0"] = float(np.exp(sol.x[0]))
            out["t_years"] = float(np.exp(sol.x[1]) / 3.156e7)
        else:
            out["A"] = float(np.exp(sol.x[0]))
            out["lam_relax_m"] = float(np.exp(sol.x[1]))
        return out
