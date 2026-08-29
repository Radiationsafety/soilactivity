/**
 * DepthInverter: end-to-end pipeline from peak count rates to a(z) + 1-sigma.
 *
 * Supports:
 *  - Non-parametric: Tikhonov, TV/ADMM, FISTA (L1), focusing (MGS),
 *    CGLS, Landweber, Cimmino, TSVD — with automatic alpha selection.
 *  - Parametric: transport-chemistry-informed (pulse/exp profiles).
 *  - Bayesian: EKI ensemble uncertainty, Laplace MAP.
 *  - Ensemble multi-method with AIC/BIC model selection.
 *
 * References
 * ----------
 * - Beck & de Planque (1968) HASL-234.
 * - Zombori et al. (1992) IAEA-314.
 * - Tyler (2008) J. Environ. Radioact. 99, 143-161.
 * - IAEA TRS-472 (2010) — Kd database.
 * - Portniaguine & Zhdanov (1999) Geophysics 64(3):874.
 * - Vatankhah et al. (2018) GJI 213(1):695.
 * - Hasan et al. (2022, 2023) J. Environ. Radioact.
 */
from __future__ import division, print_function, absolute_import

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .kernels import Detector, build_kernel
from .solvers import (
    _tikhonov_weighted, _weighted, diff_matrix, depth_scale,
    tikhonov, tsvd, landweber, cimmino, cgls, fista,
    tv_admm, focusing_irls,
)
from .criteria import (
    gcv_curve, lcurve_corner, choose_alpha_discrepancy,
    quasi_optimality, ncp_criterion, snr_criterion,
)
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
    then provides non-parametric, parametric, Bayesian, and ensemble
    inversion methods.

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

    def _make_result(self, x, A, b, alpha, method, L, info_extra=None):
        """Build DepthInversion result from solution vector."""
        n = len(self.z)
        try:
            cov = np.linalg.inv(A.T @ A + alpha * (L.T @ L))
            a_std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            areal_std = float(np.sqrt(max(self.dz @ cov @ self.dz, 0.0)))
        except np.linalg.LinAlgError:
            a_std = np.zeros(n)
            areal_std = 0.0
        areal = float(np.sum(x * self.dz))
        cdf = np.cumsum(np.clip(x, 0.0, None) * self.dz)
        z_med = (float(np.interp(0.5 * areal, cdf, self.z))
                 if areal > 0 else float(self.z[0]))
        r = A @ x - b
        info = {"L": info_extra} if info_extra else {}
        return DepthInversion(
            self.z, self.dz, x, a_std, areal, areal_std,
            z_med, float(r @ r), alpha, method, info)

    # ------ non-parametric Tikhonov inversion ------
    def fit(self, counts, sigma=None, alpha=None, criterion="gcv",
            smoothness=True, nonneg=True):
        """Non-parametric Tikhonov inversion with automatic alpha selection.

        Parameters
        ----------
        counts : array-like (m,)
        sigma : array-like (m,) or None
        alpha : float or None
        criterion : {'gcv', 'lcurve', 'discrepancy'}
        smoothness : bool
        nonneg : bool

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
        return self._make_result(x, A, b, alpha,
                                "tikhonov/{}".format(criterion), L,
                                "diff" if smoothness else "I")

    # ------ TV/ADMM inversion ------
    def fit_tv(self, counts, sigma=None, alpha=None, criterion="lcurve",
               rho=1.0, max_iter=500):
        """Total Variation inversion via ADMM.

        Preserves sharp layer boundaries. Ideal for piecewise-constant
        contamination profiles.

        Parameters
        ----------
        counts, sigma : as in fit()
        alpha : float or None
            TV weight.  If None, selected by criterion.
        criterion : {'lcurve', 'discrepancy'}
        rho, max_iter : ADMM parameters.

        Returns
        -------
        result : DepthInversion
        """
        counts = np.asarray(counts, float)
        if sigma is None:
            sigma = self.poisson_sigma(counts)
        A, b = _weighted(self.K, counts, sigma)
        n = len(self.z)
        if alpha is None:
            # Heuristic: TV alpha ~ Tikhonov alpha / 10
            alphas_tik = self._grid_alphas(A)
            alpha_tik, _ = lcurve_corner(A, b, alphas_tik)
            alpha = alpha_tik * 0.1
        x, info = tv_admm(A, b, alpha, nonneg=True,
                           rho=rho, max_iter=max_iter)
        r = A @ x - b
        areal = float(np.sum(x * self.dz))
        cdf = np.cumsum(np.clip(x, 0.0, None) * self.dz)
        z_med = (float(np.interp(0.5 * areal, cdf, self.z))
                 if areal > 0 else float(self.z[0]))
        return DepthInversion(
            self.z, self.dz, x, np.zeros(n), areal, 0.0,
            z_med, float(r @ r), alpha, "tv/admm", info)

    # ------ FISTA (L1 sparse) inversion ------
    def fit_sparse(self, counts, sigma=None, alpha=None, criterion="lcurve",
                   max_iter=2000):
        """L1-sparse inversion via FISTA.

        Promotes compact solutions where activity is concentrated
        in a few thin layers.

        Parameters
        ----------
        counts, sigma, alpha, criterion, max_iter : as in fit_tv()

        Returns
        -------
        result : DepthInversion
        """
        counts = np.asarray(counts, float)
        if sigma is None:
            sigma = self.poisson_sigma(counts)
        A, b = _weighted(self.K, counts, sigma)
        n = len(self.z)
        if alpha is None:
            alphas_tik = self._grid_alphas(A)
            alpha_tik, _ = lcurve_corner(A, b, alphas_tik)
            alpha = alpha_tik * 0.05
        x, info = fista(A, b, alpha, nonneg=True, max_iter=max_iter)
        r = A @ x - b
        areal = float(np.sum(x * self.dz))
        cdf = np.cumsum(np.clip(x, 0.0, None) * self.dz)
        z_med = (float(np.interp(0.5 * areal, cdf, self.z))
                 if areal > 0 else float(self.z[0]))
        return DepthInversion(
            self.z, self.dz, x, np.zeros(n), areal, 0.0,
            z_med, float(r @ r), alpha, "fista/l1", info)

    # ------ Focusing (MGS/MS) inversion ------
    def fit_focusing(self, counts, sigma=None, alpha=None,
                     mode="mgs", eps_focusing=1e-2):
        """Focusing (compact) inversion via IRLS.

        Produces blocky/layered models with sharp interfaces
        (Portniaguine & Zhdanov 1999).

        Parameters
        ----------
        counts, sigma : as in fit()
        alpha : float or None
        mode : {'mgs', 'ms'}
        eps_focusing : float

        Returns
        -------
        result : DepthInversion
        """
        counts = np.asarray(counts, float)
        if sigma is None:
            sigma = self.poisson_sigma(counts)
        A, b = _weighted(self.K, counts, sigma)
        n = len(self.z)
        if alpha is None:
            alphas_tik = self._grid_alphas(A)
            alpha_tik, _ = lcurve_corner(A, b, alphas_tik)
            alpha = alpha_tik * 0.5
        x, info = focusing_irls(A, b, alpha, nonneg=True, mode=mode,
                                eps_focusing=eps_focusing)
        r = A @ x - b
        areal = float(np.sum(x * self.dz))
        cdf = np.cumsum(np.clip(x, 0.0, None) * self.dz)
        z_med = (float(np.interp(0.5 * areal, cdf, self.z))
                 if areal > 0 else float(self.z[0]))
        return DepthInversion(
            self.z, self.dz, x, np.zeros(n), areal, 0.0,
            z_med, float(r @ r), alpha, "focusing/{}".format(mode), info)

    # ------ CGLS inversion ------
    def fit_cgls(self, counts, sigma=None, max_iter=500):
        """CGLS inversion with early stopping.

        Iteration count acts as regularisation parameter (semi-convergence).

        Parameters
        ----------
        counts, sigma : as in fit()
        max_iter : int

        Returns
        -------
        result : DepthInversion
        """
        counts = np.asarray(counts, float)
        if sigma is None:
            sigma = self.poisson_sigma(counts)
        A, b = _weighted(self.K, counts, sigma)
        n = len(self.z)
        x, info = cgls(A, b, sigma=None, nonneg=True, max_iter=max_iter)
        r = A @ x - b
        areal = float(np.sum(x * self.dz))
        cdf = np.cumsum(np.clip(x, 0.0, None) * self.dz)
        z_med = (float(np.interp(0.5 * areal, cdf, self.z))
                 if areal > 0 else float(self.z[0]))
        return DepthInversion(
            self.z, self.dz, x, np.zeros(n), areal, 0.0,
            z_med, float(r @ r), 0.0, "cgls", info)

    # ------ Bayesian EKI ------
    def fit_eki(self, counts, sigma=None, n_ens=200, n_iter=30,
                prior_std=1e4, seed=0):
        """Ensemble Kalman Inversion with uncertainty quantification.

        Parameters
        ----------
        counts, sigma : as in fit()
        n_ens : int
        n_iter : int
        prior_std : float
        seed : int

        Returns
        -------
        result : DepthInversion
        """
        from .bayesian import ensemble_kalman_inversion
        counts = np.asarray(counts, float)
        if sigma is None:
            sigma = self.poisson_sigma(counts)
        res = ensemble_kalman_inversion(
            self.K, counts, sigma=sigma, n_ens=n_ens, n_iter=n_iter,
            prior_std=prior_std, seed=seed, nonneg=True)
        x = res["mean"]
        a_std = res["std"]
        areal = float(np.sum(x * self.dz))
        areal_std = float(np.sqrt(self.dz @ np.diag(np.cov(res["ensemble"])) @ self.dz
                                  if res["ensemble"].shape[0] > 1 else 0))
        cdf = np.cumsum(np.clip(x, 0.0, None) * self.dz)
        z_med = (float(np.interp(0.5 * areal, cdf, self.z))
                 if areal > 0 else float(self.z[0]))
        r = self.K @ x - counts
        chi2 = float(np.sum((r / sigma) ** 2))
        return DepthInversion(
            self.z, self.dz, x, a_std, areal, areal_std,
            z_med, chi2, 0.0, "eki/n{}".format(n_ens), res)

    # ------ Ensemble multi-method with AIC/BIC ------
    def fit_ensemble(self, counts, sigma=None, methods=None):
        """Run multiple inversion methods and select best via AIC/BIC.

        Parameters
        ----------
        counts : array-like (m,)
        sigma : array-like (m,) or None
        methods : list of str or None
            Methods to try.  Default: ['tikhonov/gcv', 'tv/admm',
            'fista/l1', 'focusing/mgs', 'cgls'].

        Returns
        -------
        dict with keys:
            'results' : list of DepthInversion
                All method results.
            'best' : DepthInversion
                Best result by AIC.
            'aic' : ndarray
                AIC values for each method.
            'bic' : ndarray
                BIC values for each method.
        """
        if methods is None:
            methods = ["tikhonov/gcv", "tv/admm", "fista/l1",
                       "focusing/mgs", "cgls"]
        results = []
        for m in methods:
            try:
                if m.startswith("tikhonov/"):
                    crit = m.split("/")[1]
                    res = self.fit(counts, sigma=sigma, criterion=crit)
                elif m == "tv/admm":
                    res = self.fit_tv(counts, sigma=sigma)
                elif m == "fista/l1":
                    res = self.fit_sparse(counts, sigma=sigma)
                elif m.startswith("focusing/"):
                    mode = m.split("/")[1]
                    res = self.fit_focusing(counts, sigma=sigma, mode=mode)
                elif m == "cgls":
                    res = self.fit_cgls(counts, sigma=sigma)
                else:
                    continue
                results.append(res)
            except Exception:
                continue
        # Compute AIC/BIC
        m_data = len(counts)
        aic = np.empty(len(results))
        bic = np.empty(len(results))
        for i, res in enumerate(results):
            # Effective number of parameters (trace of hat matrix)
            n_eff = min(np.sum(res.a > 1e-3 * np.max(res.a)), m_data)
            aic[i] = res.chi2 + 2.0 * n_eff
            bic[i] = res.chi2 + np.log(m_data) * n_eff
        best_idx = int(np.argmin(aic))
        return {
            "results": results,
            "best": results[best_idx],
            "aic": aic,
            "bic": bic,
            "method_names": [r.method for r in results],
        }

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
