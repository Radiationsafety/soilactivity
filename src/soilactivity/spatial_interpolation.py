"""
Spatial Interpolation Module for soilactivity.

Provides a unified interface for multiple 2-D interpolation backends,
auto-selection via cross-validation, sparse-result interpolation with
uncertainty quantification, and measurement-sensitivity analysis.

Example
-------
>>> import numpy as np
>>> from soilactivity.spatial_interpolation import Interpolator2D
>>> x = np.array([0, 1, 2, 3, 4], dtype=float)
>>> y = np.array([0, 1, 0, 1, 0.5], dtype=float)
>>> z = np.array([10, 20, 30, 40, 25], dtype=float)
>>> interp = Interpolator2D(method='rbf_tps')
>>> interp.fit(x, y, z)
>>> xi = np.linspace(0, 4, 11)
>>> yi = np.linspace(0, 1, 6)
>>> Z, XI, YI = interp.predict_grid(xi, yi)
>>> print(Z.shape)  # (6, 11)

Dependencies
------------
- numpy, scipy  (required)
- scikit-learn  (optional - for 'gp_*' methods)
- pykrige      (optional - for 'kriging' method)
"""

from __future__ import division, print_function, absolute_import

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from scipy.interpolate import RBFInterpolator, griddata
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from scipy.spatial import cKDTree
    _HAS_CKDTree = True
except ImportError:
    _HAS_CKDTree = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (  # noqa: F401
        RBF as SklearnRBF,
        Matern,
        ConstantKernel as C,
        WhiteKernel,
    )
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from pykrige.ok import OrdinaryKriging
    _HAS_PYKRIGE = True
except ImportError:
    _HAS_PYKRIGE = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Constants & public catalogue
# ---------------------------------------------------------------------------

AVAILABLE_METHODS = {
    'rbf_tps': 'RBF thin-plate spline (smooth, default)',
    'rbf_linear': 'RBF linear',
    'rbf_cubic': 'RBF cubic',
    'rbf_gaussian': 'RBF Gaussian',
    'nearest': 'Nearest neighbor (fast, no smoothing)',
    'linear_delaunay': 'Delaunay linear triangulation',
    'cubic_delaunay': 'Delaunay Clough-Tocher cubic (C1 smooth)',
    'idw': 'Inverse distance weighting (power=2, k=12)',
    'kriging': 'Ordinary Kriging (requires pykrige)',
    'gp_rbf': 'Gaussian Process RBF kernel (requires scikit-learn)',
    'gp_matern32': 'Gaussian Process Matern 3/2 kernel',
    'gp_matern52': 'Gaussian Process Matern 5/2 kernel',
    'barnes': 'Barnes successive corrections',
    'cressman': 'Cressman scheme',
}

_RBF_KERNEL_MAP = {
    'rbf_tps': 'thin_plate_spline',
    'rbf_linear': 'linear',
    'rbf_cubic': 'cubic',
    'rbf_gaussian': 'gaussian',
}

_GRIDDATA_METHOD_MAP = {
    'nearest': 'nearest',
    'linear_delaunay': 'linear',
    'cubic_delaunay': 'cubic',
}

_GP_KERNEL_MAP = {
    'gp_rbf': ('matern', 1.5),  # fallback map
    'gp_matern32': ('matern', 1.5),
    'gp_matern52': ('matern', 2.5),
}

_EPS = 1e-12  # coincidence threshold


# ===========================================================================
#  Standalone interpolation functions
# ===========================================================================

def idw_interpolate(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    xi: NDArray,
    yi: NDArray,
    power: int = 2,
    max_neighbors: int = 12,
) -> NDArray:
    """Inverse Distance Weighting interpolation.

    For each prediction point the *max_neighbors* nearest data points are
    used with weights ``w_i = 1 / d_i**power``.  If a prediction point
    coincides with a data point (distance < eps) the data value is returned
    directly.

    Parameters
    ----------
    x, y, z : array-like, shape (N,)
        Data coordinates and values.
    xi, yi : array-like
        Prediction coordinates (any shape, will be flattened internally).
    power : int, default 2
        Distance exponent.
    max_neighbors : int, default 12
        Maximum number of nearest neighbours to use.

    Returns
    -------
    zi : np.ndarray
        Interpolated values with the broadcasted shape of *xi*, *yi*.

    Example
    -------
    >>> x = np.array([0, 1, 2.0])
    >>> y = np.array([0, 1, 0.5])
    >>> z = np.array([10, 20, 15.0])
    >>> xi = np.array([0.5])
    >>> yi = np.array([0.25])
    >>> idw_interpolate(x, y, z, xi, yi)
    array(...)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()

    orig_shape_xi = np.shape(xi)
    xi_flat = np.asarray(xi, dtype=np.float64).ravel()
    yi_flat = np.asarray(yi, dtype=np.float64).ravel()

    data_pts = np.column_stack((x, y))
    query_pts = np.column_stack((xi_flat, yi_flat))

    if not _HAS_CKDTree:
        raise ImportError(
            "scipy.spatial.cKDTree is required for IDW but could not be "
            "imported.  Please install scipy."
        )

    tree = cKDTree(data_pts)
    k = min(max_neighbors, len(x))
    dists, idxs = tree.query(query_pts, k=k)

    # Handle scalar k (scipy >= 1.6 may return shape (M,) when k==1)
    if dists.ndim == 1:
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]

    zi_flat = np.empty(len(xi_flat), dtype=np.float64)
    for i in range(len(xi_flat)):
        d = dists[i]
        idx = idxs[i]
        # coincident check
        coincident = d < _EPS
        if np.any(coincident):
            zi_flat[i] = z[idx[coincident][0]]
            continue
        w = 1.0 / (d ** power)
        zi_flat[i] = np.sum(w * z[idx]) / np.sum(w)

    return zi_flat.reshape(orig_shape_xi)


def barnes_interpolate(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    xi: NDArray,
    yi: NDArray,
    kappa: float = 5.0,
    iterations: int = 2,
) -> NDArray:
    """Barnes successive-correction interpolation.

    First pass: ``w = exp(-r^2 / kappa^2)`` weighted average.  Subsequent
    passes correct the residual field with a reduced kappa.

    Parameters
    ----------
    x, y, z : array-like, shape (N,)
        Data coordinates and values.
    xi, yi : array-like
        Prediction coordinates.
    kappa : float, default 5.0
        Scale length (meters).  Each iteration halves this value.
    iterations : int, default 2
        Number of correction passes.

    Returns
    -------
    zi : np.ndarray
        Interpolated values.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()

    orig_shape = np.shape(xi)
    xi_flat = np.asarray(xi, dtype=np.float64).ravel()
    yi_flat = np.asarray(yi, dtype=np.float64).ravel()

    # Start with first guess: simple distance-weighted average
    zi_flat = np.zeros(len(xi_flat), dtype=np.float64)

    current_kappa = float(kappa)
    for _it in range(iterations):
        zi_new = np.zeros_like(zi_flat)
        weight_sum = np.zeros_like(zi_flat)
        for j in range(len(x)):
            dx = xi_flat - x[j]
            dy = yi_flat - y[j]
            r2 = dx * dx + dy * dy
            w = np.exp(-r2 / (current_kappa ** 2))
            if _it == 0:
                zi_new += w * z[j]
            else:
                zi_new += w * (z[j] - zi_flat)  # residual
            weight_sum += w
        # avoid division by zero
        weight_sum[weight_sum == 0] = 1.0
        if _it == 0:
            zi_flat = zi_new / weight_sum
        else:
            zi_flat = zi_flat + zi_new / weight_sum
        current_kappa = current_kappa / 2.0

    return zi_flat.reshape(orig_shape)


def cressman_interpolate(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    xi: NDArray,
    yi: NDArray,
    radius: float = 5.0,
) -> NDArray:
    """Cressman analysis interpolation.

    Similar to Barnes but the weight is uniformly 1 inside the influence
    *radius* and 0 outside (actually ``w = (R^2 - r^2)/(R^2 + r^2)``).
    Only one pass is performed.

    Parameters
    ----------
    x, y, z : array-like, shape (N,)
        Data coordinates and values.
    xi, yi : array-like
        Prediction coordinates.
    radius : float, default 5.0
        Influence radius (meters).

    Returns
    -------
    zi : np.ndarray
        Interpolated values.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()

    orig_shape = np.shape(xi)
    xi_flat = np.asarray(xi, dtype=np.float64).ravel()
    yi_flat = np.asarray(yi, dtype=np.float64).ravel()

    r2_radius = radius ** 2
    zi_flat = np.zeros(len(xi_flat), dtype=np.float64)
    weight_sum = np.zeros(len(xi_flat), dtype=np.float64)

    for j in range(len(x)):
        dx = xi_flat - x[j]
        dy = yi_flat - y[j]
        r2 = dx * dx + dy * dy
        inside = r2 < r2_radius
        w = np.zeros_like(r2)
        w[inside] = (r2_radius - r2[inside]) / (r2_radius + r2[inside])
        zi_flat += w * z[j]
        weight_sum += w

    weight_sum[weight_sum == 0] = 1.0
    zi_flat = zi_flat / weight_sum

    return zi_flat.reshape(orig_shape)


# ===========================================================================
#  Interpolator2D
# ===========================================================================

class Interpolator2D(object):
    """Unified 2-D interpolation wrapper.

    Supports multiple backends: RBF (scipy), Delaunay griddata,
    inverse-distance weighting, Barnes, Cressman, Ordinary Kriging (pykrige),
    and Gaussian-Process regression (scikit-learn).

    Parameters
    ----------
    method : str
        One of the keys in ``AVAILABLE_METHODS``.
    smoothing : float, default 0
        Smoothing parameter forwarded to RBF / Kriging / GP backends.
    **kwargs
        Extra keyword arguments forwarded to the underlying backend
        constructor.

    Example
    -------
    >>> interp = Interpolator2D(method='rbf_tps', smoothing=0.1)
    >>> interp.fit(x, y, z)
    >>> Z, XI, YI = interp.predict_grid(xi, yi)
    """

    def __init__(self, method: str = 'rbf_tps', smoothing: float = 0.0, **kwargs):
        if method not in AVAILABLE_METHODS:
            raise ValueError(
                "Unknown method '{}'.  Available: {}".format(
                    method, ', '.join(sorted(AVAILABLE_METHODS.keys()))
                )
            )
        self.method = method
        self.smoothing = smoothing
        self._backend_kwargs = kwargs

        # Storage set by fit()
        self._x: Optional[NDArray] = None
        self._y: Optional[NDArray] = None
        self._z: Optional[NDArray] = None
        self._model: Any = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def fit(self, x: NDArray, y: NDArray, z: NDArray) -> 'Interpolator2D':
        """Store data points and build the backend model if needed.

        Parameters
        ----------
        x, y, z : array-like, shape (N,)
        """
        if not _HAS_SCIPY and self.method in _RBF_KERNEL_MAP:
            raise ImportError(
                "scipy is required for method '{}' but could not be "
                "imported.".format(self.method)
            )

        self._x = np.asarray(x, dtype=np.float64).ravel()
        self._y = np.asarray(y, dtype=np.float64).ravel()
        self._z = np.asarray(z, dtype=np.float64).ravel()

        if len(self._x) != len(self._y) or len(self._x) != len(self._z):
            raise ValueError(
                "x, y, z must have the same length (got {}, {}, {}).".format(
                    len(self._x), len(self._y), len(self._z)
                )
            )

        self._build_backend()
        self._fitted = True
        return self

    def predict(self, xi: NDArray, yi: NDArray) -> NDArray:
        """Interpolate to arbitrary (possibly irregular) points.

        Parameters
        ----------
        xi, yi : array-like
            Can be 1-D arrays or meshgrid arrays.

        Returns
        -------
        zi : np.ndarray
            Interpolated values.
        """
        self._check_fitted()
        xi_arr = np.asarray(xi, dtype=np.float64)
        yi_arr = np.asarray(yi, dtype=np.float64)

        return self._dispatch_predict(xi_arr, yi_arr)

    def predict_grid(
        self, xi: NDArray, yi: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Interpolate to a regular 2-D grid.

        Parameters
        ----------
        xi, yi : 1-D array-like
            Grid coordinates along each axis.

        Returns
        -------
        Z : np.ndarray, shape (len(yi), len(xi))
        XI : np.ndarray, shape (len(yi), len(xi))
        YI : np.ndarray, shape (len(yi), len(xi))
        """
        self._check_fitted()
        xi_1d = np.asarray(xi, dtype=np.float64).ravel()
        yi_1d = np.asarray(yi, dtype=np.float64).ravel()
        XI, YI = np.meshgrid(xi_1d, yi_1d)
        Z = self._dispatch_predict(XI, YI)
        return Z, XI, YI

    def uncertainty(self, xi: NDArray, yi: NDArray) -> Optional[NDArray]:
        """Return prediction standard deviation if the backend supports it.

        Currently supported for: ``'gp_*'`` and ``'kriging'``.
        Returns ``None`` for all other methods.

        Parameters
        ----------
        xi, yi : array-like
            Prediction coordinates.

        Returns
        -------
        std : np.ndarray or None
        """
        self._check_fitted()
        xi_arr = np.asarray(xi, dtype=np.float64)
        yi_arr = np.asarray(yi, dtype=np.float64)

        if self.method.startswith('gp_'):
            if not _HAS_SKLEARN:
                return None
            pts = np.column_stack((xi_arr.ravel(), yi_arr.ravel()))
            _, std = self._model.predict(pts, return_std=True)
            return std.reshape(xi_arr.shape)

        if self.method == 'kriging':
            if not _HAS_PYKRIGE:
                return None
            pts_xi = xi_arr.ravel()
            pts_yi = yi_arr.ravel()
            _, ss = self._krige_execute_predict(pts_xi, pts_yi)
            return np.sqrt(np.maximum(ss, 0.0)).reshape(xi_arr.shape)

        return None

    def get_method_name(self) -> str:
        """Return the interpolation method name."""
        return self.method

    def get_info(self) -> Dict[str, Any]:
        """Return a dict summarising the current interpolator."""
        info: Dict[str, Any] = {
            'method': self.method,
            'description': AVAILABLE_METHODS.get(self.method, ''),
            'smoothing': self.smoothing,
            'fitted': self._fitted,
            'n_points': len(self._x) if self._x is not None else 0,
            'backend_kwargs': self._backend_kwargs,
            'supports_uncertainty': self.method.startswith('gp_') or self.method == 'kriging',
        }
        return info

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "The interpolator has not been fitted yet.  Call fit() first."
            )

    def _build_backend(self):
        """Construct the backend-specific model object."""
        x = self._x
        y = self._y
        z = self._z
        pts = np.column_stack((x, y))

        # --- RBF methods ---
        if self.method in _RBF_KERNEL_MAP:
            kernel_name = _RBF_KERNEL_MAP[self.method]
            self._model = RBFInterpolator(
                pts, z,
                kernel=kernel_name,
                smoothing=self.smoothing,
                **self._backend_kwargs
            )
            return

        # --- griddata methods (no pre-build needed) ---
        if self.method in _GRIDDATA_METHOD_MAP:
            self._model = _GRIDDATA_METHOD_MAP[self.method]
            return

        # --- IDW (no pre-build needed) ---
        if self.method == 'idw':
            self._model = 'idw'
            return

        # --- Kriging ---
        if self.method == 'kriging':
            if not _HAS_PYKRIGE:
                raise ImportError(
                    "Ordinary Kriging requires the 'pykrige' package.  "
                    "Install it with:  pip install pykrige"
                )
            nlags = self._backend_kwargs.get('nlags', 20)
            variogram_model = self._backend_kwargs.get('variogram_model', 'auto')
            # Store coords as 1-D for pykrige
            self._krige_x = x.copy()
            self._krige_y = y.copy()
            self._krige_z = z.copy()
            self._model = OrdinaryKriging(
                x, y, z,
                variogram_model=variogram_model,
                nlags=nlags,
                verbose=False,
                enable_plotting=False,
            )
            return

        # --- GP methods ---
        if self.method in _GP_KERNEL_MAP:
            if not _HAS_SKLEARN:
                raise ImportError(
                    "Gaussian Process methods require scikit-learn.  "
                    "Install it with:  pip install scikit-learn"
                )
            kernel_spec = _GP_KERNEL_MAP[self.method]
            if self.method == 'gp_rbf':
                kernel = C(1.0, (1e-5, 1e5)) * SklearnRBF(length_scale=1.0)
            else:
                nu = kernel_spec[1]
                kernel = C(1.0, (1e-5, 1e5)) * Matern(length_scale=1.0, nu=nu)
            if self.smoothing > 0:
                kernel = kernel + WhiteKernel(noise_level=self.smoothing, noise_level_bounds=(1e-10, 1e5))
            n_restarts = self._backend_kwargs.get('n_restarts_optimizer', 5)
            self._model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restarts,
                alpha=self._backend_kwargs.get('alpha', 1e-10),
                **{k: v for k, v in self._backend_kwargs.items()
                   if k not in ('n_restarts_optimizer', 'alpha')}
            )
            self._model.fit(pts, z)
            return

        # --- Barnes ---
        if self.method == 'barnes':
            self._model = 'barnes'
            return

        # --- Cressman ---
        if self.method == 'cressman':
            self._model = 'cressman'
            return

    def _dispatch_predict(self, xi: NDArray, yi: NDArray) -> NDArray:
        """Route prediction to the correct backend."""
        method = self.method

        # RBF
        if method in _RBF_KERNEL_MAP:
            pts = np.column_stack((xi.ravel(), yi.ravel()))
            result = self._model(pts)
            return result.reshape(xi.shape)

        # griddata
        if method in _GRIDDATA_METHOD_MAP:
            gd_method = self._model
            pts_query = np.column_stack((xi.ravel(), yi.ravel()))
            result = griddata(
                np.column_stack((self._x, self._y)),
                self._z,
                pts_query,
                method=gd_method,
            )
            return result.reshape(xi.shape)

        # IDW
        if method == 'idw':
            power = self._backend_kwargs.get('power', 2)
            max_neighbors = self._backend_kwargs.get('max_neighbors', 12)
            return idw_interpolate(
                self._x, self._y, self._z, xi, yi,
                power=power, max_neighbors=max_neighbors,
            )

        # Kriging
        if method == 'kriging':
            z_pred, _ = self._krige_execute_predict(xi.ravel(), yi.ravel())
            return z_pred.reshape(xi.shape)

        # GP
        if method in _GP_KERNEL_MAP:
            pts = np.column_stack((xi.ravel(), yi.ravel()))
            z_pred, _ = self._model.predict(pts, return_std=True)
            return z_pred.reshape(xi.shape)

        # Barnes
        if method == 'barnes':
            kappa = self._backend_kwargs.get('kappa', 5.0)
            iterations = self._backend_kwargs.get('iterations', 2)
            return barnes_interpolate(
                self._x, self._y, self._z, xi, yi,
                kappa=kappa, iterations=iterations,
            )

        # Cressman
        if method == 'cressman':
            radius = self._backend_kwargs.get('radius', 5.0)
            return cressman_interpolate(
                self._x, self._y, self._z, xi, yi,
                radius=radius,
            )

        raise RuntimeError("Unhandled method: '{}'".format(method))

    def _krige_execute_predict(self, xi_flat: NDArray, yi_flat: NDArray):
        """Run kriging prediction, returning (z, sigma^2)."""
        z_pred, ss = self._model.execute('points', xi_flat, yi_flat)
        return np.asarray(z_pred, dtype=np.float64), np.asarray(ss, dtype=np.float64)


# ===========================================================================
#  SparseResultInterpolator
# ===========================================================================

@dataclass
class SparseResult:
    """Container for sparse-result interpolation output.

    Attributes
    ----------
    interpolated : np.ndarray, shape (ny, nx)
        Interpolated values on the dense grid.
    uncertainty : np.ndarray or None
        Prediction standard deviation (same shape) or ``None``.
    confidence_mask : np.ndarray, bool, shape (ny, nx)
        ``True`` where relative uncertainty is below the threshold.
    method_used : str
        Interpolation method that was used.
    n_input_points : int
        Number of sparse input points.
    coverage : float
        Fraction of grid cells with low uncertainty (0–1).
    """
    interpolated: NDArray
    uncertainty: Optional[NDArray] = None
    confidence_mask: NDArray = field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    method_used: str = ''
    n_input_points: int = 0
    coverage: float = 0.0


class SparseResultInterpolator(object):
    """Interpolate from a sparse set of reconstructed points to a denser grid.

    Designed for cases where a Fredholm or other inverse reconstruction
    provides results at only a few locations.  Gaussian-process methods
    are recommended for built-in uncertainty quantification.

    Parameters
    ----------
    method : str, default 'gp_rbf'
        Interpolation method (see ``AVAILABLE_METHODS``).
    uncertainty_threshold : float, default 0.3
        Relative uncertainty threshold.  Grid cells with
        ``std / max(|z|) < threshold`` are flagged as confident.
    **kwargs
        Extra arguments forwarded to ``Interpolator2D``.

    Example
    -------
    >>> spi = SparseResultInterpolator(method='gp_rbf')
    >>> spi.fit_sparse(pts, values)
    >>> result = spi.interpolate_to_grid(0, 100, 0, 50, 50, 25)
    >>> print(result.coverage)
    """

    def __init__(
        self,
        method: str = 'gp_rbf',
        uncertainty_threshold: float = 0.3,
        **kwargs,
    ):
        self.method = method
        self.uncertainty_threshold = uncertainty_threshold
        self._kwargs = kwargs
        self._interpolator: Optional[Interpolator2D] = None
        self._input_uncertainty: Optional[NDArray] = None

    def fit_sparse(
        self,
        reconstructed_points: NDArray,
        values: NDArray,
        uncertainty: Optional[NDArray] = None,
    ) -> 'SparseResultInterpolator':
        """Fit the interpolator to sparse reconstructed data.

        Parameters
        ----------
        reconstructed_points : array-like, shape (N, 2)
            (x, y) coordinates of the reconstructed points.
        values : array-like, shape (N,)
            Reconstructed values at those points.
        uncertainty : array-like, shape (N,) or None
            Optional per-point uncertainties (used as GP noise prior).
        """
        pts = np.asarray(reconstructed_points, dtype=np.float64)
        vals = np.asarray(values, dtype=np.float64).ravel()

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                "reconstructed_points must have shape (N, 2), got {}".format(
                    pts.shape
                )
            )

        # If uncertainty is provided and using GP, pass as alpha
        extra_kwargs = dict(self._kwargs)
        if uncertainty is not None:
            self._input_uncertainty = np.asarray(uncertainty, dtype=np.float64).ravel()
            if self.method.startswith('gp_'):
                extra_kwargs['alpha'] = (self._input_uncertainty ** 2).clip(min=1e-12)
        else:
            self._input_uncertainty = None

        self._interpolator = Interpolator2D(
            method=self.method, **extra_kwargs
        )
        self._interpolator.fit(pts[:, 0], pts[:, 1], vals)
        return self

    def interpolate_to_grid(
        self,
        xmin: float, xmax: float,
        ymin: float, ymax: float,
        nx: int, ny: int,
    ) -> SparseResult:
        """Interpolate onto a regular grid.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : float
            Grid extents (meters).
        nx, ny : int
            Number of grid cells along each axis.

        Returns
        -------
        result : SparseResult
        """
        if self._interpolator is None:
            raise RuntimeError("Call fit_sparse() before interpolate_to_grid().")

        xi = np.linspace(xmin, xmax, nx)
        yi = np.linspace(ymin, ymax, ny)

        Z, XI, YI = self._interpolator.predict_grid(xi, yi)

        std = self._interpolator.uncertainty(XI, YI)

        # confidence mask
        if std is not None:
            z_abs_max = np.max(np.abs(Z))
            if z_abs_max > 0:
                rel_unc = std / z_abs_max
            else:
                rel_unc = np.zeros_like(std)
            confidence_mask = rel_unc < self.uncertainty_threshold
            coverage = float(np.mean(confidence_mask))
        else:
            confidence_mask = np.ones(Z.shape, dtype=bool)
            coverage = 1.0
            std = None

        n_pts = 0
        if self._interpolator._x is not None:
            n_pts = len(self._interpolator._x)

        return SparseResult(
            interpolated=Z,
            uncertainty=std,
            confidence_mask=confidence_mask,
            method_used=self.method,
            n_input_points=n_pts,
            coverage=coverage,
        )


# ===========================================================================
#  MeasurementSensitivityAnalyzer
# ===========================================================================

class MeasurementSensitivityAnalyzer(object):
    """Analyse which measurement points most influence an interpolation.

    Provides leave-one-out and perturbation-based sensitivity measures,
    influence maps, and rankings.  Inspired by *bssunfold*'s
    ``unfold_interpret`` and the *pyoptexplain* pattern.

    Example
    -------
    >>> msa = MeasurementSensitivityAnalyzer()
    >>> msa.fit(x, y, z, method='rbf_tps')
    >>> results = msa.sensitivity_leave_one_out()
    >>> ranking = msa.ranking()
    >>> critical = msa.critical_points(threshold_percentile=90)
    """

    def __init__(self):
        self._x: Optional[NDArray] = None
        self._y: Optional[NDArray] = None
        self._z: Optional[NDArray] = None
        self._method: str = 'rbf_tps'
        self._kwargs: Dict[str, Any] = {}
        self._base_zi: Optional[NDArray] = None
        self._eval_xi: Optional[NDArray] = None
        self._eval_yi: Optional[NDArray] = None

    def fit(
        self,
        x: NDArray,
        y: NDArray,
        z: NDArray,
        method: str = 'rbf_tps',
        **kwargs,
    ) -> 'MeasurementSensitivityAnalyzer':
        """Fit the base interpolation and pre-compute the reference field.

        Parameters
        ----------
        x, y, z : array-like, shape (N,)
        method : str
        Interpolation method.
        **kwargs
            Forwarded to ``Interpolator2D``.
        """
        self._x = np.asarray(x, dtype=np.float64).ravel()
        self._y = np.asarray(y, dtype=np.float64).ravel()
        self._z = np.asarray(z, dtype=np.float64).ravel()
        self._method = method
        self._kwargs = kwargs

        # Build reference interpolation on the data convex hull
        interp = Interpolator2D(method=method, **kwargs)
        interp.fit(self._x, self._y, self._z)

        # Evaluation grid: slightly expanded bounding box
        margin_frac = 0.05
        x_range = self._x.max() - self._x.min()
        y_range = self._y.max() - self._y.min()
        xmin = self._x.min() - margin_frac * x_range
        xmax = self._x.max() + margin_frac * x_range
        ymin = self._y.min() - margin_frac * y_range
        ymax = self._y.max() + margin_frac * y_range

        n_eval = max(30, min(80, len(self._x) * 3))
        xi = np.linspace(xmin, xmax, n_eval)
        yi = np.linspace(ymin, ymax, n_eval)
        self._eval_xi, self._eval_yi = xi, yi

        self._base_zi, _, _ = interp.predict_grid(xi, yi)
        return self

    def sensitivity_leave_one_out(self) -> List[Dict[str, Any]]:
        """Leave-one-out sensitivity analysis.

        For each data point *i* the point is removed, the interpolation
        is re-built, and the change at all evaluation grid locations is
        recorded.

        Returns
        -------
        results : list of dict
            Each dict contains keys: *point_index*, *x*, *y*, *z*,
            *max_influence*, *mean_influence*, *influence_area_km2*.
        """
        self._check_ready()
        n = len(self._x)
        results = []
        grid_cell_area = self._grid_cell_area()

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            x_sub = self._x[mask]
            y_sub = self._y[mask]
            z_sub = self._z[mask]

            interp = Interpolator2D(method=self._method, **self._kwargs)
            interp.fit(x_sub, y_sub, z_sub)
            zi_sub, _, _ = interp.predict_grid(self._eval_xi, self._eval_yi)

            diff = np.abs(zi_sub - self._base_zi)
            max_change = float(np.nanmax(diff))
            mean_change = float(np.nanmean(diff))

            # "influence area" = fraction of cells above 5 % of max change
            threshold = 0.05 * max_change if max_change > 0 else 0.0
            significant = diff > threshold
            n_significant = int(np.sum(significant))
            influence_area = n_significant * grid_cell_area / 1e6  # m^2 -> km^2

            results.append({
                'point_index': int(i),
                'x': float(self._x[i]),
                'y': float(self._y[i]),
                'z': float(self._z[i]),
                'max_influence': max_change,
                'mean_influence': mean_change,
                'influence_area_km2': influence_area,
            })

        return results

    def sensitivity_perturbation(self, delta_frac: float = 0.1) -> List[Dict[str, Any]]:
        """Perturbation-based sensitivity analysis.

        Each data point's *z*-value is perturbed by ``delta_frac * z[i]``
        (minimum absolute perturbation of ``delta_frac * std(z)``), and
        the resulting change in the interpolated field is measured.

        Parameters
        ----------
        delta_frac : float, default 0.1
            Fraction of the value (or std) to perturb by.

        Returns
        -------
        results : list of dict
            Same schema as ``sensitivity_leave_one_out``.
        """
        self._check_ready()
        n = len(self._x)
        z_std = float(np.std(self._z)) if n > 1 else 1.0
        results = []
        grid_cell_area = self._grid_cell_area()

        for i in range(n):
            z_perturbed = self._z.copy()
            delta = delta_frac * max(abs(self._z[i]), z_std)
            z_perturbed[i] += delta

            interp = Interpolator2D(method=self._method, **self._kwargs)
            interp.fit(self._x, self._y, z_perturbed)
            zi_pert, _, _ = interp.predict_grid(self._eval_xi, self._eval_yi)

            diff = np.abs(zi_pert - self._base_zi)
            max_change = float(np.nanmax(diff))
            mean_change = float(np.nanmean(diff))

            threshold = 0.05 * max_change if max_change > 0 else 0.0
            significant = diff > threshold
            n_significant = int(np.sum(significant))
            influence_area = n_significant * grid_cell_area / 1e6

            results.append({
                'point_index': int(i),
                'x': float(self._x[i]),
                'y': float(self._y[i]),
                'z': float(self._z[i]),
                'max_influence': max_change,
                'mean_influence': mean_change,
                'influence_area_km2': influence_area,
            })

        return results

    def influence_map(self, xi: NDArray, yi: NDArray) -> NDArray:
        """Compute a grid showing total influence of each measurement.

        Parameters
        ----------
        xi, yi : 1-D array-like
            Grid axes.

        Returns
        -------
        influence : np.ndarray, shape (len(yi), len(xi))
            Sum of absolute changes caused by each measurement point.
        """
        self._check_ready()
        loo = self.sensitivity_leave_one_out()

        xi_1d = np.asarray(xi, dtype=np.float64).ravel()
        yi_1d = np.asarray(yi, dtype=np.float64).ravel()
        n_pts = len(self._x)
        influence = np.zeros((len(yi_1d), len(xi_1d)), dtype=np.float64)

        for entry in loo:
            i = entry['point_index']
            mask = np.ones(n_pts, dtype=bool)
            mask[i] = False

            interp = Interpolator2D(method=self._method, **self._kwargs)
            interp.fit(self._x[mask], self._y[mask], self._z[mask])
            zi_sub, _, _ = interp.predict_grid(xi_1d, yi_1d)
            influence += np.abs(zi_sub - self._base_zi)

        return influence

    def plot_influence(self, xi: NDArray, yi: NDArray):
        """Plot the influence map.

        Parameters
        ----------
        xi, yi : 1-D array-like
            Grid axes.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
            Only if matplotlib is available; otherwise returns (None, None).
        """
        if not _HAS_MATPLOTLIB:
            warnings.warn(
                "matplotlib is not installed; cannot plot.  "
                "Returning (None, None)."
            )
            return None, None

        influence = self.influence_map(xi, yi)
        fig, ax = plt.subplots(figsize=(8, 6))
        XI, YI = np.meshgrid(
            np.asarray(xi, dtype=np.float64).ravel(),
            np.asarray(yi, dtype=np.float64).ravel(),
        )
        pcm = ax.pcolormesh(XI, YI, influence, shading='auto', cmap='magma')
        fig.colorbar(pcm, ax=ax, label='Total influence')
        if self._x is not None:
            ax.scatter(self._x, self._y, c='white', edgecolors='k', s=30, zorder=5)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Measurement influence map')
        ax.set_aspect('equal')
        fig.tight_layout()
        return fig, ax

    def ranking(self) -> List[Dict[str, Any]]:
        """Return leave-one-out results sorted by ``max_influence`` descending.

        Returns
        -------
        ranked : list of dict
        """
        loo = self.sensitivity_leave_one_out()
        return sorted(loo, key=lambda d: d['max_influence'], reverse=True)

    def critical_points(self, threshold_percentile: float = 90) -> List[Dict[str, Any]]:
        """Return points whose influence exceeds a percentile threshold.

        Parameters
        ----------
        threshold_percentile : float, default 90
            Percentile (0–100) of ``max_influence`` above which a point
            is considered critical.

        Returns
        -------
        critical : list of dict
        """
        ranked = self.ranking()
        if not ranked:
            return []
        influences = [d['max_influence'] for d in ranked]
        cutoff = float(np.percentile(influences, threshold_percentile))
        return [d for d in ranked if d['max_influence'] >= cutoff]

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _check_ready(self):
        if self._x is None or self._base_zi is None:
            raise RuntimeError("Call fit() before running sensitivity analysis.")

    def _grid_cell_area(self) -> float:
        """Area (m^2) of one evaluation-grid cell."""
        if self._eval_xi is None or self._eval_yi is None:
            return 1.0
        dx = self._eval_xi[1] - self._eval_xi[0] if len(self._eval_xi) > 1 else 1.0
        dy = self._eval_yi[1] - self._eval_yi[0] if len(self._eval_yi) > 1 else 1.0
        return abs(dx * dy)


# ===========================================================================
#  InterpolationAutoSelector
# ===========================================================================

class InterpolationAutoSelector(object):
    """Automatically select the best interpolation method via cross-validation.

    Parameters
    ----------
    candidates : list of str, default ['rbf_tps', 'linear_delaunay', 'idw', 'kriging', 'gp_rbf']
        Methods to evaluate.
    cv_folds : int, default 5
        Number of cross-validation folds.  If *N < 30*, leave-one-out
        is used instead.
    metrics : list of str, default ['rmse', 'mae', 'r2']
        Metrics to compute per fold.
    min_points : int, default 10
        Minimum number of data points required for auto-selection.
    """

    def __init__(
        self,
        candidates: Optional[List[str]] = None,
        cv_folds: int = 5,
        metrics: Optional[List[str]] = None,
        min_points: int = 10,
    ):
        if candidates is None:
            candidates = ['rbf_tps', 'linear_delaunay', 'idw', 'kriging', 'gp_rbf']
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2']
        self.candidates = candidates
        self.cv_folds = cv_folds
        self.metrics = metrics
        self.min_points = min_points
        self._results: List[Dict[str, Any]] = []
        self._x: Optional[NDArray] = None
        self._y: Optional[NDArray] = None
        self._z: Optional[NDArray] = None

    def fit(self, x: NDArray, y: NDArray, z: NDArray) -> 'InterpolationAutoSelector':
        """Run cross-validation for all candidate methods.

        Parameters
        ----------
        x, y, z : array-like, shape (N,)
        """
        self._x = np.asarray(x, dtype=np.float64).ravel()
        self._y = np.asarray(y, dtype=np.float64).ravel()
        self._z = np.asarray(z, dtype=np.float64).ravel()

        n = len(self._x)
        if n < self.min_points:
            raise ValueError(
                "Need at least {} data points for auto-selection, got {}.".format(
                    self.min_points, n
                )
            )

        # Build fold indices
        if n < 30:
            folds = self._leave_one_out_folds(n)
            cv_label = 'LOO'
        else:
            folds = self._kfold_indices(n, self.cv_folds)
            cv_label = '{}-fold'.format(self.cv_folds)

        self._results = []
        for method in self.candidates:
            result = self._evaluate_method(method, folds, cv_label)
            self._results.append(result)

        return self

    def select(self) -> Dict[str, Any]:
        """Return the best method and its scores.

        Selection criterion: lowest RMSE.  Ties broken by highest R^2.

        Returns
        -------
        info : dict
            Keys: *best_method*, *best_score*, *results*.
        """
        if not self._results:
            raise RuntimeError("Call fit() before select().")

        # Filter out methods that failed
        valid = [r for r in self._results if 'error' not in r]
        if not valid:
            return {
                'best_method': None,
                'best_score': float('inf'),
                'results': self._results,
            }

        # Best = lowest RMSE, then highest R2
        best = min(valid, key=lambda r: (r['rmse'], -r['r2']))
        return {
            'best_method': best['method'],
            'best_score': best['rmse'],
            'results': self._results,
        }

    def get_ranking(self) -> List[Dict[str, Any]]:
        """Return all results sorted by RMSE ascending.

        Returns
        -------
        ranking : list of dict
        """
        if not self._results:
            raise RuntimeError("Call fit() before get_ranking().")
        valid = [r for r in self._results if 'error' not in r]
        failed = [r for r in self._results if 'error' in r]
        return sorted(valid, key=lambda r: r['rmse']) + failed

    def plot_comparison(self):
        """Bar chart comparing methods by RMSE.

        Returns
        -------
        fig, ax : matplotlib objects, or (None, None) if matplotlib is absent.
        """
        if not _HAS_MATPLOTLIB:
            warnings.warn("matplotlib is not installed; returning (None, None).")
            return None, None
        ranking = self.get_ranking()
        methods = [r['method'] for r in ranking if 'error' not in r]
        rmses = [r['rmse'] for r in ranking if 'error' not in r]

        fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.2), 5))
        bars = ax.barh(methods, rmses, color='steelblue', edgecolor='k')
        ax.set_xlabel('RMSE')
        ax.set_title('Cross-validated interpolation method comparison')
        ax.invert_yaxis()
        fig.tight_layout()
        return fig, ax

    def get_recommendation(self) -> str:
        """Return a human-readable recommendation string.

        Returns
        -------
        recommendation : str
        """
        sel = self.select()
        best = sel['best_method']
        if best is None:
            return (
                "None of the candidate methods could be evaluated.  "
                "Check that required dependencies are installed."
            )
        best_info = next(r for r in sel['results'] if r['method'] == best)
        lines = [
            "Recommended method: {}".format(best),
            "  Description: {}".format(AVAILABLE_METHODS.get(best, '')),
            "  RMSE: {:.6g}".format(best_info['rmse']),
            "  MAE:  {:.6g}".format(best_info['mae']),
            "  R^2:  {:.6g}".format(best_info['r2']),
            "  Time: {:.3f} s".format(best_info['time_s']),
        ]
        if best_info.get('notes'):
            lines.append("  Notes: {}".format(best_info['notes']))
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    #  Internal CV helpers
    # ------------------------------------------------------------------

    def _evaluate_method(
        self, method: str, folds: List[Tuple[NDArray, NDArray]], cv_label: str
    ) -> Dict[str, Any]:
        """Run CV for a single method, return result dict."""
        t0 = time.time()

        # Check dependency availability
        try:
            _check_method_available(method)
        except ImportError as exc:
            return {
                'method': method,
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2': float('-inf'),
                'time_s': 0.0,
                'notes': str(exc),
                'error': True,
            }

        all_true = []
        all_pred = []

        for train_idx, test_idx in folds:
            x_train = self._x[train_idx]
            y_train = self._y[train_idx]
            z_train = self._z[train_idx]
            x_test = self._x[test_idx]
            y_test = self._y[test_idx]
            z_test = self._z[test_idx]

            try:
                interp = Interpolator2D(method=method)
                interp.fit(x_train, y_train, z_train)
                z_pred = interp.predict(x_test, y_test)
                # Handle NaN from griddata extrapolation
                valid = ~np.isnan(z_pred)
                if not np.any(valid):
                    continue
                all_true.append(z_test[valid])
                all_pred.append(z_pred[valid])
            except Exception as exc:
                return {
                    'method': method,
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'r2': float('-inf'),
                    'time_s': time.time() - t0,
                    'notes': 'CV error: {}'.format(exc),
                    'error': True,
                }

        elapsed = time.time() - t0

        if not all_true:
            return {
                'method': method,
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2': float('-inf'),
                'time_s': elapsed,
                'notes': 'No valid predictions in CV',
                'error': True,
            }

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        notes = 'Evaluated via {} CV ({} folds)'.format(cv_label, len(folds))

        return {
            'method': method,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'time_s': elapsed,
            'notes': notes,
        }

    @staticmethod
    def _leave_one_out_folds(n: int) -> List[Tuple[NDArray, NDArray]]:
        """Generate LOO fold index pairs."""
        all_idx = np.arange(n)
        folds = []
        for i in range(n):
            test_idx = np.array([i], dtype=int)
            train_idx = np.delete(all_idx, i)
            folds.append((train_idx, test_idx))
        return folds

    @staticmethod
    def _kfold_indices(n: int, k: int) -> List[Tuple[NDArray, NDArray]]:
        """Simple k-fold split (no shuffling for reproducibility)."""
        indices = np.arange(n)
        fold_sizes = np.full(k, n // k, dtype=int)
        fold_sizes[:n % k] += 1
        folds = []
        current = 0
        for fold_size in fold_sizes:
            test_idx = indices[current:current + fold_size]
            train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
            folds.append((train_idx, test_idx))
            current += fold_size
        return folds


def _check_method_available(method: str):
    """Raise ImportError if the required package for *method* is missing."""
    if method in _RBF_KERNEL_MAP or method in _GRIDDATA_METHOD_MAP:
        if not _HAS_SCIPY:
            raise ImportError(
                "scipy is required for method '{}'".format(method)
            )
    if method == 'idw':
        if not _HAS_CKDTree:
            raise ImportError(
                "scipy.spatial.cKDTree is required for IDW"
            )
    if method == 'kriging':
        if not _HAS_PYKRIGE:
            raise ImportError(
                "pykrige is required for method 'kriging'.  "
                "Install with: pip install pykrige"
            )
    if method in _GP_KERNEL_MAP:
        if not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for method '{}'.  "
                "Install with: pip install scikit-learn".format(method)
            )
    # barnes and cressman use only numpy — always available


# ===========================================================================
#  Module-level exports
# ===========================================================================

__all__ = [
    'Interpolator2D',
    'InterpolationAutoSelector',
    'SparseResultInterpolator',
    'MeasurementSensitivityAnalyzer',
    'idw_interpolate',
    'barnes_interpolate',
    'cressman_interpolate',
    'AVAILABLE_METHODS',
]
