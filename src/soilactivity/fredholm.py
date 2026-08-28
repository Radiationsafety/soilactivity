import numpy as np
from typing import Optional, Tuple

__all__ = [
    "build_fredholm_matrix",
    "build_fredholm_matrix_no_vis",
    "raster_coords",
    "raster_to_vector",
    "vector_to_raster",
    "solve_fredholm_tikhonov",
    "solve_fredholm_tikhonov_nn",
]


def raster_coords(
    nx: int, ny: int, cell_size: float, origin: Tuple[float, float] = (0.0, 0.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2D raster cell centre coordinates.

    Parameters
    ----------
    nx, ny : int
        Number of cells along X and Y.
    cell_size : float
        Cell width (hx = hy) [m].
    origin : (x0, y0)
        Bottom-left corner of the raster [m].

    Returns
    -------
    cx, cy : np.ndarray (ny, nx)
        Centre coordinates of each cell.
    """
    x_edges = np.linspace(origin[0], origin[0] + nx * cell_size, nx + 1)
    y_edges = np.linspace(origin[1], origin[1] + ny * cell_size, ny + 1)
    cx = 0.5 * (x_edges[:-1] + x_edges[1:])  # (nx,)
    cy = 0.5 * (y_edges[:-1] + y_edges[1:])  # (ny,)
    CX, CY = np.meshgrid(cx, cy, indexing='xy')  # (ny, nx)
    return CX, CY


def raster_to_vector(arr: np.ndarray) -> np.ndarray:
    """Convert 2D raster (ny, nx) to column vector (ny*nx,). C-order (row-major)."""
    return arr.ravel(order='C')


def vector_to_raster(vec: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Convert column vector back to 2D raster (ny, nx)."""
    return vec.reshape((ny, nx), order='C')


def build_fredholm_matrix_no_vis(
    nx: int,
    ny: int,
    cell_size: float,
    height_m: float,
    kerma_constant: float,
    normalizing_factor: float,
    aperture_radius_m: Optional[float] = None,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Build Fredholm matrix F (barrier-free geometry, no visibility).

    Constructs the SLAE matrix for the discretised Fredholm equation:

        F . A = P

    where F_ij = W * K_gamma * hx * hy * vis_ij
                         / (L^2 + (xi - xj)^2 + (yi - yj)^2)

    Without visibility (vis=1 everywhere) and with aperture constraint.

    Parameters
    ----------
    nx, ny : int
        Raster dimensions (cells).
    cell_size : float
        Cell size hx = hy [m].
    height_m : float
        Detector height above surface L [m].
    kerma_constant : float
        K_gamma [aGy m^2 s^-1 Bq^-1].
    normalizing_factor : float
        W [P_unit / aGy].
    aperture_radius_m : float or None
        If set, only cells within this radius contribute (detector
        angular aperture constraint). Simulates a collimated detector.
    origin : (x0, y0)
        Raster origin.

    Returns
    -------
    np.ndarray, shape (ny*nx, ny*nx)
        Fredholm matrix F.
    """
    CX, CY = raster_coords(nx, ny, cell_size, origin)
    cx = raster_to_vector(CX)  # (N,)
    cy = raster_to_vector(CY)  # (N,)
    N = nx * ny

    hx = cell_size
    hy = cell_size
    prefactor = normalizing_factor * kerma_constant * hx * hy

    F = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        dx = cx[i] - cx
        dy = cy[i] - cy
        dist_sq = height_m ** 2 + dx ** 2 + dy ** 2

        if aperture_radius_m is not None:
            r_horiz_sq = dx ** 2 + dy ** 2
            mask = r_horiz_sq <= aperture_radius_m ** 2
            F[i, mask] = prefactor / dist_sq[mask]
        else:
            F[i, :] = prefactor / dist_sq

    return F


def build_fredholm_matrix(
    nx: int,
    ny: int,
    cell_size: float,
    height_m: float,
    kerma_constant: float,
    normalizing_factor: float,
    vis_matrix: Optional[np.ndarray] = None,
    aperture_radius_m: Optional[float] = None,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Build Fredholm matrix F with optional visibility (barrier geometry).

    Same as build_fredholm_matrix_no_vis but multiplies the kernel by
    the visibility matrix (Hadamard product):

        F = (Q * Vis)  where  * is element-wise

    If vis_matrix is None, equivalent to barrier-free geometry.

    Parameters
    ----------
    nx, ny : int
        Raster dimensions.
    cell_size : float
        Cell size [m].
    height_m : float
        Detector height L [m].
    kerma_constant : float
        K_gamma [aGy m^2 s^-1 Bq^-1].
    normalizing_factor : float
        W [P_unit / aGy].
    vis_matrix : np.ndarray or None
        Visibility matrix (N, N) with values 0 or 1.
        If None, all ones (no barriers).
    aperture_radius_m : float or None
        Detector aperture radius [m].
    origin : (x0, y0)
        Raster origin.

    Returns
    -------
    np.ndarray, shape (N, N) where N = nx * ny
        Fredholm matrix F = Q * Vis.
    """
    F = build_fredholm_matrix_no_vis(
        nx, ny, cell_size, height_m, kerma_constant,
        normalizing_factor, aperture_radius_m, origin,
    )
    if vis_matrix is not None:
        F *= vis_matrix  # Hadamard product
    return F


def solve_fredholm_tikhonov(
    F: np.ndarray,
    P: np.ndarray,
    alpha: float = 1e-10,
    smooth_order: int = 1,
) -> dict:
    """Solve the Fredholm SLAE  F . A = P  via Tikhonov regularisation.

    Solves:  (alpha * E + F^T F) A = F^T P

    Parameters
    ----------
    F : np.ndarray, shape (N, N)
        Fredholm matrix.
    P : np.ndarray, shape (N,)
        Measured ADER vector.
    alpha : float
        Regularisation parameter (>= 0).
    smooth_order : int
        Order of the smoothing matrix L (1 = first differences, 2 = second).

    Returns
    -------
    dict with keys:
        'activity' : np.ndarray (N,)  -- solution vector A
        'alpha' : float               -- regularisation parameter used
        'cond_F' : float              -- condition number of F (2-norm)
        'cond_reg' : float            -- condition number of (alpha*E + F^T F)
    """
    N = F.shape[0]
    cond_F = float(np.linalg.cond(F))

    # Smoothing matrix
    E = np.eye(N)
    if smooth_order == 1:
        L = np.eye(N) - np.diag(np.ones(N - 1), 1)
        L[-1, -1] = 1.0
    else:
        L = np.eye(N) - 2.0 * np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 2), 2)
        L[-1, -1] = 1.0
        L[-2, -1] = 0.0

    FtF = F.T @ F
    FtP = F.T @ P
    reg_matrix = alpha * L.T @ L + FtF

    cond_reg = float(np.linalg.cond(reg_matrix))
    A = np.linalg.solve(reg_matrix, FtP)

    return {
        'activity': A,
        'alpha': alpha,
        'cond_F': cond_F,
        'cond_reg': cond_reg,
    }


def solve_fredholm_tikhonov_nn(
    F: np.ndarray,
    P: np.ndarray,
    alpha: float = 1e-10,
) -> dict:
    """Solve Fredholm SLAE with Tikhonov regularisation + non-negativity.

    Uses scipy.optimize.lsq_linear for bounded least-squares:
        min ||[F; sqrt(alpha)*L] x - [P; 0]||^2   s.t. x >= 0

    Parameters
    ----------
    F : np.ndarray (N, N)
    P : np.ndarray (N,)
    alpha : float
        Regularisation parameter.

    Returns
    -------
    dict with keys:
        'activity' : np.ndarray (N,)
        'alpha' : float
        'cond_F' : float
        'success' : bool
    """
    from scipy.optimize import lsq_linear

    N = F.shape[0]
    cond_F = float(np.linalg.cond(F))

    L = np.eye(N) - np.diag(np.ones(N - 1), 1)
    L[-1, -1] = 1.0

    S_aug = np.vstack([F, np.sqrt(alpha) * L])
    P_aug = np.concatenate([P, np.zeros(N)])

    res = lsq_linear(S_aug, P_aug, bounds=(0, np.inf))

    return {
        'activity': res.x,
        'alpha': alpha,
        'cond_F': cond_F,
        'success': bool(res.success),
    }
