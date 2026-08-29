from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .radionuclides import (
    KERMA_CONSTANTS,
    get_normalizing_factor,
    mixture_kerma_constant,
)
from .fredholm import (
    build_fredholm_matrix,
    build_fredholm_matrix_no_vis,
    solve_fredholm_tikhonov,
    solve_fredholm_tikhonov_nn,
    raster_coords,
    raster_to_vector,
    vector_to_raster,
)
from .visibility import compute_visibility_matrix, visibility_radius_mask
from .mcc import mcc_ader_to_sad, mcc_total_activity, mcc_coefficient
from .lorenz import lorenz_curve, lorenz_gini_coefficient, lorenz_compactness_ratio
from .diagnostics import slae_condition_number, slae_error_bound, slae_finer_error_estimate
from .correlation import information_correlation_coefficient


__all__ = [
    "SadReconstructor",
    "SadResult",
]


@dataclass
class SadResult:
    """Result of SAD reconstruction from ADER."""
    sad: np.ndarray          # 2D raster (ny, nx), Bq per cell
    ader_input: np.ndarray   # 2D raster (ny, nx), original ADER
    ader_forward: np.ndarray # 2D raster (ny, nx), ADER recomputed from SAD
    method: str
    alpha: float
    nx: int
    ny: int
    cell_size: float
    height_m: float
    total_activity: float
    total_activity_mcc: float
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize key results to a dictionary."""
        return {
            'method': self.method,
            'alpha': self.alpha,
            'nx': self.nx,
            'ny': self.ny,
            'cell_size_m': self.cell_size,
            'height_m': self.height_m,
            'total_activity_Bq': self.total_activity,
            'total_activity_mcc_Bq': self.total_activity_mcc,
            'cond_F': self.info.get('cond_F', None),
            'gini_sad': self.info.get('gini_sad', None),
            'gini_ader': self.info.get('gini_ader', None),
        }


class SadReconstructor:
    """High-level API for reconstructing Surface Activity Density from ADER.

    Reconstructs a 2D map of surface activity density (SAD) from
    ambient dose equivalent rate (ADER) measurements by solving
    the Fredholm integral equation of the 1st kind with Tikhonov
    regularisation.

    Parameters
    ----------
    nx, ny : int
        Raster dimensions (number of cells).
    cell_size : float
        Cell width and height [m].  Must not exceed 3 m for optimal
        results per Chizhov et al (2019).
    height_m : float
        Detector height above surface [m].  Typical: 1 m (pedestrian)
        or 3-30 m (UAV).  Higher = more ill-conditioned.
    radionuclide : str
        Radionuclide name (key in KERMA_CONSTANTS), e.g. 'Cs-137'.
    kerma_constant : float or None
        Override kerma constant.  Useful for radionuclide mixtures.
    dose_quantity : str
        'H_star_10' (default), 'K_air', 'D_air', or 'X'.
    buildings : list of dict or None
        Building definitions for barrier geometry.
        Each: {'x':, 'y':, 'width':, 'height':}.
    visibility_radius : float or None
        Max line-of-sight distance [m].  Limits matrix sparsity.
    origin : (x0, y0)
        Bottom-left corner of raster [m].

    Examples
    --------
    >>> recon = SadReconstructor(nx=40, ny=40, cell_size=5.0,
    ...                         height_m=1.0, radionuclide='Cs-137')
    >>> result = recon.reconstruct(ader_map, alpha=1e-11)
    >>> result.total_activity
    6.5e8

    References
    ----------
    1. Chizhov et al (2019) J. Radiol. Prot. 39 354-372.
    2. Chizhov et al (2023) J. Radiol. Prot. 43 041505.
    3. Chizhov et al (2023) J. Radiol. Prot. 43 041506.
    4. Chizhov & Kryuchkov (2024) Nuclear Science and Technology.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        cell_size: float,
        height_m: float = 1.0,
        radionuclide: str = "Cs-137",
        kerma_constant: Optional[float] = None,
        dose_quantity: str = "H_star_10",
        buildings: Optional[List[dict]] = None,
        visibility_radius: Optional[float] = None,
        origin: tuple = (0.0, 0.0),
    ):
        self.nx = nx
        self.ny = ny
        self.cell_size = cell_size
        self.height_m = height_m
        self.radionuclide = radionuclide
        self.dose_quantity = dose_quantity
        self.buildings = buildings
        self.visibility_radius = visibility_radius
        self.origin = origin

        if kerma_constant is not None:
            self.kerma_constant = kerma_constant
        else:
            if radionuclide not in KERMA_CONSTANTS:
                raise ValueError(
                    f"Unknown radionuclide '{radionuclide}'. "
                    f"Available: {list(KERMA_CONSTANTS.keys())}"
                )
            self.kerma_constant = KERMA_CONSTANTS[radionuclide]

        self.normalizing_factor = get_normalizing_factor(
            dose_quantity, radionuclide, self.kerma_constant
        )

        # Precompute raster coordinates
        self.CX, self.CY = raster_coords(nx, ny, cell_size, origin)
        self.cx_vec = raster_to_vector(self.CX)
        self.cy_vec = raster_to_vector(self.CY)

        # Build Fredholm matrix and visibility
        self._build_matrix()

    def _build_matrix(self):
        """Build the Fredholm matrix F with optional visibility."""
        # Visibility matrix
        if self.buildings is not None and len(self.buildings) > 0:
            self.vis_matrix = compute_visibility_matrix(
                self.cx_vec, self.cy_vec,
                buildings=self.buildings,
                visibility_radius=self.visibility_radius,
            )
            self.has_barriers = True
        elif self.visibility_radius is not None:
            self.vis_matrix = visibility_radius_mask(
                self.cx_vec, self.cy_vec,
                radius_m=self.visibility_radius,
            )
            self.has_barriers = False
        else:
            self.vis_matrix = None
            self.has_barriers = False

        self.F = build_fredholm_matrix(
            self.nx, self.ny, self.cell_size,
            self.height_m, self.kerma_constant, self.normalizing_factor,
            vis_matrix=self.vis_matrix,
            origin=self.origin,
        )

    def reconstruct(
        self,
        ader: np.ndarray,
        alpha: Optional[float] = None,
        non_negative: bool = True,
        noise_fraction: float = 0.0,
    ) -> SadResult:
        """Reconstruct SAD from ADER raster.

        Parameters
        ----------
        ader : np.ndarray (ny, nx)
            Measured ADER values at each cell.
        alpha : float or None
            Tikhonov regularisation parameter.  If None, a heuristic
            default based on matrix scale is used.
        non_negative : bool
            Enforce A >= 0 constraint.
        noise_fraction : float
            Estimated relative measurement noise (for error bounds).

        Returns
        -------
        SadResult
        """
        ader = np.asarray(ader, dtype=np.float64)
        assert ader.shape == (self.ny, self.nx), (
            f"ADER shape {ader.shape} != ({self.ny}, {self.nx})"
        )

        P = raster_to_vector(ader)
        N = self.nx * self.ny

        # Default alpha: scale-relative heuristic
        if alpha is None:
            FtF = self.F.T @ self.F
            FtF_max = float(np.max(np.abs(FtF)))
            alpha = 1e-6 * FtF_max

        # Solve
        if non_negative:
            from scipy.optimize import lsq_linear
            L = np.eye(N) - np.diag(np.ones(N - 1), 1)
            L[-1, -1] = 1.0
            S_aug = np.vstack([self.F, np.sqrt(alpha) * L])
            P_aug = np.concatenate([P, np.zeros(N)])
            res = lsq_linear(S_aug, P_aug, bounds=(0, np.inf))
            A_vec = res.x
            success = bool(res.success)
        else:
            res = solve_fredholm_tikhonov(self.F, P, alpha=alpha)
            A_vec = res['activity']
            success = True

        A_2d = vector_to_raster(A_vec, self.ny, self.nx)
        P_forward = self.F @ A_vec
        P_forward_2d = vector_to_raster(P_forward, self.ny, self.nx)

        total_act = float(np.sum(A_2d))
        total_mcc = mcc_total_activity(
            ader, self.kerma_constant,
            cell_area_m2=self.cell_size ** 2,
            dose_quantity=self.dose_quantity,
            radionuclide=self.radionuclide,
        )

        # Diagnostics
        cond_F = float(np.linalg.cond(self.F))
        gini_sad = lorenz_gini_coefficient(A_2d)
        gini_ader = lorenz_gini_coefficient(ader)

        info = {
            'cond_F': cond_F,
            'gini_sad': gini_sad,
            'gini_ader': gini_ader,
            'compactness_ratio': lorenz_compactness_ratio(A_2d, ader),
            'success': success,
            'non_negative': non_negative,
        }
        if noise_fraction > 0:
            info['error_bound'] = slae_error_bound(self.F, noise_fraction)

        return SadResult(
            sad=A_2d,
            ader_input=ader,
            ader_forward=P_forward_2d,
            method='fredholm_tikhonov',
            alpha=alpha,
            nx=self.nx,
            ny=self.ny,
            cell_size=self.cell_size,
            height_m=self.height_m,
            total_activity=total_act,
            total_activity_mcc=total_mcc,
            info=info,
        )

    def mcc(self, ader: np.ndarray) -> np.ndarray:
        """Quick MCC conversion ADER -> SAD (no Fredholm solve)."""
        return mcc_ader_to_sad(
            ader, self.kerma_constant,
            dose_quantity=self.dose_quantity,
            radionuclide=self.radionuclide,
            cell_area_m2=self.cell_size ** 2,
        )

    def forward(self, sad: np.ndarray) -> np.ndarray:
        """Forward problem: SAD -> ADER (using Fredholm matrix)."""
        return vector_to_raster(
            self.F @ raster_to_vector(sad), self.ny, self.nx
        )
