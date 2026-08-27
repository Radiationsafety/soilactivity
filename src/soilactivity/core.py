import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any

from .sensitivity import calculate_analytical_sensitivity, load_sensitivity_matrix
from .solvers import solve_mlem, solve_tikhonov
from .interpolation import interpolate_and_smooth


@dataclass
class UnfoldingResult:
    """Структурированный результат реконструкции (вдохновлено bssunfold)."""
    activity_3d: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_z: np.ndarray
    solver_info: Dict[str, Any] = field(default_factory=dict)
    uncertainty_3d: Optional[np.ndarray] = None

    def save_to_file(self, filepath: str):
        """Сохранение результата в формате .npz с метаданными."""
        info = self.solver_info
        save_dict = {
            'activity_3d': self.activity_3d,
            'grid_x': self.grid_x,
            'grid_y': self.grid_y,
            'grid_z': self.grid_z,
            'method': str(info.get('method', '')),
            'iterations': int(info.get('iterations', 0)),
            'final_residual': float(info.get('final_residual', 0.0)),
            'converged': bool(info.get('converged', False)),
            'history': np.asarray(info.get('history', []), dtype=np.float64),
        }
        if self.uncertainty_3d is not None:
            save_dict['uncertainty_3d'] = self.uncertainty_3d

        np.savez(filepath, **save_dict)


class Unfolder:
    def __init__(self, method: str = 'mlem', iterations: int = 50,
                 tol: float = 1e-4, lambda_reg: float = 1e-2):
        self.method = method.lower()
        self.iterations = iterations
        self.tol = tol
        self.lambda_reg = lambda_reg

        if self.method not in ['mlem', 'tikhonov']:
            raise ValueError("Метод должен быть 'mlem' или 'tikhonov'")

    def unfold(self, data: pd.DataFrame, grids: Tuple[np.ndarray, np.ndarray, np.ndarray],
               attenuation_coeff: float = 0.1, sensitivity_file: Optional[str] = None,
               smooth_sigma: float = 0.0) -> UnfoldingResult:
        """Основной метод реконструкции."""
        grid_x, grid_y, grid_z = grids

        interp_activity, _ = interpolate_and_smooth(data, grid_x, grid_y, grid_z, sigma=smooth_sigma)

        nz, ny, nx = interp_activity.shape
        voxel_coords = self._generate_voxel_coords(grid_x, grid_y, grid_z)

        meas_coords = voxel_coords

        if sensitivity_file:
            S = load_sensitivity_matrix(sensitivity_file)
        else:
            S = calculate_analytical_sensitivity(meas_coords, voxel_coords, attenuation_coeff)

        y = interp_activity.flatten(order='C')

        if self.method == 'mlem':
            res = solve_mlem(S, y, max_iter=self.iterations, tol=self.tol)
        else:
            res = solve_tikhonov(S, y, lambda_reg=self.lambda_reg)

        activity_3d = res['activity'].reshape((nz, ny, nx), order='C')

        return UnfoldingResult(
            activity_3d=activity_3d,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            solver_info={
                'method': self.method,
                'iterations': res['iterations'],
                'final_residual': res['final_residual'],
                'converged': res['converged'],
                'history': res['history'],
            }
        )

    def _generate_voxel_coords(self, grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray) -> np.ndarray:
        """Генерация координат центров вокселей в C-order."""
        cx = (grid_x[:-1] + grid_x[1:]) / 2
        cy = (grid_y[:-1] + grid_y[1:]) / 2
        cz = (grid_z[:-1] + grid_z[1:]) / 2

        X, Y, Z = np.meshgrid(cx, cy, cz, indexing='ij')
        coords = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        return coords
