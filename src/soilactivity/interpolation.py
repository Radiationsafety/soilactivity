import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter


def interpolate_and_smooth(data: pd.DataFrame, grid_x: np.ndarray, grid_y: np.ndarray,
                           grid_z: np.ndarray, sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Интерполяция измерений на регулярную 3D-сетку с помощью кригинга (RBF)
    и опциональное сглаживание Гауссом.
    """
    nx, ny, nz = len(grid_x) - 1, len(grid_y) - 1, len(grid_z) - 1
    activity_3d = np.zeros((nz, ny, nx), dtype=np.float64)
    weights_3d = np.zeros((nz, ny, nx), dtype=np.float64)

    cx = (grid_x[:-1] + grid_x[1:]) / 2
    cy = (grid_y[:-1] + grid_y[1:]) / 2
    cz = (grid_z[:-1] + grid_z[1:]) / 2

    unique_z = np.unique(data['z'])

    for k, target_z in enumerate(cz):
        closest_z = unique_z[np.argmin(np.abs(unique_z - target_z))]
        layer_data = data[np.abs(data['z'] - closest_z) < 1e-3]

        if len(layer_data) < 3:
            continue

        rbf = Rbf(layer_data['x'], layer_data['y'], layer_data['dose_rate'], function='linear')

        X, Y = np.meshgrid(cx, cy, indexing='ij')
        Z_interp = rbf(X, Y)

        if sigma > 0:
            Z_interp = gaussian_filter(Z_interp, sigma=sigma)

        activity_3d[k, :, :] = Z_interp.T
        weights_3d[k, :, :] = 1.0

    return activity_3d, weights_3d
