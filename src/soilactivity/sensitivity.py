import numpy as np
from typing import Union


def calculate_analytical_sensitivity(meas_coords: np.ndarray, voxel_coords: np.ndarray,
                                     mu: float) -> np.ndarray:
    """
    Расчет матрицы чувствительности по формуле экспоненциального ослабления.
    S_ij = (1 / (4 * pi * d^2)) * exp(-mu * d)
    """
    n_meas = meas_coords.shape[0]
    n_vox = voxel_coords.shape[0]
    S = np.zeros((n_meas, n_vox), dtype=np.float64)

    for i in range(n_meas):
        diff = voxel_coords - meas_coords[i]
        d = np.linalg.norm(diff, axis=1)
        d = np.maximum(d, 1e-6)
        S[i, :] = (1.0 / (4.0 * np.pi * d ** 2)) * np.exp(-mu * d)

    return S


def load_sensitivity_matrix(filepath: str) -> np.ndarray:
    """Загрузка предварительно рассчитанной матрицы (например, из MCNP/Geant4)."""
    return np.load(filepath)
