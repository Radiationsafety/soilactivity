import numpy as np


def coord_to_index(x: float, y: float, z: float,
                   grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray) -> tuple[int, int, int]:
    """Преобразует физические координаты в индексы сетки (C-order)."""
    ix = np.searchsorted(grid_x, x, side='right') - 1
    iy = np.searchsorted(grid_y, y, side='right') - 1
    iz = np.searchsorted(grid_z, z, side='right') - 1

    # Ограничение границами сетки (индексы ячеек: 0..len-2)
    ix = int(np.clip(ix, 0, len(grid_x) - 2))
    iy = int(np.clip(iy, 0, len(grid_y) - 2))
    iz = int(np.clip(iz, 0, len(grid_z) - 2))

    return ix, iy, iz


def index_to_coord(ix: int, iy: int, iz: int,
                   grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray) -> tuple[float, float, float]:
    """Преобразует индексы сетки в координаты центра вокселя (C-order)."""
    cx = (grid_x[ix] + grid_x[ix + 1]) / 2.0
    cy = (grid_y[iy] + grid_y[iy + 1]) / 2.0
    cz = (grid_z[iz] + grid_z[iz + 1]) / 2.0
    return float(cx), float(cy), float(cz)


def flatten_3d_to_1d(iz: int, iy: int, ix: int, nx: int, ny: int) -> int:
    """Строгий C-order (row-major) flatten: x - самый быстрый индекс."""
    return iz * (nx * ny) + iy * nx + ix
