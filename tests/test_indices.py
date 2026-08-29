import numpy as np
from soilactivity.indices import coord_to_index, index_to_coord, flatten_3d_to_1d


def test_c_order_indexing():
    grid_x = np.array([0, 1, 2], dtype=float)
    grid_y = np.array([0, 1, 2], dtype=float)
    grid_z = np.array([0, 1, 2], dtype=float)

    ix, iy, iz = coord_to_index(0.5, 0.5, 0.5, grid_x, grid_y, grid_z)
    assert (ix, iy, iz) == (0, 0, 0)

    nx, ny, nz = 2, 2, 2
    idx = flatten_3d_to_1d(iz=0, iy=0, ix=1, nx=nx, ny=ny)
    assert idx == 1

    idx2 = flatten_3d_to_1d(iz=0, iy=1, ix=0, nx=nx, ny=ny)
    assert idx2 == 2

    cx, cy, cz = index_to_coord(0, 0, 0, grid_x, grid_y, grid_z)
    assert (cx, cy, cz) == (0.5, 0.5, 0.5)
