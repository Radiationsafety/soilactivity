import numpy as np
import pandas as pd

from soilactivity import Unfolder
from soilactivity.sensitivity import calculate_analytical_sensitivity
from soilactivity.solvers import solve_mlem, solve_tikhonov


def _make_voxel_coords(nx, ny, nz, sx=1.0, sy=1.0, sz=1.0):
    cx = np.arange(0.5 * sx, (nx + 0.5) * sx, sx)
    cy = np.arange(0.5 * sy, (ny + 0.5) * sy, sy)
    cz = np.arange(0.5 * sz, (nz + 0.5) * sz, sz)
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing='ij')
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


def test_mlem_recovers_known_activity():
    """Детерминированный тест (bssunfold-стиль): прямое проецирование -> решение -> восстановление."""
    rng = np.random.default_rng(0)
    nx, ny, nz = 4, 4, 3
    coords = _make_voxel_coords(nx, ny, nz)

    S = calculate_analytical_sensitivity(coords, coords, mu=0.1)

    true = np.abs(rng.normal(size=S.shape[1]))
    true[5] = 100.0
    true /= true.sum()

    y = S @ true
    y = np.maximum(y + rng.normal(0, 1e-4, size=y.shape), 0.0)

    res = solve_mlem(S, y, max_iter=500, tol=1e-7)

    assert res['activity'].shape == (S.shape[1],)
    assert np.all(res['activity'] >= -1e-9)
    corr = np.corrcoef(res['activity'], true)[0, 1]
    assert corr > 0.9, f"Корреляция восстановления слишком низкая: {corr:.3f}"


def test_tikhonov_runs_and_nonneg():
    rng = np.random.default_rng(1)
    coords = _make_voxel_coords(3, 3, 2)
    S = calculate_analytical_sensitivity(coords, coords, mu=0.2)
    true = np.abs(rng.normal(size=S.shape[1]))
    true /= true.sum()
    y = np.maximum(S @ true, 0.0)

    res = solve_tikhonov(S, y, lambda_reg=1e-2)
    assert np.all(res['activity'] >= -1e-9)
    assert res['activity'].shape == (S.shape[1],)


def test_end_to_end_unfold_smoke():
    """Smoke-тест полного пайплайна Unfolder (структура выхода корректна)."""
    nx, ny, nz = 4, 4, 3
    grid_x = np.linspace(0, 4, nx + 1)
    grid_y = np.linspace(0, 4, ny + 1)
    grid_z = np.linspace(0, 3, nz + 1)

    rng = np.random.default_rng(2)
    df_data = pd.DataFrame({
        'x': rng.uniform(0, 4, 30),
        'y': rng.uniform(0, 4, 30),
        'z': rng.choice(grid_z, 30),
        'dose_rate': rng.uniform(10, 50, 30),
    })

    uf = Unfolder(method='mlem', iterations=50, tol=1e-5)
    result = uf.unfold(df_data, (grid_x, grid_y, grid_z), attenuation_coeff=0.1)

    assert result.activity_3d.shape == (nz, ny, nx)
    assert np.all(result.activity_3d >= -1e-9)
    assert 'final_residual' in result.solver_info


def test_save_to_file_roundtrip(tmp_path):
    nx, ny, nz = 3, 3, 2
    grid_x = np.linspace(0, 3, nx + 1)
    grid_y = np.linspace(0, 3, ny + 1)
    grid_z = np.linspace(0, 2, nz + 1)

    rng = np.random.default_rng(3)
    df_data = pd.DataFrame({
        'x': rng.uniform(0, 3, 20),
        'y': rng.uniform(0, 3, 20),
        'z': rng.choice(grid_z, 20),
        'dose_rate': rng.uniform(10, 50, 20),
    })

    uf = Unfolder(method='tikhonov', lambda_reg=1e-2)
    result = uf.unfold(df_data, (grid_x, grid_y, grid_z), attenuation_coeff=0.1)

    out = tmp_path / "res.npz"
    result.save_to_file(str(out))

    loaded = np.load(str(out))
    assert loaded['activity_3d'].shape == (nz, ny, nx)
    assert loaded['iterations'] >= 1
    assert loaded['history'].dtype == np.float64
