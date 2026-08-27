import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(cache=True, fastmath=True)
def _mlem_iteration_numba(activity: np.ndarray, S: np.ndarray, y: np.ndarray,
                          S_sum: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Один шаг MLEM, ускоренный через Numba (вдохновлено bssunfold)."""
    n_meas = y.shape[0]
    n_vox = activity.shape[0]

    proj = np.zeros(n_meas, dtype=np.float64)
    for i in range(n_meas):
        acc = 0.0
        for j in range(n_vox):
            acc += S[i, j] * activity[j]
        proj[i] = acc

    ratio = np.zeros(n_meas, dtype=np.float64)
    for i in range(n_meas):
        ratio[i] = y[i] / (proj[i] + eps)

    backproj = np.zeros(n_vox, dtype=np.float64)
    for j in range(n_vox):
        acc = 0.0
        for i in range(n_meas):
            acc += S[i, j] * ratio[i]
        backproj[j] = acc

    new_activity = np.zeros(n_vox, dtype=np.float64)
    for j in range(n_vox):
        new_activity[j] = activity[j] * (backproj[j] / (S_sum[j] + eps))

    return new_activity


def solve_mlem(S: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-4) -> dict:
    """Решение обратной задачи методом MLEM с контролем сходимости."""
    n_voxels = S.shape[1]
    activity = np.full(n_voxels, np.mean(y), dtype=np.float64)
    S_sum = np.sum(S, axis=0)

    history = []
    diff = np.inf

    for iteration in range(max_iter):
        if NUMBA_AVAILABLE:
            new_activity = _mlem_iteration_numba(activity, S, y, S_sum)
        else:
            proj = S @ activity
            ratio = y / (proj + 1e-10)
            backproj = S.T @ ratio
            new_activity = activity * (backproj / (S_sum + 1e-10))

        diff = np.linalg.norm(new_activity - activity) / (np.linalg.norm(activity) + 1e-10)
        history.append(float(diff))
        activity = new_activity

        if diff < tol:
            break

    residual = np.sum((y - S @ activity) ** 2 / (S @ activity + 1e-10))

    return {
        "activity": activity,
        "iterations": iteration + 1,
        "history": history,
        "final_residual": float(residual),
        "converged": diff < tol,
    }


def solve_tikhonov(S: np.ndarray, y: np.ndarray, lambda_reg: float = 1e-2) -> dict:
    """Решение с регуляризацией Тихонова (с неотрицательными ограничениями).

    Минимизируется ||S*x - y||^2 + lambda * ||L*x||^2, x >= 0, где L -
    матрица первых разностей (сглаживание соседей).
    """
    from scipy.optimize import lsq_linear

    n = S.shape[1]
    L = np.diag(np.ones(n)) - np.diag(np.ones(n - 1), 1)
    L[-1, -1] = 1.0  # граничное условие

    S_aug = np.vstack([S, np.sqrt(lambda_reg) * L])
    y_aug = np.concatenate([y, np.zeros(n)])

    res = lsq_linear(S_aug, y_aug, bounds=(0, np.inf))

    residual = np.sum((y - S @ res.x) ** 2)

    return {
        "activity": res.x,
        "iterations": 1,
        "history": [],
        "final_residual": float(residual),
        "converged": bool(res.success),
    }
