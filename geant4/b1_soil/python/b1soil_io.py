"""b1soil_io.py - общие константы, разбор CSV-результатов b1soil и
преобразование откликов детекторов в H*(10).

Используется и для результатов Geant4 (пример geant4/b1_soil), и для
Python-Монте-Карло (demo/python_mc.py): формат CSV у них идентичен.

Формат CSV (b1soil_version=1.0):
  строка-комментарий "# b1soil_version=1.0 nx=5 ny=5 cellSize_m=2.0 ..."
  затем CSV-строки по одному детектору за прогон.

Основные величины:
  отклик детектора на распад  H*(10)/распад = sum_b N_b * h*(10)/Phi(E_b) / (pi R^2)
  матрица чувствительности    A[i,j] = H*(10)/распад от 1 Бк в ячейке j
                              в точке детектора i  [Зв/с на Бк]  (1 Бк = 1 распад/с)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import soilactivity as sa

# ----------------------------------------------------------------------------
# Энергетическая сетка спектра вошедших фотонов (копия B1Run.hh - синхронизировать!)
# ----------------------------------------------------------------------------
NBINS = 48
EMIN_MEV = 0.01
EMAX_MEV = 3.0
BIN_EDGES = np.logspace(math.log10(EMIN_MEV), math.log10(EMAX_MEV), NBINS + 1)
BIN_CENTERS = np.sqrt(BIN_EDGES[:-1] * BIN_EDGES[1:])  # геом. середины

# h*(10)/Ф (ICRP 74) для всех бинов, Зв*см^2 на фотон
H10_PER_PHI_BINS = np.array([sa.h_star_10_over_phil(e) for e in BIN_CENTERS])

# Плотность воздуха G4_AIR (NIST), г/см^3 - для массы детектора в kerma-канале
G4_AIR_DENSITY_G_CM3 = 1.20479e-3


# ----------------------------------------------------------------------------
@dataclass
class GridConfig:
    """Геометрия b1soil (в метрах)."""
    nx: int = 5
    ny: int = 5
    cell_size: float = 2.0
    src_depth: float = 0.10
    soil_depth: float = 2.0
    det_height: float = 1.0
    det_radius: float = 0.15
    soil_density: float = 1.6
    n_bins: int = NBINS
    emin_mev: float = EMIN_MEV
    emax_mev: float = EMAX_MEV

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def cell_xy(self, idx: int) -> tuple[float, float]:
        ix, iy = idx % self.nx, idx // self.nx
        return (ix - 0.5 * (self.nx - 1)) * self.cell_size, \
               (iy - 0.5 * (self.ny - 1)) * self.cell_size

    def detector_area_cm2(self) -> float:
        """Площадь поперечного сечения сферы, см^2 (пи R^2)."""
        return math.pi * (self.det_radius * 100.0) ** 2

    def detector_mass_g(self) -> float:
        """Масса воздушной сферы, г (для kerma-канала edep)."""
        r_cm = self.det_radius * 100.0
        return 4.0 / 3.0 * math.pi * r_cm**3 * G4_AIR_DENSITY_G_CM3


_HEADER_RE = re.compile(r"#\s*b1soil_version=([\d.]+)\s+(.*)")


def parse_header(path: str | Path) -> GridConfig:
    """Читает строку-шапку CSV и возвращает конфигурацию геометрии."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _HEADER_RE.match(line.strip())
            if m:
                kv = dict(p.split("=") for p in m.group(2).split())
                cfg = GridConfig(
                    nx=int(kv["nx"]), ny=int(kv["ny"]),
                    cell_size=float(kv["cellSize_m"]),
                    src_depth=float(kv["srcDepth_m"]),
                    soil_depth=float(kv.get("soilDepth_m", 2.0)),
                    det_height=float(kv["detHeight_m"]),
                    det_radius=float(kv["detRadius_m"]),
                    soil_density=float(kv.get("soilDensity_g_cm3", 1.6)),
                    n_bins=int(kv.get("nBins", NBINS)),
                    emin_mev=float(kv.get("emin_MeV", EMIN_MEV)),
                    emax_mev=float(kv.get("emax_MeV", EMAX_MEV)),
                )
                return cfg
    raise ValueError(f"В {path} не найдена строка-шапка '# b1soil_version=...'")


def read_runs(path: str | Path):
    """Читает CSV b1soil -> (DataFrame строк, GridConfig)."""
    import pandas as pd

    cfg = parse_header(path)
    n_spec_cols = cfg.n_bins
    names = (["run_type", "src_index", "src_x_m", "src_y_m", "src_z_m",
              "det_mode", "det_index", "det_x_m", "det_y_m", "det_z_m",
              "n_decays", "n_in", "edep_sum_MeV", "edep_rms_MeV"]
             + [f"sp_{i:03d}" for i in range(n_spec_cols)])
    df = pd.read_csv(path, comment="#", header=None, names=names)
    return df, cfg


def spectrum_to_h10_per_decay(spec_counts: np.ndarray, cfg: GridConfig,
                              n_decays: float) -> float:
    """Отклик детектора на ОДИН распад: H*(10), Зв/распад.

    H*(10)/распад = sum_b (N_b/N) * h*(10)/Phi(E_b) / (пи R^2),
    где N_b - число вошедших фотонов в бине b за N распадов,
    N_b/N - флюенс на распад через сечение сферы (пи R^2).
    Для 1 Бк численно равно мощности H*(10), Зв/с.
    """
    edges = np.logspace(math.log10(cfg.emin_mev), math.log10(cfg.emax_mev),
                        cfg.n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    h10 = np.array([sa.h_star_10_over_phil(e) for e in centers])
    counts = spec_counts / float(n_decays)   # флюенс на распад
    return float(np.dot(counts, h10) / cfg.detector_area_cm2())


def runs_to_A(df, cfg: GridConfig, symmetrize: bool = True):
    """Строит матрицу чувствительности A (n_det x n_src) из SENSITIVITY-строк.

    Возвращает (A, A_rel_err): A[i,j] в Зв/с на Бк; A_rel_err - относительная
    статистическая погрешность элементов (1/sqrt(N), усреднённая по симметрии).

    symmetrize=True усредняет элементы с одинаковым (|dx|,|dy|) смещением -
    геометрия квадратной сетки на однородном грунте симметрична, это
    уменьшает статистический шум матрицы в несколько раз.
    """
    sens = df[df["run_type"] == "SENSITIVITY"]
    n = cfg.n_cells
    spec_cols = [f"sp_{i:03d}" for i in range(cfg.n_bins)]

    A = np.zeros((n, n))
    A_err = np.zeros((n, n))

    for _, row in sens.iterrows():
        j = int(row["src_index"])
        i = int(row["det_index"])
        counts = row[spec_cols].to_numpy(dtype=float)
        A[i, j] = spectrum_to_h10_per_decay(counts, cfg, row["n_decays"])
        n_in = float(row["n_in"])
        A_err[i, j] = 1.0 / math.sqrt(n_in) if n_in > 0 else 1.0

    if symmetrize:
        # ключ симметрии: (|dx|,|dy|) -> сортированная пара (D4-симметрия квадрата)
        groups: dict[tuple, list[tuple[int, int]]] = {}
        for i in range(n):
            xi, yi = cfg.cell_xy(i)
            for j in range(n):
                xj, yj = cfg.cell_xy(j)
                key = tuple(sorted((abs(round(xi - xj, 6)), abs(round(yi - yj, 6)))))
                groups.setdefault(key, []).append((i, j))

        A_sym = A.copy()
        err_sym = A_err.copy()
        for pairs in groups.values():
            vals = np.array([A[i, j] for i, j in pairs])
            errs = np.array([A_err[i, j] for i, j in pairs])
            mask = vals > 0
            if not mask.any():
                continue
            w = 1.0 / errs[mask] ** 2          # веса = N_k (errs - относительные: 1/sqrt(N))
            mean = float(np.average(vals[mask], weights=w))
            mean_err_rel = float(1.0 / np.sqrt(w.sum()))  # относительная ошибка группы
            for i, j in pairs:
                A_sym[i, j] = mean
                err_sym[i, j] = mean_err_rel
        A, A_err = A_sym, err_sym

    return A, A_err


def runs_to_b(df, cfg: GridConfig) -> np.ndarray:
    """Вектор измерений b[i] (Зв/с) из MODEL-строк, по одному на детектор i."""
    model = df[df["run_type"] == "MODEL"]
    n = cfg.n_cells
    spec_cols = [f"sp_{i:03d}" for i in range(cfg.n_bins)]
    b = np.zeros(n)
    # несколько MODEL-наборов строк (стратифицированные прогоны) суммируются
    for _, row in model.iterrows():
        i = int(row["det_index"])
        counts = row[spec_cols].to_numpy(dtype=float)
        b[i] += spectrum_to_h10_per_decay(counts, cfg, row["n_decays"])
    return b


def h10_per_decay_to_uSv_per_h(h10_per_decay: float) -> float:
    """1 Бк -> H*(10)/с; в мкЗв/ч: * 3600 с/ч * 1e6 мкЗв/Зв."""
    return h10_per_decay * 3.6e9
