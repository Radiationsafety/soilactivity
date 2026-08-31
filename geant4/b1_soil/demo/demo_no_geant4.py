"""demo_no_geant4.py - полный пайплайн b1soil БЕЗ установки Geant4.

Вместо Geant4 используется numpy Монте-Карло (python_mc.py) с тем же
CSV-форматом, поэтому все последующие шаги (build_sensitivity.py,
reconstruct.py) идентичны работе с реальным Geant4.

Шаги:
  1. model.json - модельный источник (3 горячих пятна, 3.5e8 Бк суммарно)
  2. MC-прогоны матрицы чувствительности: 25 источников x 6e5 распадов
  3. MC-прогон модельного источника ("измерение" МАЭД, 4e6 распадов)
  4. build_sensitivity.py -> A.npz (25x25, Зв/с на Бк)
  5. reconstruct.py (MLEM + Тихонов пакета soilactivity) -> сверка с истиной
  6. сверка MC с аналитическим point-kernel (ослабление + накопление ANS-6.4.3)

Запуск:  python demo_no_geant4.py [--fast]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PY = HERE.parent / "python"
sys.path.insert(0, str(PY))
sys.path.insert(0, str(HERE))

MODEL = {
    "grid": {"nx": 5, "ny": 5, "cell_size_m": 2.0, "src_depth_m": 0.10},
    "cells": [
        {"ix": 1, "iy": 1, "activity_Bq": 1.0e8},
        {"ix": 2, "iy": 2, "activity_Bq": 2.0e8},
        {"ix": 3, "iy": 2, "activity_Bq": 5.0e7},
    ],
}


def point_kernel_row(cfg, xs, ys):
    """Аналитический отклик (point-kernel): прямой луч через 2 слоя
    + фактор накопления ANS-6.4.3 для грунта.

    A_pk[i,j] = n_gamma/(4 pi d^2) * exp(-mu_soil*l_soil) * B(E, mu*l_soil)
                * exp(-mu_air*l_air) * h*(10)/Phi(E)
    Внимание: point-kernel учитывает рассеяние лишь феноменологически (B
    вдоль прямой), поэтому для смещённых детекторов занижает MC-отклик -
    воздух прозрачен для рассеянного излучения, вышедшего из грунта.
    """
    import soilactivity as sa

    mu_soil = sa.linear_attenuation(sa.NIST_SOIL_COMPOSITION, 1.6, 0.6617)
    mu_air = sa.linear_attenuation(sa.NIST_AIR_DRY_COMPOSITION, 1.20479e-3, 0.6617)
    h10 = sa.h_star_10_over_phil(0.6617)
    B_soil = sa.buildup_for_mixture(
        sa.NIST_SOIL_COMPOSITION, 0.6617, mu_soil * 10.0,
        coeff_lookup=lambda el, E: (sa.lookup_mu_en_rho(el, E),
                                    sa.lookup_mu_rho(el, E)))

    n = cfg.n_cells
    A = np.zeros((n, n))
    for i in range(n):
        xi, yi = cfg.cell_xy(i)
        for j in range(n):
            xj, yj = cfg.cell_xy(j)
            d3 = math.dist((xi, yi, cfg.det_height), (xj, yj, -cfg.src_depth))
            t_soil = cfg.src_depth / (cfg.det_height + cfg.src_depth)
            l_soil_cm = t_soil * d3 * 100.0
            l_air_cm = (1.0 - t_soil) * d3 * 100.0
            mfp = mu_soil * l_soil_cm
            # фактор накопления интерполируем по mfp относительно значения на глубине надира
            B = 1.0 + (B_soil - 1.0) * (mfp / max(mu_soil * 10.0, 1e-9))
            A[i, j] = (0.851 / (4.0 * math.pi * (d3 * 100.0) ** 2)
                       * math.exp(-mfp) * B
                       * math.exp(-mu_air * l_air_cm) * h10)
    return A


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="ускоренный прогон (меньше фотонов, для проверки)")
    args = ap.parse_args()

    n_ph_sens = 150_000 if args.fast else 600_000
    n_ph_model = 1_000_000 if args.fast else 4_000_000

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    for f in ("results_sensitivity.csv", "results_model.csv", "A.npz",
              "pk_vs_mc.csv"):
        (HERE / f).unlink(missing_ok=True)
    for f in ("reconstructed.npz", "recon_map.png", "reconstruct_summary.csv"):
        (out / f).unlink(missing_ok=True)

    model_path = HERE / "model.json"
    model_path.write_text(json.dumps(MODEL, indent=2), encoding="utf-8")
    print(f"Модельный источник: {len(MODEL['cells'])} ячейки, "
          f"суммарно {sum(c['activity_Bq'] for c in MODEL['cells']):.2e} Бк")

    def run(script, *args_):
        cmd = [sys.executable, str(script), *map(str, args_)]
        print("\n$", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=HERE)

    # 2-3. Монте-Карло (аналог запуска Geant4: sensitivity.mac + model_source.mac)
    run(HERE / "python_mc.py", "--runType", "SENSITIVITY",
        "--out", "results_sensitivity.csv",
        "--nPhotons", n_ph_sens, "--seed", "11")
    run(HERE / "python_mc.py", "--runType", "MODEL", "--model", "model.json",
        "--out", "results_model.csv",
        "--nPhotons", n_ph_model, "--seed", "77")

    # 4. Матрица чувствительности
    run(PY / "build_sensitivity.py", "results_sensitivity.csv", "--out", "A.npz")

    # 5. Реконструкция методами пакета + сверка с истиной
    run(PY / "reconstruct.py", "A.npz", "results_model.csv", "model.json",
        "--outdir", "out", "--noise", "0.02", "--iters", "200")

    # 6. Сверка с point-kernel (аналитика пакета: XCOM + ANS-6.4.3 + ICRP 74)
    from b1soil_io import GridConfig, read_runs, runs_to_A, \
        h10_per_decay_to_uSv_per_h
    import csv

    df, cfg = read_runs(HERE / "results_sensitivity.csv")
    A_mc, A_err = runs_to_A(df, cfg, symmetrize=True)
    A_pk = point_kernel_row(cfg, None, None)
    n = cfg.n_cells

    print("\n=== Сверка Монте-Карло с point-kernel (прямой луч + накопление) ===")
    print("смещение детектора, м | MC, мкЗв/ч на ГБк | PK | PK/MC")
    rows_csv = []
    seen = set()
    for j in range(n):
        xj, yj = cfg.cell_xy(j)
        key = (abs(round(xj)), abs(round(yj)))
        if key in seen:
            continue
        seen.add(key)
        mc = A_mc[n // 2, j] * 3.6e18
        pk = A_pk[n // 2, j] * 3.6e18
        ratio = pk / mc if mc > 0 else float("nan")
        print(f"  ({xj:+.0f},{yj:+.0f})            | {mc:.4f}          | {pk:.4f} | {ratio:.3f}")
        rows_csv.append({"offset_x_m": xj, "offset_y_m": yj,
                         "A_mc_uSv_h_per_GBq": mc, "A_pk_uSv_h_per_GBq": pk,
                         "ratio_pk_mc": ratio})
    with open(out / ".." / "pk_vs_mc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print("Сохранено: pk_vs_mc.csv")

    # центральный элемент: ожидание физическое
    c = A_mc[n // 2, n // 2]
    print(f"\nОтклик в надире: {c * 3.6e18:.1f} мкЗв/ч на 1 ГБк "
          f"(совместимо с гамма-постоянной Cs-137 ~76-120 мкЗв/ч/(ГБк·м) с учётом "
          f"ослабления в грунте и накопления)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
