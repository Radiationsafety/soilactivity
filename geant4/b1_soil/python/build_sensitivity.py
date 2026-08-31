"""build_sensitivity.py - сборка матрицы чувствительности A из CSV b1soil.

A[i,j] = мощность H*(10) в точке детектора i (высота 1 м над ячейкой i)
от источника единичной активности (1 Бк) в ячейке j грунта, [Зв/с на Бк].

Вход  : results_sensitivity.csv (Geant4 или Python-MC, формат идентичен)
Выход : A.npz (матрица A, относительные ошибки, конфигурация)

Запуск:
  python build_sensitivity.py ../results_sensitivity.csv --out A.npz
  python build_sensitivity.py ... --no-symmetrize   # без усреднения по симметрии
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1soil_io import GridConfig, read_runs, runs_to_A  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="CSV b1soil -> матрица A (.npz)")
    ap.add_argument("runs_csv")
    ap.add_argument("--out", default="A.npz")
    ap.add_argument("--no-symmetrize", action="store_true",
                    help="не усреднять элементы с одинаковой геометрией")
    args = ap.parse_args()

    df, cfg = read_runs(args.runs_csv)
    sens = df[df["run_type"] == "SENSITIVITY"]
    if sens.empty:
        print("ОШИБКА: в файле нет строк SENSITIVITY", file=sys.stderr)
        return 1

    n_src = sens["src_index"].nunique()
    n_det_rows = sens["det_index"].nunique()
    print(f"Загружено {len(sens)} прогонов "
          f"({n_src} источников x {n_det_rows} детекторов, "
          f"det_mode={sens['det_mode'].iloc[0]})")

    A, A_err = runs_to_A(df, cfg, symmetrize=not args.no_symmetrize)

    n = cfg.n_cells
    cond = float(np.linalg.cond(A))
    central = A[n // 2, n // 2]
    print(f"A: {A.shape}, отклик над центром от источника в центре: "
          f"{central:.4e} Зв/с на Бк  ({central * 3.6e18:.1f} мкЗв/ч на 1 ГБк)")
    print(f"Число обусловленности cond(A) = {cond:.3e}")
    med = float(np.median(A_err[A > 0]))
    print(f"Медианная относительная стат. погрешность элементов: {med * 100:.2f}%")

    from dataclasses import asdict
    np.savez(args.out, A=A, A_rel_err=A_err,
             config=json.dumps(asdict(cfg)))
    print(f"Сохранено: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
