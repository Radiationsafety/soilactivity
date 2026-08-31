"""reconstruct.py - восстановление активности модельного источника
методами пакета soilactivity и сверка с истинным (модельным) распределением.

Пайплайн:
  1. b = измеренный вектор МАЭД (results_model.csv, MODEL-строки)
  2. A = матрица чувствительности (A.npz из build_sensitivity.py)
  3. x_mlem = MLEM(A, b); x_tik = Тихонов(A, b)   <- soilactivity.solvers
  4. сравнение с x_true (model.json): ошибки по ячейкам, суммарная активность,
     косинусная близость, информационный коэффициент корреляции (Linfoot),
     невязка ||A x - b||/||b||
  5. графики: карты истина/MLEM/Тихонов + карта измерений b

Запуск:
  python reconstruct.py A.npz results_model.csv model.json \
      [--iters 200] [--lambda 1e-2] [--noise 0.02] --outdir out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import soilactivity as sa
from soilactivity.solvers import solve_mlem, solve_tikhonov
from soilactivity.correlation import information_correlation_coefficient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1soil_io import GridConfig, read_runs, runs_to_b, h10_per_decay_to_uSv_per_h  # noqa: E402


def relative_diff(x: np.ndarray, y: np.ndarray) -> float:
    d = np.linalg.norm(x - y)
    return float(d / np.linalg.norm(y)) if np.linalg.norm(y) > 0 else 0.0


def make_plots(cfg: GridConfig, x_true, x_mlem, x_tik, b, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = cfg.n_cells
    xs = np.array([cfg.cell_xy(i)[0] for i in range(n)])
    ys = np.array([cfg.cell_xy(i)[1] for i in range(n)])
    extent = [xs.min() - cfg.cell_size / 2, xs.max() + cfg.cell_size / 2,
              ys.min() - cfg.cell_size / 2, ys.max() + cfg.cell_size / 2]
    shape = (cfg.ny, cfg.nx)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    panels = [
        (x_true.reshape(shape), "Истинный источник, Бк", True),
        (x_mlem.reshape(shape), f"MLEM, Бк", True),
        (x_tik.reshape(shape), "Тихонов (x>=0), Бк", True),
        (b.reshape(shape) * 3.6e9, "Измеренная МАЭД, мкЗв/ч", False),
    ]
    for ax, (img, title, log) in zip(axes, panels):
        vmax = img.max()
        if log and vmax > 0:
            im = ax.imshow(img, origin="upper", extent=extent, cmap="hot",
                           norm=matplotlib.colors.LogNorm(vmin=max(img.min(), vmax * 1e-4),
                                                          vmax=vmax))
        else:
            im = ax.imshow(img, origin="upper", extent=extent, cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x, м")
        ax.set_ylabel("y, м")
    fig.suptitle("Реконструкция активности по матрице чувствительности A "
                 "(детектор на 1 м, источник на 10 см в грунте)", fontsize=11)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Реконструкция активности (soilactivity)")
    ap.add_argument("A_npz")
    ap.add_argument("model_csv")
    ap.add_argument("model_json")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-2)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="доп. относительный шум измерений (0..1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- A ---------------------------------------------------------------------
    A_npz = np.load(args.A_npz, allow_pickle=False)
    A = A_npz["A"]
    cfg_dict = json.loads(str(A_npz["config"]))
    cfg = GridConfig(**{k: cfg_dict[k] for k in GridConfig.__dataclass_fields__})

    # --- b ---------------------------------------------------------------------
    df, cfg_csv = read_runs(args.model_csv)
    b_per_decay = runs_to_b(df, cfg_csv)

    # --- истина ----------------------------------------------------------------
    model = json.loads(Path(args.model_json).read_text(encoding="utf-8"))
    x_true = np.zeros(cfg.n_cells)
    for c in model["cells"]:
        j = int(c["iy"]) * cfg.nx + int(c["ix"])
        x_true[j] = float(c["activity_Bq"])

    # CSV хранит отклик НА РАСПАД модельного источника (MC моделирует распады,
    # а не секунды). Физическое измерение: источник суммарной активностью
    # x_total Бк "наблюдался" 1 с, поэтому b = отклик_на_распад * x_total.
    # Масштаб задаётся суммарной активностью (известной из постановки
    # эксперимента), пространственное РАСПРЕДЕЛЕНИЕ - то, что восстанавливаем.
    x_total = float(np.sum(x_true))
    b = b_per_decay * x_total
    if args.noise > 0:
        rng = np.random.default_rng(args.seed)
        b = b * (1.0 + rng.normal(0.0, args.noise, size=b.shape))

    # --- методы пакета ----------------------------------------------------------
    # Масштабирование задачи: солверы пакета содержат абсолютный порог eps=1e-10,
    # рассчитанный на O(1)-величины. Переходим в единицы "мкЗв/ч на Бк" и
    # дополнительно нормируем A на максимум. Решение x при этом не меняется
    # (линейная замена переменных), численная устойчивость - восстанавливается.
    A_u = A * 3.6e9                      # мкЗв/ч на Бк
    b_u = b * 3.6e9                      # мкЗв/ч
    a_max = float(A_u.max())
    A_n = A_u / a_max
    b_n = b_u / a_max

    mlem = solve_mlem(A_n, b_n, max_iter=args.iters, tol=args.tol)
    x_mlem = mlem["activity"]

    tik = solve_tikhonov(A_n, b_n, lambda_reg=args.lam)
    x_tik = tik["activity"]

    # --- метрики ---------------------------------------------------------------
    rows = []
    for name, x, info in (("MLEM", x_mlem, mlem), ("Тихонов", x_tik, tik)):
        rel_err_cells = []
        for j in range(cfg.n_cells):
            if x_true[j] > 0:
                rel_err_cells.append(abs(x[j] - x_true[j]) / x_true[j])
        rec = {
            "метод": name,
            "отн_ошибка_суммарной_активности": float(
                abs(np.sum(x) - np.sum(x_true)) / np.sum(x_true)),
            "отн_ошибка_распределения (L2)": relative_diff(x, x_true),
            "средняя_ошибка_горячих_ячеек": float(np.mean(rel_err_cells)) if rel_err_cells else 0.0,
            "косинусная_близость": float(np.dot(x, x_true) /
                                         (np.linalg.norm(x) * np.linalg.norm(x_true) + 1e-300)),
            "информационная_корреляция_Linfoot": float(
                information_correlation_coefficient(x, x_true)),
            "невязка ||Ax-b||/||b||": relative_diff(A @ x, b),
            "итого_Бк": float(np.sum(x)),
        }
        rows.append(rec)

    # невязка истинного источника (проверка согласованности A и b)
    truth_residual = relative_diff(A @ x_true, b)

    rows.append({
        "метод": "ИСТИНА (модель)",
        "отн_ошибка_суммарной_активности": 0.0,
        "отн_ошибка_распределения (L2)": 0.0,
        "средняя_ошибка_горячих_ячеек": 0.0,
        "косинусная_близость": 1.0,
        "информационная_корреляция_Linfoot": 1.0,
        "невязка ||Ax-b||/||b||": truth_residual,
        "итого_Бк": float(np.sum(x_true)),
    })

    # --- печать -----------------------------------------------------------------
    print("\n=== Сверка реконструкции с модельным (истинным) источником ===")
    hdr = f"{'Метод':<10} {'Итого, Бк':>12} {'err Сумма':>10} {'err L2':>8} " \
          f"{'err горяч.':>10} {'cos':>7} {'Linfoot':>8} {'невязка':>8}"
    print(hdr)
    for r in rows:
        print(f"{r['метод']:<10} {r['итого_Бк']:>12.4e} "
              f"{r['отн_ошибка_суммарной_активности']:>10.2%} "
              f"{r['отн_ошибка_распределения (L2)']:>8.2%} "
              f"{r['средняя_ошибка_горячих_ячеек']:>10.2%} "
              f"{r['косинусная_близость']:>7.4f} "
              f"{r['информационная_корреляция_Linfoot']:>8.4f} "
              f"{r['невязка ||Ax-b||/||b||']:>8.4f}")

    print("\nMLEM:", f"сошёлся за {mlem['iterations']} итераций"
          if mlem["converged"] else f"достигнут лимит {mlem['iterations']} итераций")

    print("\nПо ячейкам (истина -> MLEM -> Тихонов), Бк:")
    for j in range(cfg.n_cells):
        if x_true[j] > 0 or x_mlem[j] > 0.01 * x_mlem.max():
            ix, iy = j % cfg.nx, j // cfg.nx
            print(f"  ячейка ({ix},{iy}): {x_true[j]:>10.3e} -> "
                  f"{x_mlem[j]:>10.3e} -> {x_tik[j]:>10.3e}")

    print(f"\nИзмеренная МАЭД: {b.min()*3.6e9:.3f}..{b.max()*3.6e9:.3f} мкЗв/ч "
          f"(ожидание для {float(np.sum(x_true))/1e9:.2f} ГБк на 10 см глубине)")

    # --- сохранение --------------------------------------------------------------
    import csv
    with open(outdir / "reconstruct_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    np.savez(outdir / "reconstructed.npz",
             x_true=x_true, x_mlem=x_mlem, x_tikhonov=x_tik, b=b, A=A)

    make_plots(cfg, x_true, x_mlem, x_tik, b, outdir / "recon_map.png")
    print(f"\nСохранено: {outdir}/reconstruct_summary.csv, "
          f"{outdir}/reconstructed.npz, {outdir}/recon_map.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
