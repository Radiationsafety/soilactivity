"""python_mc.py - лёгкий Монте-Карло эмулятор фотонного переноса b1soil
(заместитель Geant4 для проверки пайплайна без установки Geant4).

Физическая модель (Cs-137, 661.7 кэВ):
  * геометрия: слой грунта z<0 (ICRU 53 sandy loam, rho=1.6 г/см3, пакет
    soilactivity NIST XCOM) + воздух z>0 (G4_AIR);
  * детекторы: 25 воздушных сфер R=15 см на высоте 1 м над ячейками сетки
    (режим "all" - физически эквивалентен поочерёдному размещению);
  * источник: точка в грунте на глубине 10 см, изотропно 4*пи;
  * взаимодействия:
      - комптоновское рассеяние: сечение - аналитический Клейн-Нишина,
        выборка E' отбором (макс. f = 2), поворот направления;
      - (мю - мю_Compton) при E > 150 кэВ трактуется как когерентное
        (упругое) рассеяние с углом ~ (1+cos^2 theta)/2;
      - при E <= 150 кэВ - фотоэффект (поглощение); фотон ниже 10 кэВ
        поглощается;
  * скоринг (как в Geant4-примере b1soil): флюенс фотонов, вошедших в
    сферу через границу, по 48 лог-бинам энергии (10 кэВ...3 МэВ).
    H*(10)/распад = sum_b N_b h*(10)/Phi(E_b) / (pi R^2) - в b1soil_io.

ВЫХОД: CSV в формате b1soil_version=1.0, идентичном Geant4-примеру.

Приближения: электронный транспорт не моделируется, когерентное рассеяние
без атомных форм-факторов; систематика абсолютных значений - единицы
процентов. Для точных расчётов используйте Geant4-пример.

Запуск:
  python python_mc.py --runType SENSITIVITY --out results_sensitivity.csv \
      --nPhotons 400000 --seed 11
  python python_mc.py --runType MODEL --model model.json \
      --out results_model.csv --nPhotons 2000000 --seed 77
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import soilactivity as sa

NBINS = 48
EMIN, EMAX = 0.01, 3.0        # МэВ, лог-сетка спектра (синхронно с B1Run.hh)
E_CUT = 0.010                  # поглощение фотонов ниже 10 кэВ
ELASTIC_THRESHOLD = 0.150      # МэВ: выше - когерентное~упругое, ниже - фотоэффект
ELECTRON_MASS = 0.51099895     # МэВ
RHO_AIR_G_CM3 = 1.20479e-3     # G4_AIR, г/см^3
E0_CS137 = 0.6617              # МэВ
EDGES = np.logspace(math.log10(EMIN), math.log10(EMAX), NBINS + 1)
EDGES_LOG = np.log(EDGES)
BIN_W = NBINS / (EDGES_LOG[-1] - EDGES_LOG[0])

# Z/A элементов (для mu_Compton = N_A Z/A sigma_KN)
_Z_OVER_A = {
    "H": 1 / 1.008, "C": 6 / 12.011, "N": 7 / 14.007, "O": 8 / 15.999,
    "Ar": 18 / 39.948, "Al": 13 / 26.982, "Si": 14 / 28.086,
    "K": 19 / 39.098, "Ca": 20 / 40.078, "Fe": 26 / 55.845,
}


def _kn_total_sigma_cm2(E_MeV):
    """Полное сечение Клейна-Нишина на электрон, см^2."""
    x = np.asarray(E_MeV, dtype=float) / ELECTRON_MASS
    lx = np.log1p(2.0 * x)
    t1 = (1.0 + x) / (x * x) * (2.0 * (1.0 + x) / (1.0 + 2.0 * x) - lx / x)
    t2 = 0.5 * lx - (1.0 + 3.0 * x) / (1.0 + 2.0 * x) ** 2
    return 2.0 * np.pi * (2.8179403262e-13) ** 2 * (t1 + t2)


class Materials:
    """Таблицы mu(E) и mu_Compton(E) грунта и воздуха на лог-сетке."""

    def __init__(self):
        self.E = np.logspace(math.log10(E_CUT), math.log10(EMAX), 200)
        self.lE = np.log(self.E)
        mats = {"soil": (sa.NIST_SOIL_COMPOSITION, 1.6),
                "air": (sa.NIST_AIR_DRY_COMPOSITION, RHO_AIR_G_CM3)}
        self.mu = {}
        self.mu_c = {}
        for name, (comp, rho) in mats.items():
            mu_rho = np.array([sa.mixture_mu_rho(comp, e) for e in self.E])
            z_a = sum(w * _Z_OVER_A[el] for el, w in comp.items())
            mu_c_rho = z_a * 6.02214076e23 * _kn_total_sigma_cm2(self.E)
            self.mu[name] = mu_rho * rho
            self.mu_c[name] = np.minimum(mu_c_rho, mu_rho) * rho


def mu_interp(E, table, lE_ref, E_ref):
    return np.interp(np.log(E), lE_ref, table)


def sample_compton(E0, rng, max_iter=60):
    """Выборка E' и cos(theta) по Клейну-Нишина (векторный отбор).

    f(x) = x^2 (x + 1/x - sin^2 theta) <= 2 при x = E'/E in [xmin, 1].
    Возвращает (E_new, cos_theta).
    """
    alpha = E0 / ELECTRON_MASS
    xmin = 1.0 / (1.0 + 2.0 * alpha)
    x = np.full(E0.shape, np.nan)
    cos_s = np.full(E0.shape, np.nan)
    remain = np.ones(E0.shape, dtype=bool)
    for _ in range(max_iter):
        if not remain.any():
            break
        idx = np.nonzero(remain)[0]
        xt = rng.uniform(xmin[idx], 1.0)
        cs = 1.0 - (1.0 / xt - 1.0) / alpha[idx]
        sin2 = np.clip(1.0 - cs * cs, 0.0, 1.0)
        f = xt * xt * (xt + 1.0 / xt - sin2)
        acc = rng.random(xt.size) * 2.0 <= f
        ok = idx[acc]
        x[ok] = xt[acc]
        cos_s[ok] = cs[acc]
        remain[ok] = False
    # непринятые после max_iter (крайне редки): рассеяние вперёд
    x[remain] = 1.0
    cos_s[remain] = 1.0
    return x * E0, cos_s


def sample_rayleigh_cos(n, rng):
    """cos(theta) для (1+cos^2 theta)/2 отбором."""
    out = np.empty(n)
    remain = np.ones(n, dtype=bool)
    while remain.any():
        idx = np.nonzero(remain)[0]
        c = rng.uniform(-1.0, 1.0, idx.size)
        acc = rng.random(idx.size) <= (1.0 + c * c) / 2.0
        ok = idx[acc]
        out[ok] = c[acc]
        remain[ok] = False
    return out


def rotate_dirs(u, cos_t, phi):
    """Поворот направлений u на полярный угол arccos(cos_t) и азимут phi."""
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (u.shape[0], 1))
    flip = np.abs(u[:, 2]) > 0.9
    helper[flip] = np.array([1.0, 0.0, 0.0])
    v1 = np.cross(u, helper)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + 1e-300
    v2 = np.cross(u, v1)
    return (u * cos_t[:, None]
            + (v1 * np.cos(phi)[:, None] + v2 * np.sin(phi)[:, None]) * sin_t[:, None])


def run_one_source(n_photons, src_xy, src_z, det_centers, det_r, mats, rng,
                   batch=200_000):
    """Прогон n_photons распадов из фиксированной точки источника.

    Возвращает hist (n_det, NBINS) - спектр энергий вошедших фотонов,
    n_in (n_det,) - число вошедших фотонов.
    """
    n_det = det_centers.shape[0]
    hist = np.zeros((n_det, NBINS))
    n_in = np.zeros(n_det, dtype=np.int64)
    det_z = det_centers[0, 2]
    pre_z_min = det_z - 2.0 * det_r   # ниже этого уровня сферы недостижимы за шаг

    soil_mu, soil_muc = mats.mu["soil"], mats.mu_c["soil"]
    air_mu, air_muc = mats.mu["air"], mats.mu_c["air"]
    lE = mats.lE

    done = 0
    while done < n_photons:
        n = min(batch, n_photons - done)
        done += n

        cos_t = rng.uniform(-1.0, 1.0, n)
        phi0 = rng.uniform(0.0, 2.0 * np.pi, n)
        sin_t = np.sqrt(1.0 - cos_t**2)
        u = np.stack([sin_t * np.cos(phi0), sin_t * np.sin(phi0), cos_t], axis=1)
        pos = np.tile(np.array([src_xy[0], src_xy[1], src_z]), (n, 1))
        E = np.full(n, E0_CS137)
        mat = np.zeros(n, dtype=np.int8)      # 0 - грунт, 1 - воздух
        alive = np.ones(n, dtype=bool)

        # --- транспорт
        while alive.any():
            idx = np.nonzero(alive)[0]
            p = pos[idx]
            d = u[idx]
            e = E[idx]
            m = mat[idx]

            mu = np.empty(idx.size)
            ms0 = m == 0
            if ms0.all():
                mu = mu_interp(e, soil_mu, lE, mats.E)
            elif (~ms0).all():
                mu = mu_interp(e, air_mu, lE, mats.E)
            else:
                mu[ms0] = mu_interp(e[ms0], soil_mu, lE, mats.E)
                mu[~ms0] = mu_interp(e[~ms0], air_mu, lE, mats.E)

            s = -np.log(rng.random(idx.size)) / mu

            # граница грунт/воздух и границы мира
            with np.errstate(divide="ignore", invalid="ignore"):
                t_plane = np.where(d[:, 2] != 0, -p[:, 2] / d[:, 2], np.inf)
            t_plane = np.where((t_plane > 1e-9) & np.isfinite(t_plane), t_plane, np.inf)

            t_exit = np.full(idx.size, np.inf)
            for k, (lo_k, hi_k) in enumerate(((-15.0, 15.0), (-15.0, 15.0),
                                              (-3.0, 3.0))):
                with np.errstate(divide="ignore", invalid="ignore"):
                    t1 = (lo_k - p[:, k]) / d[:, k]
                    t2 = (hi_k - p[:, k]) / d[:, k]
                t_exit = np.minimum(t_exit, np.where((t1 > 1e-9) & np.isfinite(t1), t1, np.inf))
                t_exit = np.minimum(t_exit, np.where((t2 > 1e-9) & np.isfinite(t2), t2, np.inf))

            s_min = np.minimum(s, np.minimum(t_plane, t_exit))
            p_new = p + d * s_min[:, None]

            # --- вход фотонов в сферу-детектор (только потенциально достижимые)
            cand = p_new[:, 2] > pre_z_min
            if cand.any():
                ci = idx[cand]
                pc = p[cand]            # (m,3)
                pnc = p_new[cand]       # (m,3)
                dd = pnc - pc           # (m,3) вектор шага
                oc = pc[:, None, :] - det_centers[None, :, :]  # (m,n_det,3)
                a2 = np.sum(dd * dd, axis=1)[:, None]          # (m,1)
                b2 = 2.0 * (dd[:, None, :] * oc).sum(axis=2)   # (m,n_det)
                c2 = (oc * oc).sum(axis=2) - det_r**2          # (m,n_det)
                disc = b2 * b2 - 4.0 * a2 * c2
                ok = (disc > 0) & (c2 > 0)
                with np.errstate(invalid="ignore"):
                    t1 = (-b2 - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a2)
                hit = ok & (t1 > 0.0) & (t1 <= 1.0)
                if hit.any():
                    ii, jj = np.nonzero(hit)
                    # ближайшая сфера для каждого фотона (по t1)
                    order = np.argsort(t1[ii, jj])
                    ii, jj = ii[order], jj[order]
                    _, first_pos = np.unique(ii, return_index=True)
                    ii = ii[first_pos]
                    jj = jj[first_pos]
                    ph = ci[ii]
                    Eb = E[ph]
                    bins = np.clip(((np.log(Eb) - EDGES_LOG[0]) * BIN_W).astype(int),
                                   0, NBINS - 1)
                    np.add.at(hist, (jj, bins), 1.0)
                    np.add.at(n_in, jj, 1)

            # --- перенос состояния
            pos[idx] = p_new
            mat[idx] = (p_new[:, 2] > 0.0).astype(np.int8)  # материал по знаку z

            escaped = t_exit < np.minimum(s, t_plane) + 1e-12
            alive[idx[escaped]] = False

            crossed_plane = (~escaped) & (t_plane <= s + 1e-12)
            interact = (~escaped) & (~crossed_plane)
            ai = idx[interact]
            if ai.size == 0:
                continue

            ea = E[ai]
            ma = mat[ai]
            mu_a = np.empty(ai.size)
            mu_c_a = np.empty(ai.size)
            ms = ma == 0
            if ms.any():
                mu_a[ms] = mu_interp(ea[ms], soil_mu, lE, mats.E)
                mu_c_a[ms] = mu_interp(ea[ms], soil_muc, lE, mats.E)
            if (~ms).any():
                mu_a[~ms] = mu_interp(ea[~ms], air_mu, lE, mats.E)
                mu_c_a[~ms] = mu_interp(ea[~ms], air_muc, lE, mats.E)

            xi = rng.random(ai.size)
            is_compton = xi < (mu_c_a / mu_a)
            is_absorb = (~is_compton) & (ea <= ELASTIC_THRESHOLD)
            is_elastic = (~is_compton) & (~is_absorb)

            # комптон
            ic = ai[is_compton]
            if ic.size:
                E_new, cos_s = sample_compton(E[ic], rng)
                phi_s = rng.random(ic.size) * 2.0 * np.pi
                u[ic] = rotate_dirs(u[ic], cos_s, phi_s)
                E[ic] = E_new
                alive[ic[E_new < E_CUT]] = False

            # фотоэффект
            alive[ai[is_absorb]] = False

            # когерентное (упругое) рассеяние
            ie = ai[is_elastic]
            if ie.size:
                cos_s = sample_rayleigh_cos(ie.size, rng)
                phi_s = rng.random(ie.size) * 2.0 * np.pi
                u[ie] = rotate_dirs(u[ie], cos_s, phi_s)

    return hist, n_in


def cell_xy(idx, nx, ny, cell):
    ix, iy = idx % nx, idx // nx
    return (ix - 0.5 * (nx - 1)) * cell, (iy - 0.5 * (ny - 1)) * cell


def write_rows(f, run_type, src_index, src_xyz, det_centers, n_dec, hist, n_in):
    n_det = det_centers.shape[0]
    for i in range(n_det):
        f.write(f"{run_type},{src_index},{src_xyz[0]:.4f},{src_xyz[1]:.4f},"
                f"{src_xyz[2]:.4f},all,{i},"
                f"{det_centers[i,0]:.4f},{det_centers[i,1]:.4f},{det_centers[i,2]:.4f},"
                f"{n_dec},{int(n_in[i])},0.000000e+00,0.000000e+00")
        f.write("," + ",".join(f"{v:.0f}" for v in hist[i]) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Python-MC эмулятор b1soil")
    ap.add_argument("--runType", default="SENSITIVITY",
                    choices=["SENSITIVITY", "MODEL"])
    ap.add_argument("--srcIndex", type=int, nargs="*", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--nPhotons", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="results_py_mc.csv")
    ap.add_argument("--nx", type=int, default=5)
    ap.add_argument("--ny", type=int, default=5)
    ap.add_argument("--cell", type=float, default=2.0)
    ap.add_argument("--depth", type=float, default=0.10)
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--radius", type=float, default=0.15)
    args = ap.parse_args()

    n = args.nx * args.ny
    det_centers = np.array([[*cell_xy(i, args.nx, args.ny, args.cell), args.height]
                            for i in range(n)])
    det_r = args.radius
    mats = Materials()

    # --- план прогонов: (метка src_index, (sx,sy,sz), n_photons)
    jobs = []
    if args.runType == "MODEL":
        model = json.loads(Path(args.model).read_text(encoding="utf-8"))
        grid = model.get("grid", {})
        nx, ny = int(grid.get("nx", args.nx)), int(grid.get("ny", args.ny))
        cs = float(grid.get("cell_size_m", args.cell))
        depth = float(grid.get("src_depth_m", args.depth))
        cells = [((int(c["ix"]) - 0.5 * (nx - 1)) * cs,
                  (int(c["iy"]) - 0.5 * (ny - 1)) * cs,
                  -depth, float(c["activity_Bq"])) for c in model["cells"]]
        w = np.array([c[3] for c in cells])
        w = w / w.sum()
        labels = [f"({int(round(c[0] / cs + (nx - 1) / 2))},{int(round(c[1] / cs + (ny - 1) / 2))})"
                  for c in cells]
        print(f"MODEL: {len(cells)} ячеек, доли событий "
              f"{dict(zip(labels, np.round(w, 4)))}")
        for k, (sx, sy, sz, _a) in enumerate(cells):
            jobs.append((-1, (sx, sy, sz), int(round(args.nPhotons * w[k]))))
    else:
        idxs = args.srcIndex if args.srcIndex else list(range(n))
        for i in idxs:
            x, y = cell_xy(i, args.nx, args.ny, args.cell)
            jobs.append((int(i), (x, y, -args.depth), args.nPhotons))

    out_path = Path(args.out)
    need_header = not out_path.exists()
    t0 = time.time()
    total_decays = 0
    with open(out_path, "a", encoding="utf-8") as f:
        if need_header:
            f.write(f"# b1soil_version=1.0 nx={args.nx} ny={args.ny} "
                    f"cellSize_m={args.cell} srcDepth_m={args.depth} "
                    f"soilDepth_m=2.0 detHeight_m={args.height} "
                    f"detRadius_m={args.radius} soilDensity_g_cm3=1.6 "
                    f"nBins={NBINS} emin_MeV={EMIN} emax_MeV={EMAX}\n")
            f.write("# python_mc (numpy): формат идентичен Geant4 b1soil\n")
            cols = (["run_type", "src_index", "src_x_m", "src_y_m", "src_z_m",
                     "det_mode", "det_index", "det_x_m", "det_y_m", "det_z_m",
                     "n_decays", "n_in", "edep_sum_MeV", "edep_rms_MeV"]
                    + [f"sp_{i:03d}" for i in range(NBINS)])
            f.write("#" + ",".join(cols) + "\n")

        # MODEL: стратифицированный прогон - вклады всех ячеек аккумулируются
        # и пишутся ОДНИМ набором строк (как в Geant4 с выборкой ячейки
        # пропорционально активности); SENSITIVITY: строка-набор на источник.
        model_accum = None
        model_decays = 0
        for k, (src_idx, sxyz, nph) in enumerate(jobs):
            rng = np.random.default_rng(args.seed + k * 7919)
            hist, n_in = run_one_source(nph, sxyz[:2], sxyz[2], det_centers,
                                        det_r, mats, rng, batch=args.batch)
            total_decays += nph
            if args.runType == "MODEL":
                if model_accum is None:
                    model_accum = (hist, n_in)
                else:
                    model_accum = (model_accum[0] + hist, model_accum[1] + n_in)
                model_decays += nph
            else:
                write_rows(f, args.runType, src_idx, sxyz, det_centers, nph,
                           hist, n_in)
            f.flush()
            dt = time.time() - t0
            print(f"  [{args.runType} {k+1}/{len(jobs)}] src={sxyz[0]:+.1f},{sxyz[1]:+.1f} "
                  f"вошло {int(n_in.sum())} ({int(n_in.sum())/nph*100:.3f}%), "
                  f"{total_decays/dt if dt>0 else 0:.1e} распад/с")
        if args.runType == "MODEL":
            write_rows(f, "MODEL", -1, (float("nan"),) * 3, det_centers,
                       model_decays, model_accum[0], model_accum[1])
    print(f"Готово за {time.time()-t0:.1f} с -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
