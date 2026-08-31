"""visualize.py - визуализация примера b1_soil: геометрия, 3D-модели,
матрица чувствительности и карты активности.

Генерирует (по умолчанию в demo/out):
  geometry_3d.png        - 3D-сцена: грунт, сетка источников, детекторы (2 режима)
  geometry_3d.html       - интерактивная 3D-сцена (plotly, наведение = координаты/активность)
  geometry_sections.png  - разрез XZ (вертикальный) и вид сверху XY с размерами
  sensitivity_matrix.png - матрица A (25x25) + профили отклика
  activity_maps.png      - карты: истина / MLEM / Тихонов + невязки + измерения
  activity_3d.png        - 3D-бары распределения активности (истина/MLEM/Тихонов)

Скрипт автономен: нужны только numpy, matplotlib (plotly - для HTML).
Запуск (из каталога b1_soil):
  python3 python/visualize.py
  python3 python/visualize.py --A demo/A.npz --model demo/model.json \
      --recon demo/out/reconstructed.npz --outdir demo/out
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402

BASE = Path(__file__).resolve().parents[1]

# --- палитра -----------------------------------------------------------------
C_SOIL = "#8a6642"      # грунт
C_SOIL_TOP = "#a9865c"  # поверхность земли
C_SRC = "#23233b"       # источники (сетка)
C_DET = "#1f6f9c"       # детекторы
C_HOT = "#d62728"       # горячие пятна
C_AIR = "#e8f0f8"


# ----------------------------------------------------------------------------
# Конфигурация сетки (локальная копия GridConfig - без зависимости от пакета)
# ----------------------------------------------------------------------------
@dataclass
class Cfg:
    nx: int = 5
    ny: int = 5
    cell_size: float = 2.0
    src_depth: float = 0.10
    soil_depth: float = 2.0
    det_height: float = 1.0
    det_radius: float = 0.15

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def cell_xy(self, idx: int) -> tuple[float, float]:
        ix, iy = idx % self.nx, idx // self.nx
        return (ix - 0.5 * (self.nx - 1)) * self.cell_size, \
               (iy - 0.5 * (self.ny - 1)) * self.cell_size

    @property
    def half(self) -> float:
        """Полуширина площадки (м): край внешних ячеек."""
        return 0.5 * self.nx * self.cell_size


def load_cfg(A_npz: Path) -> Cfg:
    """GridConfig сохранён в A.npz['config'] как JSON-строка."""
    d = np.load(A_npz, allow_pickle=False)
    raw = json.loads(str(d["config"]))
    keys = {f for f in Cfg.__dataclass_fields__}
    return Cfg(**{k: float(raw[k]) if k not in ("nx", "ny") else int(raw[k])
                  for k in keys if k in raw})


def load_xtrue(cfg: Cfg, model_json: Path) -> np.ndarray:
    model = json.loads(Path(model_json).read_text(encoding="utf-8"))
    x = np.zeros(cfg.n_cells)
    for c in model.get("cells", []):
        j = int(c["iy"]) * cfg.nx + int(c["ix"])
        x[j] = float(c["activity_Bq"])
    return x


# ----------------------------------------------------------------------------
# 1. 3D-сцена (matplotlib): чувствительность и модельный источник
# ----------------------------------------------------------------------------
def _sphere(ax, x0, y0, z0, r, color, alpha, n=12, lw=0.25, edge=True):
    u = np.linspace(0, 2 * np.pi, 2 * n)
    v = np.linspace(0, np.pi, n)
    x = x0 + r * np.outer(np.cos(u), np.sin(v))
    y = y0 + r * np.outer(np.sin(u), np.sin(v))
    z = z0 + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=lw if edge else 0,
                    edgecolor=color if edge else "none", shade=False, zorder=3)


def _soil_box(ax, cfg: Cfg, alpha=0.16):
    h, d = cfg.half, cfg.soil_depth
    x = np.array([-h, h])
    y = np.array([-h, h])
    X, Y = np.meshgrid(x, y)
    # верхняя грань (поверхность земли)
    ax.plot_surface(X, Y, np.zeros_like(X), color=C_SOIL_TOP, alpha=min(alpha * 2.2, 0.95),
                    linewidth=0.4, edgecolor=C_SOIL, shade=False, zorder=1)
    # дно
    ax.plot_surface(X, Y, np.full_like(X, -d), color=C_SOIL, alpha=alpha,
                    linewidth=0, shade=False, zorder=1)
    # боковые грани
    for (xa, ya, xb, yb) in ((-h, -h, h, -h), (-h, h, h, h), (-h, -h, -h, h), (h, -h, h, h)):
        xs = np.array([xa, xb])
        ys = np.array([ya, yb])
        XS, Z = np.meshgrid(xs, np.array([-d, 0.0]))
        ZS = np.broadcast_to(Z, XS.shape)
        YS = np.broadcast_to(np.linspace(ya, yb, 2)[None, :], XS.shape) \
            if ya != yb else np.full_like(XS, ya)
        ax.plot_surface(XS, YS, ZS, color=C_SOIL, alpha=alpha,
                        linewidth=0, shade=False, zorder=1)


def _draw_scene(ax, cfg: Cfg, x_true: np.ndarray | None, title: str):
    _soil_box(ax, cfg)

    # детекторы: сферы R = det_radius на высоте det_height
    for i in range(cfg.n_cells):
        dx, dy = cfg.cell_xy(i)
        _sphere(ax, dx, dy, cfg.det_height, cfg.det_radius, C_DET, 0.10, n=10, lw=0.2)

    # источники: точки сетки на глубине src_depth
    zs = -cfg.src_depth
    for j in range(cfg.n_cells):
        sx, sy = cfg.cell_xy(j)
        ax.scatter([sx], [sy], [zs], color=C_SRC, s=14, depthshade=False, zorder=4)

    # горячие пятна модельного источника
    hot_handles = []
    if x_true is not None and x_true.max() > 0:
        vmax = x_true.max()
        for j in np.where(x_true > 0)[0]:
            sx, sy = cfg.cell_xy(j)
            r = cfg.cell_size * 0.22 * (x_true[j] / vmax) ** (1.0 / 3.0) + 0.08
            _sphere(ax, sx, sy, zs, r, C_HOT, 0.9, n=14, lw=0.4)
            hot_handles.append(Line2D([], [], marker="o", ls="none", color=C_HOT,
                                      markersize=7,
                                      label=f"горячее пятно: {x_true[j]/1e9:.2f} ГБк"))

    ax.set_xlim(-cfg.half, cfg.half)
    ax.set_ylim(-cfg.half, cfg.half)
    ax.set_zlim(-cfg.soil_depth - 0.4, cfg.det_height + 0.7)
    ax.set_box_aspect((1, 1, (cfg.soil_depth + cfg.det_height + 1.1) / (2 * cfg.half)))
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_zlabel("z, м")
    ax.view_init(elev=16, azim=-58)
    ax.set_title(title, fontsize=10.5, pad=10)

    handles = [
        Line2D([], [], marker="o", ls="none", color=C_SRC, markersize=6,
               label="точка источника (глубина 10 см)"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="none",
               markeredgecolor=C_DET, markersize=9,
               label=f"детектор-сфера R = {cfg.det_radius*100:.0f} см (h = 1 м)"),
        Patch(facecolor=C_SOIL_TOP, alpha=0.8, label="поверхность грунта"),
        Patch(facecolor=C_SOIL, alpha=0.45, label=f"грунт {cfg.soil_depth:.0f} м "
                                                  f"(ρ = 1.6 г/см³)"),
    ] + hot_handles
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=7.5, framealpha=0.9)


def fig_geometry_3d(cfg: Cfg, x_true: np.ndarray, out_png: Path):
    fig = plt.figure(figsize=(14.5, 6.6), constrained_layout=True)
    ax1 = fig.add_subplot(121, projection="3d")
    _draw_scene(ax1, cfg, None,
                "Режим матрицы чувствительности:\nисточник поочерёдно во всех 25 ячейках")
    ax2 = fig.add_subplot(122, projection="3d")
    _draw_scene(ax2, cfg, x_true,
                "Модельный источник: 3 горячих пятна\n(объём ∝ активности)")
    fig.suptitle("Геометрия b1_soil: источник в грунте - детекторы на 1 м над землёй "
                 "(сетка 5×5, шаг 2 м)", fontsize=12)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 2. Интерактивная 3D-сцена (plotly -> HTML)
# ----------------------------------------------------------------------------
def _pl_sphere_traces(x0, y0, z0, r, color, opacity, name, hover, n=16,
                      showlegend=False, legendgroup=None):
    import plotly.graph_objects as go
    u = np.linspace(0, 2 * np.pi, 2 * n)
    v = np.linspace(0, np.pi, n)
    x = x0 + r * np.outer(np.cos(u), np.sin(v)).ravel()
    y = y0 + r * np.outer(np.sin(u), np.sin(v)).ravel()
    z = z0 + r * np.outer(np.ones_like(u), np.cos(v)).ravel()
    X, Y, Z = [], [], []
    nu, nv = len(u), len(v)
    idx = lambda i, j2: i * nv + j2
    for i in range(nu - 1):
        for j2 in range(nv - 1):
            X += [x[idx(i, j2)], x[idx(i + 1, j2)], x[idx(i + 1, j2 + 1)],
                  x[idx(i, j2)], x[idx(i + 1, j2 + 1)], x[idx(i, j2 + 1)]]
            Y += [y[idx(i, j2)], y[idx(i + 1, j2)], y[idx(i + 1, j2 + 1)],
                  y[idx(i, j2)], y[idx(i + 1, j2 + 1)], y[idx(i, j2 + 1)]]
            Z += [z[idx(i, j2)], z[idx(i + 1, j2)], z[idx(i + 1, j2 + 1)],
                  z[idx(i, j2)], z[idx(i + 1, j2 + 1)], z[idx(i, j2 + 1)]]
    return go.Mesh3d(x=X, y=Y, z=Z, color=color, opacity=opacity, name=name,
                     legendgroup=legendgroup, showlegend=showlegend,
                     hovertemplate=hover, lighting=dict(ambient=0.7, diffuse=0.4),
                     flatshading=True)


def fig_html_3d(cfg: Cfg, x_true: np.ndarray, out_html: Path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly не установлен - HTML-сцена пропущена")
        return
    fig = go.Figure()

    # грунт: бокс
    h, d = cfg.half, cfg.soil_depth
    fig.add_trace(go.Mesh3d(
        x=[-h, h, h, -h, -h, h, h, -h],
        y=[-h, -h, h, h, -h, -h, h, h],
        z=[0, 0, 0, 0, -d, -d, -d, -d],
        i=[0, 0, 4, 4, 0, 1, 1, 2, 2, 3, 3, 0],
        j=[1, 3, 5, 7, 4, 5, 2, 6, 7, 7, 6, 4],
        k=[2, 2, 6, 6, 5, 6, 6, 7, 3, 2, 2, 1],
        color=C_SOIL, opacity=0.28, name="грунт",
        hovertemplate="грунт: 10×10×2 м, ρ = 1.6 г/см³<extra></extra>",
        showlegend=True, flatshading=True))
    # поверхность земли (подсветка)
    fig.add_trace(go.Mesh3d(
        x=[-h, h, h, -h], y=[-h, -h, h, h], z=[0, 0, 0, 0],
        i=[0, 1, 2], j=[0, 2, 3], color=C_SOIL_TOP, opacity=0.85,
        name="поверхность земли", showlegend=True,
        hovertemplate="поверхность грунта<extra></extra>", flatshading=True))

    # источники
    sx, sy = map(np.array, zip(*[cfg.cell_xy(j) for j in range(cfg.n_cells)]))
    fig.add_trace(go.Scatter3d(
        x=sx, y=sy, z=np.full(cfg.n_cells, -cfg.src_depth), mode="markers",
        marker=dict(size=3.2, color=C_SRC), name="точки источников (25 ячеек)",
        hovertemplate="ячейка источника (%{x:.1f}, %{y:.1f}) м, глубина 10 см<extra></extra>"))

    # детекторы
    for i in range(cfg.n_cells):
        dx, dy = cfg.cell_xy(i)
        fig.add_trace(_pl_sphere_traces(
            dx, dy, cfg.det_height, cfg.det_radius, C_DET, 0.30,
            "детекторы (R = 15 см, h = 1 м)",
            "детектор (%{x:.2f}, %{y:.2f}) м<extra></extra>",
            n=10, showlegend=(i == 0), legendgroup="det"))

    # горячие пятна
    if x_true.max() > 0:
        vmax = x_true.max()
        for j in np.where(x_true > 0)[0]:
            hx, hy = cfg.cell_xy(j)
            r = 0.22 * (x_true[j] / vmax) ** (1 / 3) + 0.05
            fig.add_trace(_pl_sphere_traces(
                hx, hy, -cfg.src_depth, r, C_HOT, 0.95, "горячие пятна",
                f"активность {x_true[j]/1e9:.2f} ГБк<extra></extra>",
                n=18, showlegend=bool(j == np.where(x_true > 0)[0][0]), legendgroup="hot"))

    fig.update_layout(
        title="b1_soil: источники в грунте и детекторы на 1 м "
              "(сетка 5×5, шаг 2 м) — интерактивная модель",
        template="plotly_white", width=1000, height=720,
        legend=dict(itemsizing="constant", font=dict(size=11)),
        scene=dict(aspectmode="data",
                   xaxis_title="x, м", yaxis_title="y, м", zaxis_title="z, м",
                   camera=dict(eye=dict(x=1.7, y=-1.5, z=0.85))),
        margin=dict(l=0, r=0, t=56, b=0))
    fig.write_html(out_html, include_plotlyjs="cdn")


# ----------------------------------------------------------------------------
# 3. Разрезы с размерами
# ----------------------------------------------------------------------------
def fig_geometry_sections(cfg: Cfg, x_true: np.ndarray, out_png: Path):
    fig, (axz, axy) = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    h = cfg.half

    # --- вертикальный разрез XZ (y = 0) --------------------------------------
    axz.add_patch(Rectangle((-h, -cfg.soil_depth), 2 * h, cfg.soil_depth,
                            facecolor=C_SOIL, alpha=0.35, edgecolor=C_SOIL, lw=1.2))
    axz.axhline(0, color=C_SOIL_TOP, lw=2.4)
    axz.add_patch(Rectangle((-h, 0), 2 * h, cfg.det_height + 0.5,
                            facecolor=C_AIR, alpha=0.5, edgecolor="none"))
    # источники на разрезе (только центральный ряд y=0)
    for j in range(cfg.n_cells):
        sx, sy = cfg.cell_xy(j)
        if abs(sy) < 1e-9:
            axz.plot(sx, -cfg.src_depth, "o", color=C_SRC, ms=6, zorder=5)
    # горячие пятна
    vmax = x_true.max() if x_true.max() > 0 else 1.0
    for j in np.where(x_true > 0)[0]:
        sx, sy = cfg.cell_xy(j)
        if abs(sy) < 1e-9:
            axz.plot(sx, -cfg.src_depth, "o", color=C_HOT, ms=13, zorder=6)
    # детекторы центрального ряда
    for i in range(cfg.n_cells):
        dx, dy = cfg.cell_xy(i)
        if abs(dy) < 1e-9:
            axz.add_patch(Circle((dx, cfg.det_height), cfg.det_radius,
                                 facecolor="none", edgecolor=C_DET, lw=1.4, zorder=5))
    # размеры
    ann = dict(arrowstyle="<->", color="0.25", lw=1.0)
    axz.annotate("", (4 - 1, cfg.det_height + 0.32), (4 + 1, cfg.det_height + 0.32),
                 arrowprops=ann, fontsize=8)
    axz.text(0, cfg.det_height + 0.38, "шаг сетки 2 м", ha="center", fontsize=8.5)
    axz.annotate("", (5.45, -cfg.src_depth), (5.45, 0), arrowprops=ann)
    axz.text(5.6, -cfg.src_depth / 2, "глубина\n10 см", va="center", fontsize=8.5)
    axz.annotate("", (-5.45, 0), (-5.45, cfg.det_height), arrowprops=ann)
    axz.text(-5.6, cfg.det_height / 2, "высота\n1 м", va="center", ha="right", fontsize=8.5)
    axz.annotate("R = 15 см", xy=(-2, cfg.det_height + cfg.det_radius * 0.7),
                 xytext=(-3.6, cfg.det_height + 0.42), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9))
    axz.set_xlim(-6.6, 8.0)
    axz.set_ylim(-cfg.soil_depth - 0.35, cfg.det_height + 0.8)
    axz.set_aspect("equal")
    axz.set_xlabel("x, м")
    axz.set_ylabel("z, м")
    axz.set_title("Вертикальный разрез (y = 0): грунт 2 м, источники,\n"
                  "детекторы-сферы на 1 м", fontsize=10.5)
    axz.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=C_SRC, label="источник"),
        Line2D([], [], marker="o", ls="none", color=C_HOT, label="горячее пятно"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="none",
               markeredgecolor=C_DET, label="детектор"),
        Patch(facecolor=C_SOIL, alpha=0.4, label="грунт"),
        Patch(facecolor=C_AIR, alpha=0.6, label="воздух")],
        loc="lower left", fontsize=8, framealpha=0.9)

    # --- вид сверху XY ---------------------------------------------------------
    for j in range(cfg.n_cells):
        cx, cy = cfg.cell_xy(j)
        axy.add_patch(Rectangle((cx - 1, cy - 1), 2, 2,
                                facecolor=C_SOIL_TOP, alpha=0.12,
                                edgecolor=C_SOIL, lw=0.8))
        axy.plot(cx, cy, "o", color=C_SRC, ms=5)
    for j in np.where(x_true > 0)[0]:
        cx, cy = cfg.cell_xy(j)
        axy.plot(cx, cy, "o", color=C_HOT, ms=12)
        axy.annotate(f"{x_true[j]/1e8:.1f}e8 Бк", (cx, cy),
                     textcoords="offset points", xytext=(10, 8), fontsize=8.5,
                     color=C_HOT, fontweight="bold")
    for i in range(cfg.n_cells):
        dx, dy = cfg.cell_xy(i)
        axy.add_patch(Circle((dx, dy), cfg.det_radius, facecolor="none",
                             edgecolor=C_DET, lw=1.0, linestyle="--"))
    axy.set_xlim(-h - 1, h + 1)
    axy.set_ylim(-h - 1, h + 1)
    axy.set_aspect("equal")
    axy.set_xlabel("x, м")
    axy.set_ylabel("y, м")
    axy.set_title("Вид сверху: 25 ячеек источника (2×2 м),\n"
                  "детекторы-сферы (пунктир) и горячие пятна", fontsize=10.5)
    fig.suptitle("Геометрия b1_soil — разрезы", fontsize=12)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Матрица чувствительности
# ----------------------------------------------------------------------------
def fig_sensitivity(cfg: Cfg, A: np.ndarray, out_png: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    im = ax1.imshow(A, origin="upper", cmap="magma",  # матрица - не карта: ориентация условна
                    norm=mcolors.LogNorm(vmin=A[A > 0].min(), vmax=A.max()),
                    aspect="auto")
    fig.colorbar(im, ax=ax1, shrink=0.85, label="A[i,j], Зв/с на Бк (log)")
    ax1.set_xlabel("индекс ячейки-источника j")
    ax1.set_ylabel("индекс детектора i")
    ax1.set_title(f"Матрица чувствительности A ({cfg.nx*cfg.ny}×{cfg.nx*cfg.ny})\n"
                  f"cond(A) = {np.linalg.cond(A):.1f}, D4-симметризация", fontsize=10.5)
    # сетка блоков 5x5
    for k in range(4, 25, 5):
        ax1.axhline(k - 0.5, color="w", lw=0.4, alpha=0.35)
        ax1.axvline(k - 0.5, color="w", lw=0.4, alpha=0.35)

    # профили отклика: детекторы в ТОМ ЖЕ ряду y, что и источник (PSF по ряду)
    det_x = np.array([cfg.cell_xy(i)[0] for i in range(cfg.n_cells)])

    def row_of(j):
        iy = j // cfg.nx
        return np.arange(iy * cfg.nx, (iy + 1) * cfg.nx)

    for j, lbl, col in ((12, "источник в центре (0, 0)", C_DET),
                        (24, "источник в углу (+4, +4)", C_HOT),
                        (20, "источник в углу (-4, +4)", "#2ca02c")):
        idx = row_of(j)
        yy = np.where(A[idx, j] > 0, A[idx, j], np.nan)
        ax2.semilogy(det_x[idx], yy, "o-", color=col, ms=5, lw=1.1, label=lbl)
    ax2.set_xlabel("координата детектора x в ряду источника, м")
    ax2.set_ylabel("A[i,j], Зв/с на Бк (log)")
    ax2.set_title("Отклик детекторов в ряду источника (y = const):\n"
                  "функция отклика точки (PSF)", fontsize=10.5)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8.5)
    fig.suptitle("Матрица чувствительности «25 источников × 25 детекторов» "
                 "(H*(10) на распад, Cs-137)", fontsize=12)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Карты активности (2×3) + 6. 3D-бары
# ----------------------------------------------------------------------------
def _cell_grid(cfg: Cfg):
    xs = np.array([cfg.cell_xy(j)[0] for j in range(cfg.n_cells)])
    ys = np.array([cfg.cell_xy(j)[1] for j in range(cfg.n_cells)])
    ext = [xs.min() - 1, xs.max() + 1, ys.min() - 1, ys.max() + 1]
    return xs, ys, ext


def _annotate_hotspots(ax, x, cfg: Cfg):
    vmax = x.max()
    for j in np.where(x > 0.02 * vmax)[0]:
        cx, cy = cfg.cell_xy(j)
        # на светлых (горячих) ячейках тёмный текст, на тёмных - белый
        col = "0.1" if x[j] > 0.25 * vmax else "w"
        ax.text(cx, cy, f"{x[j]/1e9:.2f}\nГБк", ha="center", va="center",
                fontsize=7.5, color=col, fontweight="bold")


def fig_activity_maps(cfg: Cfg, x_true, x_mlem, x_tik, b, out_png: Path):
    xs, ys, ext = _cell_grid(cfg)
    shape = (cfg.ny, cfg.nx)
    maps = [("Истинный источник", x_true), ("MLEM", x_mlem), ("Тихонов (x ≥ 0)", x_tik)]
    vmax = max(m.max() for _, m in maps)
    vmin = max(min(m[m > 0].min() for _, m in maps if (m > 0).any()), vmax * 1e-4)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), constrained_layout=True)
    for k, (name, img) in enumerate(maps):
        ax = axes[0, k]
        im = ax.imshow(img.reshape(shape), origin="lower", extent=ext, cmap="hot",
                       norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
        for j in range(cfg.n_cells):
            cx, cy = cfg.cell_xy(j)
            ax.add_patch(Rectangle((cx - 1, cy - 1), 2, 2, fill=False,
                                   ec="0.5", lw=0.5, alpha=0.6))
        fig.colorbar(im, ax=ax, shrink=0.82, label="активность, Бк (log)")
        ax.plot(xs, ys, "o", ms=1.2, color="w", alpha=0.35)
        _annotate_hotspots(ax, img, cfg)
        ax.set_title(f"{name}  (Σ = {img.sum()/1e9:.3f} ГБк)", fontsize=10.5)
        ax.set_xlabel("x, м")
        ax.set_ylabel("y, м")

    # нижний ряд: относительные разности и измерения
    for k, (name, img) in enumerate(maps[1:]):
        ax = axes[1, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(x_true > 0, (img - x_true) / x_true, np.nan)
        cm = matplotlib.colormaps["RdBu_r"].copy()
        cm.set_bad("0.88")
        im = ax.imshow(rel.reshape(shape), origin="lower", extent=ext, cmap=cm,
                       vmin=-0.35, vmax=0.35)
        fig.colorbar(im, ax=ax, shrink=0.82, format=mticker.PercentFormatter(1.0, decimals=0),
                     label="относительная разность")
        ax.set_title(f"{name} − истина (серое = ячейки без источника)", fontsize=10)
        ax.set_xlabel("x, м")
        ax.set_ylabel("y, м")

    ax = axes[1, 2]
    im = ax.imshow((b * 3.6e9).reshape(shape), origin="lower", extent=ext,
                   cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.82, label="МАЭД, мкЗв/ч")
    ax.plot(xs, ys, "o", ms=1.5, color="w", alpha=0.4)
    ax.set_title("Измеренный вектор b\n(МАЭД на детекторах, 1 с)", fontsize=10)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")

    err_m = abs(x_mlem.sum() - x_true.sum()) / x_true.sum()
    err_t = abs(x_tik.sum() - x_true.sum()) / x_true.sum()
    cos_m = float(np.dot(x_mlem, x_true) /
                  (np.linalg.norm(x_mlem) * np.linalg.norm(x_true) + 1e-300))
    fig.suptitle(f"Реконструкция активности: истина {x_true.sum()/1e9:.2f} ГБк в 3 ячейках — "
                 f"MLEM: err Σ = {err_m:.2%}, cos = {cos_m:.4f} | "
                 f"Тихонов: err Σ = {err_t:.2%}", fontsize=12)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_activity_3d(cfg: Cfg, x_true, x_mlem, x_tik, out_png: Path):
    fig = plt.figure(figsize=(16.2, 5.6), constrained_layout=True)
    xs, ys, _ = _cell_grid(cfg)
    dz = 1.4
    cmap = matplotlib.colormaps["inferno"].copy()
    cmap.set_under("0.92")  # ячейки без активности - светло-серые
    vmax = max(x_true.max(), x_mlem.max(), x_tik.max())
    norm = mcolors.LogNorm(vmin=max(vmax * 1e-3, 1.0), vmax=vmax)

    for k, (name, img) in enumerate((("Истинный источник", x_true),
                                     ("MLEM", x_mlem),
                                     ("Тихонов (x ≥ 0)", x_tik))):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        heights = np.where(img > 0, img, vmax * 1e-4)
        cols = cmap(norm(heights))
        cols[img <= 0, 3] = 0.35  # полупрозрачные «пустые» ячейки
        ax.bar3d(xs - dz / 2, ys - dz / 2, 0, dz, dz, heights,
                 color=cols, shade=True, edgecolor="k", linewidth=0.15)
        ax.set_zlim(0, vmax * 1.12)
        ax.set_zticks([0, 1e8, 2e8])
        ax.set_zticklabels(["0", "1", "2"])
        ax.tick_params(axis="z", labelsize=8, pad=3)
        ax.set_xlabel("x, м", labelpad=2)
        ax.set_ylabel("y, м", labelpad=2)
        ax.set_zlabel("активность, ×10⁸ Бк", labelpad=6)
        ax.view_init(elev=28, azim=-58)
        ax.set_box_aspect((1, 1, 0.55))
        ax.set_title(f"{name}\nΣ = {img.sum()/1e9:.3f} ГБк", fontsize=10.5, pad=0)

    fig.suptitle("3D-модели распределения активности по ячейкам "
                 "(высота бара ∝ активности, лог. цвет)", fontsize=12)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Визуализация b1_soil")
    ap.add_argument("--A", default=str(BASE / "demo/A.npz"))
    ap.add_argument("--model", default=str(BASE / "demo/model.json"))
    ap.add_argument("--recon", default=str(BASE / "demo/out/reconstructed.npz"))
    ap.add_argument("--outdir", default=str(BASE / "demo/out"))
    ap.add_argument("--no-html", action="store_true", help="не строить интерактивный HTML")
    args = ap.parse_args()

    for p in (args.A, args.model):
        if not Path(p).exists():
            print(f"Нет файла {p} — сначала выполните build_sensitivity.py / демо")
            return 1
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(Path(args.A))
    x_true = load_xtrue(cfg, Path(args.model))

    A = np.load(args.A)["A"]
    fig_geometry_3d(cfg, x_true, outdir / "geometry_3d.png")
    print("ok  geometry_3d.png")
    fig_geometry_sections(cfg, x_true, outdir / "geometry_sections.png")
    print("ok  geometry_sections.png")
    fig_sensitivity(cfg, A, outdir / "sensitivity_matrix.png")
    print("ok  sensitivity_matrix.png")

    if not args.no_html:
        fig_html_3d(cfg, x_true, outdir / "geometry_3d.html")
        print("ok  geometry_3d.html (plotly)")

    if Path(args.recon).exists():
        rec = np.load(args.recon)
        fig_activity_maps(cfg, rec["x_true"], rec["x_mlem"], rec["x_tikhonov"],
                          rec["b"], outdir / "activity_maps.png")
        print("ok  activity_maps.png")
        fig_activity_3d(cfg, rec["x_true"], rec["x_mlem"], rec["x_tikhonov"],
                        outdir / "activity_3d.png")
        print("ok  activity_3d.png")
    else:
        print(f"--  {args.recon} не найден: карты активности пропущены "
              f"(запустите reconstruct.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
