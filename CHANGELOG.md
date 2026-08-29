# Changelog

## [0.5.0] - 2026-08-29
### Added
- **`soilactivity.spatial_interpolation`** — модуль пространственной
  интерполяции с 14 методами, автоподбором и анализом чувствительности.

  **`Interpolator2D`** — единый интерфейс для 14 backends:
  - RBF (scipy): `rbf_tps` (thin-plate spline, default), `rbf_linear`,
    `rbf_cubic`, `rbf_gaussian`
  - Delaunay (scipy griddata): `nearest`, `linear_delaunay`,
    `cubic_delaunay` (Clough-Tocher C1)
  - IDW (inverse distance weighting, k=12 nearest, power=2)
  - Метеорологические: `barnes` (последовательные поправки, kappa/
    iterations), `cressman` (радиус влияния)
  - Геостатистика: `kriging` (ordinary kriging, pykrige)
  - Gaussian Process (scikit-learn): `gp_rbf`, `gp_matern32`,
    `gp_matern52`
  - Методы с оценкой неопределённости: `gp_*` и `kriging`
    (`.uncertainty()` возвращает std)

  **`InterpolationAutoSelector`** — автоматический подбор лучшего метода:
  - k-fold CV (N >= 30) или leave-one-out (N < 30)
  - Метрики: RMSE, MAE, R², время
  - Критерий отбора: минимальный RMSE, тай-брейк — максимальный R²
  - `.select()` — лучший метод, `.get_ranking()` — все методы,
    `.plot_comparison()` — столбчатая диаграмма,
    `.get_recommendation()` — текстовая рекомендация

  **`MeasurementSensitivityAnalyzer`** — анализ влияния точек измерений
  (аналог `unfold_interpret` из bssunfold + pyoptexplain):
  - `sensitivity_leave_one_out()` — удаление каждой точки,
    пересчёт поля, `max_influence`, `mean_influence`,
    `influence_area_km2`
  - `sensitivity_perturbation(delta_frac)` — возмущение z[i],
    измерение изменения интерполяционного поля
  - `ranking()` — сортировка по влиянию (descending)
  - `critical_points(percentile)` — точки с влиянием выше порога
  - `influence_map(xi, yi)` — суммарная карта влияния
  - `plot_influence(xi, yi)` — heatmap + точки измерений

  **`SparseResultInterpolator`** — интерполяция разреженных результатов
  реконструкции на плотную сетку:
  - `.fit_sparse(points, values, uncertainty)` — обучить на N точках
  - `.interpolate_to_grid(xmin, xmax, ymin, ymax, nx, ny)` — плотная сетка
  - Возвращает `SparseResult`: `interpolated`, `uncertainty`,
    `confidence_mask`, `method_used`, `n_input_points`, `coverage`
  - Если GP + uncertainty → передаётся как noise prior (alpha)
  - `confidence_mask`: relative std < `uncertainty_threshold`

  **Standalone-функции:**
  - `idw_interpolate(x, y, z, xi, yi, power, max_neighbors)`
  - `barnes_interpolate(x, y, z, xi, yi, kappa, iterations)`
  - `cressman_interpolate(x, y, z, xi, yi, radius)`
  - `AVAILABLE_METHODS` — справочник 14 методов

- **`examples/example06_interpretation.ipynb`** (15 ячеек):
  - Каталог 14 методов (`AVAILABLE_METHODS`)
  - Синтетические данные МАЭД: 3 горячих пятна + фон, 100 точек,
    lognormal noise
  - Сравнение 6 методов интерполяции (RBF TPS, linear Delaunay,
    IDW, Barnes, Cressman, GP RBF) через `InterpolationAutoSelector`
  - Визуализация: 2×3 subplot сетка карт интерполяции
  - Столбчатая диаграмма RMSE + текстовая рекомендация
  - Leave-one-out анализ чувствительности: `ranking()`,
    `critical_points(90)`
  - Карта влияния (`influence_map`, `plot_influence`)
  - Интерполяция разреженных результатов (8 точек → 50×50 сетка)
    через `SparseResultInterpolator` с confidence mask

### Changed
- `soilactivity.__init__` экспортирует `Interpolator2D`,
  `InterpolationAutoSelector`, `SparseResultInterpolator`,
  `MeasurementSensitivityAnalyzer`, `idw_interpolate`,
  `barnes_interpolate`, `cressman_interpolate`, `AVAILABLE_METHODS`
- Версия → 0.5.0
- Обновлён README.md: полный раздел пространственной интерполяции
  с таблицей 14 методов, примерами для каждого класса, таблицей зависимостей
- Обновлены Sphinx docs: methods.rst, api.rst, examples.rst, index.rst

## [0.4.0] - 2026-08-29
### Added
- **Примеры реального применения** (examples/):
  - `example03_chernobyl.ipynb` (25 ячеек) — Чернобыльская зона отчуждения:
    Cs-137 и Sr-90 карты загрязнения, реконструкция МАЭД, решение уравнения
    Фредгольма для ПДА, сравнение гамма-спектрометрии и радиохимии,
    кривые Лоренца, коэффициент Джини, вертикальные профили по почвам.
    Данные по Kashparov V. et al. (2018, 2020) ESSD.
  - `example04_semei.ipynb` (28 ячеек) — Семипалатинский испытательный полигон:
    мультинуклидная модель (Cs-137, Sr-90, Co-60) для трёх площадок
    (Опытное поле, Балапан, Дегелен), Фредгольм-реконструкция,
    оценка доз для населённых пунктов.
    Данные по OSTI, PMC, IAEA INIS публикациям.
  - `example05_co60.ipynb` (22 ячейек) — Промышленное загрязнение Co-60:
    реконструкция ПДА с барьерами (здания), теневой эффект,
    сравнение с/без барьеров, анализ вклада Cs-137 и Co-60.
    Сценарий по данным Al Tuwaitha, Hanford, Plymouth Pilgrim.

- **Обновлён README.md**: добавлены разделы реконструкции ПДА,
  таблицы примеров (синтетические, реальные данные, bssunfold),
  пространственной статистики.

## [0.3.0] - 2026-08-27
### Added
- **`soilactivity.attenuation`** module: NIST XCOM mass attenuation (μ/ρ) and
  mass energy-absorption (μ_en/ρ) coefficients for all 92 elements (Z=1..92),
  525 energy points 1 keV - 20 MeV, with K-edge aware log-log interpolation.
  - `lookup_mu_rho(element, E_MeV)` — μ/ρ in cm²/g.
  - `lookup_mu_en_rho(element, E_MeV)` — μ_en/ρ in cm²/g, computed from partial
    cross sections using the Compton energy-transfer factor T(E) (Klein-Nishina,
    16-point Gauss-Legendre integration).
  - `mixture_mu_rho(composition, E)` and `mixture_mu_en_rho(composition, E)`
    — rule of mixtures for compounds and mixtures.
  - `linear_attenuation(composition, density, E)` — μ in cm⁻¹.
  - `validate_k_edges()` — returns K-edge energies and jump ratios (Pb K-edge
    at 88.004 keV, jump ~4×).
  - Built-in compositions: `NIST_AIR_DRY_COMPOSITION`, `NIST_WATER_COMPOSITION`,
    `NIST_CONCRETE_COMPOSITION`, `NIST_SOIL_COMPOSITION`, `NIST_TISSUE_SOFT_COMPOSITION`.

- **`soilactivity.dosimetry`** module: ICRP 74 photon dosimetry conversion
  coefficients.
  - `h_star_10_over_Ka(E)` — ambient dose equivalent / air kerma, Sv/Gy
    (ICRP 74 Table A.21, 25-point grid 10 keV - 10 MeV, log-log interp).
  - `kerma_per_fluence_air(E)` — air kerma per unit fluence, Ka/Φ (Gy·cm² per
    photon) via μ_en/ρ of air from the attenuation module.
  - `h_star_10_over_phil(E)` — combined fluence-to-ambient-dose-equivalent,
    h*(10)/Φ (Sv·cm² per photon).
  - `point_source_dose_rate(activity, gamma_lines, distance, ...)` — full
    point-source dose rate calculation including 1/r² geometry, optional
    shield attenuation, and optional ANS-6.4.3 buildup factor.

- **`soilactivity.data/nist_xcom_elements.json`** (1.7 MB) — NIST XCOM data for
  92 elements × 525 energies. Source: Dale-Black/XrayAttenuation.jl (Julia
  package with embedded Float64 arrays, CC-BY-4.0).
  - μ/ρ (total mass attenuation coefficient).
  - μ_en/ρ computed as `(μ_incoh/ρ)·T(E) + (μ_photo/ρ) + (μ_pair/ρ)·(1-2mc²/E)`
    where T(E) is the Klein-Nishina Compton energy-transfer factor.

- **`tests/test_attenuation.py`** (28 tests): spot-checks against NIST reference
  values for 12 elements × energies, K-edge jump validation for Pb/W/Cu, mixture
  rule validation for water/air/lead, μ_en/ρ ≤ μ/ρ invariant.

- **`tests/test_dosimetry.py`** (22 tests): ICRP 74 grid point verification,
  Cs-137 (662 keV) and Co-60 (1.17/1.33 MeV) dose rate validation against
  ISO 4037-3 gamma constants (Cs-137: ~121 μSv/h per GBq at 1 m, ref 124;
  Co-60: ~457 μSv/h per GBq at 1 m, ref ~357-380).

### Changed
- Bumped version to 0.3.0.
- Added `[project.optional-dependencies] physics = ["xraylib>=4.0.0"]` for
  users who want an alternative backend (not required; bundled JSON is the
  default).
- `soilactivity/__init__.py` now exports the full physics API: `lookup_mu_rho`,
  `mixture_mu_rho`, `linear_attenuation`, `h_star_10_over_Ka`,
  `kerma_per_fluence_air`, `h_star_10_over_phil`, `point_source_dose_rate`.

### Known Limitations
- μ_en/ρ is computed from partial cross sections using the Compton
  energy-transfer factor T(E); fluorescence escape and radiative loss
  corrections (g(E)) are ignored. Result: ±5% accuracy at 50 keV - 2 MeV,
  ~30% underestimate at E > 5 MeV due to missing pair-production corrections.
- K-edge discontinuities are handled by nearest-grid-point fallback (no
  sub-edge interpolation). The bundled XCOM grid does not include sub-edge
  points for the U K-edge at 115.6 keV (the source file omits this), so the
  U K-edge jump is not visible in the data.

## [0.2.0] - 2026-08-27
### Added
- **`soilactivity.buildup`** module: ANSI/ANS-6.4.3-1991 exposure buildup
  factors B(E, x) for 26 materials (23 elements + water/air/concrete).
  Source: Trubey 1988 (ORNL/RSIC-49), 4.1 MB PDF, 128 pages.
  - `get_buildup(material, E, x)` — table lookup with 2D log-log interpolation.
  - `gp_buildup_water(E, x)` — Geometric Progression (Harima) formula for
    water; validated against tabulated values to within 0.5-3 %.
  - `buildup_for_mixture(composition, E, x)` — rule of mixtures via Zeq.
- **`soilactivity.data`** package: bundled JSON data files.
- **`tests/test_buildup.py`**: 33 test cases.

## [0.1.0] - 2026-08-26
### Added
- Initial release of `soilactivity`.
- Core `Unfolder` class with MLEM and Tikhonov solvers.
- Numba-accelerated MLEM inner loop (inspired by `bssunfold`).
- Strict C-order indexing utilities (`index_to_coord`, `coord_to_index`).
- Sensitivity matrix analytical calculation and `.npy` loading.
- Kriging-based interpolation with optional Gaussian smoothing.
- `UnfoldingResult` dataclass with `.save_to_file()` method.
- Basic `pytest` suite including `test_against_bssunfold_style`.
