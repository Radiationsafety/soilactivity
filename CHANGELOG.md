# Changelog

## [0.2.0] - 2026-08-27
### Added
- **`soilactivity.buildup`** module: ANSI/ANS-6.4.3-1991 exposure buildup
  factors B(E, x) for 26 materials (23 elements + water/air/concrete).
  Source: Trubey 1988 (ORNL/RSIC-49), 4.1 MB PDF, 128 pages.
  - `get_buildup(material, E, x)` — table lookup with 2D log-log interpolation.
  - `gp_buildup_water(E, x)` — Geometric Progression (Harima) formula for
    water; validated against tabulated values to within 0.5-3 % across
    the full energy/depth grid (0.015-15 MeV, 0-40 mfp).
  - `buildup_for_mixture(composition, E, x)` — rule of mixtures via the
    equivalent atomic number Zeq (standard ANS-6.4.3 procedure).
- **`soilactivity.data`** package: bundled JSON data files
  - `buildup_factors_ans643.json` (191 KB) — full B(E,x) tables.
  - `gp_coefficients_water_ans643.json` (5 KB) — GP fitting coefficients
    for water (water-kerma + air-kerma response).
- **`tests/test_buildup.py`**: 30+ test cases including reference-value
  spot-checks against published ANS-6.4.3 tables and GP-vs-table cross-check.
- Energy grid: 25 points (0.015-15 MeV), depth grid: 16 points (0.5-40 mfp).
- Total coverage: 26 materials × 25 energies × 16 depths = 10400 cells,
  0 missing values (extrapolated via 2D log-log where the source PDF omits
  low-energy cells for Pb and U).

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
