# Changelog

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
