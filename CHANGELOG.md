# Changelog

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
