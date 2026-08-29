Available Radionuclides
=========================

SoilActivity ships with kerma constants :math:`K_\gamma` for the following
radionuclides. The kerma constant represents the air kerma rate at 1 m from
a unit-activity point source (neglecting scatter), in units of
aGy·m²·s⁻¹·Bq⁻¹.

.. list-table::
   :header-rows: 1
   :widths: 15 20 30 25
   :stub-columns: 0

   * - Radionuclide
     - :math:`K_\gamma` (aGy·m²/s/Bq)
     - Key Gamma Lines
     - Half-life
   * - Cs-137
     - 21.3
     - 661.7 keV (via ¹³⁷ᵐBa, 94.6%)
     - 30.2 y
   * - Cs-134
     - 57.6
     - 605, 796, 802 keV
     - 2.06 y
   * - Co-60
     - 137.0
     - 1173.2 + 1332.5 keV
     - 5.27 y
   * - Co-58
     - 59.1
     - 810.8 keV
     - 70.9 d
   * - Eu-152
     - 122.0
     - Multi-line (122–1408 keV)
     - 13.5 y
   * - Eu-154
     - 68.0
     - Multi-line (123–1596 keV)
     - 8.6 y
   * - I-131
     - 43.3
     - 364.5 keV (dominant)
     - 8.02 d
   * - Ba-140
     - 27.0
     - 537.3 keV
     - 12.8 d
   * - Zr-95
     - 27.4
     - 756.7, 724.2 keV
     - 64.0 d
   * - Nb-95
     - 28.3
     - 765.8 keV
     - 34.9 d
   * - Ru-103
     - 17.7
     - 497.1 keV
     - 39.3 d
   * - Ru-106
     - 7.5
     - 511.9 keV (via Rh-106)
     - 373.6 d
   * - Ce-141
     - 2.9
     - 145.4 keV
     - 32.5 d
   * - Ce-144
     - 1.0
     - 133.5 keV (via Pr-144)
     - 284.9 d
   * - La-140
     - 69.0
     - 1596.5 keV
     - 1.68 d
   * - Mn-54
     - 147.0
     - 834.8 keV
     - 312.1 d
   * - Fe-59
     - 133.0
     - 1099.3, 1291.6 keV
     - 44.5 d
   * - Zn-65
     - 75.0
     - 1115.5 keV
     - 244.3 d
   * - Sb-124
     - 187.0
     - Multi-line (602–1691 keV)
     - 60.2 d
   * - Am-241
     - 22.3
     - 59.5 keV
     - 432.6 y
   * - Sr-90
     - 0.0
     - Pure beta (no gamma)
     - 28.8 y
   * - Y-90
     - 0.0
     - Pure beta (no gamma)
     - 2.67 d

Radionuclide Mixtures
---------------------

The package also supports activity-weighted mixtures via
:func:`~soilactivity.radionuclides.mixture_kerma_constant`. A pre-defined
Chernobyl Unit 4 fuel composition at 131 days post-accident is available as
:data:`~soilactivity.radionuclides.CHERNOBYL_FUEL_VECTOR_131D`.

The Chernobyl fuel vector includes Nb-95, Zr-95, Ru-103, Ru-106, Cs-134,
Cs-137, Ce-141, and Ce-144 with their activity fractions, kerma constants,
and specific air kerma rates (SAKR).

Data Sources
------------

- Mashkovich V, Kudryavtseva A (1995) *Protection from Ionizing Radiation*.
  Energoatomizdat, Moscow.
- Ninkovic M, Adrovic F (2012) Air Kerma Rate Constants for Nuclides
  Important to Gamma Ray Dosimetry. DOI: 10.5772/39170.
- Jacob P et al (1990) GSF-2/90. Calculation of organ doses from
  environmental gamma-rays using human phantoms and Monte Carlo methods.
- ICRP Publication 74 (1996) Conversion Coefficients for use in
  Radiological Protection against External Radiation.

Usage Example
-------------

::

    from soilactivity import KERMA_CONSTANTS, mixture_kerma_constant

    # Single radionuclide
    print(KERMA_CONSTANTS["Cs-137"])  # 21.3

    # Mixture (Chernobyl fuel at day 131)
    K_mix = mixture_kerma_constant()
    print(K_mix)  # ~14.6 aGy m^2 s^-1 Bq^-1
